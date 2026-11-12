"""Embedding bằng Gemini API, có retry khi bị giới hạn tốc độ và cache vector ra đĩa.

Vị trí trong pipeline:
    chunk_store.py (ChunkRecord, embed_input)  ->  embedder.py (vector)  ->  vector_store.py (Chroma)

Quyết định thiết kế:
  * Model: `gemini-embedding-001` (xem CLAUDE.md, mục "Gemini embedding"): nhận batch, trả đúng 1 vector cho
    mỗi văn bản, có task_type. `gemini-embedding-2` KHÔNG dùng ở đây vì nó gộp cả list thành 1 vector.
  * task_type: chunk dùng "RETRIEVAL_DOCUMENT", câu hỏi dùng "RETRIEVAL_QUERY" (bài toán bất đối xứng).
  * Cache: mỗi chunk lưu (chunk_id, sha256 của chuỗi đã embed, vector). Chạy lại chỉ gọi API cho chunk MỚI hoặc
    chunk có nội dung đổi (hash khác). Cache được ghi sau MỖI batch, nên bị ngắt giữa chừng cũng không mất công.
    Lý do: gọi API tốn quota (free tier có giới hạn token/phút và request/ngày) và tốn thời gian.
  * Vector được chuẩn hoá về độ dài 1 và lưu float32 (3072 số x 4 byte = 12KB mỗi vector).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from google import genai
from google.genai import errors, types

from src.config import EMBED_CACHE_DIR, require_api_key

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 3072
DOC_TASK = "RETRIEVAL_DOCUMENT"  # cho chunk được lưu
QUERY_TASK = "RETRIEVAL_QUERY"  # cho câu hỏi của người dùng

# Gọi API: mỗi lần gửi tối đa BATCH_SIZE văn bản. 32 chunk x ~260 token ≈ 8k token mỗi lần: đủ nhỏ để không
# vướng giới hạn token/phút, đủ lớn để không phải gọi quá nhiều lần.
BATCH_SIZE = 32
MAX_RETRIES = 6  # số lần thử lại khi lỗi tạm thời (429 quá quota, 5xx quá tải, timeout)
BASE_BACKOFF_S = 15.0  # chờ 15s, 30s, 60s, ... (nhân đôi mỗi lần, tối đa 120s)


def make_client() -> genai.Client:
    """Client với timeout 60s và TẮT retry ngầm của SDK (attempts=1): SDK retry rất chậm và không phân biệt hạn mức theo
    ngày; việc thử lại do embed_texts / RagPipeline._generate tự làm (có backoff, và báo lỗi ngay khi hết hạn mức ngày)."""
    return genai.Client(
        api_key=require_api_key(),
        http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)),
    )


def input_sha(text: str) -> str:
    """Dấu vân tay của chuỗi đã embed. Chuỗi đổi (sửa chunker, đổi header...) -> hash đổi -> embed lại."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_daily_quota(exc: Exception) -> bool:
    """429 do hết hạn mức THEO NGÀY (vd 'Embed content requests per day limit exceeded'): thử lại vài giây/phút vô ích."""
    s = str(exc)
    return "PerDay" in s or "per day" in s


def is_retryable(exc: Exception) -> bool:
    """Lỗi tạm thời đáng thử lại: 429 (quá giới hạn theo phút), 5xx (server quá tải) hoặc timeout mạng.
    Hết hạn mức theo NGÀY thì KHÔNG thử lại: báo lỗi ngay (hạn mức hồi sau nhiều giờ)."""
    if isinstance(exc, errors.APIError):
        if exc.code == 429 and is_daily_quota(exc):
            return False
        return exc.code == 429 or (exc.code is not None and exc.code >= 500)
    # Timeout/ngắt kết nối của thư viện HTTP bên dưới (httpx) không phải APIError.
    return type(exc).__name__ in {"ReadTimeout", "ConnectTimeout", "TimeoutException", "ConnectError", "RemoteProtocolError"}


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    model: str = EMBED_MODEL,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> np.ndarray:
    """Embed MỘT batch văn bản -> ma trận (n, 3072) đã chuẩn hoá. Tự thử lại khi lỗi tạm thời.

    Chốt chặn quan trọng: số vector trả về phải bằng số văn bản gửi đi. Nếu lệch (vd dùng nhầm model gộp list),
    báo lỗi ngay thay vì lưu nhầm vector cho chunk khác mà không ai biết.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.models.embed_content(
                model=model, contents=texts, config=types.EmbedContentConfig(task_type=task_type)
            )
            break
        except Exception as exc:  # noqa: BLE001 - phân loại ngay bên dưới, lỗi không retry được sẽ raise lại
            if attempt == MAX_RETRIES or not is_retryable(exc):
                raise
            wait = min(BASE_BACKOFF_S * 2**attempt, 120.0)
            if on_retry:
                on_retry(attempt + 1, exc, wait)
            time.sleep(wait)
    if len(resp.embeddings) != len(texts):
        raise RuntimeError(f"{model}: gửi {len(texts)} văn bản nhưng nhận {len(resp.embeddings)} vector")
    mat = np.array([e.values for e in resp.embeddings], dtype=np.float64)
    return (mat / np.linalg.norm(mat, axis=1, keepdims=True)).astype(np.float32)


def embed_query_cached(client: genai.Client, question: str, world_no: int = 1, model: str = EMBED_MODEL) -> np.ndarray:
    """Vector của MỘT câu hỏi (RETRIEVAL_QUERY), có cache theo NỘI DUNG câu hỏi trong file queries.

    Tại sao cần: gói miễn phí giới hạn số yêu cầu embedding mỗi NGÀY (đã chạm 1000/ngày khi chạy lặp bộ câu hỏi), nên
    một câu hỏi đã embed rồi thì không được gọi API lần nữa. Tra theo hash nội dung nên dùng được cả vector do
    embed_with_cache (khóa là mã câu như 'Q01') đã lưu; câu mới lưu dưới khóa 'q:<hash>'.
    """
    path = cache_path(world_no, "queries", model)
    meta = cache_meta(model, QUERY_TASK)
    cache = load_cache(path, expect_meta=meta)
    sha = input_sha(question)
    for stored_sha, vec in cache.values():
        if stored_sha == sha:
            return vec
    vec = embed_texts(client, [question], QUERY_TASK, model)[0]
    cache[f"q:{sha[:16]}"] = (sha, vec)
    save_cache(path, cache, meta)
    return vec


# ---------------------------------------------------------------------------
# Cache vector trên đĩa
# ---------------------------------------------------------------------------


def cache_path(world_no: int, variant: str, model: str = EMBED_MODEL, cache_dir: Path = EMBED_CACHE_DIR) -> Path:
    """vd data/embeddings/w01_header_gemini-embedding-001.npz  (variant: "plain" hoặc "header")."""
    return cache_dir / f"w{world_no:02d}_{variant}_{model}.npz"


def cache_meta(model: str, task_type: str, dims: int = EMBED_DIMS) -> dict:
    """Phần còn lại của khóa cache ngoài nội dung text: cùng một text nhưng khác model / task_type / số chiều
    là vector KHÁC NHAU. Meta được ghi một lần cho cả file (mỗi file một cấu hình) thay vì lặp lại ở từng record."""
    return {"model": model, "task_type": task_type, "dims": dims}


def load_meta(path: Path) -> dict | None:
    """Đọc meta ghi kèm file cache. File chưa migrate (định dạng cũ, không có meta) hoặc chưa tồn tại -> None."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])) if "meta" in z.files else None


def load_cache(path: Path, expect_meta: dict | None = None) -> dict[str, tuple[str, np.ndarray]]:
    """Đọc cache -> {khóa: (sha256 chuỗi đã embed, vector)}. Chưa có file thì trả dict rỗng.

    Truyền `expect_meta` (từ cache_meta) để chặn lỗi im lặng khó thấy nhất: trỏ nhầm vào file embed bằng
    cấu hình khác và lặng lẽ dùng vector sai. Meta lệch hoặc file chưa migrate -> báo lỗi ngay.
    """
    if not path.exists():
        return {}
    if expect_meta is not None:
        meta = load_meta(path)
        if meta is None:
            raise RuntimeError(
                f"{path.name}: file cache định dạng cũ chưa có meta. Chạy: python scripts/v1_step10a_migrate_cache.py"
            )
        if meta != expect_meta:
            raise RuntimeError(f"{path.name}: meta trong file {meta} khác cấu hình đang dùng {expect_meta}.")
    with np.load(path, allow_pickle=False) as z:
        return {str(i): (str(s), v) for i, s, v in zip(z["ids"], z["shas"], z["vecs"])}


def save_cache(path: Path, entries: dict[str, tuple[str, np.ndarray]], meta: dict | None = None) -> None:
    """Ghi cache (ghi file tạm rồi đổi tên, để lỡ bị ngắt giữa chừng cũng không hỏng file cũ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(entries)
    tmp = path.with_name(path.stem + ".tmp.npz")
    arrays: dict = dict(
        ids=np.array(ids),
        shas=np.array([entries[i][0] for i in ids]),
        vecs=np.stack([entries[i][1] for i in ids]) if ids else np.zeros((0, 0), dtype=np.float32),
    )
    if meta is not None:
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(tmp, **arrays)
    tmp.replace(path)


def embed_with_cache(
    client: genai.Client,
    items: list[tuple[str, str]],
    cache_file: Path,
    task_type: str = DOC_TASK,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, dict[str, int]]:
    """Embed danh sách (chunk_id, chuỗi_đưa_vào_embedding), dùng cache khi có.

    Trả về (ma trận vector theo đúng thứ tự `items`, thống kê {"cached": .., "api": .., "batches": ..}).
    """
    meta = cache_meta(model, task_type)
    cache = load_cache(cache_file, expect_meta=meta)
    shas = {cid: input_sha(txt) for cid, txt in items}
    # Chunk cần gọi API: chưa có trong cache, hoặc có nhưng chuỗi đã embed khác (hash lệch).
    todo = [(cid, txt) for cid, txt in items if cid not in cache or cache[cid][0] != shas[cid]]
    stats = {"cached": len(items) - len(todo), "api": len(todo), "batches": 0}

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        t0 = time.time()
        vecs = embed_texts(
            client,
            [txt for _, txt in batch],
            task_type,
            model,
            on_retry=lambda n, exc, wait: log(f"    ! lần thử {n}: {type(exc).__name__}, chờ {wait:.0f}s rồi thử lại"),
        )
        for (cid, _), v in zip(batch, vecs):
            cache[cid] = (shas[cid], v)
        save_cache(cache_file, cache, meta)  # ghi ngay sau mỗi batch
        stats["batches"] += 1
        log(f"    batch {stats['batches']}: {len(batch)} chunk trong {time.time() - t0:.1f}s "
            f"({start + len(batch)}/{len(todo)})")

    return np.stack([cache[cid][1] for cid, _ in items]), stats
