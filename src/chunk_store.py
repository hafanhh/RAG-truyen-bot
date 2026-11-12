"""Chunk store: bổ sung metadata cho chunk (liên kết prev/next, header ngữ cảnh, token_count) và ghi/đọc JSONL.

Vị trí trong pipeline:
    loader.py (World/Chapter)  ->  chunker.py (Chunk)  ->  chunk_store.py (ChunkRecord -> file .jsonl)  ->  V1: embedding

Quyết định thiết kế quan trọng:
  * `context_header` được LƯU RIÊNG, không trộn vào `text`. Nhờ vậy ở V1 ta có thể thử embedding CÓ header
    và KHÔNG header để so sánh, mà không phải cắt lại chunk.
  * `token_count` được đếm trên ĐÚNG chuỗi sẽ đưa vào embedding model, tức `embed_input(header, text)`,
    bằng đúng tokenizer của model đó (BAAI/bge-m3), CÓ tính token đặc biệt, và KHÔNG tự cắt ngắn
    (để phát hiện đầu vào vượt giới hạn thay vì bị cắt lặng lẽ).
  * `prev_id/next_id` nối chunk theo thứ tự đọc truyện của CẢ thế giới (kể cả qua ranh giới chương),
    để sau này mở rộng ngữ cảnh khi retrieve. Chunk đầu có prev_id=None, chunk cuối có next_id=None.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from src.chunker import Chunk
from src.config import CHUNKS_DIR

# Tokenizer/model dùng để đếm token và (ở V1) để embedding. Ghi rõ tên để mỗi con số token_count truy được nguồn.
TOKENIZER_NAME = "BAAI/bge-m3"
# Giới hạn độ dài đầu vào của bge-m3 (đã xác nhận: tokenizer.model_max_length == 8192).
MODEL_MAX_TOKENS = 8192
# Chuỗi nối giữa header và text khi đưa vào embedding. Dùng CHUNG một hằng số cho cả đếm token lẫn embedding thật.
EMBED_SEPARATOR = "\n\n"


@dataclass
class ChunkRecord:
    """Một dòng trong file JSONL: chunk + toàn bộ metadata cần cho retrieval, citation và kiểm tra."""

    chunk_id: str  # vd "w01-c012-k02"
    world_no: int
    world_title: str
    chapter_no: int
    chunk_index: int  # thứ tự trong chương, từ 0
    para_start: int  # đoạn đầu trong chương (1-based, dùng cho citation)
    para_end: int  # đoạn cuối (1-based, bao gồm)
    prev_id: str | None  # chunk liền trước trong cả thế giới (None nếu là chunk đầu)
    next_id: str | None  # chunk liền sau (None nếu là chunk cuối)
    context_header: str  # vd "Thế giới 1: Phú thiếu bạn gái cũ | Chương 12 | Đoạn 3-7"
    text: str  # nội dung chunk, KHÔNG gồm header
    n_chars: int
    token_count: int  # số token của embed_input(context_header, text), có special tokens, không cắt ngắn
    tokenizer: str  # tokenizer đã dùng để ra token_count


# ---------------------------------------------------------------------------
# Header ngữ cảnh và chuỗi đưa vào embedding
# ---------------------------------------------------------------------------


def make_context_header(world_no: int, world_title: str, chapter_no: int, para_start: int, para_end: int) -> str:
    """Header ngắn cho biết chunk nằm ở đâu trong truyện (cũng là mẫu cho citation ở V2)."""
    return f"Thế giới {world_no}: {world_title} | Chương {chapter_no} | Đoạn {para_start}-{para_end}"


def embed_input(context_header: str, text: str) -> str:
    """Chuỗi ĐÚNG như sẽ đưa vào embedding model (header + text). Mọi phép đếm token phải dùng hàm này."""
    return f"{context_header}{EMBED_SEPARATOR}{text}"


# ---------------------------------------------------------------------------
# Đếm token
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_tokenizer():
    """Tải tokenizer của bge-m3 (chỉ file tokenizer nhỏ, không tải trọng số model). lru_cache: chỉ tải một lần."""
    from transformers import AutoTokenizer  # import muộn để các module khác không phải nạp thư viện nặng

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(s: str, add_special_tokens: bool = True) -> int:
    """Đếm token của chuỗi `s` bằng tokenizer bge-m3.

    add_special_tokens=True: tính cả <s> và </s> mà model thật sự nhận (mặc định, dùng cho token_count).
    truncation=False: KHÔNG cắt ngắn, để thấy độ dài thật và so với MODEL_MAX_TOKENS.
    """
    return len(get_tokenizer()(s, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])


# ---------------------------------------------------------------------------
# Dựng record
# ---------------------------------------------------------------------------


def build_records(chunks: list[Chunk]) -> list[ChunkRecord]:
    """Chuyển list Chunk (đã theo thứ tự đọc truyện) thành list ChunkRecord đủ metadata."""
    records = []
    for i, c in enumerate(chunks):
        header = make_context_header(c.world_no, c.world_title, c.chapter_no, c.para_start, c.para_end)
        records.append(
            ChunkRecord(
                chunk_id=c.chunk_id,
                world_no=c.world_no,
                world_title=c.world_title,
                chapter_no=c.chapter_no,
                chunk_index=c.chunk_index,
                para_start=c.para_start,
                para_end=c.para_end,
                # Liên kết theo vị trí trong danh sách: chunk trước/sau, None ở hai đầu.
                prev_id=chunks[i - 1].chunk_id if i > 0 else None,
                next_id=chunks[i + 1].chunk_id if i + 1 < len(chunks) else None,
                context_header=header,
                text=c.text,
                n_chars=c.n_chars,
                token_count=count_tokens(embed_input(header, c.text)),
                tokenizer=TOKENIZER_NAME,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Đường dẫn, ghi/đọc file
# ---------------------------------------------------------------------------


def chunks_path(world_no: int, out_dir: Path = CHUNKS_DIR) -> Path:
    return out_dir / f"chunks_w{world_no:02d}.jsonl"


def manifest_path(world_no: int, out_dir: Path = CHUNKS_DIR) -> Path:
    return out_dir / f"chunks_w{world_no:02d}.manifest.json"


def sha256_file(path: Path) -> str:
    """Dấu vân tay của file nguồn: nếu file truyện bị sửa sau khi cắt chunk, hash sẽ khác -> phát hiện được."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(records: list[ChunkRecord], path: Path) -> None:
    """Ghi mỗi record thành một dòng JSON. ensure_ascii=False để tiếng Việt đọc được bằng mắt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[ChunkRecord]:
    """Đọc lại file JSONL thành list ChunkRecord (bỏ qua dòng trống)."""
    with Path(path).open(encoding="utf-8") as f:
        return [ChunkRecord(**json.loads(line)) for line in f if line.strip()]


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
