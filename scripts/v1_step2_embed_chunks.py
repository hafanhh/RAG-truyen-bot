"""V1 - Bước 2: Embed các chunk bằng Gemini và lưu vào Chroma (2 collection: plain và header).

Làm gì:
  1. Đọc data/chunks/chunks_w01.jsonl (393 chunk, kết quả Step 5).
  2. Với mỗi biến thể, tạo chuỗi đưa vào embedding rồi gọi Gemini (gemini-embedding-001, task RETRIEVAL_DOCUMENT):
        plain  : chỉ `text`
        header : `context_header + "\\n\\n" + text`  (embed_input trong chunk_store.py)
     Mục đích A/B: header ("Thế giới 1 ... | Chương 12 | Đoạn 3-7") có giúp retrieval không? Đo ở Bước 3 và 6.
  3. Lưu vector vào cache trên đĩa (data/embeddings/) rồi ghi vào Chroma (data/chroma/), mỗi biến thể một
     collection (w01_plain, w01_header), khoảng cách cosine.
  4. Tự kiểm tra: đếm số bản ghi, số chiều, vector trong Chroma khớp cache, mỗi chunk tự truy vấn ra chính nó.

An toàn khi chạy lại: có cache (chỉ gọi API cho chunk mới/đổi) và dùng upsert (không bị trùng).

Chạy từ thư mục gốc:
    python scripts/v1_step2_embed_chunks.py --limit 10     # chạy thử 10 chunk đầu (xem tốc độ, quota)
    python scripts/v1_step2_embed_chunks.py                # chạy đủ toàn bộ chunk (phần đã có trong cache được bỏ qua)
    python scripts/v1_step2_embed_chunks.py --reset        # xoá 2 collection trong Chroma rồi ghi lại (cache vẫn giữ)
LƯU Ý: script này GỬI văn bản chunk của World 01 lên Gemini API (đã được bạn đồng ý ngày 2026-11-11).
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.chunk_store import chunks_path, embed_input, read_jsonl  # noqa: E402
from src.config import CHROMA_DIR, ROOT  # noqa: E402
from src.embedder import DOC_TASK, EMBED_MODEL, cache_path, embed_with_cache, load_cache, make_client  # noqa: E402
from src.vector_store import collection_name, get_client, get_collection, upsert_records  # noqa: E402

# Cách dựng chuỗi đưa vào embedding cho từng biến thể.
VARIANTS = {
    "plain": lambda r: r.text,
    "header": lambda r: embed_input(r.context_header, r.text),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="chỉ embed N chunk đầu (để chạy thử)")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--reset", action="store_true", help="xoá collection trong Chroma trước khi ghi (cache giữ nguyên)")
    args = ap.parse_args()

    records = read_jsonl(chunks_path(args.world))
    if args.limit:
        records = records[: args.limit]
    print(f"Thế giới {args.world:02d}: {len(records)} chunk | model {EMBED_MODEL} | task {DOC_TASK}")
    print(f"Chroma: {CHROMA_DIR.relative_to(ROOT)}\n")

    client = make_client()
    chroma = get_client()
    t_all = time.time()
    vectors = {}  # variant -> ma trận (n, dim), dùng cho bước kiểm tra cuối

    for variant in args.variants:
        name = collection_name(args.world, variant)
        print(f"=== Biến thể '{variant}' -> collection '{name}' ===")
        if args.reset:
            try:
                chroma.delete_collection(name)
                print("  đã xoá collection cũ")
            except Exception:  # noqa: BLE001 - chưa tồn tại thì thôi
                pass

        items = [(r.chunk_id, VARIANTS[variant](r)) for r in records]
        t0 = time.time()
        vecs, st = embed_with_cache(client, items, cache_path(args.world, variant), log=lambda s: print(s))
        print(f"  embedding: {st['cached']} lấy từ cache, {st['api']} gọi API ({st['batches']} batch) "
              f"trong {time.time() - t0:.1f}s | vector {vecs.shape}")
        vectors[variant] = vecs

        col = get_collection(chroma, args.world, variant)
        upsert_records(col, records, vecs)
        print(f"  Chroma '{name}': {col.count()} bản ghi\n")

    # ------------------------------------------------------------------ kiểm tra
    print("=== Tự kiểm tra ===")
    ids = [r.chunk_id for r in records]
    rng = random.Random(42)
    sample = rng.sample(range(len(records)), k=min(5, len(records)))
    ok_all = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {'✓' if cond else '✗'} {msg}")

    for variant in args.variants:
        col = get_collection(chroma, args.world, variant)
        vecs = vectors[variant]
        in_chroma = set(col.get(ids=ids, include=[])["ids"])
        check(in_chroma == set(ids), f"[{variant}] cả {len(ids)} chunk đều có trong Chroma (Chroma đang có tổng {col.count()})")
        check(vecs.shape[1] == 3072 and np.isfinite(vecs).all(), f"[{variant}] vector {vecs.shape[1]} chiều, không có NaN/inf")
        check(np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4), f"[{variant}] mọi vector có độ dài ≈ 1")

        # Vector lưu trong Chroma phải khớp vector trong cache; mỗi chunk tự truy vấn phải ra chính nó.
        stored = col.get(ids=[ids[i] for i in sample], include=["embeddings"])
        by_id = {i: np.array(e) for i, e in zip(stored["ids"], stored["embeddings"])}
        match = all(np.allclose(by_id[ids[i]], vecs[i], atol=1e-5) for i in sample)
        check(match, f"[{variant}] vector trong Chroma khớp cache ({len(sample)} chunk ngẫu nhiên)")
        selfhit = []
        for i in sample:
            res = col.query(query_embeddings=[vecs[i].tolist()], n_results=1, include=["distances"])
            selfhit.append(res["ids"][0][0] == ids[i] and res["distances"][0][0] < 1e-3)
        check(all(selfhit), f"[{variant}] truy vấn bằng vector của chính chunk -> trả về đúng chunk đó, distance ≈ 0")

        cached = load_cache(cache_path(args.world, variant))
        check(all(i in cached for i in ids), f"[{variant}] cache trên đĩa có đủ {len(ids)} chunk")

    if len(vectors) == 2:
        # Header làm vector thay đổi bao nhiêu? cosine giữa vector plain và header của CÙNG một chunk.
        cos = np.sum(vectors["plain"] * vectors["header"], axis=1)
        print(f"\n  Cosine giữa vector plain và header của cùng một chunk: "
              f"min {cos.min():.3f} | trung bình {cos.mean():.3f} | max {cos.max():.3f}")
        print("  (càng thấp = header càng làm vector khác đi)")

    print(f"\nTổng thời gian: {time.time() - t_all:.1f}s | " + ("TẤT CẢ ĐẠT" if ok_all else "CÓ KIỂM TRA THẤT BẠI"))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
