"""Step 5 - Dựng metadata cho chunk và ghi ra data/chunks/ (JSONL + manifest).

Sinh 2 file:
    data/chunks/chunks_w01.jsonl          mỗi dòng một chunk + metadata (xem ChunkRecord trong src/chunk_store.py)
    data/chunks/chunks_w01.manifest.json  tham số đã dùng, tokenizer, hash file nguồn... để truy vết/kiểm tra

Chạy từ thư mục gốc:
    python scripts/step5_build_chunks.py
    python scripts/step5_build_chunks.py --size 800 --overlap 150   # thử tham số khác (sẽ ghi đè file)
Sau đó kiểm tra bằng:  python scripts/step5_verify_chunks.py
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_store import (  # noqa: E402
    MODEL_MAX_TOKENS,
    TOKENIZER_NAME,
    EMBED_SEPARATOR,
    build_records,
    chunks_path,
    count_tokens,
    get_tokenizer,
    manifest_path,
    sha256_file,
    write_jsonl,
    write_manifest,
)
from src.chunker import chunk_world  # noqa: E402
from src.config import ROOT  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=250)
    args = ap.parse_args()

    src_file = find_world_file(args.world)
    world = load_world(src_file)
    chunks = chunk_world(world, size=args.size, overlap=args.overlap, min_chars=args.min_chars)
    print(f"Thế giới {world.world_no:02d}: {world.title} | {len(world.chapters)} chương -> {len(chunks)} chunk "
          f"(size={args.size}, overlap={args.overlap}, min_chars={args.min_chars})")

    print(f"Đang đếm token bằng tokenizer {TOKENIZER_NAME} ...")
    records = build_records(chunks)

    # Token đặc biệt mà tokenizer tự thêm: đo bằng cách đếm cùng một chuỗi có/không có special tokens.
    sample = records[0].context_header
    special_added = count_tokens(sample, True) - count_tokens(sample, False)

    toks = [r.token_count for r in records]
    over = [r.chunk_id for r in records if r.token_count > MODEL_MAX_TOKENS]
    print(f"token_count (header + text, đã gồm {special_added} special tokens, không cắt ngắn):")
    print(f"  min {min(toks)} | trung bình {sum(toks) // len(toks)} | max {max(toks)} | giới hạn model {MODEL_MAX_TOKENS}")
    print(f"  chunk vượt giới hạn model: {len(over)}")
    chars = [r.n_chars for r in records]
    print(f"  ký tự/token trung bình: {sum(chars) / sum(toks):.2f}  (số này cho biết văn bản convert này 'tốn token' cỡ nào)")

    out = chunks_path(world.world_no)
    write_jsonl(records, out)
    write_manifest(
        {
            "world_no": world.world_no,
            "world_title": world.title,
            "source_file": str(src_file.relative_to(ROOT)),
            "source_sha256": sha256_file(src_file),
            "chunker": {"size_chars": args.size, "overlap_chars": args.overlap, "min_chars": args.min_chars},
            "n_chapters": len(world.chapters),
            "n_chunks": len(records),
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_model_max_length": get_tokenizer().model_max_length,
            "max_input_tokens": MODEL_MAX_TOKENS,
            "special_tokens_added": special_added,
            "token_count_definition": f"len(tokenizer(context_header + {EMBED_SEPARATOR!r} + text, add_special_tokens=True, truncation=False))",
            "token_count_min_mean_max": [min(toks), sum(toks) // len(toks), max(toks)],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        manifest_path(world.world_no),
    )
    print(f"\nĐã ghi: {out.relative_to(ROOT)} ({len(records)} dòng) và {manifest_path(world.world_no).relative_to(ROOT)}")
    print("Ví dụ bản ghi đầu tiên:")
    r = records[0]
    print(f"  {r.chunk_id} | prev={r.prev_id} next={r.next_id} | {r.token_count} token")
    print(f"  header: {r.context_header}")


if __name__ == "__main__":
    main()
