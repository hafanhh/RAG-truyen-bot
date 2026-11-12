"""V1 - Bước 10a: gắn meta (model + task_type + dims) vào các file cache embedding định dạng cũ.

Vì sao: khóa cache đúng nghĩa phải là nội_dung + model + task_type + cấu hình embedding. Trước giờ model và
task_type chỉ nằm NGẦM trong tên file (w01_queries_... hiểu ngầm là RETRIEVAL_QUERY): nếu một script trỏ nhầm
file, cache trả về vector embed bằng cấu hình khác mà không ai phát hiện (lỗi im lặng). Sau bước này, mọi lần
đọc qua embed_with_cache / embed_query_cached đều validate meta và báo lỗi ngay khi lệch.

Không gọi API, không re-embed: chỉ đọc từng file .npz, backup thành .npz.bak, ghi lại kèm meta, rồi tự kiểm
chứng (cùng tập khóa, cùng sha, vector bằng nhau từng bit so với backup). Task_type suy từ biến thể trong tên
file: plain / header / paragraphs -> RETRIEVAL_DOCUMENT, queries -> RETRIEVAL_QUERY. Chạy lại lần nữa vô hại
(file đã có meta đúng thì bỏ qua). Exit code 1 nếu có file lỗi.

Chạy: python scripts/v1_step10a_migrate_cache.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBED_CACHE_DIR  # noqa: E402
from src.embedder import DOC_TASK, QUERY_TASK, cache_meta, load_cache, load_meta, save_cache  # noqa: E402

VARIANT_TASK = {"plain": DOC_TASK, "header": DOC_TASK, "paragraphs": DOC_TASK, "queries": QUERY_TASK}


def main() -> None:
    files = sorted(p for p in EMBED_CACHE_DIR.glob("*.npz") if ".tmp" not in p.name)
    if not files:
        print(f"Không có file cache nào trong {EMBED_CACHE_DIR}")
        return
    failed = False
    for path in files:
        parts = path.stem.split("_", 2)  # vd "w01_header_gemini-embedding-001" -> [w01, header, model]
        if len(parts) != 3 or parts[1] not in VARIANT_TASK:
            print(f"LỖI  {path.name}: không nhận ra biến thể trong tên file, không migrate")
            failed = True
            continue
        _, variant, model = parts
        entries = load_cache(path)  # không validate: file có thể chưa có meta
        dims = next(iter(entries.values()))[1].shape[0] if entries else 3072
        meta = cache_meta(model, VARIANT_TASK[variant], dims)
        if load_meta(path) == meta:
            print(f"OK   {path.name}: đã có meta đúng, bỏ qua ({len(entries)} vector)")
            continue
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        save_cache(path, entries, meta)
        # Tự kiểm chứng: đọc lại CÓ validate meta, so từng record với bản backup.
        again = load_cache(path, expect_meta=meta)
        old = load_cache(backup)
        same = (
            set(again) == set(old)
            and all(again[k][0] == old[k][0] for k in old)
            and all(np.array_equal(again[k][1], old[k][1]) for k in old)
        )
        if same:
            print(f"OK   {path.name}: {len(entries)} vector, meta={meta}, backup={backup.name}")
        else:
            print(f"LỖI  {path.name}: SAI LỆCH sau migrate — file gốc còn nguyên ở {backup.name}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
