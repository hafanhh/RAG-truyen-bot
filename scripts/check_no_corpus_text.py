"""Quét các file sẽ được commit để chắc không chứa văn bản truyện (có bản quyền) nguyên văn.

Repo này public, còn Books/ và dữ liệu sinh ra bị .gitignore. Script lấy mọi cụm N từ liên tiếp (mặc định 6, bỏ qua hoa/thường
và dấu câu) của toàn bộ chương trong World 01, rồi tìm các cụm đó trong những file git theo dõi hoặc sắp được theo dõi
(`git ls-files --cached --others --exclude-standard`). File nào chứa cụm trùng -> in đoạn trùng và thoát với mã 1.

Quy ước của repo: trích dẫn trong file commit phải <= N-1 từ liên tiếp, hoặc ghi bằng tham chiếu (chương, đoạn). Đoạn cần đọc
văn bản thì nạp lúc chạy từ Books/ qua src/loader.py.

Chạy: python scripts/check_no_corpus_text.py [--words 6] [--world 1]
Thoát: 0 sạch | 1 có văn bản trùng | 2 không đọc được Books/ (không quét được).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402

SKIP_SUFFIXES = (".pyc", ".png", ".jpg", ".jpeg", ".gif", ".npz", ".sqlite3", ".gitkeep")
WORD = re.compile(r"\w+")
MIN_DISTINCT = 4  # một cụm phải có ít nhất ngần này từ KHÁC NHAU mới tính là trích văn bản


def corpus_shingles(world_no: int, n: int) -> set[tuple[str, ...]]:
    world = load_world(find_world_file(world_no))
    grams: set[tuple[str, ...]] = set()
    for ch in world.chapters:
        for para in ch.paragraphs:
            toks = [m.group().lower() for m in WORD.finditer(para)]
            # Cụm chỉ gồm vài từ lặp lại (vd tên nhân vật nối nhau trong danh sách) không phải văn bản truyện: bỏ qua.
            grams.update(g for g in (tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)) if len(set(g)) >= MIN_DISTINCT)
    return grams


def candidate_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    )
    return sorted({f for f in out.split("\n") if f and not f.endswith(SKIP_SUFFIXES)})


def find_overlaps(text: str, grams: set[tuple[str, ...]], n: int) -> list[tuple[int, int, int]]:
    """Các đoạn (bắt đầu, kết thúc, số từ) của `text` nằm trong cụm trùng với truyện; các cụm chồng nhau được gộp."""
    matches = list(WORD.finditer(text))
    toks = [m.group().lower() for m in matches]
    spans: list[list[int]] = []
    for i in range(len(toks) - n + 1):
        if tuple(toks[i : i + n]) in grams:
            if spans and i <= spans[-1][1]:
                spans[-1][1] = i + n  # nối vào đoạn trước (đầu mút loại trừ, tính theo chỉ số từ)
            else:
                spans.append([i, i + n])
    return [(matches[a].start(), matches[b - 1].end(), b - a) for a, b in spans]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", type=int, default=6, help="số từ liên tiếp tối thiểu bị coi là trích nguyên văn")
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()

    try:
        grams = corpus_shingles(args.world, args.words)
    except Exception as exc:  # noqa: BLE001 - thiếu Books/ hoặc lỗi loader: không được báo "sạch" giả
        print(f"Không quét được (cần Books/ cục bộ): {type(exc).__name__}: {exc}")
        return 2

    dirty = 0
    files = candidate_files()
    for f in files:
        try:
            text = (ROOT / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        spans = find_overlaps(text, grams, args.words)
        if not spans:
            continue
        dirty += 1
        print(f"{f}: {sum(s[2] for s in spans)} từ trùng nguyên văn, {len(spans)} đoạn")
        for a, b, w in spans:
            snippet = " ".join(text[a:b].split())
            print(f"    [{w} từ] {snippet[:90]}{'…' if len(snippet) > 90 else ''}")
    print(f"Đã quét {len(files)} file: " + (f"{dirty} file CÓ văn bản truyện" if dirty else "sạch"))
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
