"""Step 5 - Kiểm tra file chunk trên đĩa có đúng không (độc lập với bước build).

Script này ĐỌC LẠI data/chunks/chunks_wNN.jsonl và đối chiếu với file truyện gốc, để bắt lỗi của cả chunker
lẫn bước ghi file. Các nhóm kiểm tra:
  1. Đủ số bản ghi (mặc định 393) và khớp manifest.
  2. chunk_id không trùng, đúng định dạng, chunk_index liên tục trong từng chương.
  3. Liên kết prev/next hợp lệ: trỏ tới id tồn tại, hai chiều khớp nhau, tạo thành MỘT chuỗi duy nhất đi qua mọi chunk.
  4. Nguồn đúng: file nguồn không đổi (sha256), world/chương/số đoạn khớp, text khớp đúng các đoạn gốc.
  5. Văn bản không bị mất: ghép các chunk (bỏ phần overlap) khôi phục lại ĐÚNG từng chương gốc.
  6. Token: tokenizer ghi đúng, đếm lại khớp token_count, không chunk nào vượt giới hạn model.
  7. Vòng đọc-ghi: file trên đĩa giống hệt kết quả chạy chunker mới từ file gốc.

Chạy:  python scripts/step5_verify_chunks.py            (exit code 0 = tất cả OK, 1 = có lỗi)
       python scripts/step5_verify_chunks.py --expect 393
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_store import (  # noqa: E402
    MODEL_MAX_TOKENS,
    TOKENIZER_NAME,
    chunks_path,
    count_tokens,
    embed_input,
    make_context_header,
    manifest_path,
    read_jsonl,
    sha256_file,
)
from src.chunker import chunk_world  # noqa: E402
from src.config import ROOT  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402

failures: list[str] = []


def check(name: str, errors: list[str], detail: str = "") -> None:
    """In một dòng kết quả; nếu có lỗi thì in tối đa 3 lỗi đầu và nhớ lại để báo exit code."""
    if errors:
        print(f"  ✗ {name}: {len(errors)} lỗi")
        for e in errors[:3]:
            print(f"      - {e}")
        failures.append(name)
    else:
        print(f"  ✓ {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--expect", type=int, default=393, help="số bản ghi mong đợi")
    args = ap.parse_args()

    path, mpath = chunks_path(args.world), manifest_path(args.world)
    recs = read_jsonl(path)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    src_file = find_world_file(args.world)
    world = load_world(src_file)
    print(f"Kiểm tra {path.relative_to(ROOT)} so với {src_file.relative_to(ROOT)}\n")

    # ---- 1. Số lượng ----
    print("1. Số lượng bản ghi")
    errs = []
    if len(recs) != args.expect:
        errs.append(f"có {len(recs)} bản ghi, mong đợi {args.expect}")
    if manifest["n_chunks"] != len(recs):
        errs.append(f"manifest ghi {manifest['n_chunks']} chunk nhưng file có {len(recs)}")
    n_lines = sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())
    if n_lines != len(recs):
        errs.append(f"số dòng JSONL ({n_lines}) khác số bản ghi đọc được ({len(recs)})")
    check(f"đủ {args.expect} bản ghi, khớp manifest và số dòng file", errs)

    # ---- 2. ID ----
    print("2. ID")
    ids = [r.chunk_id for r in recs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    check("chunk_id không trùng", [f"trùng: {d}" for d in dup])
    pat = re.compile(r"^w(\d{2})-c(\d{3})-k(\d{2})$")
    errs = []
    for r in recs:
        m = pat.match(r.chunk_id)
        if not m or (int(m[1]), int(m[2]), int(m[3])) != (r.world_no, r.chapter_no, r.chunk_index):
            errs.append(f"{r.chunk_id}: id không khớp world/chapter/chunk_index ({r.world_no},{r.chapter_no},{r.chunk_index})")
    check("chunk_id đúng định dạng và khớp trường world/chương/chỉ số", errs)
    by_chapter: dict[int, list] = {}
    for r in recs:
        by_chapter.setdefault(r.chapter_no, []).append(r)
    errs = [f"chương {c}: chunk_index {[r.chunk_index for r in rs]} không liên tục từ 0"
            for c, rs in by_chapter.items() if [r.chunk_index for r in rs] != list(range(len(rs)))]
    check("chunk_index liên tục 0..k trong từng chương", errs)

    # ---- 3. Liên kết prev/next ----
    print("3. Liên kết prev/next")
    by_id = {r.chunk_id: r for r in recs}
    errs = []
    for r in recs:
        for field in ("prev_id", "next_id"):
            target = getattr(r, field)
            if target is not None and target not in by_id:
                errs.append(f"{r.chunk_id}.{field} = {target} không tồn tại")
    check("prev_id/next_id đều trỏ tới chunk có thật", errs)
    errs = []
    for r in recs:
        if r.next_id and by_id[r.next_id].prev_id != r.chunk_id:
            errs.append(f"{r.chunk_id} -> next {r.next_id}, nhưng prev của nó là {by_id[r.next_id].prev_id}")
        if r.prev_id and by_id[r.prev_id].next_id != r.chunk_id:
            errs.append(f"{r.chunk_id} <- prev {r.prev_id}, nhưng next của nó là {by_id[r.prev_id].next_id}")
    check("hai chiều khớp nhau (next của prev = chính nó, prev của next = chính nó)", errs)
    heads = [r for r in recs if r.prev_id is None]
    tails = [r for r in recs if r.next_id is None]
    errs = []
    if len(heads) != 1 or len(tails) != 1:
        errs.append(f"cần đúng 1 chunk đầu và 1 chunk cuối, thấy {len(heads)} đầu / {len(tails)} cuối")
    else:
        walk, seen, cur = [], set(), heads[0]
        while cur is not None and cur.chunk_id not in seen:  # đi theo next_id, chặn vòng lặp
            seen.add(cur.chunk_id)
            walk.append(cur.chunk_id)
            cur = by_id.get(cur.next_id) if cur.next_id else None
        if len(walk) != len(recs):
            errs.append(f"đi từ chunk đầu theo next_id chỉ qua {len(walk)}/{len(recs)} chunk (chuỗi bị đứt hoặc lặp)")
        if walk != ids:
            errs.append("thứ tự chuỗi liên kết khác thứ tự dòng trong file")
        if heads[0].chunk_id != "w%02d-c001-k00" % args.world:
            errs.append(f"chunk đầu là {heads[0].chunk_id}, mong đợi chương 1 chunk 0")
    check("tạo thành MỘT chuỗi duy nhất từ chunk đầu đến cuối, đúng thứ tự đọc truyện", errs)

    # ---- 4. Nguồn ----
    print("4. Nguồn đúng")
    errs = []
    if sha256_file(src_file) != manifest["source_sha256"]:
        errs.append("file truyện đã thay đổi kể từ lúc cắt chunk (sha256 khác manifest)")
    if manifest["source_file"] != str(src_file.relative_to(ROOT)):
        errs.append(f"manifest trỏ tới {manifest['source_file']}")
    check("file truyện gốc không đổi (sha256) và manifest trỏ đúng file", errs)
    chapters = {c.chapter_no: c for c in world.chapters}
    errs = []
    for r in recs:
        c = chapters.get(r.chapter_no)
        if c is None:
            errs.append(f"{r.chunk_id}: chương {r.chapter_no} không có trong file gốc")
            continue
        if (r.world_no, r.world_title) != (world.world_no, world.title):
            errs.append(f"{r.chunk_id}: world sai ({r.world_no}, {r.world_title!r})")
        if not 1 <= r.para_start <= r.para_end <= len(c.paragraphs):
            errs.append(f"{r.chunk_id}: đoạn {r.para_start}-{r.para_end} ngoài phạm vi 1..{len(c.paragraphs)}")
            continue
        expected = "\n\n".join(c.paragraphs[r.para_start - 1 : r.para_end])
        if r.text != expected:
            errs.append(f"{r.chunk_id}: text KHÁC các đoạn gốc {r.para_start}-{r.para_end} của chương {r.chapter_no}")
        if r.n_chars != len(r.text):
            errs.append(f"{r.chunk_id}: n_chars={r.n_chars} nhưng len(text)={len(r.text)}")
        if r.context_header != make_context_header(r.world_no, r.world_title, r.chapter_no, r.para_start, r.para_end):
            errs.append(f"{r.chunk_id}: context_header không khớp metadata")
    check("world/chương/đoạn đúng và text khớp từng ký tự với các đoạn gốc", errs,
          f"{len(recs)} chunk, so từng đoạn gốc")

    # ---- 5. Không mất văn bản ----
    print("5. Văn bản không bị mất")
    errs = []
    for no, c in chapters.items():
        rebuilt, covered = [], 0  # covered = số đoạn đã khôi phục (đoạn 1..covered)
        for r in by_chapter.get(no, []):
            if r.para_start > covered + 1:
                errs.append(f"chương {no}: hở, thiếu đoạn {covered + 1}..{r.para_start - 1}")
            # Chỉ lấy phần MỚI (đoạn > covered); phần đầu chunk trùng với overlap thì bỏ qua.
            for pi in range(max(r.para_start, covered + 1), r.para_end + 1):
                rebuilt.append(c.paragraphs[pi - 1])
            covered = max(covered, r.para_end)
        # Ghép từ CHÍNH TEXT của chunk (không phải từ file gốc) để chứng minh text trong file đủ nội dung:
        text_rebuilt = _rebuild_from_chunk_text(by_chapter.get(no, []))
        if covered != len(c.paragraphs) or rebuilt != c.paragraphs:
            errs.append(f"chương {no}: phủ {covered}/{len(c.paragraphs)} đoạn")
        if text_rebuilt != c.text:
            errs.append(f"chương {no}: ghép text các chunk (bỏ overlap) KHÔNG bằng nguyên văn chương")
    total_chars = sum(len(c.text) for c in world.chapters)
    check("ghép các chunk (bỏ overlap) khôi phục đúng nguyên văn cả 81 chương", errs,
          f"{len(chapters)} chương, {total_chars:,} ký tự")

    # ---- 6. Token ----
    print("6. Token")
    errs = [f"{r.chunk_id}: tokenizer={r.tokenizer!r}" for r in recs if r.tokenizer != TOKENIZER_NAME]
    check(f"mọi bản ghi ghi rõ tokenizer = {TOKENIZER_NAME}", errs)
    errs = []
    for r in recs:
        again = count_tokens(embed_input(r.context_header, r.text))
        if again != r.token_count:
            errs.append(f"{r.chunk_id}: lưu {r.token_count}, đếm lại {again}")
    check("đếm lại (header + text, có special tokens) khớp token_count đã lưu", errs)
    over = [f"{r.chunk_id}: {r.token_count} > {MODEL_MAX_TOKENS}" for r in recs if r.token_count > MODEL_MAX_TOKENS]
    check(f"không chunk nào vượt giới hạn {MODEL_MAX_TOKENS} token của model", over,
          f"max thực tế {max(r.token_count for r in recs)}")

    # ---- 7. Vòng đọc-ghi ----
    print("7. Đối chiếu với kết quả chunker chạy mới")
    cfg = manifest["chunker"]
    fresh = chunk_world(world, size=cfg["size_chars"], overlap=cfg["overlap_chars"], min_chars=cfg["min_chars"])
    errs = []
    if [c.chunk_id for c in fresh] != ids:
        errs.append("danh sách chunk_id khác kết quả chunker chạy mới với cùng tham số")
    else:
        errs = [f"{c.chunk_id}: text trên đĩa khác chunker chạy mới" for c, r in zip(fresh, recs) if c.text != r.text]
    check("file trên đĩa giống hệt kết quả chunker chạy mới từ file gốc (cùng tham số trong manifest)", errs)

    print()
    if failures:
        print(f"✗ KẾT QUẢ: {len(failures)} nhóm kiểm tra có lỗi: {failures}")
        sys.exit(1)
    print("✓ KẾT QUẢ: tất cả kiểm tra đều đạt")


def _rebuild_from_chunk_text(chunks: list) -> str:
    """Ghép lại văn bản chương từ CHÍNH text của các chunk, bỏ phần overlap.

    Mỗi chunk gồm các đoạn cách nhau "\\n\\n". Với mỗi chunk, chỉ lấy các đoạn có số thứ tự > số đoạn đã ghép.
    Vì text của chunk gồm đúng các đoạn para_start..para_end theo thứ tự, ta có thể tách theo "\\n\\n".
    """
    out: list[str] = []
    covered = 0
    for r in chunks:
        paras = r.text.split("\n\n")
        if len(paras) != r.para_end - r.para_start + 1:
            return f"<chunk {r.chunk_id}: số đoạn trong text ({len(paras)}) khác para_end-para_start+1>"
        for k, p in enumerate(paras):
            if r.para_start + k > covered:
                out.append(p)
        covered = max(covered, r.para_end)
    return "\n\n".join(out)


if __name__ == "__main__":
    main()
