"""Step 4b - Kiểm tra: một chunk ĐỨNG MỘT MÌNH có đủ thông tin để biết "ai" và "chuyện gì" không?

Vì sao quan trọng: khi RAG truy xuất, LLM chỉ nhận vài chunk riêng lẻ, không thấy phần còn lại của
chương. Truyện này dùng rất nhiều đại từ ("hắn", "nàng"); nếu một chunk toàn đại từ mà không nêu tên
ai thì LLM sẽ không biết "nàng" là ai (hiện tượng "mất ngữ cảnh" - lost context).

Script làm 3 việc:
  1) Thống kê TOÀN thế giới: bao nhiêu % chunk không có tên nhân vật nào / đại từ xuất hiện trước tên.
  2) In các chunk quanh một sự kiện (mặc định: Chương 1) kèm các phép đo, để BẠN đọc và chấm.
  3) (--llm) Đưa TỪNG chunk riêng lẻ cho Gemini hỏi "nhân vật là ai? chuyện gì?" -> xem LLM có xác định được không.

Chạy từ thư mục gốc:
    python scripts/step4b_inspect_chunks.py                    # chỉ in, không gọi API
    python scripts/step4b_inspect_chunks.py --llm              # thêm phép thử với Gemini
    python scripts/step4b_inspect_chunks.py --chapter 40 --after 0 --brief   # chương khác, in gọn
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import Chunk, chunk_world  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402

# Danh sách tên nhân vật CHỦ YẾU của Thế giới 01, ghi tay (gazetteer). Mỗi nhân vật có thể có nhiều
# cách gọi (alias). Hạn chế: nhân vật phụ không có trong danh sách, nên chỉ số "không có tên" bên
# dưới là cận trên, có thể chunk chứa tên một nhân vật phụ nhưng vẫn bị tính là "không tên".
CHARACTERS = {
    "Đường Quả": ["Đường Quả", "Tiểu Quả"],
    "Lãnh Tử Việt": ["Lãnh Tử Việt"],
    "Lục Kỳ": ["Lục Kỳ", "Kỳ Kỳ"],
}
# Đại từ ngôi thứ ba: không cho biết ai là ai nếu chunk không nêu tên.
PRONOUNS = ["hắn", "nàng"]


def find_names(text: str) -> tuple[dict[str, int], int | None]:
    """Đếm số lần mỗi nhân vật được nêu TÊN; trả về (đếm, vị trí tên xuất hiện sớm nhất hoặc None)."""
    counts: dict[str, int] = {}
    first: int | None = None
    for canon, aliases in CHARACTERS.items():
        for alias in aliases:
            for m in re.finditer(re.escape(alias), text):
                counts[canon] = counts.get(canon, 0) + 1
                first = m.start() if first is None else min(first, m.start())
    return counts, first


def find_pronouns(text: str) -> tuple[dict[str, int], int | None]:
    """Đếm đại từ (\\b = ranh giới từ, tránh khớp nhầm bên trong từ khác; IGNORECASE cho 'Hắn' đầu câu)."""
    counts: dict[str, int] = {}
    first: int | None = None
    for p in PRONOUNS:
        for m in re.finditer(rf"\b{p}\b", text, flags=re.IGNORECASE):
            counts[p] = counts.get(p, 0) + 1
            first = m.start() if first is None else min(first, m.start())
    return counts, first


def inspect(chunk: Chunk) -> dict:
    """Phân tích một chunk: có tên không, có đại từ không, đại từ có đứng trước tên không."""
    names, first_name = find_names(chunk.text)
    prons, first_pron = find_pronouns(chunk.text)
    flags = []
    if prons and not names:
        flags.append("KHÔNG CÓ TÊN (chỉ có đại từ)")
    elif prons and first_pron is not None and first_name is not None and first_pron < first_name:
        flags.append("ĐẠI TỪ XUẤT HIỆN TRƯỚC TÊN")
    return dict(names=names, prons=prons, flags=flags,
                no_name=not names, pron_before_name=bool(prons and names and first_pron < first_name))


def print_summary(chunks: list[Chunk]) -> None:
    """Thống kê toàn thế giới, tách chunk đầu chương (k=0) và các chunk sau (k>0)."""
    rows = [(c, inspect(c)) for c in chunks]

    def line(label: str, subset: list) -> None:
        n = len(subset)
        if not n:
            return
        no_name = sum(1 for _, r in subset if r["no_name"])
        pbn = sum(1 for _, r in subset if r["pron_before_name"])
        print(f"  {label:<26} {n:>4} chunk | không có tên nào: {no_name:>3} ({100 * no_name / n:4.1f}%) "
              f"| đại từ trước tên: {pbn:>3} ({100 * pbn / n:4.1f}%)")

    print("THỐNG KÊ TOÀN THẾ GIỚI (chỉ tính 3 nhân vật chủ yếu trong CHARACTERS)")
    line("tất cả chunk", rows)
    line("chunk ĐẦU chương (k=0)", [x for x in rows if x[0].chunk_index == 0])
    line("chunk SAU (k>0)", [x for x in rows if x[0].chunk_index > 0])
    print()


def select_event(chunks: list[Chunk], chapter: int, after: int) -> list[Chunk]:
    """Lấy mọi chunk của `chapter`, cộng thêm `after` chunk liền sau (có thể sang chương kế tiếp)."""
    idxs = [i for i, c in enumerate(chunks) if c.chapter_no == chapter]
    if not idxs:
        raise SystemExit(f"Không có chunk nào của chương {chapter}")
    return chunks[idxs[0] : idxs[-1] + 1 + after]


def print_chunk(chunk: Chunk, brief: bool) -> None:
    r = inspect(chunk)
    print("=" * 100)
    print(f"{chunk.chunk_id} | chương {chunk.chapter_no}, đoạn {chunk.para_start}-{chunk.para_end} | {chunk.n_chars} ký tự")
    print(f"  tên nêu rõ : {r['names'] or 'KHÔNG CÓ'}")
    print(f"  đại từ     : {r['prons'] or 'không có'}")
    if r["flags"]:
        print(f"  ⚠ CẢNH BÁO : {'; '.join(r['flags'])}")
    paras = chunk.text.split("\n\n")
    shown = paras[:2] + ["   [...]"] + paras[-1:] if brief and len(paras) > 3 else paras
    for p in shown:
        print("  │ " + p)


# ---------------------------------------------------------------------------
# Phép thử với Gemini: đưa MỘT chunk riêng lẻ, xem LLM có xác định được nhân vật/diễn biến không
# ---------------------------------------------------------------------------

LLM_PROMPT = """Dưới đây là MỘT đoạn trích từ một truyện. Bạn CHỈ có đoạn này, không có phần nào khác của truyện.
Tuyệt đối không dùng kiến thức bên ngoài đoạn trích. Trả lời bằng tiếng Việt, ngắn gọn, đúng mẫu:

1) Nhân vật: liệt kê từng người. Nếu đoạn trích nêu rõ tên thì ghi tên; nếu chỉ có đại từ như "hắn"/"nàng" mà
   không biết tên thì ghi "không rõ tên (đại từ: ...)".
2) Diễn biến: tối đa 2 câu.
3) Độ chắc chắn về danh tính nhân vật: CAO / TRUNG BÌNH / THẤP, kèm lý do ngắn.

=== ĐOẠN TRÍCH ===
{chunk}
=== HẾT ĐOẠN TRÍCH ==="""


def run_llm_test(chunks: list[Chunk]) -> None:
    from google import genai  # import muộn: chạy không --llm thì không cần SDK/key
    from google.genai import types

    from src.config import require_api_key

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # timeout 60 giây (đơn vị mili-giây) để một lần gọi lỗi/quá tải không treo cả script.
    client = genai.Client(api_key=require_api_key(), http_options=types.HttpOptions(timeout=60_000))
    print("\n" + "#" * 100)
    print(f"PHÉP THỬ GEMINI ({model}): mỗi chunk được đưa RIÊNG LẺ, không kèm chunk nào khác")
    for c in chunks:
        print("=" * 100)
        print(f"{c.chunk_id} (chương {c.chapter_no}, đoạn {c.para_start}-{c.para_end})")
        try:
            resp = client.models.generate_content(model=model, contents=LLM_PROMPT.format(chunk=c.text))
            print((resp.text or "(trả lời rỗng)").strip())
        except Exception as e:  # in lỗi gọn thay vì cả traceback dài
            print(f"  ✗ Lỗi khi gọi API: {type(e).__name__}: {str(e)[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--chapter", type=int, default=1, help="chương chứa sự kiện cần xem")
    ap.add_argument("--after", type=int, default=1, help="số chunk liền SAU chương đó cũng được in (sự kiện có thể kéo sang chương sau)")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--brief", action="store_true", help="chỉ in 2 đoạn đầu + 1 đoạn cuối của mỗi chunk")
    ap.add_argument("--llm", action="store_true", help="chạy thêm phép thử với Gemini (tốn vài lần gọi API)")
    args = ap.parse_args()

    world = load_world(find_world_file(args.world))
    chunks = chunk_world(world, size=args.size, overlap=args.overlap)
    print(f"Thế giới {world.world_no:02d}: {world.title} | size={args.size}, overlap={args.overlap} | {len(chunks)} chunk\n")

    print_summary(chunks)
    event = select_event(chunks, args.chapter, args.after)
    print(f"SỰ KIỆN: chương {args.chapter} + {args.after} chunk liền sau = {len(event)} chunk")
    for c in event:
        print_chunk(c, args.brief)
    if args.llm:
        run_llm_test(event)


if __name__ == "__main__":
    main()
