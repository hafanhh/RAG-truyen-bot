"""Step 3 - Naive chunking: cắt theo số KÝ TỰ cố định, cố tình KHÔNG quan tâm câu/đoạn/chương.

Mục đích: tự thấy các vấn đề của cách cắt "ngu" để hiểu vì sao Step 4 cần chunking thông minh hơn.
Mỗi kích thước chunk được đo bằng 5 chỉ số "xấu" rồi in ví dụ đường cắt (seam) cụ thể.

Chạy từ thư mục gốc:
    python scripts/step3_naive_chunking.py                       # so sánh size 500, 1000, 2000
    python scripts/step3_naive_chunking.py --sizes 800 --show 6  # 1 size, xem 6 đường cắt xấu
    python scripts/step3_naive_chunking.py --sizes 1000 --overlap 200   # thử overlap
    python scripts/step3_naive_chunking.py --world 1 --phrases "bút máy" "trà sữa"
"""
import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc project vào sys.path để `import src...` hoạt động khi chạy script từ scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import World, find_world_file, load_world  # noqa: E402

# Tập các ký tự được coi là "kết thúc một câu". Một chunk kết thúc bằng ký tự KHÁC (vd giữa chữ)
# nghĩa là bị cắt giữa câu. Lưu ý: 】 đóng lời "hệ thống" 【 ... 】 trong truyện, cũng là kết câu hợp lệ.
SENTENCE_END = set(".!?…”\"】")
# Các chi tiết trong Chương 1 (dùng ở Step 1) để kiểm tra xem có bị đường cắt chặt đôi không.
DEFAULT_PHRASES = ["trà sữa", "bút máy", "năm ngàn vạn", "chi phiếu", "Lục Kỳ"]


def build_text(world: World) -> tuple[str, list[tuple[int, int, int]]]:
    """Nối mọi chương thành MỘT chuỗi dài (đúng kiểu naive: bỏ qua ranh giới chương).

    Trả về (text, spans) với spans = [(số_chương, vị_trí_bắt_đầu, vị_trí_kết_thúc), ...]
    để sau này biết một đoạn [start, end) của text thuộc những chương nào.
    """
    parts, spans, pos = [], [], 0
    for c in world.chapters:
        t = c.text
        spans.append((c.chapter_no, pos, pos + len(t)))
        parts.append(t)
        pos += len(t) + 2  # +2 vì giữa các chương sẽ chèn "\n\n" (2 ký tự)
    return "\n\n".join(parts), spans


def naive_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Cắt [start, end) mỗi `size` ký tự, lùi `overlap` ký tự cho chunk kế tiếp.

    Trả về list các cặp (start, end) là VỊ TRÍ trong `text`, không phải bản copy văn bản
    (tiết kiệm bộ nhớ và dễ kiểm tra "chunk này nằm ở đâu").
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap phải >= 0 và < size")
    step = size - overlap  # mỗi chunk bắt đầu cách chunk trước `step` ký tự
    # range(0, len, step): các vị trí bắt đầu. Điều kiện `if` loại bỏ chunk cuối chỉ toàn overlap
    # (không có nội dung mới nào); riêng s == 0 luôn giữ lại.
    return [(s, min(s + size, len(text))) for s in range(0, len(text), step) if s < len(text) - overlap or s == 0]


def at_paragraph_start(text: str, i: int) -> bool:
    """Vị trí i có phải đầu một đoạn văn không (đầu text, hoặc ngay sau dòng trống)?"""
    return i == 0 or text[i - 2 : i] == "\n\n"


def at_paragraph_end(text: str, i: int) -> bool:
    """Vị trí i có phải cuối một đoạn văn không (cuối text, hoặc ngay trước dòng trống)?"""
    return i >= len(text) or text[i : i + 2] == "\n\n"


def analyse(text: str, spans: list[tuple[int, int, int]], chunks: list[tuple[int, int]], phrases: list[str]) -> dict:
    """Đo 5 chỉ số "xấu" của một cách chia chunk. Trả về dict các con số."""
    n = len(chunks)
    cut_para = end_sentence = quotes = cross = 0
    bad_seams = []  # chỉ số các chunk kết thúc giữa câu (để in ví dụ)
    for idx, (s, e) in enumerate(chunks):
        body = text[s:e]
        # (1) Chunk không bắt đầu ở đầu đoạn HOẶC không kết thúc ở cuối đoạn => cắt giữa đoạn.
        if not (at_paragraph_start(text, s) and at_paragraph_end(text, e)):
            cut_para += 1
        # (2) Ký tự cuối (sau khi bỏ khoảng trắng cuối) không phải dấu kết câu => cắt giữa câu.
        #     [-1:] thay vì [-1] để không lỗi khi chuỗi rỗng.
        if body.rstrip()[-1:] not in SENTENCE_END:
            end_sentence += 1
            if idx + 1 < n:  # chunk cuối không có "đường cắt" tiếp theo nên không đưa vào ví dụ
                bad_seams.append(idx)
        # (3) Số dấu “ mở khác số dấu ” đóng => có lời thoại bị cắt dở.
        if body.count("“") != body.count("”"):
            quotes += 1
        # (4) Chunk chạm tới từ 2 chương trở lên => trộn chương.
        #     Chunk [s,e) chạm chương [a,b) khi hai khoảng giao nhau: s < b và e > a.
        if sum(1 for _, a, b in spans if s < b and e > a) > 1:
            cross += 1
    # (5) Cụm từ quan trọng bị cắt đôi: tìm mọi lần xuất hiện của cụm từ trong text, xem có
    #     ranh giới chunk nào nằm ngay BÊN TRONG cụm từ không (i < b < i + len).
    boundaries = {s for s, _ in chunks[1:]} | {e for _, e in chunks[:-1]}
    cut_phrases = {}
    for ph in phrases:
        total = cut = 0
        i = text.find(ph)
        while i != -1:  # str.find trả về -1 khi hết chỗ khớp
            total += 1
            if any(i < b < i + len(ph) for b in boundaries):
                cut += 1
            i = text.find(ph, i + 1)  # tìm tiếp từ vị trí kế tiếp (cho phép khớp chồng nhau)
        cut_phrases[ph] = (cut, total)
    return dict(n=n, cut_para=cut_para, end_sentence=end_sentence, quotes=quotes, cross=cross,
                bad_seams=bad_seams, cut_phrases=cut_phrases)


def pct(x: int, n: int) -> str:
    """Định dạng 'số lượng (phần trăm%)' cho bảng, ví dụ '  22 ( 5.3%)'."""
    return f"{x:>4} ({100 * x / n:4.1f}%)"


def show_seam(text: str, chunks: list[tuple[int, int]], idx: int, width: int = 90) -> None:
    """In "đường cắt" giữa chunk idx và idx+1: phần cuối chunk trước ✂ phần đầu chunk sau."""
    _, e = chunks[idx]  # chỗ chunk idx kết thúc
    nxt_s, _ = chunks[idx + 1]  # chỗ chunk idx+1 bắt đầu (khác e nếu có overlap)
    # Thay "\n" bằng ký hiệu ⏎ để thấy ranh giới đoạn mà vẫn in trên một dòng.
    left = text[max(0, e - width) : e].replace("\n", "⏎")
    right = text[nxt_s : nxt_s + width].replace("\n", "⏎")
    print(f"   chunk {idx:>3} → {idx + 1:<3}: ...{left}  ✂  {right}...")


def main() -> None:
    # argparse: đọc tham số dòng lệnh (--sizes, --overlap, ...). RawDescriptionHelpFormatter giữ
    # nguyên định dạng của docstring khi in --help.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000], help="kích thước chunk (ký tự)")
    ap.add_argument("--overlap", type=int, default=0, help="số ký tự chồng lấn (ký tự)")
    ap.add_argument("--show", type=int, default=4, help="số đường cắt xấu in ra cho mỗi size")
    ap.add_argument("--phrases", nargs="+", default=DEFAULT_PHRASES, help="các cụm từ cần kiểm tra có bị cắt đôi không")
    args = ap.parse_args()

    world = load_world(find_world_file(args.world))
    text, spans = build_text(world)
    print(f"Thế giới {world.world_no:02d}: {world.title} | {len(world.chapters)} chương | {len(text):,} ký tự | overlap={args.overlap}\n")

    # Với mỗi kích thước: cắt, đo, rồi in bảng chỉ số + vài đường cắt xấu làm ví dụ.
    for size in args.sizes:
        chunks = naive_chunks(text, size, args.overlap)
        r = analyse(text, spans, chunks, args.phrases)
        n = r["n"]
        print("=" * 100)
        print(f"SIZE = {size} ký tự  →  {n} chunks")
        print(f"  bắt đầu/kết thúc GIỮA đoạn văn : {pct(r['cut_para'], n)}")
        print(f"  kết thúc GIỮA câu              : {pct(r['end_sentence'], n)}")
        print(f"  lời thoại “ ” bị cắt dở        : {pct(r['quotes'], n)}")
        print(f"  chunk TRỘN 2+ chương           : {pct(r['cross'], n)}")
        print("  cụm từ quan trọng bị cắt đôi   : " + ", ".join(f"'{p}' {c}/{t}" for p, (c, t) in r["cut_phrases"].items()))
        if r["bad_seams"] and args.show:
            print(f"  --- {min(args.show, len(r['bad_seams']))} đường cắt giữa câu đầu tiên ---")
            for idx in r["bad_seams"][: args.show]:
                show_seam(text, chunks, idx)
        print()


if __name__ == "__main__":
    main()
