"""Step 4 - So sánh chunker theo đoạn (src/chunker.py) với naive chunking của Step 3, cùng bộ chỉ số.

Chạy từ thư mục gốc:
    python scripts/step4_compare_chunking.py                       # size 500, 1000, 2000, overlap 200
    python scripts/step4_compare_chunking.py --sizes 800 --overlap 0
    python scripts/step4_compare_chunking.py --sizes 1000 --show 3   # in thử 3 chunk mẫu
    python scripts/step4_compare_chunking.py --sizes 1000 --overlap 0 100 200 400   # so sánh overlap
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # để import src.*
# Khi chạy một script, Python tự thêm thư mục chứa script (scripts/) vào sys.path, nên có thể
# import trực tiếp step3_naive_chunking để TÁI SỬ DỤNG đúng các hàm đo của Step 3. Nhờ vậy hai
# cách chunk được đo bằng CÙNG một thước, so sánh mới công bằng.
from step3_naive_chunking import SENTENCE_END, analyse, build_text, naive_chunks  # noqa: E402

from src.chunker import Chunk, chunk_world  # noqa: E402
from src.loader import World, find_world_file, load_world  # noqa: E402


def verify(world: World, chunks: list[Chunk], size: int, min_chars: int) -> list[str]:
    """Kiểm tra các "bất biến" (điều luôn phải đúng) của chunker. Trả về list lỗi; rỗng = ổn.

    Đây là phiên bản thủ công của unit test: nếu chunker có bug làm mất đoạn, để hở, hay tạo
    chunk quá khổ thì hàm này sẽ báo ngay.
    """
    errors = []
    # Gom chunk theo chương: {số_chương: [các chunk của chương đó theo thứ tự]}
    by_chapter: dict[int, list[Chunk]] = {}
    for c in chunks:
        by_chapter.setdefault(c.chapter_no, []).append(c)
    for ch in world.chapters:
        cs = by_chapter.get(ch.chapter_no, [])
        if not cs:
            errors.append(f"chương {ch.chapter_no}: không có chunk")
            continue
        # Bất biến 1 - Phủ kín: chunk đầu phải bắt đầu từ đoạn 1, chunk cuối phải tới đoạn cuối.
        if cs[0].para_start != 1 or cs[-1].para_end != len(ch.paragraphs):
            errors.append(f"chương {ch.chapter_no}: không phủ hết đoạn 1..{len(ch.paragraphs)}")
        # Bất biến 2 - Liền mạch: chunk sau không được bắt đầu SAU đoạn cuối của chunk trước + 1
        # (nếu có thì ở giữa bị hở, mất đoạn); và không được lùi lại so với chunk trước.
        for a, b in zip(cs, cs[1:]):
            if b.para_start > a.para_end + 1:
                errors.append(f"chương {ch.chapter_no}: hở giữa chunk {a.chunk_index} và {b.chunk_index}")
            if b.para_end < a.para_end:
                errors.append(f"chương {ch.chapter_no}: chunk {b.chunk_index} lùi lại so với chunk trước")
        # Bất biến 3 - Không mất/sửa nội dung: mỗi đoạn văn gốc phải xuất hiện NGUYÊN VĂN trong
        # ít nhất một chunk. (Bỏ qua đoạn dài hơn size vì chúng bị cắt theo câu có chủ đích.)
        all_text = "\n".join(c.text for c in cs)
        for pi, p in enumerate(ch.paragraphs, 1):
            if len(p) <= size and p not in all_text:
                errors.append(f"chương {ch.chapter_no}: đoạn {pi} bị mất/sửa nội dung")
    # Bất biến 4 - Kích thước: chunk có thể vượt size, nhưng tối đa min_chars (do gộp đuôi ngắn).
    for c in chunks:
        if c.n_chars > size + min_chars:
            errors.append(f"{c.chunk_id}: {c.n_chars} ký tự > size+min_chars")
        if not c.text.strip():
            errors.append(f"{c.chunk_id}: rỗng")
    return errors


def pct(x: int, n: int) -> str:
    """Định dạng 'số lượng (phần trăm%)' cho bảng, ví dụ '  22 ( 5.3%)'."""
    return f"{x:>4} ({100 * x / n:4.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--overlap", type=int, nargs="+", default=[200])  # có thể truyền nhiều giá trị để so sánh
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--show", type=int, default=0, help="số chunk mẫu in ra cho mỗi cấu hình")
    args = ap.parse_args()

    world = load_world(find_world_file(args.world))
    text, spans = build_text(world)  # chuỗi liền cho naive chunking
    paras = [p for c in world.chapters for p in c.paragraphs]  # list mọi đoạn văn (để tính đường cơ sở)
    chapter_chars = sum(len(c.text) for c in world.chapters)
    print(f"Thế giới {world.world_no:02d}: {world.title} | {len(world.chapters)} chương | {len(paras)} đoạn | {chapter_chars:,} ký tự")
    # "Đường cơ sở": tỉ lệ vấn đề đã có sẵn trong DỮ LIỆU GỐC (không do chunker gây ra). Cần biết để
    # không đổ lỗi nhầm: nếu 0.8% đoạn gốc đã có dấu “ ” lệch thì chunk nào chứa nó cũng sẽ lệch.
    print(f"Đường cơ sở (dữ liệu gốc): đoạn không kết thúc bằng dấu câu = {pct(sum(1 for p in paras if p[-1] not in SENTENCE_END), len(paras))}, "
          f"đoạn có “ ” lệch = {pct(sum(1 for p in paras if p.count('“') != p.count('”')), len(paras))}\n")

    for size in args.sizes:
        for overlap in args.overlap:
            if overlap >= size:  # tổ hợp không hợp lệ, bỏ qua
                continue
            # --- Chạy chunker theo đoạn (Step 4) ---
            chunks = chunk_world(world, size=size, overlap=overlap, min_chars=args.min_chars)
            n = len(chunks)
            lens = [c.n_chars for c in chunks]
            # --- Chạy naive (Step 3) cùng tham số để so sánh ---
            naive = naive_chunks(text, size, overlap)
            nr = analyse(text, spans, naive, [])
            errs = verify(world, chunks, size, args.min_chars)

            # --- In bảng so sánh. Cột trái là tên chỉ số, hai cột phải là hai phương pháp. ---
            print("=" * 100)
            print(f"SIZE = {size}, OVERLAP = {overlap}")
            print(f"{'':36}{'NAIVE (Step 3)':>18}{'THEO ĐOẠN (Step 4)':>24}")
            print(f"{'số chunk':<36}{nr['n']:>18}{n:>24}")
            # Hai chỉ số dưới đây bằng 0 "theo thiết kế": chunker chỉ ghép đoạn nguyên vẹn và
            # chunk từng chương riêng, nên không thể xảy ra. verify() ở trên kiểm chứng điều đó.
            print(f"{'bắt đầu/kết thúc GIỮA đoạn văn':<36}{pct(nr['cut_para'], nr['n']):>18}{'0 (theo thiết kế)':>24}")
            print(f"{'kết thúc GIỮA câu':<36}{pct(nr['end_sentence'], nr['n']):>18}"
                  f"{pct(sum(1 for c in chunks if c.text.rstrip()[-1:] not in SENTENCE_END), n):>24}")
            print(f"{'lời thoại “ ” bị cắt dở':<36}{pct(nr['quotes'], nr['n']):>18}"
                  f"{pct(sum(1 for c in chunks if c.text.count('“') != c.text.count('”')), n):>24}")
            print(f"{'chunk TRỘN 2+ chương':<36}{pct(nr['cross'], nr['n']):>18}{'0 (theo thiết kế)':>24}")
            print(f"{'độ dài chunk min/TB/max (ký tự)':<36}{'':>18}{f'{min(lens)}/{sum(lens) // n}/{max(lens)}':>24}")
            # Tỉ lệ nội dung bị lặp = (tổng độ dài mọi chunk / độ dài truyện gốc) - 1.
            # Ví dụ 12% nghĩa là cứ 100 ký tự truyện thì lưu thêm 12 ký tự lặp (do overlap).
            print(f"{'tỉ lệ nội dung bị lặp (overlap)':<36}{f'{100 * (sum(e - s for s, e in naive) / len(text) - 1):.0f}%':>18}"
                  f"{f'{max(0.0, 100 * (sum(lens) / chapter_chars - 1)):.0f}%':>24}")  # max(): bỏ số âm lẻ do mất dấu "\n\n" giữa các chunk
            long_paras = sum(1 for p in paras if len(p) > size)
            print(f"{'đoạn dài > size (phải cắt theo câu)':<36}{'':>18}{long_paras:>24}")
            print(f"kiểm tra bất biến: {'✓ OK' if not errs else '✗ ' + str(len(errs)) + ' lỗi: ' + '; '.join(errs[:3])}")
            # In vài chunk mẫu (nếu có --show) để đọc bằng mắt; cắt ở 300 ký tự cho gọn.
            for c in chunks[: args.show]:
                print(f"\n--- {c.chunk_id} | chương {c.chapter_no}, đoạn {c.para_start}-{c.para_end} | {c.n_chars} ký tự ---")
                print(c.text[:300] + ("..." if c.n_chars > 300 else ""))
            print()


if __name__ == "__main__":
    main()
