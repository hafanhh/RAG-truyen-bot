"""Chunker: cắt mỗi chương thành các chunk theo ĐOẠN VĂN nguyên vẹn (kích thước đo bằng ký tự).

Quy tắc:
  1. Không bao giờ vượt ranh giới chương (mỗi chương được chunk độc lập).
  2. Gom các đoạn văn liền nhau cho tới khi thêm đoạn nữa sẽ vượt `size` ký tự thì đóng chunk.
  3. Đoạn văn dài hơn `size` mới bị cắt - và chỉ cắt theo CÂU (fallback, hiếm khi xảy ra).
  4. Overlap theo đoạn: chunk sau lặp lại vài đoạn cuối của chunk trước, tổng không quá `overlap` ký tự.
  5. Đuôi chương quá ngắn (< `min_chars` nội dung mới) được gộp vào chunk trước, nên chunk có thể
     vượt `size` tối đa `min_chars` ký tự.

Ví dụ (size=1000, overlap=200). Giả sử chương có các đoạn P1..P6 với độ dài:
    P1=400  P2=450  P3=300  P4=150  P5=500  P6=100
  - Chunk 0: P1 (400) + P2 (450) = ~852 (cộng "\\n\\n" giữa các đoạn). Thêm P3 sẽ thành ~1154 > 1000
    => đóng chunk 0 với [P1, P2].
  - Overlap: lấy các đoạn CUỐI của chunk 0 sao cho tổng <= 200. P2 dài 450 > 200 => không lấy gì.
  - Chunk 1 bắt đầu từ P3: P3 (300) + P4 (150) = ~452, thêm P5 (500) thành ~954 <= 1000 => nhận P5.
    Thêm P6 thành ~1056 > 1000 => đóng chunk 1 với [P3, P4, P5].
  - Overlap: P5 dài 500 > 200 nên không lấy; nhưng P4 mà đứng cuối thì sẽ được lấy (150 <= 200).
  - Chunk 2 chỉ có P6 (100 < min_chars=250) => là "đuôi quá ngắn" => gộp ngược vào chunk 1.

Chưa có ở bước này (làm ở Step 5): prev_id/next_id, token_count, ghi ra JSONL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.loader import Chapter, World

# Giá trị mặc định (có thể đổi khi gọi hàm). Tất cả đều tính bằng KÝ TỰ, chưa phải token.
DEFAULT_SIZE = 1000  # kích thước mục tiêu của một chunk
DEFAULT_OVERLAP = 200  # tối đa bao nhiêu ký tự lặp lại từ chunk trước sang chunk sau
DEFAULT_MIN_CHARS = 250  # đuôi chương ngắn hơn mức này sẽ bị gộp vào chunk trước

# Regex tách câu (chỉ dùng cho trường hợp đoạn văn quá dài). Giải thích:
#   (?<=[.!?…”])  "lookbehind": vị trí NGAY SAU một dấu kết câu (. ! ? … ”) - không "ăn" ký tự đó
#   \s+            rồi tới khoảng trắng => đây là chỗ tách. Dấu câu vẫn ở lại cuối câu trước.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…”])\s+")

# "Unit" = đơn vị nhỏ nhất mà chunker làm việc: (chỉ số đoạn văn gốc trong chương, văn bản).
# Thường 1 unit = 1 đoạn văn nguyên vẹn. Chỉ khi đoạn văn quá dài mới bị tách thành nhiều unit
# (các unit đó cùng chỉ số đoạn). Nhớ chỉ số đoạn để sau này làm citation "đoạn 12-15".
Unit = tuple[int, str]


@dataclass
class Chunk:
    """Một chunk: văn bản + "địa chỉ" của nó trong truyện (để truy ngược nguồn / trích dẫn)."""

    chunk_id: str  # id duy nhất, vd "w01-c012-k02" = thế giới 1, chương 12, chunk số 2
    world_no: int
    world_title: str
    chapter_no: int
    chunk_index: int  # thứ tự chunk trong chương, từ 0
    para_start: int  # đoạn đầu (1-based, để dùng làm citation)
    para_end: int  # đoạn cuối (1-based, bao gồm)
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Các hàm phụ (bắt đầu bằng "_" = dùng nội bộ trong file này)
# ---------------------------------------------------------------------------


def _join(units: list[Unit]) -> str:
    """Nối các unit thành văn bản chunk.

    Hai unit cùng thuộc một đoạn gốc (đoạn bị cắt theo câu) thì nối bằng dấu cách để khôi phục
    đoạn nguyên; khác đoạn thì nối bằng dòng trống ("\\n\\n") như trong file gốc.
    """
    out = ""
    for k, (pi, t) in enumerate(units):
        if k:  # từ unit thứ 2 trở đi mới cần dấu nối
            out += " " if units[k - 1][0] == pi else "\n\n"
        out += t
    return out


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Cắt một đoạn quá dài (> max_chars) thành các mảnh <= max_chars, ưu tiên cắt theo câu.

    Hiếm khi chạy: ở World 01 chỉ 1 đoạn dài hơn 500 ký tự và 0 đoạn dài hơn 1000.
    """
    pieces: list[str] = []
    cur = ""  # mảnh đang gom
    for sent in _SENTENCE_SPLIT.split(text):  # duyệt từng câu
        # Trường hợp hiếm nhất: BẢN THÂN một câu đã dài hơn max_chars -> phải cắt cứng.
        while len(sent) > max_chars:
            if cur:  # đẩy mảnh đang gom ra trước
                pieces.append(cur)
                cur = ""
            # Cắt tại dấu cách cuối cùng trước max_chars để không cắt giữa một từ.
            # rfind trả về -1 nếu không có dấu cách; khi đó đành cắt đúng max_chars.
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()  # phần còn lại tiếp tục vòng while
        # Thêm câu vào mảnh hiện tại nếu còn vừa, không thì đóng mảnh và mở mảnh mới.
        if cur and len(cur) + 1 + len(sent) > max_chars:  # +1 cho dấu cách nối
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def _to_units(paragraphs: list[str], max_chars: int) -> list[Unit]:
    """Biến list đoạn văn thành list unit: đoạn vừa thì giữ nguyên, đoạn quá dài thì tách theo câu."""
    units: list[Unit] = []
    for pi, p in enumerate(paragraphs):
        if len(p) <= max_chars:
            units.append((pi, p))
        else:
            # Mọi mảnh của đoạn này giữ cùng chỉ số đoạn `pi`.
            units.extend((pi, piece) for piece in _split_long_paragraph(p, max_chars))
    return units


def _overlap_tail(units: list[Unit], overlap: int) -> list[Unit]:
    """Chọn các unit CUỐI của chunk vừa đóng để lặp lại ở đầu chunk kế tiếp (overlap).

    Chỉ lấy unit NGUYÊN VẸN (không cắt đôi một đoạn), tổng độ dài <= overlap.
    Luôn chừa lại ít nhất 1 unit "mới" để chunk kế tiếp chắc chắn tiến lên (không lặp vô hạn).
    """
    tail: list[Unit] = []
    # reversed(units[1:]): đi từ cuối về đầu; bỏ units[0] để không bao giờ lấy toàn bộ chunk.
    for u in reversed(units[1:]):
        if len(_join([u] + tail)) > overlap:  # thêm nữa sẽ vượt ngân sách overlap -> dừng
            break
        tail.insert(0, u)  # chèn vào ĐẦU để giữ đúng thứ tự gốc
    return tail


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------


def chunk_chapter(
    chapter: Chapter,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Cắt MỘT chương thành list Chunk (thuật toán "greedy packing": gom tham lam)."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Cần size > 0 và 0 <= overlap < size")
    units = _to_units(chapter.paragraphs, size)

    # groups: mỗi phần tử là (các unit của 1 chunk, số unit ĐẦU TIÊN là phần overlap lặp lại).
    # Cần biết số unit overlap để: (a) không đóng chunk khi nó chỉ toàn overlap,
    # (b) khi gộp đuôi, biết đâu là phần "nội dung mới".
    groups: list[tuple[list[Unit], int]] = []
    cur: list[Unit] = []  # chunk đang gom
    carried = 0  # số unit đầu của `cur` là phần overlap
    for u in units:
        # Đóng chunk khi: có nội dung MỚI (len(cur) > carried) và thêm `u` sẽ vượt `size`.
        if cur and len(cur) > carried and len(_join(cur + [u])) > size:
            groups.append((cur, carried))
            # Chuẩn bị chunk kế: bắt đầu bằng overlap lấy từ cuối chunk vừa đóng...
            carry = _overlap_tail(cur, overlap)
            # ...nhưng phải còn chỗ cho unit mới `u`; nếu không, bỏ bớt overlap từ phía đầu.
            while carry and len(_join(carry + [u])) > size:
                carry.pop(0)
            cur, carried = carry + [u], len(carry)
        else:
            cur.append(u)  # còn vừa -> nhận thêm
    if cur:  # chunk cuối cùng
        groups.append((cur, carried))

    # Đuôi chương quá ngắn -> gộp vào chunk trước, để tránh một chunk nhỏ tí hon
    # (chunk quá ngắn ít ngữ cảnh, khi embedding sẽ cho vector kém chất lượng).
    if len(groups) >= 2:
        last, last_carried = groups[-1]
        new_part = last[last_carried:]  # chỉ phần nội dung mới của chunk cuối (bỏ phần overlap)
        if len(_join(new_part)) < min_chars:
            groups.pop()
            # Nối phần mới vào chunk trước. Phần overlap của chunk cuối vốn chính là đuôi chunk
            # trước, nên chỉ cần nối `new_part` là liền mạch, không bị lặp.
            groups[-1] = (groups[-1][0] + new_part, groups[-1][1])

    # Đổi từng nhóm unit thành đối tượng Chunk kèm metadata vị trí.
    chunks = []
    for k, (us, _) in enumerate(groups):
        chunks.append(
            Chunk(
                chunk_id=f"w{chapter.world_no:02d}-c{chapter.chapter_no:03d}-k{k:02d}",
                world_no=chapter.world_no,
                world_title=chapter.world_title,
                chapter_no=chapter.chapter_no,
                chunk_index=k,
                # Chỉ số đoạn trong unit là 0-based; +1 để thành số đoạn "người đọc" (1-based).
                para_start=us[0][0] + 1,
                para_end=us[-1][0] + 1,
                text=_join(us),
            )
        )
    return chunks


def chunk_world(world: World, **kwargs) -> list[Chunk]:
    """Cắt cả thế giới: chunk từng chương một rồi nối kết quả. **kwargs = size/overlap/min_chars."""
    return [c for chapter in world.chapters for c in chunk_chapter(chapter, **kwargs)]
