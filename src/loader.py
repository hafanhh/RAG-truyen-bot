"""Document Loader: đọc một "thế giới" (1 file .txt) thành cấu trúc World -> Chapter -> paragraphs.

Định dạng file (đã kiểm tra trên file 01):
    TÊN TRUYỆN
    ====================

    1 tên truyện ( 1 )          <- heading chương: dòng không rỗng + dòng gạch ----- ngay bên dưới
    ----------------------------
    <đoạn văn>                  <- mỗi đoạn văn là đúng 1 dòng, cách nhau bằng dòng trống
    ...
    81 tên truyện ( xong )      <- chương cuối ghi "xong"

Loader chỉ lo việc ĐỌC + TÁCH CHƯƠNG. Chunking nằm ở bước sau (src/chunker.py).
Chạy thử:  python -m src.loader            (file 01)
           python -m src.loader 1          (thế giới số 1)
"""
# `from __future__ import annotations`: cho phép viết type hint kiểu `Chapter | None`,
# `list[str]` mà không lỗi (hint được xử lý lười, không chạy lúc định nghĩa).
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.config import STORY_DIR

# ---------------------------------------------------------------------------
# Các regex nhận diện cấu trúc file (biên dịch sẵn một lần cho nhanh)
# ---------------------------------------------------------------------------

# Dòng gạch dưới heading: từ 5 dấu "-" trở lên. fullmatch => cả dòng phải là gạch.
_UNDERLINE = re.compile(r"-{5,}")

# Heading chương, ví dụ "12 phú thiếu bạn gái cũ ( 12 )" hoặc "81 phú thiếu bạn gái cũ ( xong )".
#   ^(\d+)          nhóm 1: số chương ở đầu dòng
#   \s+(.*?)\s*     nhóm 2: tên (lười - lấy ít nhất có thể) để dành chỗ cho phần "( ... )" ở cuối
#   \(\s*(\d+|xong)\s*\)$  nhóm 3: bên trong ngoặc là một số, hoặc chữ "xong" (chương cuối)
_HEADING = re.compile(r"^(\d+)\s+(.*?)\s*\(\s*(\d+|xong)\s*\)\s*$")

# Tên file, ví dụ "01 - Phú thiếu bạn gái cũ.txt": nhóm 1 = số thế giới, nhóm 2 = tên truyện.
_FILENAME = re.compile(r"^(\d+)\s*-\s*(.+)\.txt$")


# ---------------------------------------------------------------------------
# Cấu trúc dữ liệu
# ---------------------------------------------------------------------------
# @dataclass: Python tự sinh __init__, __repr__ ... từ danh sách field, đỡ phải viết tay.


@dataclass
class Chapter:
    """Một chương: các đoạn văn + thông tin để sau này gắn metadata/citation."""

    world_no: int  # số thế giới (1..20), lấy từ tên file
    world_title: str  # tên thế giới, lấy từ tên file
    chapter_no: int  # số chương, lấy từ heading, bắt đầu từ 1
    heading: str  # dòng heading gốc, vd "12 phú thiếu bạn gái cũ ( 12 )"
    # field(default_factory=list): mỗi Chapter có list RIÊNG của nó. (Viết `= []` trực tiếp là
    # lỗi kinh điển: mọi object sẽ dùng chung một list.)
    paragraphs: list[str] = field(default_factory=list)
    is_last: bool = False  # heading ghi "( xong )" => đây là chương cuối của thế giới

    # @property: truy cập như một thuộc tính (chapter.text) nhưng thực chất là tính toán mỗi lần gọi.
    @property
    def text(self) -> str:
        """Toàn văn chương: các đoạn nối bằng một dòng trống."""
        return "\n\n".join(self.paragraphs)

    @property
    def n_words(self) -> int:
        """Số "từ" (tách theo khoảng trắng). Với tiếng Việt, mỗi từ này thực ra là một âm tiết."""
        return sum(len(p.split()) for p in self.paragraphs)


@dataclass
class World:
    """Một thế giới = một file = một truyện hoàn chỉnh, gồm nhiều chương."""

    world_no: int
    title: str
    source_file: Path  # đường dẫn file gốc, để truy ngược khi cần
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(c.n_words for c in self.chapters)


# ---------------------------------------------------------------------------
# Tìm file và đọc file
# ---------------------------------------------------------------------------


def find_world_file(world_no: int, story_dir: Path = STORY_DIR) -> Path:
    """Tìm file của thế giới số `world_no`, ví dụ 1 -> '01 - Phú thiếu bạn gái cũ.txt'."""
    # f"{world_no:02d}" => số nguyên đệm 0 cho đủ 2 chữ số (1 -> "01"). glob("01 - *.txt") khớp
    # mọi file bắt đầu bằng "01 - ". Phải khớp ĐÚNG 1 file, nếu không thì báo lỗi rõ ràng.
    matches = sorted(story_dir.glob(f"{world_no:02d} - *.txt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Cần đúng 1 file cho thế giới {world_no:02d} trong {story_dir}, thấy {len(matches)}")
    return matches[0]


def load_world(path: Path) -> World:
    """Đọc 1 file thành World. Raise ValueError nếu cấu trúc chương bất thường.

    Cách làm: duyệt file từng dòng như một "máy trạng thái" đơn giản:
      - gặp heading  -> mở một Chapter mới
      - gặp dòng chữ -> thêm vào chương đang mở như một đoạn văn
    Mọi bất thường (heading sai, sai thứ tự, thiếu "xong"...) đều báo lỗi ngay, để lỗi dữ liệu
    lộ ra sớm thay vì âm thầm làm hỏng các bước sau (chunking, embedding...).
    """
    path = Path(path)

    # --- Lấy số thế giới + tên từ tên file ---
    m = _FILENAME.match(path.name)
    if not m:
        raise ValueError(f"Tên file không theo dạng 'NN - Tên.txt': {path.name}")
    world_no, world_title = int(m.group(1)), m.group(2).strip()

    # --- Đọc file ---
    # NFC: chuẩn hóa Unicode. Một chữ như "ế" có thể được lưu bằng 1 ký tự dựng sẵn (NFC) hoặc
    # 2 ký tự "e + dấu" (NFD). Nếu không thống nhất, "ế" ở chỗ này sẽ KHÔNG bằng "ế" ở chỗ kia
    # khi so khớp chuỗi / tìm kiếm. Ép về NFC để luôn nhất quán.
    raw = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    # Đổi xuống dòng kiểu Windows (\r\n) về \n rồi tách thành list các dòng.
    lines = raw.replace("\r\n", "\n").split("\n")

    world = World(world_no=world_no, title=world_title, source_file=path)
    current: Chapter | None = None  # chương đang mở; None nghĩa là chưa gặp heading nào
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Dòng kế tiếp (nếu đã hết file thì coi như chuỗi rỗng)
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        # Heading = dòng không rỗng + dòng ngay dưới là hàng gạch "-----".
        if line and _UNDERLINE.fullmatch(next_line):
            hm = _HEADING.match(line)
            if not hm:
                # i + 1 vì dòng trong editor đếm từ 1 còn index list đếm từ 0
                raise ValueError(f"{path.name}:{i + 1}: heading không đúng dạng: {line!r}")
            chapter_no = int(hm.group(1))
            # Số chương phải liên tục 1, 2, 3...: chương mới phải có số = (số chương đã đọc) + 1.
            if chapter_no != len(world.chapters) + 1:
                raise ValueError(f"{path.name}:{i + 1}: chương {chapter_no} sai thứ tự (mong đợi {len(world.chapters) + 1})")
            current = Chapter(world_no, world_title, chapter_no, line, is_last=hm.group(3) == "xong")
            world.chapters.append(current)
            i += 2  # nhảy qua cả dòng heading lẫn dòng gạch
            continue

        # Dòng thường: nếu đã có chương đang mở và dòng không rỗng thì đó là một đoạn văn.
        # Phần trước chương 1 (tên truyện + dòng "====") bị bỏ qua tự nhiên vì current còn là None.
        if current is not None and line:
            current.paragraphs.append(line)
        i += 1

    # --- Kiểm tra toàn vẹn sau khi đọc xong ---
    if not world.chapters:
        raise ValueError(f"{path.name}: không tìm thấy chương nào")
    if not world.chapters[-1].is_last:
        # Chương cuối phải ghi "( xong )"; nếu không, file có thể bị cắt cụt giữa chừng.
        raise ValueError(f"{path.name}: chương cuối không có '( xong )' - file có thể bị cụt")
    empty = [c.chapter_no for c in world.chapters if not c.paragraphs]
    if empty:
        raise ValueError(f"{path.name}: chương rỗng: {empty}")
    return world


# ---------------------------------------------------------------------------
# Thống kê nhanh (để nhìn dữ liệu bằng mắt, và chạy thử bằng `python -m src.loader`)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _cached_world(world_no: int) -> World:
    return load_world(find_world_file(world_no))


def paragraph_text(chapter_no: int, para_no: int, world_no: int = 1, through: str | None = None) -> str:
    """Văn bản của MỘT đoạn theo tham chiếu (chương, đoạn), nạp lúc chạy từ Books/.

    Dùng ở những chỗ cần mẫu văn bản (thí nghiệm, test) thay vì chép nguyên văn vào file được commit: repo public còn
    Books/ thì không. `through`: cắt ngay sau lần xuất hiện đầu tiên của cụm này (rồi kết thúc bằng dấu chấm)."""
    chapter = _cached_world(world_no).chapters[chapter_no - 1]
    if not 1 <= para_no <= len(chapter.paragraphs):
        raise ValueError(f"chương {chapter_no} không có đoạn {para_no} (có {len(chapter.paragraphs)})")
    text = chapter.paragraphs[para_no - 1]
    if through is not None:
        pos = text.find(through)
        if pos < 0:
            raise ValueError(f"chương {chapter_no} đoạn {para_no} không chứa {through!r}")
        text = text[: pos + len(through)].rstrip(",;") + "."
    return text


def print_stats(world: World) -> None:
    ws = [c.n_words for c in world.chapters]  # số từ mỗi chương
    ps = [len(c.paragraphs) for c in world.chapters]  # số đoạn mỗi chương
    print(f"Thế giới {world.world_no:02d}: {world.title}")
    print(f"  file: {world.source_file.name}")
    print(f"  {len(world.chapters)} chương | {sum(ps)} đoạn | {world.n_words} từ")
    print(f"  từ/chương: min {min(ws)}, trung bình {sum(ws) // len(ws)}, max {max(ws)}")
    print(f"  đoạn/chương: min {min(ps)}, trung bình {sum(ps) // len(ps)}, max {max(ps)}")
    longest = max(len(p.split()) for c in world.chapters for p in c.paragraphs)
    print(f"  đoạn dài nhất: {longest} từ")
    # min/max với key=...: chọn Chapter có n_words nhỏ nhất / lớn nhất
    short = min(world.chapters, key=lambda c: c.n_words)
    long_ = max(world.chapters, key=lambda c: c.n_words)
    print(f"  chương ngắn nhất: {short.chapter_no} ({short.n_words} từ) | dài nhất: {long_.chapter_no} ({long_.n_words} từ)")


# Chạy trực tiếp: `python -m src.loader [số thế giới]`. Không có tham số thì mặc định thế giới 1.
if __name__ == "__main__":
    no = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print_stats(load_world(find_world_file(no)))
