"""Cấu hình trung tâm của project: đường dẫn thư mục và API key.

Nguyên tắc: MỌI file khác chỉ `from src.config import ...` để lấy đường dẫn / key,
không tự đọc biến môi trường hay tự ghép đường dẫn. Nhờ vậy muốn đổi thư mục dữ liệu
hay cách nạp key thì chỉ phải sửa ở một nơi duy nhất.
"""
import os
from pathlib import Path

from dotenv import load_dotenv  # thư viện python-dotenv: đọc file .env rồi nạp vào biến môi trường

# ROOT = thư mục gốc của project (thư mục chứa src/, Books/, data/, .env ...).
#   __file__                 -> đường dẫn tới file này:  <ROOT>/src/config.py
#   .resolve()               -> đổi thành đường dẫn tuyệt đối (gỡ bỏ "..", symlink)
#   .parent                  -> <ROOT>/src
#   .parent.parent           -> <ROOT>
# Tính từ vị trí file (thay vì từ thư mục đang đứng khi chạy lệnh) để code chạy đúng
# dù bạn chạy từ đâu.
ROOT = Path(__file__).resolve().parent.parent

# Nạp file .env vào os.environ. File này chứa GEMINI_API_KEY và KHÔNG được commit
# (đã nằm trong .gitignore). Nếu .env không tồn tại, load_dotenv chỉ lặng lẽ bỏ qua.
load_dotenv(ROOT / ".env")

# ---- Đường dẫn dữ liệu ----
BOOKS_DIR = ROOT / "Books"  # thư mục chứa mọi sách
STORY_DIR = BOOKS_DIR / "Truyen"  # corpus đang dùng: 20 thế giới, file dạng "NN - Tên.txt"
CHUNKS_DIR = ROOT / "data" / "chunks"  # nơi sẽ ghi các chunk đã cắt (Step 5, thư mục bị git bỏ qua)
EMBED_CACHE_DIR = ROOT / "data" / "embeddings"  # cache vector đã gọi API (V1 Step 2), để khỏi gọi lại và tốn quota
CHROMA_DIR = ROOT / "data" / "chroma"  # VectorDB Chroma lưu trên đĩa (V1 Step 2)
EVAL_DIR = ROOT / "data" / "eval"  # bộ câu hỏi đánh giá viết tay + đáp án nguồn (V1 Step 3, 6)
TRACES_DIR = ROOT / "data" / "traces"  # trace JSON của từng lần answer() (V1 Step 5, chứa văn bản truyện nên git bỏ qua)

# os.getenv(tên, mặc định): lấy biến môi trường, nếu không có thì dùng chuỗi rỗng.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Model LLM dùng để trả lời (đổi bằng biến môi trường GEMINI_MODEL). gemini-3.5-flash-lite là model đã thử chạy được.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def require_api_key() -> str:
    """Trả về API key, hoặc báo lỗi rõ ràng nếu chưa cấu hình.

    Gọi hàm này ở NGAY chỗ cần dùng key (không phải lúc import), để các script không cần
    Gemini (loader, chunker) vẫn chạy được khi chưa có key.
    """
    # Hai trường hợp coi là "chưa điền": rỗng, hoặc vẫn còn giá trị mẫu copy từ .env.example.
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise RuntimeError("Chưa đặt GEMINI_API_KEY trong file .env")
    return GEMINI_API_KEY
