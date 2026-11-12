"""Step 1 - Demo "vì sao cần RAG".

Hỏi cùng một câu hỏi theo 2 cách:
  A) KHÔNG có context  -> LLM chỉ dựa vào trí nhớ (parametric knowledge)
  B) CÓ context        -> ta đưa đúng đoạn truyện vào prompt (Retrieve -> Augment -> Generate)

Ở đây "Retrieve" là làm tay (ta tự lấy Chương 1). Từ V1, bước đó sẽ do hệ thống tự làm.

Chạy từ thư mục gốc:  python scripts/step1_why_rag.py
Đổi câu hỏi:          sửa list QUESTIONS bên dưới, hoặc truyền câu hỏi qua argv.
Đổi model:            GEMINI_MODEL=gemini-3.5-flash python scripts/step1_why_rag.py
"""
import os
import re
import sys
from pathlib import Path

# Khi chạy `python scripts/xxx.py`, Python chỉ thêm thư mục scripts/ vào đường tìm module,
# nên `import src...` sẽ lỗi. Dòng này thêm thư mục gốc của project vào đầu sys.path để
# import được package `src`. (Script nằm ở <ROOT>/scripts/ -> .parent.parent = <ROOT>.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hai import dưới đặt SAU dòng sys.path ở trên nên linter báo E402; "noqa: E402" là
# để tắt cảnh báo đó, vì thứ tự này là cố ý.
from google import genai  # noqa: E402  (SDK google-genai để gọi Gemini)

from src.config import BOOKS_DIR, require_api_key  # noqa: E402

# Tên model lấy từ biến môi trường GEMINI_MODEL nếu có, không thì dùng giá trị mặc định.
# (Các model gemini-2.5-* không còn dùng được với tài khoản mới.)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
STORY_FILE = BOOKS_DIR / "Truyen" / "01 - Phú thiếu bạn gái cũ.txt"
STORY_TITLE = "Phú thiếu bạn gái cũ"  # chỉ dùng để nhắc tên truyện trong prompt không-context

# 3 câu có đáp án trong Chương 1, 1 câu KHÔNG có đáp án (để test "không đủ thông tin")
QUESTIONS = [
    "Lúc Lãnh Tử Việt đưa chi phiếu, Đường Quả đang cầm vật gì trong tay?",
    "Lãnh Tử Việt đưa cho Đường Quả loại bút gì để điền chi phiếu?",
    "Đường Quả định điền số tiền bao nhiêu vào chi phiếu?",
    "Lục Kỳ làm nghề gì?",  # không có trong Chương 1
]


def load_chapter_1() -> str:
    """Tạm thời cắt Chương 1 bằng tay. Loader đúng nghĩa sẽ làm ở Step 2."""
    lines = STORY_FILE.read_text(encoding="utf-8").split("\n")
    # Tìm các dòng heading chương: dòng không rỗng mà dòng NGAY DƯỚI là một hàng gạch "-----".
    #   lines[:-1]            : bỏ dòng cuối vì nó không có "dòng ngay dưới"
    #   re.fullmatch(r"-{5,}") : cả dòng chỉ gồm từ 5 dấu gạch trở lên
    heads = [i for i, l in enumerate(lines[:-1]) if l.strip() and re.fullmatch(r"-{5,}", lines[i + 1].strip())]
    # Nội dung Chương 1 nằm từ sau dòng gạch của heading đầu (heads[0] + 2)
    # tới trước heading của chương 2 (heads[1]).
    return "\n".join(lines[heads[0] + 2 : heads[1]]).strip()


def ask(client: genai.Client, prompt: str) -> str:
    """Gửi một prompt tới Gemini và trả về phần text của câu trả lời."""
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    # resp.text có thể là None (vd bị chặn nội dung) nên `or ""` để tránh lỗi .strip() trên None.
    return (resp.text or "").strip()


def prompt_no_context(q: str) -> str:
    """Prompt A: chỉ nêu tên truyện + câu hỏi. LLM phải tự "nhớ" hoặc... tự bịa."""
    return (
        f"Trả lời ngắn gọn bằng tiếng Việt về truyện '{STORY_TITLE}' "
        f"(series xuyên nhanh, nhân vật Đường Quả).\nCâu hỏi: {q}"
    )


def prompt_with_context(q: str, context: str) -> str:
    """Prompt B: đây chính là bước "Augment" của RAG.

    Hai điểm mấu chốt của một prompt RAG tốt:
      1. Đưa đoạn truyện (context) vào, kẹp giữa hai dòng đánh dấu để LLM thấy rõ ranh giới.
      2. Ràng buộc "CHỈ trả lời dựa trên đoạn truyện" + một câu từ chối cố định khi thiếu thông tin
         -> ngăn LLM bịa (grounding).
    """
    return (
        "Bạn là trợ lý đọc truyện. CHỈ trả lời dựa trên đoạn truyện bên dưới. "
        "Nếu đoạn truyện không có thông tin để trả lời, hãy nói đúng câu: "
        "'Không đủ thông tin trong đoạn truyện.' Trả lời ngắn gọn bằng tiếng Việt.\n\n"
        f"=== ĐOẠN TRUYỆN ===\n{context}\n=== HẾT ĐOẠN TRUYỆN ===\n\nCâu hỏi: {q}"
    )


def main() -> None:
    # Câu hỏi lấy từ dòng lệnh nếu có (python ... "câu 1" "câu 2"), không thì dùng QUESTIONS mặc định.
    questions = sys.argv[1:] or QUESTIONS
    client = genai.Client(api_key=require_api_key())
    context = load_chapter_1()
    print(f"Model: {MODEL} | Context: Chương 1, {len(context.split())} từ\n")
    for i, q in enumerate(questions, 1):
        print(f"{'=' * 70}\nCâu {i}: {q}")
        # Cùng một câu hỏi, hỏi 2 lần: không context (A) và có context (B), rồi so sánh.
        print(f"\n[A] KHÔNG context:\n{ask(client, prompt_no_context(q))}")
        print(f"\n[B] CÓ context (RAG thủ công):\n{ask(client, prompt_with_context(q, context))}\n")


# Chỉ chạy main() khi file được chạy trực tiếp, không chạy khi bị import từ file khác.
if __name__ == "__main__":
    main()
