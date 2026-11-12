"""[kiểm tra viên v2: thêm Claim.fits_question; kết quả Step 7-8 dùng v1]
Lượt KIỂM TRA bằng LLM cho một câu trả lời đã có (dùng ở Step 7 và vòng sửa Step 8).

Đầu vào của kiểm tra viên = ĐÚNG những gì lượt trả lời đã thấy (câu hỏi + các chunk đã gửi, mỗi đoạn có số [đN]) + câu trả
lời và trích dẫn của nó. KHÔNG bao giờ đưa đáp án chuẩn / đáp án nguồn cho kiểm tra viên: nó mô phỏng lúc hệ thống phục vụ
người dùng thật. Đáp án chuẩn chỉ dùng ở ngoài, để chấm lại kết quả kiểm tra.

Đầu ra (JSON, lược đồ Verification): đúng nhân vật-cảnh-thời điểm? | từng nhận định có căn cứ? | bỏ sót thông tin cần thiết
ĐÃ CÓ trong các đoạn? | địa chỉ trích dẫn (chương, số đoạn) có đúng? | hành động đề xuất: giữ / sửa từ bằng chứng hiện có /
cần tìm thêm.

Giới hạn đã biết (xem CLAUDE.md, Experiment A): cùng model tự kiểm tra; bắt tốt sai cảnh/thời điểm và sai địa chỉ trích dẫn,
bắt kém chuyện bỏ sót thông tin đã có và độ chặt về khoảnh khắc; một trích dẫn có thật chưa chắc đúng ý.
"""
from __future__ import annotations

import json
import time
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from src.embedder import MAX_RETRIES, is_retryable


class Check(BaseModel):
    verdict: Literal["đạt", "không đạt", "chưa xác định"]
    reason: str


class Claim(BaseModel):
    claim: str
    supported: bool
    support: str  # "mã_chunk đN" nơi có căn cứ, hoặc chuỗi rỗng nếu không có
    fits_question: bool  # (kiểm tra viên v2) nhận định có thuộc đúng nhân vật/cảnh/thời điểm mà CÂU HỎI hỏi không


class Omission(BaseModel):
    chunk_id: str
    paragraph: int
    quote: str  # trích NGUYÊN VĂN ngắn từ đoạn đó
    why: str


class CitationCheck(BaseModel):
    chunk_id: str
    stated_paragraph: int
    correct: bool
    actual_paragraph: int  # số đoạn đúng nếu tìm được; 0 nếu không rõ
    note: str


class Verification(BaseModel):
    scene_check: Check
    claims: list[Claim]
    omissions: list[Omission]
    citation_checks: list[CitationCheck]
    action: Literal["giữ", "sửa từ bằng chứng hiện có", "cần tìm thêm"]
    action_reason: str
    missing_info: str


SYSTEM = """Bạn là kiểm tra viên cho một hệ thống hỏi đáp về truyện. Bạn nhận: CÂU HỎI, các đoạn truyện đã được gửi cho \
hệ thống trả lời (mỗi đoạn có số [đN]), và CÂU TRẢ LỜI CẦN KIỂM TRA cùng các trích dẫn của nó. Hãy kiểm tra câu trả lời \
CHỈ dựa vào các đoạn được cung cấp; không dùng kiến thức bên ngoài; không viết lại câu trả lời.

Trả về JSON gồm:
1. scene_check: câu trả lời có đúng NHÂN VẬT, CẢNH và THỜI ĐIỂM mà câu hỏi hỏi không? "đạt" / "không đạt" / "chưa xác định" \
(dùng "chưa xác định" nếu các đoạn không cho biết thời điểm hay cảnh). Nếu câu hỏi nêu vị trí trong truyện (vd "ở đầu \
truyện") thì đoạn được dẫn phải thực sự thuộc vị trí đó; một cảnh tương tự ở nơi khác là "không đạt". reason nêu lý do ngắn.
2. claims: tách câu trả lời thành từng nhận định về truyện; mỗi nhận định ghi supported (có đoạn nói rõ), support \
("mã_chunk đN" nơi có căn cứ, hoặc rỗng nếu thiếu căn cứ) và fits_question (true nếu nhận định thuộc đúng nhân vật, cảnh và \
thời điểm mà CÂU HỎI hỏi; false nếu nó nói về cảnh/thời điểm khác dù có thật trong truyện). Nếu câu trả lời chỉ là \
"không đủ thông tin" thì claims để trống.
3. omissions: thông tin CẦN cho câu hỏi, ĐÃ có trong các đoạn được cung cấp nhưng câu trả lời bỏ sót. Mỗi mục: chunk_id, \
paragraph (số đoạn), quote (trích NGUYÊN VĂN ngắn từ đoạn đó), why. Không nêu thông tin không liên quan đến câu hỏi.
4. citation_checks: với MỖI trích dẫn có cấu trúc của câu trả lời: đoạn trích có nằm đúng ở số đoạn đã ghi không \
(correct); nếu sai, ghi actual_paragraph là số đoạn đúng (0 nếu không rõ).
5. action: "giữ" nếu câu trả lời đã đúng và đủ trong phạm vi bằng chứng hiện có; "sửa từ bằng chứng hiện có" nếu có thể \
sửa (bỏ sót, nhận định thiếu căn cứ, sai địa chỉ trích dẫn) mà KHÔNG cần đoạn mới; "cần tìm thêm" nếu các đoạn hiện có \
không đủ để trả lời đúng câu hỏi (thiếu thông tin hoặc sai cảnh). Kèm action_reason; nếu "cần tìm thêm" thì missing_info \
mô tả cần tìm thông tin gì (ngược lại để chuỗi rỗng).
Chỉ nêu lỗi khi có căn cứ trong các đoạn; nếu câu trả lời đúng thì để các danh sách lỗi trống và chọn "giữ".
"""


def build_input(trace: dict) -> str:
    """Giữ nguyên câu hỏi + chunk đã gửi (trace['prompt'], KHÔNG gồm ghi chú sửa lỗi nếu có) rồi thêm câu trả lời cần kiểm tra."""
    a = trace["llm"]
    cites = "\n".join(f"- [{c['chunk_id']}, đoạn {c['paragraph']}] “{c['quote']}”" for c in a["citations"]) or "(không có)"
    return (trace["prompt"] + "\n\n=== CÂU TRẢ LỜI CẦN KIỂM TRA ===\n"
            f"verdict của hệ thống: {a['verdict']}\ncâu trả lời: {a['answer']}\n"
            f"phần còn thiếu hệ thống nêu: {a['missing'] or '(không)'}\ntrích dẫn có cấu trúc:\n{cites}")


def call_verifier(client, model: str, text: str) -> tuple[Verification | None, dict, str | None]:
    """Gọi kiểm tra viên. Trả về (kết quả | None, usage gồm token và latency_s, lỗi | None)."""
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.0,
                                      response_mime_type="application/json", response_schema=Verification)
    t0 = time.time()
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=text, config=cfg)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_RETRIES or not is_retryable(exc):
                return None, {"latency_s": round(time.time() - t0, 1)}, f"{type(exc).__name__}: {str(exc)[:300]}"
            time.sleep(min(10.0 * 2**attempt, 90.0))
    u = resp.usage_metadata
    usage = {"prompt": getattr(u, "prompt_token_count", None), "output": getattr(u, "candidates_token_count", None),
             "latency_s": round(time.time() - t0, 1)}
    try:
        return Verification.model_validate(json.loads(resp.text)), usage, None
    except Exception as exc:  # noqa: BLE001
        return None, usage, f"đầu ra không hợp lệ ({type(exc).__name__}): {(resp.text or '')[:200]!r}"
