"""Chính sách SỬA CÂU TRẢ LỜI: cổng kích hoạt rẻ, vòng sửa một lần, và chính sách cuối (từ chối khi không còn bằng chứng).

    câu trả lời ban đầu
        │  cổng kích hoạt RẺ (không gọi LLM): verdict ≠ "đủ", hoặc kiểm tra tự động báo lỗi trích dẫn
        ├─ không kích hoạt ──► GIỮ NGUYÊN, KHÔNG qua kiểm tra  (!!! giới hạn: lỗi "sai cảnh nhưng trích dẫn sạch" lọt qua đây)
        └─ kích hoạt ──► kiểm tra (src/verifier.py)
                           ├─ giữ                       → giữ
                           ├─ sửa từ bằng chứng hiện có → trả lời lại từ chính các chunk đã gửi (kèm ghi chú kiểm tra viên)
                           └─ cần tìm thêm              → thêm chunk lân cận (±1, theo ID đã lưu) → trả lời lại
                                  ↓ kiểm tra lại → DỪNG (đúng một lần sửa)
                           ↓
                    CHÍNH SÁCH CUỐI (final_policy) trên lượt kiểm tra cuối cùng:
                        loại các nhận định không được hỗ trợ (supported=false) hoặc không thuộc đúng cảnh câu hỏi hỏi
                        (fits_question=false; nếu kiểm tra viên báo cảnh "không đạt" mà không đánh dấu nhận định nào thì coi
                        MỌI nhận định là không thuộc cảnh); nếu không còn nhận định nào → từ chối "không đủ thông tin".

Giới hạn phải nói rõ khi báo kết quả: chính sách chỉ bảo vệ những câu ĐI QUA lượt kiểm tra. Câu có verdict "đủ" và trích dẫn
sạch (như Q01 ở prompt v2 hoặc k5+mở rộng: trả lời tự tin bằng cảnh chương 59) không kích hoạt cổng nên KHÔNG được bảo vệ.
"""
from __future__ import annotations

import re

from src.chunk_store import ChunkRecord


def cheap_triggers(trace: dict) -> list[str]:
    """Tín hiệu RẺ (không LLM) cho biết câu trả lời đáng được kiểm tra. Danh sách rỗng = không kích hoạt."""
    a, v = trace.get("llm"), trace.get("verification")
    if a is None:
        return ["lỗi LLM"]
    out = []
    if a["verdict"] != "đủ":
        out.append(f"verdict={a['verdict']}")
    if v:
        if v.get("quotes_other_para", 0):
            out.append("trích dẫn đúng chunk nhưng sai đoạn")
        if v.get("quotes_none", 0):
            out.append("đoạn trích không khớp văn bản")
        if v.get("inline_bad_label"):
            out.append("nhãn chương/đoạn sai")
        if v.get("cited_not_sent"):
            out.append("dẫn chunk không được gửi")
    return out


def make_feedback(prev: dict, v) -> str:
    """Ghi chú của kiểm tra viên đưa kèm khi trả lời lại 'từ bằng chứng hiện có'. Chỉ để tham khảo, mô hình phải tự đối chiếu."""
    lines = ["=== GHI CHÚ CỦA KIỂM TRA VIÊN VỀ CÂU TRẢ LỜI TRƯỚC (chỉ để tham khảo; hãy tự đối chiếu với các đoạn truyện ở trên, "
             "đừng chép lại mù quáng) ===",
             f"câu trả lời trước: {prev['llm']['answer']}", f"lý do cần sửa: {v.action_reason}"]
    if v.omissions:
        lines.append("thông tin có thể đã bị bỏ sót (đã có trong các đoạn ở trên):")
        lines += [f"- [{o.chunk_id}, đoạn {o.paragraph}] “{o.quote}” ({o.why})" for o in v.omissions]
    bad = [c for c in v.citation_checks if not c.correct]
    if bad:
        lines.append("địa chỉ trích dẫn trước bị sai: " + "; ".join(
            f"{c.chunk_id} ghi đoạn {c.stated_paragraph}, đúng là đoạn {c.actual_paragraph or '?'}" for c in bad))
    unsup = [c.claim for c in v.claims if not c.supported]
    if unsup:
        lines.append("nhận định trước thiếu căn cứ: " + " | ".join(unsup))
    lines.append("Hãy trả lời lại câu hỏi đúng theo các quy tắc, chỉ dựa trên các đoạn truyện.")
    return "\n".join(lines)


_SUPPORT = re.compile(r"(w\d+-c\d+-k\d+)\D*(\d+)")


def _label(support: str, by_id: dict[str, ChunkRecord]) -> str:
    """'w01-c033-k00 đ7' -> '[w01-c033-k00, chương 33 đoạn 7]' (chương lấy từ metadata, không tin mô hình)."""
    m = _SUPPORT.search(support or "")
    if not m or m.group(1) not in by_id:
        return ""
    return f"[{m.group(1)}, chương {by_id[m.group(1)].chapter_no} đoạn {m.group(2)}]"


def final_policy(llm: dict, v, by_id: dict[str, ChunkRecord]) -> tuple[dict, dict]:
    """Áp chính sách cuối lên một câu trả lời đã được kiểm tra. Trả về (câu trả lời cuối, ghi chú về những gì đã làm).

    Quy tắc: bỏ nhận định có supported=false hoặc fits_question=false. Nếu kiểm tra viên báo cảnh "không đạt" mà không nhận
    định nào bị đánh dấu thì coi mọi nhận định là không thuộc cảnh (bảo thủ). Nếu không còn nhận định nào → "không đủ thông tin".
    Nếu còn một phần → dựng lại câu trả lời từ các nhận định còn lại (kèm nhãn nguồn lấy từ metadata), verdict "một phần".
    Câu trả lời vốn là "không đủ thông tin" hoặc không bị loại gì thì giữ nguyên.
    """
    note = {"scene": v.scene_check.verdict, "removed": [], "kept": 0, "action": "giữ nguyên"}
    if not v.claims:  # câu trả lời vốn là từ chối (hoặc không có nhận định nào để lọc)
        return llm, note
    scene_fail = v.scene_check.verdict == "không đạt"
    flagged = [c for c in v.claims if (not c.supported) or (not c.fits_question)]
    drop = list(v.claims) if (scene_fail and not flagged) else flagged
    kept = [c for c in v.claims if c not in drop]
    note["removed"] = [c.claim for c in drop]
    note["kept"] = len(kept)
    if not drop:
        return llm, note
    if not kept:
        note["action"] = "TỪ CHỐI: không còn nhận định nào được hỗ trợ và đúng cảnh"
        return {"verdict": "không đủ thông tin", "answer": "không đủ thông tin",
                "missing": "Các đoạn tìm được không đủ bằng chứng đúng cảnh/thời điểm cho câu hỏi.", "citations": []}, note
    note["action"] = f"LƯỢC {len(drop)} nhận định, giữ {len(kept)}"
    text = " ".join(f"{c.claim.rstrip('.')} {_label(c.support, by_id)}".strip() + "." for c in kept)
    return {"verdict": "một phần", "answer": text,
            "missing": "Đã lược các nhận định thiếu căn cứ hoặc không đúng cảnh: " + " | ".join(note["removed"]),
            "citations": [c for c in llm["citations"] if any(c["chunk_id"] in (k.support or "") for k in kept)]}, note


# ---------------------------------------------------------------------------
# Chạy cả chính sách cho MỘT câu trả lời đã có (dùng ở Step 9)
# ---------------------------------------------------------------------------


def _tok(usage: dict) -> int:
    return (usage.get("prompt") or usage.get("api_prompt_tokens") or 0) + (usage.get("output") or usage.get("api_output_tokens") or 0)


def run_policy(pipe, trace0: dict, model: str, force: bool = False) -> dict:
    """Cổng kích hoạt -> (kiểm tra -> sửa một lần -> kiểm tra lại) -> chính sách cuối. Không gọi embedding.

    force=True: bắt buộc qua kiểm tra dù cổng không kích hoạt (dùng để đo chính sách trên câu ĐÃ ĐÚNG, hoặc để cho thấy
    chính sách làm gì với câu mà cổng sẽ bỏ qua). Trả về dict ghi đủ: tín hiệu kích hoạt, hành động của hai lượt kiểm tra,
    chunk đã thêm, câu trả lời trước / sau sửa / cuối cùng, ghi chú chính sách cuối và chi phí thêm.
    """
    from src.rag import DEFAULT_MAX_CONTEXT_TOKENS  # import muộn tránh vòng phụ thuộc khi nạp module
    from src.verifier import build_input, call_verifier

    triggers = cheap_triggers(trace0)
    res = {"triggers": triggers, "engaged": bool(triggers) or force, "forced": force and not triggers,
           "answer_before": trace0["llm"], "action1": None, "action2": None, "added": [], "answer_repaired": None,
           "policy": None, "final": trace0["llm"], "cost": {"tokens": 0, "seconds": 0.0}, "error": None}
    if not res["engaged"]:
        return res
    q, cost = trace0["question"], res["cost"]

    v1, u1, err = call_verifier(pipe.client, model, build_input(trace0))
    if not v1:
        res["error"] = f"kiểm tra lần 1: {err}"
        return res
    cost["tokens"] += _tok(u1); cost["seconds"] += u1["latency_s"]
    res["action1"] = v1.action
    res["verify1"] = {"reason": v1.action_reason, "scene": v1.scene_check.model_dump(), "missing_info": v1.missing_info,
                      "omissions": [o.model_dump() for o in v1.omissions],
                      "unsupported": [c.claim for c in v1.claims if not c.supported or not c.fits_question]}

    t1 = None
    if v1.action == "sửa từ bằng chứng hiện có":
        t1 = pipe.answer_from_chunks(q, trace0["sent_ids"], make_feedback(trace0, v1), model)
    elif v1.action == "cần tìm thêm":
        ctx = pipe.build_context([r["chunk_id"] for r in trace0["retrieved"]], 1, DEFAULT_MAX_CONTEXT_TOKENS)
        new_ids = [c["chunk_id"] for c in ctx if c["sent"]]
        res["added"] = [c for c in new_ids if c not in set(trace0["sent_ids"])]
        t1 = pipe.answer_from_chunks(q, new_ids, None, model)

    last_trace, last_v = trace0, v1
    if t1 is not None:
        if not t1["llm"]:
            res["error"] = f"trả lời lại: {t1['error']}"
        else:
            cost["tokens"] += _tok(t1["tokens"]); cost["seconds"] += t1["latency_s"]
            res["answer_repaired"] = t1["llm"]
            v2, u2, err2 = call_verifier(pipe.client, model, build_input(t1))
            last_trace = t1
            if v2:
                cost["tokens"] += _tok(u2); cost["seconds"] += u2["latency_s"]
                res["action2"], last_v = v2.action, v2
            else:
                res["error"] = f"kiểm tra lần 2: {err2}"
    res["final"], res["policy"] = final_policy(last_trace["llm"], last_v, pipe.by_id)
    res["final_sent_ids"] = last_trace["sent_ids"]
    return res
