"""V1 - Bước 7 (thử nghiệm nhỏ): LLM KIỂM TRA một câu trả lời đã có, chưa sửa gì.

Mục tiêu: xem một lượt kiểm tra bằng LLM có (a) bắt được câu trả lời cần can thiệp và (b) để yên câu đã đúng không.

Cách làm (mô phỏng lúc hệ thống phục vụ người dùng):
  * Đầu vào của lượt kiểm tra = ĐÚNG những gì lượt trả lời đã thấy (câu hỏi + các chunk đã gửi, lấy nguyên từ trace) +
    câu trả lời cũ và các trích dẫn của nó. Không đổi câu hỏi, bằng chứng hay câu trả lời.
  * KHÔNG đưa đáp án chuẩn / đáp án mong đợi / đáp án nguồn cho LLM kiểm tra. Đáp án chuẩn chỉ dùng ở đây, trong code, để
    CHẤM LẠI kết quả kiểm tra.
  * Lượt kiểm tra trả về (JSON): đúng nhân vật-cảnh-thời điểm? | từng nhận định có căn cứ? | bỏ sót thông tin cần thiết đã có?
    | địa chỉ trích dẫn (chương, số đoạn) có đúng? | hành động đề xuất: giữ / sửa từ bằng chứng hiện có / cần tìm thêm.
  * Thử trên cả câu CẦN can thiệp (dương tính) và câu ĐÃ ĐÚNG (âm tính, để đo chặn/sửa nhầm).

KỲ VỌNG ĐƯỢC ĐĂNG KÝ TRƯỚC KHI CHẠY (xem EXPECT bên dưới); kết quả chấm theo đúng các kỳ vọng này.

Giới hạn cần nhớ: người kiểm tra là CÙNG model với người trả lời (tự kiểm tra), và chỉ có 6 ca dương tính + 6 ca âm tính.
Chạy:
    python scripts/v1_step7_verify_llm.py
    python scripts/v1_step7_verify_llm.py --only H03-base Q01-v2
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.chunk_store import chunks_path, read_jsonl  # noqa: E402
from src.config import GEMINI_MODEL, TRACES_DIR  # noqa: E402
from src.embedder import make_client  # noqa: E402
from src.rag import paragraphs, quote_match  # noqa: E402
from src.verifier import Verification, build_input, call_verifier  # noqa: E402  (lược đồ, prompt, hàm gọi: xem src/verifier.py)

BASE = "data/traces/20261112-090852-hybrid-header-k5-e0"  # lần chạy mốc (prompt v3, hybrid-header, k5, e0)
V2 = "data/traces/20261112-090700-hybrid-header-k5-e0"  # prompt v2: Q01 từng bị trả lời tự tin SAI
K5E1 = "data/traces/20261112-091217-hybrid-header-k5-e1"  # prompt v3 + mở rộng: Q01 lại bị trả lời tự tin SAI

# ---------------------------------------------------------------------------
# KỲ VỌNG ĐĂNG KÝ TRƯỚC (viết trước khi chạy lượt kiểm tra lần nào)
#   ok_actions : các hành động đề xuất được chấp nhận. Âm tính: chỉ "giữ".
#   scene_not  : kết quả mục "đúng nhân vật/cảnh/thời điểm" KHÔNG được là giá trị này (dương tính về cảnh/thời điểm sai).
#   must_omit  : (chunk, số đoạn) đã được gửi mà câu trả lời bỏ sót; lượt kiểm tra phải nêu.
#   must_flag  : (chunk, số đoạn ghi) trích dẫn có cấu trúc bị ghi sai địa chỉ; lượt kiểm tra phải đánh dấu "không đúng".
# ---------------------------------------------------------------------------
FIX = {"sửa từ bằng chứng hiện có", "cần tìm thêm"}
CASES = {
    # --- dương tính: câu trả lời cần can thiệp ---
    "H03-base": dict(qid="H03", dir=BASE, kind="dương tính", ok_actions=FIX,
                     must_omit=[("w01-c001-k02", 17)], must_flag=[("w01-c001-k00", 5), ("w01-c001-k00", 6)],
                     why="bỏ sót ch1 đ17 (nàng cười) đã được gửi; hai trích dẫn đảo số đoạn 5/6; đ10 không được gửi"),
    "Q01-v2": dict(qid="Q01", dir=V2, kind="dương tính", ok_actions=FIX, scene_not="đạt",
                   why="trả lời tự tin bằng cảnh ở chương 59, không phải cảnh mở đầu truyện"),
    "Q01-k5e1": dict(qid="Q01", dir=K5E1, kind="dương tính", ok_actions=FIX, scene_not="đạt",
                     why="trả lời tự tin 'đoạn cuối truyện (chương 59)', không đúng cảnh mở đầu"),
    "Q09-base": dict(qid="Q09", dir=BASE, kind="dương tính", ok_actions=FIX,
                     why="từ chối vì chunk gửi đi thiếu ch75 đ29 (ai đưa sách); cần thêm ngữ cảnh lân cận"),
    "Q07-base": dict(qid="Q07", dir=BASE, kind="dương tính", ok_actions={"cần tìm thêm"},
                     why="chỉ trả lời được nửa đầu; nửa sau (Lãnh Duệ nói nàng là gì) không có trong chunk đã gửi"),
    "H01-base": dict(qid="H01", dir=BASE, kind="dương tính", ok_actions=FIX, scene_not="đạt",
                     why="nói cô cầm cả tờ chi phiếu 'ở đầu truyện' và lấy ly trà sữa từ đ19 thay vì khoảnh khắc mở đầu (đ5)"),
    # --- âm tính: câu trả lời đã đúng, phải GIỮ ---
    "Q02-base": dict(qid="Q02", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
    "Q04-base": dict(qid="Q04", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
    "Q08-base": dict(qid="Q08", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
    "H02-base": dict(qid="H02", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
    "H08-base": dict(qid="H08", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
    "H10-base": dict(qid="H10", dir=BASE, kind="âm tính", ok_actions={"giữ"}, why="đã đúng"),
}

# ---------------------------------------------------------------------------
# Chấm lại kết quả kiểm tra bằng sự thật đã biết (đáp án nguồn, kiểm tra tự động)
# ---------------------------------------------------------------------------


def grade(case: dict, v: Verification, trace: dict, by_id) -> dict:
    g = {}
    g["action_ok"] = v.action in case["ok_actions"]
    if "scene_not" in case:
        g["scene_ok"] = v.scene_check.verdict != case["scene_not"]
    for chunk, para in case.get("must_omit", []):
        text = paragraphs(by_id[chunk])[para - by_id[chunk].para_start]
        g[f"nêu đúng bỏ sót {chunk} đ{para}"] = any(
            (o.chunk_id == chunk and o.paragraph == para) or quote_match(o.quote, text) != "none" for o in v.omissions)
    for chunk, para in case.get("must_flag", []):
        g[f"đánh dấu sai địa chỉ {chunk} đ{para}"] = any(
            c.chunk_id == chunk and c.stated_paragraph == para and not c.correct for c in v.citation_checks)
    # kiểm tra chính lượt kiểm tra: trích nguyên văn của mục bỏ sót có thật không?
    bad_quotes = 0
    for o in v.omissions:
        r = by_id.get(o.chunk_id)
        ok = bool(r) and r.para_start <= o.paragraph <= r.para_end and \
            quote_match(o.quote, paragraphs(r)[o.paragraph - r.para_start]) != "none"
        bad_quotes += not ok
    g["_omission_quotes_bịa"] = bad_quotes
    # so khớp đánh giá địa chỉ trích dẫn của LLM với kiểm tra tự động (kết quả match của lượt trả lời)
    truth = {(c["chunk_id"], c["paragraph"]): c["match"] in ("exact", "close") for c in trace["verification"]["checks"]}
    agree = total = 0
    for c in v.citation_checks:
        k = (c.chunk_id, c.stated_paragraph)
        if k in truth:
            total += 1
            agree += (truth[k] == c.correct)
    g["_citation_agree"] = (agree, total)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(CASES))
    ap.add_argument("--model", default=GEMINI_MODEL)
    args = ap.parse_args()

    by_id = {r.chunk_id: r for r in read_jsonl(chunks_path(1))}
    client = make_client()
    out_dir = TRACES_DIR / f"verify-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Lượt kiểm tra bằng {args.model} | kết quả lưu ở {out_dir}\n")

    results = []
    for name, case in CASES.items():
        if args.only and name not in args.only:
            continue
        trace = json.loads((Path(case["dir"]) / f"{case['qid']}.json").read_text(encoding="utf-8"))
        v, usage, err = call_verifier(client, args.model, build_input(trace))
        rec = {"case": name, "kind": case["kind"], "usage": usage, "error": err,
               "verification": v.model_dump() if v else None}
        print(f"--- {name} [{case['kind']}] {trace['question']}")
        print(f"    câu trả lời cũ ({trace['llm']['verdict']}): {trace['llm']['answer'][:220]}")
        if not v:
            print(f"    LỖI lượt kiểm tra: {err}\n")
            results.append((name, case, None, None))
            (out_dir / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            continue
        gr = grade(case, v, trace, by_id)
        rec["grade"] = {k: (list(x) if isinstance(x, tuple) else x) for k, x in gr.items()}
        (out_dir / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        unsupported = [c.claim for c in v.claims if not c.supported]
        print(f"    1) cảnh/thời điểm: {v.scene_check.verdict} - {v.scene_check.reason[:150]}")
        print(f"    2) nhận định thiếu căn cứ: {len(unsupported)}/{len(v.claims)}" + (f" -> {unsupported}" if unsupported else ""))
        print(f"    3) bỏ sót: " + ("; ".join(f"{o.chunk_id} đ{o.paragraph}" for o in v.omissions) or "không"))
        wrong_c = [f"{c.chunk_id} đ{c.stated_paragraph}→đ{c.actual_paragraph}" for c in v.citation_checks if not c.correct]
        print(f"    4) trích dẫn sai địa chỉ: {len(wrong_c)}/{len(v.citation_checks)}" + (f" -> {wrong_c}" if wrong_c else ""))
        print(f"    5) hành động đề xuất: {v.action} | {v.action_reason[:140]}" + (f" | cần tìm: {v.missing_info[:100]}" if v.missing_info else ""))
        checks = {k: x for k, x in gr.items() if not k.startswith("_")}
        print(f"    chấm (kỳ vọng đăng ký trước: {sorted(case['ok_actions'])}): "
              + " | ".join(f"{k}: {'ĐÚNG' if x else 'SAI'}" for k, x in checks.items()))
        print(f"    kiểm tra chính lượt kiểm tra: trích bỏ sót bịa {gr['_omission_quotes_bịa']} | "
              f"đánh giá địa chỉ khớp kiểm tra tự động {gr['_citation_agree'][0]}/{gr['_citation_agree'][1]}\n")
        results.append((name, case, v, gr))

    # ---------------- tổng hợp ----------------
    done = [(n, c, v, g) for n, c, v, g in results if v]
    pos = [(n, g) for n, c, v, g in done if c["kind"] == "dương tính"]
    neg = [(n, c, v, g) for n, c, v, g in done if c["kind"] == "âm tính"]
    print("=" * 78)
    print("TỔNG HỢP (chấm theo kỳ vọng đăng ký trước)")
    if pos:
        full = [n for n, g in pos if all(x for k, x in g.items() if not k.startswith("_"))]
        print(f"  dương tính: đạt ĐỦ mọi tiêu chí {len(full)}/{len(pos)}: {full}")
        for n, g in pos:
            miss = [k for k, x in g.items() if not k.startswith("_") and not x]
            if miss:
                print(f"     {n} trượt: {miss}")
    if neg:
        kept = [n for n, c, v, g in neg if g["action_ok"]]
        flagged = [(n, sum(not c2.supported for c2 in v.claims), len(v.omissions),
                    sum(not c2.correct for c2 in v.citation_checks)) for n, c, v, g in neg]
        print(f"  âm tính: được GIỮ đúng {len(kept)}/{len(neg)}; chặn/sửa nhầm: {[n for n, c, v, g in neg if not g['action_ok']]}")
        print("     (kể cả khi giữ, số cờ phụ: nhận định thiếu căn cứ / bỏ sót / trích dẫn sai) "
              + str({n: (a, b, c) for n, a, b, c in flagged}))
    ag = sum(g["_citation_agree"][0] for _, _, _, g in done)
    tt = sum(g["_citation_agree"][1] for _, _, _, g in done)
    bq = sum(g["_omission_quotes_bịa"] for _, _, _, g in done)
    print(f"  độ tin cậy của chính lượt kiểm tra: đánh giá địa chỉ trích dẫn khớp kiểm tra tự động {ag}/{tt}; trích bỏ sót không có thật: {bq}")


if __name__ == "__main__":
    main()
