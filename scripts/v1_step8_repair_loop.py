"""V1 - Bước 8 (thử nghiệm nhỏ): VÒNG SỬA khép kín sau lượt kiểm tra, dừng sau đúng MỘT lần sửa.

    Câu trả lời ban đầu (lấy nguyên từ trace đã lưu)
        ↓ Kiểm tra (src/verifier.py)
        ├─ giữ                       → giữ nguyên
        ├─ sửa từ bằng chứng hiện có → trả lời lại từ CHÍNH các chunk đã gửi (kèm ghi chú của kiểm tra viên)
        └─ cần tìm thêm              → thêm chunk LÂN CẬN (±1, theo prev_id/next_id, không qua chương) → trả lời lại
                                         ↓
                                   Kiểm tra lại
                                         ↓
                          DỪNG, kể cả khi vẫn thiếu thông tin (chỉ ghi lại, không sửa lần hai)

Không gọi embedding: chunk được lấy bằng ID đã lưu trong trace. Chỉ tốn hạn mức LLM (trả lời lại + kiểm tra).
Mỗi ca lưu (data/traces/repair-<giờ>/<ca>.json): câu trả lời trước và sau; lý do kiểm tra yêu cầu sửa/tìm thêm; chunk mới
được thêm; bằng chứng có được gửi đủ / có dẫn chunk đúng hơn không (chấm bằng đáp án nguồn, KHÔNG đưa cho LLM); token và
thời gian tăng thêm. Đúng/sai nội dung câu trả lời do người đọc đối chiếu (script in trước/sau cạnh nhau).

Đối chứng: các câu ĐÃ ĐÚNG. Hai loại:
  * tự nhiên : để lượt kiểm tra tự quyết (kỳ vọng "giữ": vòng sửa không chạy);
  * ÉP nhánh "tìm thêm": bắt buộc thêm chunk lân cận và trả lời lại, để xem MỘT lượt sửa có làm câu đúng sai đi không.

Chạy: python scripts/v1_step8_repair_loop.py [--only CA ...]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_DIR, GEMINI_MODEL, TRACES_DIR  # noqa: E402
from src.evalset import gold_groups, load_questions  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.rag import DEFAULT_MAX_CONTEXT_TOKENS, RagPipeline  # noqa: E402
from src.repair import make_feedback  # noqa: E402
from src.verifier import build_input, call_verifier  # noqa: E402

BASE = "data/traces/20261112-090852-hybrid-header-k5-e0"
V2 = "data/traces/20261112-090700-hybrid-header-k5-e0"
K5E1 = "data/traces/20261112-091217-hybrid-header-k5-e1"

# mode: "tự nhiên" = lượt kiểm tra quyết định | "ép tìm thêm" = bắt buộc nhánh thêm chunk lân cận (đối chứng)
CASES = {
    "Q09-base": dict(qid="Q09", dir=BASE, mode="tự nhiên", kind="cần can thiệp"),
    "Q01-v2": dict(qid="Q01", dir=V2, mode="tự nhiên", kind="cần can thiệp"),
    "Q01-k5e1": dict(qid="Q01", dir=K5E1, mode="tự nhiên", kind="cần can thiệp"),
    "Q07-base": dict(qid="Q07", dir=BASE, mode="tự nhiên", kind="cần can thiệp"),
    "H03-base": dict(qid="H03", dir=BASE, mode="tự nhiên", kind="cần can thiệp"),
    "H01-base": dict(qid="H01", dir=BASE, mode="tự nhiên", kind="cần can thiệp (kiểm tra có thể bỏ lỡ)"),
    "Q02-base": dict(qid="Q02", dir=BASE, mode="tự nhiên", kind="đối chứng đã đúng"),
    "Q04-base": dict(qid="Q04", dir=BASE, mode="tự nhiên", kind="đối chứng đã đúng"),
    "H08-base": dict(qid="H08", dir=BASE, mode="tự nhiên", kind="đối chứng đã đúng"),
    "Q04-ép": dict(qid="Q04", dir=BASE, mode="ép tìm thêm", kind="đối chứng đã đúng (ép)"),
    "Q08-ép": dict(qid="Q08", dir=BASE, mode="ép tìm thêm", kind="đối chứng đã đúng (ép)"),
    "H02-ép": dict(qid="H02", dir=BASE, mode="ép tìm thêm", kind="đối chứng đã đúng (ép)"),
    "H10-ép": dict(qid="H10", dir=BASE, mode="ép tìm thêm", kind="đối chứng đã đúng (ép)"),
}


def tok(usage: dict) -> int:
    return (usage.get("prompt") or usage.get("api_prompt_tokens") or 0) + (usage.get("output") or usage.get("api_output_tokens") or 0)


def summarize_answer(t: dict, groups, records_by_id) -> dict:
    """Tóm tắt một lượt trả lời: verdict, nguồn dẫn, trích dẫn tốt/xấu, và (nếu có đáp án nguồn) bằng chứng gửi đủ / dẫn đúng chunk."""
    a, v = t["llm"], t["verification"]
    sent, cited = set(t["sent_ids"]), set(v["cited_ids"]) if v else set()
    gold = set().union(*groups) if groups else set()
    return {
        "verdict": a["verdict"] if a else "LỖI", "answer": a["answer"] if a else t.get("error"),
        "missing": a["missing"] if a else "", "cited": sorted(cited),
        "n_sent": len(sent),
        "evidence_sent": (all(g & sent for g in groups) if groups else None),
        "cited_gold": (bool(cited & gold) if groups else None),
        # .get(..., 0): trace của prompt v2 được tạo bằng mã cũ, chưa có trường quotes_other_para
        "quotes": (f"{v['quotes_exact']}+{v['quotes_close']}+{v.get('quotes_other_para', 0)}+{v['quotes_none']}/{v['n_citations']}"
                   if v else "-"),
        "bad_label": bool(v and v.get("inline_bad_label")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(CASES))
    ap.add_argument("--model", default=GEMINI_MODEL)
    args = ap.parse_args()

    pipe = RagPipeline(1)
    world = load_world(find_world_file(1))
    qs = {q["id"]: q for f in ("questions_w01.json", "questions_w01_user.json") for q in load_questions(EVAL_DIR / f)["questions"]}
    out = TRACES_DIR / f"repair-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Vòng sửa (1 lần) | model {args.model} | lưu ở {out}\n")

    rows = []
    for name, case in CASES.items():
        if args.only and name not in args.only:
            continue
        t0 = json.loads((Path(case["dir"]) / f"{case['qid']}.json").read_text(encoding="utf-8"))
        q = qs[case["qid"]]
        groups = gold_groups(q, pipe.records, world) if q["evidence"] else []
        rec = {"case": name, **{k: case[k] for k in ("mode", "kind")}, "question": t0["question"]}
        before = summarize_answer(t0, groups, pipe.by_id)
        rec["before"] = before
        cost = {"tokens": 0, "seconds": 0.0}

        # ---- 1) kiểm tra (hoặc ép nhánh) ----
        if case["mode"] == "ép tìm thêm":
            action, reason, v1 = "cần tìm thêm", "ÉP nhánh 'tìm thêm' (đối chứng, không hỏi kiểm tra viên)", None
        else:
            v1, u1, err = call_verifier(pipe.client, args.model, build_input(t0))
            if not v1:
                print(f"--- {name} LỖI kiểm tra lần 1: {err}\n")
                rec["error"] = err
                (out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                continue
            cost["tokens"] += tok(u1); cost["seconds"] += u1["latency_s"]
            action, reason = v1.action, v1.action_reason
            rec["verify1"] = {"action": action, "reason": reason, "scene": v1.scene_check.model_dump(),
                              "missing_info": v1.missing_info, "omissions": [o.model_dump() for o in v1.omissions],
                              "bad_citations": [c.model_dump() for c in v1.citation_checks if not c.correct],
                              "unsupported": [c.claim for c in v1.claims if not c.supported], "usage": u1}
        rec["action"], rec["reason"] = action, reason

        # ---- 2) sửa hoặc thêm chunk, trả lời lại ----
        t1, added = None, []
        if action == "sửa từ bằng chứng hiện có":
            t1 = pipe.answer_from_chunks(t0["question"], t0["sent_ids"], make_feedback(t0, v1), args.model)
        elif action == "cần tìm thêm":
            retrieved = [r["chunk_id"] for r in t0["retrieved"]]
            ctx = pipe.build_context(retrieved, 1, DEFAULT_MAX_CONTEXT_TOKENS)
            new_ids = [c["chunk_id"] for c in ctx if c["sent"]]
            added = [c for c in new_ids if c not in set(t0["sent_ids"])]
            t1 = pipe.answer_from_chunks(t0["question"], new_ids, None, args.model)
        rec["added_chunks"] = added

        after = v2 = None
        if t1:
            if not t1["llm"]:
                rec["error"] = t1["error"]
            else:
                cost["tokens"] += tok(t1["tokens"]); cost["seconds"] += t1["latency_s"]
                after = summarize_answer(t1, groups, pipe.by_id)
                # ---- 3) kiểm tra lại rồi DỪNG ----
                v2, u2, err2 = call_verifier(pipe.client, args.model, build_input(t1))
                if v2:
                    cost["tokens"] += tok(u2); cost["seconds"] += u2["latency_s"]
                    rec["verify2"] = {"action": v2.action, "reason": v2.action_reason, "scene": v2.scene_check.model_dump(),
                                      "missing_info": v2.missing_info,
                                      "bad_citations": [c.model_dump() for c in v2.citation_checks if not c.correct]}
                else:
                    rec["verify2_error"] = err2
        rec["after"] = after
        rec["cost_extra"] = {**cost, "baseline_tokens": tok(t0["tokens"]), "baseline_seconds": t0.get("latency_s")}
        (out / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append((name, case, rec, before, after, v2))

        # ---- in ----
        print(f"--- {name} [{case['kind']} | {case['mode']}] {t0['question']}")
        print(f"    TRƯỚC ({before['verdict']}): {before['answer'][:260]}")
        print(f"    kiểm tra 1 -> {action}: {reason[:200]}")
        if added:
            print(f"    chunk mới thêm: {added}")
        if after:
            print(f"    SAU   ({after['verdict']}): {after['answer'][:300]}")
            if v2:
                print(f"    kiểm tra 2 -> {v2.action} | cảnh: {v2.scene_check.verdict}"
                      + (f" | cần tìm: {v2.missing_info[:100]}" if v2.missing_info else ""))
            f = lambda b: "-" if b is None else ("✓" if b else "✗")
            print(f"    bằng chứng gửi đủ: {f(before['evidence_sent'])} → {f(after['evidence_sent'])} | dẫn chunk đúng: "
                  f"{f(before['cited_gold'])} → {f(after['cited_gold'])} | trích dẫn: {before['quotes']} → {after['quotes']} "
                  f"| chunk gửi: {before['n_sent']} → {after['n_sent']}")
        else:
            print("    (không sửa, giữ nguyên)")
        print(f"    chi phí thêm: +{cost['tokens']} token, +{cost['seconds']:.1f}s (lượt gốc: {tok(t0['tokens'])} token, {t0.get('latency_s')}s)\n")

    # ---- tổng hợp ----
    print("=" * 78 + "\nTỔNG HỢP")
    by = {}
    for name, case, rec, before, after, v2 in rows:
        by.setdefault(rec["action"], []).append(name)
    for a, names in by.items():
        print(f"  {a}: {names}")
    tot_tok = sum(r[2]["cost_extra"]["tokens"] for r in rows)
    tot_sec = sum(r[2]["cost_extra"]["seconds"] for r in rows)
    base_tok = sum(r[2]["cost_extra"]["baseline_tokens"] for r in rows)
    print(f"  chi phí thêm cả vòng: +{tot_tok} token (gấp {tot_tok / max(base_tok, 1):.1f} lần token của các lượt gốc), +{tot_sec:.0f}s")


if __name__ == "__main__":
    main()
