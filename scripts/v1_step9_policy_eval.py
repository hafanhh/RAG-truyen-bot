"""V1 - Bước 9: đo CHÍNH SÁCH (a) = cổng kích hoạt rẻ + vòng sửa một lần + chính sách cuối, trên 21 câu của mốc.

Chính sách (src/repair.py):
  1. Cổng RẺ (không LLM): chỉ câu có verdict ≠ "đủ" hoặc kiểm tra tự động báo lỗi trích dẫn mới qua lượt kiểm tra bằng LLM.
  2. Qua cổng: kiểm tra -> giữ | sửa từ bằng chứng hiện có | thêm chunk lân cận -> trả lời lại -> kiểm tra lại -> DỪNG.
  3. Chính sách cuối: loại nhận định không được hỗ trợ / không đúng cảnh câu hỏi; không còn gì thì từ chối.

GIỚI HẠN (phải đọc cùng kết quả): chính sách chỉ bảo vệ các câu ĐI QUA lượt kiểm tra. Câu verdict "đủ" + trích dẫn sạch, nhưng sai cảnh
(Q01 ở prompt v2 và ở k5+mở rộng) KHÔNG kích hoạt cổng nên không được bảo vệ. Script đo điều này bằng cách (i) chạy cổng trên các
trace đó (kỳ vọng: không kích hoạt) và (ii) ÉP chúng qua kiểm tra để cho thấy chính sách sẽ làm gì NẾU được kích hoạt.

Các phần:
  A. Cổng trên mọi câu (không gọi LLM): câu nào kích hoạt, câu nào lọt qua dù chưa đúng (đối chiếu điểm chấm tay).
  B. Câu qua cổng: chạy nhiều lần (--runs) vì kiểm tra viên không ổn định giữa các lần chạy.
  C. Câu NGOÀI cổng nhưng sai (Q01-v2, Q01-k5e1): ép qua kiểm tra.
  D. Đối chứng: câu đã đúng, ép qua kiểm tra + chính sách cuối, xem có bị lược/từ chối nhầm không.
Chạy: python scripts/v1_step9_policy_eval.py [--runs 3]
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
from src.rag import RagPipeline  # noqa: E402
from src.repair import _tok, cheap_triggers, run_policy  # noqa: E402

BASE = Path("data/traces/20261112-090852-hybrid-header-k5-e0")
OUTSIDE = {"Q01-v2": ("Q01", Path("data/traces/20261112-090700-hybrid-header-k5-e0")),
           "Q01-k5e1": ("Q01", Path("data/traces/20261112-091217-hybrid-header-k5-e1"))}
CONTROLS = ["Q02", "Q04", "Q08", "H02", "H08", "H10"]  # đã đúng ở mốc
REFUSE = "không đủ thông tin"


def load(path: Path, qid: str) -> dict:
    return json.loads((path / f"{qid}.json").read_text(encoding="utf-8"))


def show(name: str, k: int, qtext: str, res: dict, flags: dict) -> None:
    tag = "ÉP qua kiểm tra" if res["forced"] else "kích hoạt: " + "; ".join(res["triggers"])
    print(f"--- {name} lần {k} [{tag}]")
    if res.get("error"):
        print(f"    LỖI: {res['error']}")
    if res["action1"]:
        print(f"    kiểm tra 1 -> {res['action1']}" + (f" | thêm {len(res['added'])} chunk" if res["added"] else "")
              + (f" | kiểm tra 2 -> {res['action2']}" if res["action2"] else ""))
    if res["policy"]:
        print(f"    chính sách cuối: {res['policy']['action']} (cảnh: {res['policy']['scene']})")
        for c in res["policy"]["removed"]:
            print(f"      - lược: {c[:120]}")
    f = res["final"]
    print(f"    TRƯỚC ({res['answer_before']['verdict']}): {res['answer_before']['answer'][:200]}")
    if res["answer_repaired"]:
        print(f"    SAU SỬA ({res['answer_repaired']['verdict']}): {res['answer_repaired']['answer'][:260]}")
    print(f"    CUỐI  ({f['verdict']}): {f['answer'][:300]}")
    print(f"    {flags} | chi phí thêm +{res['cost']['tokens']} token, +{res['cost']['seconds']:.0f}s\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=GEMINI_MODEL)
    ap.add_argument("--out", help="folder kết quả có sẵn: case đã có file thì nạp lại, chỉ chạy case còn thiếu")
    args = ap.parse_args()

    pipe = RagPipeline(1)
    world = load_world(find_world_file(1))
    qs = {q["id"]: q for f in ("questions_w01.json", "questions_w01_user.json") for q in load_questions(EVAL_DIR / f)["questions"]}
    grades = json.loads((EVAL_DIR / "manual_grades_w01.json").read_text(encoding="utf-8"))["grades"]
    base = {p.stem: load(BASE, p.stem) for p in sorted(BASE.glob("*.json"))}
    out = Path(args.out) if args.out else TRACES_DIR / f"policy-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- A. cổng trên mọi câu (không gọi LLM) ----------------
    print("=" * 78 + "\nA. CỔNG KÍCH HOẠT trên 21 câu của mốc (không gọi LLM)")
    trig = {qid: cheap_triggers(t) for qid, t in base.items()}
    fired = [q for q, tr in trig.items() if tr]
    for q in fired:
        print(f"  kích hoạt {q}: {'; '.join(trig[q])}")
    leaked = [q for q, tr in trig.items() if not tr and grades.get(q, {}).get("grade") in ("một phần", "sai")]
    print(f"  => {len(fired)}/{len(base)} câu kích hoạt: {fired}")
    print(f"  => câu KHÔNG kích hoạt nhưng chưa đúng (lỗi lọt qua cổng, theo điểm chấm tay): {leaked}")
    for name, (qid, d) in OUTSIDE.items():
        tr = cheap_triggers(load(d, qid))
        print(f"  trace {name}: " + ("kích hoạt " + "; ".join(tr) if tr else "KHÔNG kích hoạt (câu trả lời tự tin sai cảnh, trích dẫn sạch) -> LỌT QUA CỔNG"))
    print()

    results = {}  # tên ca -> danh sách kết quả các lần chạy

    def run_case(name, qid, trace, force, runs):
        q = qs[qid]
        groups = gold_groups(q, pipe.records, world) if q["evidence"] else []
        gold = set().union(*groups) if groups else set()
        for k in range(1, runs + 1):
            saved = out / f"{name}-r{k}.json"
            if saved.exists():
                res = json.loads(saved.read_text(encoding="utf-8"))
            else:
                res = run_policy(pipe, trace, args.model, force=force)
                fin_cited = {c["chunk_id"] for c in res["final"]["citations"]}
                res["name"], res["run"], res["qid"] = name, k, qid
                res["final_cites_gold"] = bool(fin_cited & gold) if groups else None
                res["final_refuses"] = res["final"]["verdict"] == REFUSE
                saved.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            results.setdefault(name, []).append(res)
            flag = ("dẫn chunk đúng: " + ("✓" if res["final_cites_gold"] else "✗")) if groups else ("từ chối: " + ("✓" if res["final_refuses"] else "✗ (trả lời)"))
            show(name, k, trace["question"], res, flag)

    # ---------------- B. câu qua cổng ----------------
    print("=" * 78 + f"\nB. CÂU QUA CỔNG, chạy {args.runs} lần mỗi câu")
    for qid in fired:
        run_case(qid, qid, base[qid], False, args.runs)

    # ---------------- C. câu ngoài cổng nhưng sai ----------------
    print("=" * 78 + "\nC. CÂU NGOÀI CỔNG nhưng sai cảnh: ÉP qua kiểm tra để xem chính sách sẽ làm gì nếu được kích hoạt (2 lần)")
    for name, (qid, d) in OUTSIDE.items():
        run_case(name, qid, load(d, qid), True, 2)

    # ---------------- D. đối chứng đã đúng ----------------
    print("=" * 78 + "\nD. ĐỐI CHỨNG đã đúng: ÉP qua kiểm tra + chính sách cuối (1 lần)")
    for qid in CONTROLS:
        run_case(qid + "-đối chứng", qid, base[qid], True, 1)

    # ---------------- tổng hợp ----------------
    print("=" * 78 + "\nTỔNG HỢP (đúng/sai nội dung cần đọc đối chiếu expected_answer ở các khối trên)")
    for name, rs in results.items():
        fin = [r["final"]["verdict"] for r in rs]
        a1 = [r["action1"] for r in rs]
        pol = [r["policy"]["action"].split(":")[0] if r["policy"] else "-" for r in rs]
        cg = [r["final_cites_gold"] for r in rs]
        print(f"  {name:<16} kiểm tra 1: {a1} | chính sách cuối: {pol} | verdict cuối: {fin} | dẫn chunk đúng: {cg}")
    base_tok = sum(_tok(t["tokens"]) for t in base.values())
    base_sec = sum(t.get("latency_s", 0) for t in base.values())
    mean_extra = sum(sum(r["cost"]["tokens"] for r in rs) / len(rs) for n, rs in results.items() if n in fired)
    mean_sec = sum(sum(r["cost"]["seconds"] for r in rs) / len(rs) for n, rs in results.items() if n in fired)
    print(f"\nChi phí trên 21 câu của mốc: lượt gốc {base_tok} token / {base_sec:.0f}s; chính sách thêm TRUNG BÌNH {mean_extra:.0f} token / "
          f"{mean_sec:.0f}s (chỉ {len(fired)} câu qua cổng) => gấp {(base_tok + mean_extra) / base_tok:.2f} lần token.")
    print("(so với kiểm tra MỌI câu ở Step 8: +2.7-2.9k token mỗi câu, tức gần gấp đôi chỉ riêng lượt kiểm tra)")
    print("GIỚI HẠN: chính sách không bảo vệ câu không qua cổng (xem phần A, C): lỗi kiểu Q01-v2 / Q01-k5e1 vẫn lọt.")
    print(f"Kết quả chi tiết: {out}")


if __name__ == "__main__":
    main()
