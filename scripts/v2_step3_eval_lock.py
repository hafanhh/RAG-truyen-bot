"""V2 - Bước 3: khóa bộ câu hỏi sạch bằng sha256, và kiểm tra bộ đó chưa bị sửa.

Vì sao: bộ cũ (questions_w01*.json) bị sửa gold 4 lần SAU KHI thấy kết quả, nên số liệu bị nhiễm. Với bộ v2, gold được
ghi trước mọi lần chạy, rồi khóa: sha256 của các trường GOLD của từng câu (id, question, evidence, set, pair, expected_answer;
KHÔNG gồm note và type, là phần mô tả) nằm trong data/eval/questions_w01_v2.lock.json. Script chấm điểm gọi --check; lệch hash -> báo lỗi (exit 1).
Nghi vấn sau khi khóa ghi vào questions_w01_v2.notes.json (không nằm trong hash), KHÔNG sửa gold.

  python scripts/v2_step3_eval_lock.py --validate   # đoạn gold có tồn tại, được chunk phủ (không gọi API)
  python scripts/v2_step3_eval_lock.py --lock       # ghi lock (từ chối nếu đã có lock)
  python scripts/v2_step3_eval_lock.py --check      # so hash hiện tại với lock
  python scripts/v2_step3_eval_lock.py --relock "lý do"   # khóa lại, ghi lý do + hash cũ vào lịch sử (dùng hiếm, phải nói rõ lý do)
  python scripts/v2_step3_eval_lock.py --diff-gold [REF]  # so các trường gold với git REF (mặc định HEAD), chứng minh cái gì đổi
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_store import read_jsonl  # noqa: E402
from src.config import CHUNKS_DIR, EVAL_DIR  # noqa: E402
from src.evalset import gold_groups  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402

QUESTIONS = EVAL_DIR / "questions_w01_v2.json"
LOCK = EVAL_DIR / "questions_w01_v2.lock.json"
GOLD_FIELDS = ("id", "question", "evidence", "set", "pair", "expected_answer")


def gold_view(questions: list[dict]) -> list[dict]:
    """Chỉ các trường gold của mỗi câu (bỏ note/type: là mô tả, sửa được mà không đổi cách chấm)."""
    return [{k: q[k] for k in GOLD_FIELDS if k in q} for q in questions]


def questions_sha(path: Path = QUESTIONS) -> str:
    qs = gold_view(json.loads(path.read_text(encoding="utf-8"))["questions"])
    canon = json.dumps(qs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def verify_lock() -> None:
    """Dùng trong script chấm điểm: dừng nếu bộ câu hỏi đã bị sửa so với lúc khóa."""
    if not LOCK.exists():
        raise SystemExit("Bộ v2 chưa được khóa (thiếu questions_w01_v2.lock.json).")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["sha256"] != questions_sha():
        raise SystemExit("Bộ câu hỏi v2 ĐÃ BỊ SỬA sau khi khóa: sha256 không khớp. Không chấm điểm.")


def validate() -> bool:
    world = load_world(find_world_file(1))
    records = read_jsonl(CHUNKS_DIR / "chunks_w01.jsonl")
    doc = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    ok = True
    ids = [q["id"] for q in doc["questions"]]
    if len(ids) != len(set(ids)):
        print("LỖI: id trùng")
        ok = False
    for q in doc["questions"]:
        if (q["set"] == "unanswerable") != (q["evidence"] == []):
            print(f"LỖI {q['id']}: set và evidence không khớp")
            ok = False
            continue
        try:
            groups = gold_groups(q, records, world) if q["evidence"] else []
        except ValueError as exc:
            print(f"LỖI {q['id']}: {exc}")
            ok = False
            continue
        desc = " + ".join("{" + ",".join(sorted(g)) + "}" for g in groups) or "(không có đáp án)"
        print(f"  {q['id']:<4} [{q['set']:<12}] {q['question'][:46]:<46} -> {desc}")
    return ok


def diff_gold(ref: str) -> int:
    """So các trường gold với phiên bản ở git `ref`. Exit 1 nếu question/evidence/set/pair/id đổi (đó mới là đổi gold thật);
    expected_answer đổi chỉ được LIỆT KÊ để người dùng duyệt (ví dụ rút trích dẫn nguyên văn mà giữ nghĩa)."""
    import subprocess

    rel = QUESTIONS.relative_to(QUESTIONS.parents[2]).as_posix()
    old = {q["id"]: q for q in json.loads(subprocess.check_output(["git", "show", f"{ref}:{rel}"], text=True))["questions"]}
    new = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    hard = False
    if list(old) != list(new):
        print("LỖI: danh sách/thứ tự id khác", list(old) ^ list(new) if set(old) != set(new) else "(thứ tự)")
        hard = True
    for qid in new:
        if qid not in old:
            continue
        for f in GOLD_FIELDS:
            if old[qid].get(f) != new[qid].get(f):
                if f == "expected_answer":
                    print(f"  {qid}.expected_answer ĐỔI (cần duyệt):\n      cũ: {old[qid][f]}\n      mới: {new[qid][f]}")
                else:
                    print(f"  {qid}.{f} ĐỔI  <- gold thật bị đổi")
                    hard = True
    print("question/evidence/set/pair/id: " + ("CÓ THAY ĐỔI" if hard else f"giữ nguyên từng byte so với {ref}"))
    return 1 if hard else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--lock", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--relock", metavar="LÝ_DO")
    g.add_argument("--diff-gold", nargs="?", const="HEAD", metavar="REF")
    args = ap.parse_args()

    if args.diff_gold:
        sys.exit(diff_gold(args.diff_gold))

    if args.validate:
        sys.exit(0 if validate() else 1)
    if args.check:
        verify_lock()
        print("OK: bộ v2 khớp lock", json.loads(LOCK.read_text(encoding="utf-8"))["sha256"][:16])
        return
    if not validate():
        sys.exit("Không khóa: gold có lỗi.")
    sha, now = questions_sha(), datetime.now().isoformat(timespec="seconds")
    if args.lock:
        if LOCK.exists():
            sys.exit("Đã có lock. Dùng --check, hoặc --relock 'lý do' nếu chưa từng chạy pipeline trên bộ này.")
        lock = {"sha256": sha, "locked_at": now, "n_questions": len(json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]),
                "history": []}
    else:
        old = json.loads(LOCK.read_text(encoding="utf-8"))
        old["history"].append({"relocked_at": now, "previous_sha256": old["sha256"], "reason": args.relock})
        lock = {**old, "sha256": sha, "locked_at": now}
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã khóa: sha256={sha}")


if __name__ == "__main__":
    main()
