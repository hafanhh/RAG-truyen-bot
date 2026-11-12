"""V1 - Bước 10b (Experiment B): query REWRITING (viết lại bằng từ vựng truyện) và SPLITTING (tách câu hỏi).

Nhắm vào nguyên nhân lỗi MISSING evidence đã phân loại ở Bước 6:
  * Q01/H01: hỏi bằng từ ngữ tự nhiên, không trùng từ vựng convert của truyện -> BM25 vô dụng (Q01 rank 178),
    RRF kéo hybrid xuống. Thử: LLM viết lại câu hỏi sang từ vựng convert (KHÔNG được xem đáp án/bằng chứng),
    embed biến thể, trộn các ranking bằng RRF đa truy vấn.
  * Q07: hai mảnh bằng chứng ở hai chương khác nhau, không bao giờ cùng vào top-5. Thử: tách thành các câu hỏi
    con độc lập, retrieval từng câu, trộn RRF.
  * Đối chứng Q03/Q08 (đang first=1): rewrite + trộn KHÔNG được phá câu đang tốt.

KỲ VỌNG ĐĂNG KÝ TRƯỚC (viết trước khi chạy, xem CASES): tiêu chí PASS/FAIL in ở cuối.

Biến thể do LLM sinh được LƯU vào data/eval/rewrites_w01.json và tái dùng ở lần chạy sau (ổn định + không tốn
LLM call lại); xoá file đó nếu muốn sinh lại. Mỗi biến thể mới tốn 1 request embedding (cache lại theo nội dung,
chạy lại không tốn). Hết hạn mức embedding theo ngày -> script dừng sớm với thông báo, chạy lại sau khi hồi.

Chạy: python scripts/v1_step10b_rewrite_split.py [--model ...] [--only Q01,Q07]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402
from google.genai import types  # noqa: E402

from src.config import EVAL_DIR, GEMINI_MODEL, TRACES_DIR  # noqa: E402
from src.embedder import MAX_RETRIES, embed_query_cached, is_daily_quota, is_retryable  # noqa: E402
from src.evalset import gold_groups, group_ranks, load_questions, summarize  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.rag import RagPipeline  # noqa: E402
from src.retriever import METHODS, rank_all_methods, rrf  # noqa: E402

REWRITES_FILE = EVAL_DIR / "rewrites_w01.json"
BASE_METHOD = "hybrid-header"  # phương pháp baseline đã chốt (2026-11-12)

# ---- kỳ vọng đăng ký trước (PASS/FAIL chấm máy ở cuối; baseline in kèm để đối chiếu) ----
CASES = {
    "Q01": {"kind": "rewrite", "check": {"fused_need_max": 10, "variant_bm25_first_max": 50},
            "why": "từ vựng tự nhiên, BM25 rank 178 kéo hybrid xuống 25"},
    "H01": {"kind": "rewrite", "check": {"fused_need_max": 10},
            "why": "cùng kiểu Q01 (hỏi 'đầu truyện' bằng lời tự nhiên)"},
    "Q07": {"kind": "split", "check": {"fused_need_max": 7, "each_group_top5_in_a_subq": True},
            "why": "hai mảnh bằng chứng ở hai chương, mảnh 2 hybrid rank 10"},
    "Q03": {"kind": "rewrite", "check": {"fused_first_max": 2}, "why": "đối chứng: baseline first=1, không được phá"},
    "Q08": {"kind": "rewrite", "check": {"fused_first_max": 2}, "why": "đối chứng: baseline first=1, không được phá"},
}

REWRITE_PROMPT = (
    "Người dùng hỏi về một truyện 'xuyên nhanh' dịch convert từ tiếng Trung, văn phong Hán-Việt đặc trưng "
    "(ví dụ: 'nữ hài' = cô gái, 'nam nhân', 'ngươi/ta', 'hảo', 'a di', 'bà bà', 'kim chủ', 'cự tuyệt'...). "
    "Viết lại câu hỏi sau thành 2 biến thể cùng nghĩa nhưng dùng từ vựng kiểu convert đó, để khớp từ khóa với "
    "văn bản truyện khi tìm kiếm. Giữ nguyên MỌI điều kiện của câu hỏi, không thêm chi tiết mới, không suy đoán "
    "nội dung truyện.\n\nCâu hỏi: {question}"
)
SPLIT_PROMPT = (
    "Tách câu hỏi sau thành các câu hỏi con độc lập (thường là 2), mỗi câu hỏi con hỏi đúng MỘT sự việc và tự "
    "đứng được một mình (lặp lại tên nhân vật thay vì dùng đại từ như 'nàng/hắn'). Giữ nguyên từ ngữ gốc ở mức "
    "có thể, không thêm chi tiết mới, không suy đoán nội dung truyện.\n\nCâu hỏi: {question}"
)


class Variants(BaseModel):
    variants: list[str]


def gen_variants(pipe: RagPipeline, model: str, kind: str, question: str) -> list[str]:
    prompt = (REWRITE_PROMPT if kind == "rewrite" else SPLIT_PROMPT).format(question=question)
    cfg = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", response_schema=Variants)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = pipe.client.models.generate_content(model=model, contents=prompt, config=cfg)
            break
        except Exception as exc:  # noqa: BLE001 - 503 quá tải / 429 theo phút là tạm thời, còn lại raise lại
            if attempt == MAX_RETRIES or not is_retryable(exc):
                raise
            wait = min(10.0 * 2**attempt, 90.0)
            print(f"    ! LLM lần thử {attempt + 1}: {type(exc).__name__}, chờ {wait:.0f}s rồi thử lại")
            time.sleep(wait)
    out = [v.strip() for v in Variants.model_validate(json.loads(resp.text)).variants if v.strip()]
    if not out:
        raise RuntimeError(f"LLM không trả biến thể nào cho: {question!r}")
    return out


def load_or_gen_variants(pipe: RagPipeline, model: str, qid: str, kind: str, question: str) -> list[str]:
    store = json.loads(REWRITES_FILE.read_text(encoding="utf-8")) if REWRITES_FILE.exists() else {}
    if qid in store and store[qid]["question"] == question:
        return store[qid]["variants"]
    variants = gen_variants(pipe, model, kind, question)
    store[qid] = {"kind": kind, "question": question, "variants": variants, "model": model,
                  "generated_at": datetime.now().isoformat(timespec="seconds")}
    REWRITES_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    return variants


def rankings_for(pipe: RagPipeline, text: str) -> dict[str, list[str]]:
    qvec = embed_query_cached(pipe.client, text, pipe.world_no)
    return rank_all_methods(text, qvec, pipe.cols, pipe.bm25, pipe.ids)


def fmt(s: dict) -> str:
    return f"first={s['first']} need={s['need']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=GEMINI_MODEL, help="model LLM sinh biến thể")
    ap.add_argument("--only", help="chỉ chạy các câu này, vd Q01,Q07")
    args = ap.parse_args()

    pipe = RagPipeline(1)
    world = load_world(find_world_file(pipe.world_no))
    qs = {q["id"]: q for f in ("questions_w01.json", "questions_w01_user.json")
          for q in load_questions(EVAL_DIR / f)["questions"]}
    only = set(args.only.split(",")) if args.only else set(CASES)
    out = {"model": args.model, "base_method": BASE_METHOD, "time": datetime.now().isoformat(timespec="seconds"),
           "cases": {}}
    verdicts: dict[str, bool] = {}

    for qid, case in CASES.items():
        if qid not in only:
            continue
        q = qs[qid]
        groups = gold_groups(q, pipe.records, world)
        print("=" * 78 + f"\n{qid} [{case['kind']}] — {case['why']}\n  câu gốc: {q['question']}")
        try:
            variants = load_or_gen_variants(pipe, args.model, qid, case["kind"], q["question"])
            r_orig = rankings_for(pipe, q["question"])
            r_vars = [rankings_for(pipe, v) for v in variants]
        except Exception as exc:  # noqa: BLE001
            if is_daily_quota(exc):
                print("  !! HẾT HẠN MỨC EMBEDDING THEO NGÀY — dừng, chạy lại khi quota hồi.")
                break
            raise

        base = {m: summarize(group_ranks(r_orig[m], groups)) for m in METHODS}
        print(f"  baseline ({BASE_METHOD}): {fmt(base[BASE_METHOD])}   "
              f"(dense-header {fmt(base['dense-header'])}, bm25 {fmt(base['bm25'])})")
        var_sums = []
        for v, rv in zip(variants, r_vars):
            s = {m: summarize(group_ranks(rv[m], groups)) for m in METHODS}
            var_sums.append(s)
            print(f"  biến thể: {v}\n    {BASE_METHOD} {fmt(s[BASE_METHOD])} | bm25 {fmt(s['bm25'])} "
                  f"| dense-header {fmt(s['dense-header'])}")

        fused_ids = rrf([r_orig[BASE_METHOD]] + [rv[BASE_METHOD] for rv in r_vars])
        fused = summarize(group_ranks(fused_ids, groups))
        print(f"  => TRỘN RRF (gốc + {len(variants)} biến thể, {BASE_METHOD}): {fmt(fused)}")

        ok, notes = True, []
        c = case["check"]
        if "fused_need_max" in c:
            cond = fused["need"] is not None and fused["need"] <= c["fused_need_max"]
            ok &= cond
            notes.append(f"fused need<={c['fused_need_max']}: {'ĐẠT' if cond else 'KHÔNG (' + str(fused['need']) + ')'}")
        if "fused_first_max" in c:
            cond = fused["first"] is not None and fused["first"] <= c["fused_first_max"]
            ok &= cond
            notes.append(f"fused first<={c['fused_first_max']}: {'ĐẠT' if cond else 'KHÔNG (' + str(fused['first']) + ')'}")
        if "variant_bm25_first_max" in c:
            best = min((s["bm25"]["first"] for s in var_sums if s["bm25"]["first"]), default=None)
            cond = best is not None and best <= c["variant_bm25_first_max"]
            ok &= cond
            notes.append(f"bm25 first của biến thể tốt nhất<={c['variant_bm25_first_max']}: "
                         f"{'ĐẠT (' + str(best) + ')' if cond else 'KHÔNG (' + str(best) + ')'}")
        if c.get("each_group_top5_in_a_subq"):
            per_group = [min((gr for gr in (group_ranks(rv[BASE_METHOD], [g])[0] for rv in r_vars) if gr), default=None)
                         for g in groups]
            cond = all(gr is not None and gr <= 5 for gr in per_group)
            ok &= cond
            notes.append(f"mỗi nhóm bằng chứng top-5 ở một câu con: {'ĐẠT' if cond else 'KHÔNG'} {per_group}")
        verdicts[qid] = ok
        print(f"  KỲ VỌNG: {' | '.join(notes)}  ->  {'PASS' if ok else 'FAIL'}")
        out["cases"][qid] = {"question": q["question"], "variants": variants, "baseline": base,
                             "variant_summaries": var_sums, "fused": fused, "pass": ok, "notes": notes}

    if out["cases"]:
        path = TRACES_DIR / f"rewrite-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("=" * 78 + f"\nPASS {sum(verdicts.values())}/{len(verdicts)}: {verdicts}\nKết quả chi tiết: {path}")


if __name__ == "__main__":
    main()
