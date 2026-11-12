"""V1 - Bước 5: answer() đầy đủ (tìm -> gửi LLM -> trả lời có dẫn nguồn) với TRACE từng bước.

Trace gồm: câu hỏi + cấu hình (model, cách tìm, k), bộ lọc chương, các chunk tìm được (hạng, id, chương/đoạn),
các chunk THỰC SỰ gửi cho LLM (và chunk bị loại kèm lý do), số token, câu trả lời và nguồn được dẫn (đã kiểm chứng).
"Tìm được" và "đã gửi" được tách riêng: nếu ngân sách token cắt mất chunk, trace báo rõ.

Chạy từ thư mục gốc:
    python scripts/v1_step5_answer.py --ask "Ai là cha nuôi của Lãnh Tử Việt?"
    python scripts/v1_step5_answer.py --ask "Ở chương 58, Đường Quả bảo Lãnh Tử Việt gọi nàng là gì?"   # tự lọc chương 58
    python scripts/v1_step5_answer.py --ask "..." --k 10 --expand 1 --show-prompt
    python scripts/v1_step5_answer.py --ask "..." --max-context-tokens 600 --no-llm    # xem chunk bị loại vì ngân sách, không gọi LLM
    python scripts/v1_step5_answer.py --eval                      # chạy cả hai bộ câu hỏi (bỏ câu trùng), in bảng tóm tắt
    python scripts/v1_step5_answer.py --eval --only Q05 H08 U1    # chỉ vài câu
Trace JSON (kèm prompt đầy đủ) được lưu trong data/traces/ khi dùng --eval hoặc --save.
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
from src.rag import (  # noqa: E402
    DEFAULT_K, DEFAULT_MAX_CONTEXT_TOKENS, DEFAULT_METHOD, RagPipeline, format_citations, format_trace,
)
from src.retriever import METHODS, parse_chapter_set  # noqa: E402

DEFAULT_SETS = [EVAL_DIR / "questions_w01.json", EVAL_DIR / "questions_w01_user.json"]


def check_row(q: dict, tr: dict, groups: list[set[str]]) -> dict:
    """Đối chiếu một trace với đáp án nguồn: bằng chứng có được TÌM, có được GỬI, có được DẪN không."""
    retrieved = {r["chunk_id"] for r in tr["retrieved"]}
    sent = set(tr["sent_ids"])
    gold = set().union(*groups) if groups else set()
    v, a = tr["verification"], tr["llm"]
    return {
        "id": q["id"],
        "answerable": bool(groups),
        "verdict": a["verdict"] if a else "LỖI",
        "retrieved_ok": all(g & retrieved for g in groups) if groups else None,
        "sent_ok": all(g & sent for g in groups) if groups else None,
        "cited_gold": bool(set(v["cited_ids"]) & gold) if (v and groups) else None,
        "quotes": (f"{v['quotes_exact']}+{v['quotes_close']}+{v['quotes_other_para']}+{v['quotes_none']}/{v['n_citations']}"
                   if v else "-"),
        "quotes_none": (v["quotes_none"] + v["quotes_other_para"]) if v else 0,
        "warn": bool(v and (v["cited_not_sent"] or v["inline_bad_label"])),
        "no_cite": bool(v and a["verdict"] != "không đủ thông tin" and not v["cited_ids"]),
    }


def run_eval(pipe: RagPipeline, args) -> None:
    world = load_world(find_world_file(1))
    rows, qs = [], []
    for path in args.questions:
        for q in load_questions(path)["questions"]:
            if "TRÙNG" in q["type"]:  # H09 trùng Q09: chỉ chạy một lần
                continue
            if args.only and q["id"] not in args.only:
                continue
            qs.append(q)
    out = TRACES_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.method}-k{args.k}-e{args.expand}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Chạy {len(qs)} câu | model {args.model} | {args.method} k={args.k} mở rộng ±{args.expand} | trace -> {out}\n")

    failed = []  # câu bị lỗi API (vd embedding hết lượt thử lại): ghi nhận rồi chạy tiếp, không làm mất cả lần chạy
    for q in qs:
        try:
            tr = pipe.answer(q["question"], method=args.method, k=args.k, expand=args.expand,
                             max_context_tokens=args.max_context_tokens, model=args.model,
                             auto_chapters=not args.no_auto_chapters)
        except Exception as exc:  # noqa: BLE001
            failed.append(q["id"])
            print(f"--- {q['id']} LỖI, bỏ qua: {type(exc).__name__}: {str(exc)[:150]}\n")
            continue
        (out / f"{q['id']}.json").write_text(json.dumps({"expected": q["expected_answer"], **tr}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
        groups = gold_groups(q, pipe.records, world) if q["evidence"] else []
        row = check_row(q, tr, groups)
        rows.append(row)
        a = tr["llm"]
        print(f"--- {q['id']} [{q['type']}] {q['question']}")
        print(f"    đáp án mong đợi: {q['expected_answer']}")
        if a:
            print(f"    trả lời ({a['verdict']}): {a['answer'][:700]}")
            if a["missing"]:
                print(f"    còn thiếu: {a['missing'][:300]}")
        else:
            print(f"    LỖI: {tr['error']}")
        flag = lambda b: "-" if b is None else ("✓" if b else "✗")
        print(f"    tìm được bằng chứng {flag(row['retrieved_ok'])} | gửi {flag(row['sent_ok'])} | dẫn chunk đúng {flag(row['cited_gold'])} "
              f"| trích (nguyên văn+gần+sai đoạn+không khớp)/tổng {row['quotes']} | cảnh báo nguồn {'CÓ' if row['warn'] else 'không'} "
              f"| tokens API {tr['tokens'].get('api_prompt_tokens')}\n")

    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    n = len(ans)
    print("=" * 78)
    print(f"TÓM TẮT ({len(rows)} câu: {n} có đáp án, {len(un)} không có trong truyện)"
          + (f" | {len(failed)} câu LỖI API, không tính: {failed}" if failed else ""))
    print(f"  bằng chứng được TÌM : {sum(r['retrieved_ok'] for r in ans)}/{n}")
    print(f"  bằng chứng được GỬI : {sum(r['sent_ok'] for r in ans)}/{n}   (khác số trên = bị loại giữa hai bước)")
    print(f"  dẫn ít nhất một chunk đúng: {sum(bool(r['cited_gold']) for r in ans)}/{n}")
    dist = {}
    for r in ans:
        dist[r["verdict"]] = dist.get(r["verdict"], 0) + 1
    print(f"  verdict ở câu có đáp án: {dist}")
    print(f"  câu không có trong truyện -> verdict: {[(r['id'], r['verdict']) for r in un]}")
    print(f"  câu có cảnh báo nguồn (dẫn chunk không gửi / ghi sai chương-đoạn): {[r['id'] for r in rows if r['warn']]}")
    print(f"  câu có đoạn trích sai đoạn hoặc không khớp văn bản gốc: {[r['id'] for r in rows if r['quotes_none']]}")
    print(f"  câu trả lời (không phải 'không đủ thông tin') mà không dẫn nguồn nào: {[r['id'] for r in rows if r['no_cite']]}")
    print("Đúng/sai của nội dung câu trả lời cần đối chiếu bằng mắt với 'đáp án mong đợi' ở trên.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ask", metavar="CÂU_HỎI")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--questions", nargs="+", type=Path, default=DEFAULT_SETS)
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--expand", type=int, default=0, help="thêm n chunk lân cận mỗi phía")
    ap.add_argument("--chapters", metavar="N|A-B", help="lọc thủ công theo chương")
    ap.add_argument("--no-auto-chapters", action="store_true", help="không tự nhận 'chương N' trong câu hỏi")
    ap.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    ap.add_argument("--model", default=GEMINI_MODEL)
    ap.add_argument("--no-llm", action="store_true", help="chỉ tìm và dựng ngữ cảnh, không gọi LLM")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--save", action="store_true", help="lưu trace JSON (với --ask)")
    args = ap.parse_args()
    if not (args.ask or args.eval):
        ap.error("cần --ask hoặc --eval")

    pipe = RagPipeline(1)
    if args.eval:
        run_eval(pipe, args)
        return
    chapters = parse_chapter_set(f"chương {args.chapters}") if args.chapters else None
    tr = pipe.answer(args.ask, method=args.method, k=args.k, expand=args.expand, chapters=chapters,
                     auto_chapters=not args.no_auto_chapters, max_context_tokens=args.max_context_tokens,
                     model=args.model, call_llm=not args.no_llm)
    print(format_trace(tr, show_prompt=args.show_prompt))
    if tr.get("answer") and tr["answer"].get("citations"):
        print("NGUỒN (dạng đọc được):")
        for line in format_citations(tr["answer"]["citations"], pipe.by_id):
            print(f"  - {line}")
    if args.save:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        p = TRACES_DIR / f"ask-{datetime.now():%Y%m%d-%H%M%S}.json"
        p.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"đã lưu trace: {p}")


if __name__ == "__main__":
    main()
