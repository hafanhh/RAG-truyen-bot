"""V1 - Bước 4: Mở rộng chunk lân cận (prev/next) và lọc theo metadata. (chưa dùng LLM)

A) MỞ RỘNG LÂN CẬN: lấy top-k rồi thêm chunk liền trước/sau mỗi chunk (đi theo prev_id/next_id, không qua chương).
   Câu hỏi công bằng: mở rộng có tốt hơn việc CHỈ LẤY NHIỀU CHUNK HƠN không? Vì mở rộng làm tăng số chunk (k -> tối đa
   3k), ta so ba cách với CÙNG ngân sách chunk:
        topk      : chỉ top-k
        mở rộng   : top-k + lân cận                 (kích thước = N chunk)
        topN      : top-N thẳng từ bảng xếp hạng, N = kích thước của "mở rộng"
   Nếu "mở rộng" không hơn "topN" thì lân cận chẳng có giá trị riêng trên bộ câu hỏi này.
B) LỌC METADATA (kiểu "oracle"): giả sử người dùng nói đúng chương chứa đáp án ("ở chương 58..."), chỉ tìm trong các
   chương đó (Chroma `where`). Đây là CẬN TRÊN: thực tế phải biết chương từ câu hỏi, và câu hỏi hay không nói chương.
C) --ask: hỏi tự do. Tự nhận ra "chương N", "chương A-B" trong câu hỏi để lọc; có thể thêm --chapters thủ công.

Chạy từ thư mục gốc:
    python scripts/v1_step4_expand_filter.py                      # A và B trên bộ câu hỏi
    python scripts/v1_step4_expand_filter.py --n 2                # mở rộng 2 chunk mỗi phía
    python scripts/v1_step4_expand_filter.py --ask "Ở chương 58, Đường Quả bảo Lãnh Tử Việt gọi nàng là gì?" --expand 1
    python scripts/v1_step4_expand_filter.py --ask "Đường Quả uống gì?" --chapters 1-2 --k 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25  # noqa: E402
from src.chunk_store import chunks_path, read_jsonl  # noqa: E402
from src.embedder import QUERY_TASK, cache_path, embed_texts, embed_with_cache, make_client  # noqa: E402
from src.evalset import DEFAULT_QUESTIONS, gold_groups, group_ranks, load_questions, summarize  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.retriever import (  # noqa: E402
    chapter_where, dense_ranking, expand_neighbors, parse_chapter_set, rank_all_methods,
)
from src.vector_store import get_client, get_collection  # noqa: E402


def enough(ids, groups) -> bool:
    s = set(ids)
    return all(g & s for g in groups)


# ---------------------------------------------------------------------------
# A) mở rộng lân cận
# ---------------------------------------------------------------------------


def eval_expand(questions, groups_by_q, rankings, by_id, methods, ks, n) -> None:
    print(f"=== A) Mở rộng lân cận (±{n} chunk, không qua chương) so với chỉ lấy nhiều chunk hơn ===")
    print("Mỗi ô: ✓ nếu mọi nhóm bằng chứng có mặt. (kích thước) = số chunk. 'topN' có cùng số chunk với 'mở rộng'.\n")
    for m in methods:
        for k in ks:
            tot = {"topk": 0, "exp": 0, "topn": 0}
            sizes, chars_e, chars_n, chars_k = [], [], [], []
            rows = []
            for q in questions:
                rk, groups = rankings[q["id"]][m], groups_by_q[q["id"]]
                hits = rk[:k]
                exp = expand_neighbors(hits, by_id, n=n)
                topn = rk[: len(exp)]
                a, b, c = enough(hits, groups), enough(exp, groups), enough(topn, groups)
                tot["topk"] += a; tot["exp"] += b; tot["topn"] += c
                sizes.append(len(exp))
                chars_k.append(sum(by_id[x].n_chars for x in hits))
                chars_e.append(sum(by_id[x].n_chars for x in exp))
                chars_n.append(sum(by_id[x].n_chars for x in topn))
                rows.append(f"    {q['id']:<4} top{k}: {'✓' if a else '✗'}   mở rộng ({len(exp):>2}): {'✓' if b else '✗'}   "
                            f"top{len(exp):<2}: {'✓' if c else '✗'}")
            nq = len(questions)
            print(f"[{m}] k={k}:  chỉ top-{k} {tot['topk']}/{nq}  |  mở rộng {tot['exp']}/{nq} (TB {sum(sizes) / nq:.1f} chunk, "
                  f"~{sum(chars_e) / nq:.0f} ký tự)  |  topN cùng cỡ {tot['topn']}/{nq} (~{sum(chars_n) / nq:.0f} ký tự)")
            if k == ks[-1]:
                print("\n".join(rows))
            print()


# ---------------------------------------------------------------------------
# B) lọc metadata kiểu oracle
# ---------------------------------------------------------------------------


def eval_filter(questions, groups_by_q, qvecs, cols, records, world) -> None:
    print("=== B) Lọc metadata kiểu ORACLE: chỉ tìm trong (các) chương chứa đáp án (dense-plain) ===")
    print("Ô = 'đầu/đủ' (hạng chunk đúng đầu tiên / k tối thiểu để đủ bằng chứng); 'pool' = số chunk còn lại sau lọc.\n")
    print(f"{'câu':<5}{'chương lọc':<16}{'pool':>6}{'không lọc':>12}{'có lọc':>10}")
    gain = 0
    for q, qvec in zip(questions, qvecs):
        chapters = sorted({ch for grp in q["evidence"] for ch, _ in grp})
        where = {"chapter_no": {"$in": chapters}} if len(chapters) > 1 else {"chapter_no": chapters[0]}
        pool = sum(r.chapter_no in chapters for r in records)
        groups = groups_by_q[q["id"]]
        s0 = summarize(group_ranks([i for i, _ in dense_ranking(cols["plain"], qvec)], groups))
        s1 = summarize(group_ranks([i for i, _ in dense_ranking(cols["plain"], qvec, where=where)], groups))
        fmt = lambda s: f"{s['first'] or '-'}/{s['need'] or '-'}"
        gain += (s1["need"] or 99) < (s0["need"] or 99)
        print(f"{q['id']:<5}{','.join(map(str, chapters)):<16}{pool:>6}{fmt(s0):>12}{fmt(s1):>10}")
    print(f"\nLọc đưa 'k tối thiểu để đủ bằng chứng' xuống thấp hơn ở {gain}/{len(questions)} câu. "
          f"(Cận trên: cần biết trước chương chứa đáp án.)\n")


# ---------------------------------------------------------------------------
# C) hỏi tự do
# ---------------------------------------------------------------------------


def run_ask(args, records, by_id) -> None:
    chapters = parse_chapter_set(f"chương {args.chapters}") if args.chapters else parse_chapter_set(args.ask)
    where = chapter_where(chapters)
    print(f"Câu hỏi: {args.ask}")
    print(f"Bộ lọc metadata: {where if where else 'không (tìm trong toàn bộ truyện)'}")
    qvec = embed_texts(make_client(), [args.ask], QUERY_TASK)[0]
    col = get_collection(get_client(), 1, args.variant)
    ranking = dense_ranking(col, qvec, where=where)
    sim = dict(ranking)
    hits = [i for i, _ in ranking[: args.k]]
    ids = expand_neighbors(hits, by_id, n=args.expand) if args.expand else sorted(hits, key=list(by_id).index)
    print(f"Top-{args.k}" + (f" + lân cận ±{args.expand} -> {len(ids)} chunk, theo thứ tự đọc truyện" if args.expand else "") + ":\n")
    for cid in ids:
        r = by_id[cid]
        tag = f"★ top-{hits.index(cid) + 1} ({sim[cid]:.3f})" if cid in hits else "· lân cận"
        print(f"  {tag:<18} {cid}  ch{r.chapter_no} đ{r.para_start}-{r.para_end}")
        if args.text:
            print(f"      {r.text[:200].replace(chr(10), ' ')}...")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["dense-plain", "hybrid-plain"])
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--n", type=int, default=1, help="mở rộng n chunk mỗi phía")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="file câu hỏi (mặc định bộ của Claude)")
    ap.add_argument("--ask", metavar="CÂU_HỎI")
    ap.add_argument("--chapters", metavar="N|A-B", help="lọc thủ công theo chương cho --ask")
    ap.add_argument("--k", type=int, default=3, help="số chunk top cho --ask")
    ap.add_argument("--expand", type=int, default=0, help="mở rộng n chunk mỗi phía cho --ask")
    ap.add_argument("--variant", choices=["plain", "header"], default="plain")
    ap.add_argument("--text", action="store_true", help="in kèm nội dung chunk (cho --ask)")
    args = ap.parse_args()

    data = load_questions(args.questions)
    records = read_jsonl(chunks_path(data["world_no"]))
    by_id = {r.chunk_id: r for r in records}  # giữ thứ tự đọc truyện
    if args.ask:
        run_ask(args, records, by_id)
        return

    world = load_world(find_world_file(data["world_no"]))
    questions = [q for q in data["questions"] if q["evidence"]]
    ids = [r.chunk_id for r in records]
    qvecs, _ = embed_with_cache(make_client(), [(q["id"], q["question"]) for q in questions],
                                cache_path(world.world_no, "queries"), task_type=QUERY_TASK, log=lambda s: None)
    chroma = get_client()
    cols = {v: get_collection(chroma, world.world_no, v) for v in ("plain", "header")}
    bm25 = BM25([r.text for r in records])
    groups_by_q = {q["id"]: gold_groups(q, records, world) for q in questions}
    rankings = {q["id"]: rank_all_methods(q["question"], qv, cols, bm25, ids) for q, qv in zip(questions, qvecs)}

    eval_expand(questions, groups_by_q, rankings, by_id, args.methods, args.ks, args.n)
    eval_filter(questions, groups_by_q, qvecs, cols, records, world)


if __name__ == "__main__":
    main()
