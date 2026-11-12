"""V1 - Bước 3b: Chẩn đoán ĐỘ SÂU tìm kiếm: bằng chứng nằm ở hạng nào trong toàn bộ 393 chunk? (chưa dùng LLM)

Bước 3 chỉ nhìn top-5. Ở đây ta xếp hạng ĐẦY ĐỦ mọi chunk cho mỗi câu hỏi, rồi hỏi:
  * hạng của chunk đúng đầu tiên là mấy?                       (cột "đầu")
  * phải lấy top-k bằng bao nhiêu mới ĐỦ mọi nhóm bằng chứng?  (cột "đủ")   <- con số quan trọng nhất
Và so sánh 5 cách xếp hạng cho cùng một bộ câu hỏi:
  dense-plain    : tìm theo ý nghĩa, vector của text (collection w01_plain)
  dense-header   : như trên nhưng vector của header+text (collection w01_header)
  bm25           : tìm theo TỪ KHÓA (src/bm25.py), trên text
  hybrid-plain   : trộn dense-plain + bm25 bằng RRF (src/retriever.py)
  hybrid-header  : trộn dense-header + bm25 bằng RRF
Không chỉnh tham số nào theo bộ câu hỏi (k1, b, hằng số RRF để mặc định). Bộ câu hỏi chỉ có 10 câu và do chính
người viết đáp án soạn từ văn bản, nên BM25/hybrid có thể được lợi hơn thực tế (đặc biệt câu Q02 trích từ của sách).

Chạy từ thư mục gốc:
    python scripts/v1_step3b_rank_depth.py
    python scripts/v1_step3b_rank_depth.py --methods dense-plain dense-header     # chỉ xem dense (việc "thử top-10/20")
    python scripts/v1_step3b_rank_depth.py --only Q09 --verbose                    # hạng từng nhóm bằng chứng của một câu
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25  # noqa: E402
from src.chunk_store import chunks_path, read_jsonl  # noqa: E402
from src.embedder import QUERY_TASK, cache_path, embed_with_cache, make_client  # noqa: E402
from src.evalset import DEFAULT_QUESTIONS, gold_groups, group_ranks, load_questions, summarize  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.retriever import METHODS, rank_all_methods  # noqa: E402
from src.vector_store import get_client, get_collection  # noqa: E402


def cell(s: dict) -> str:
    first = s["first"] if s["first"] else "-"
    need = s["need"] if s["need"] else "-"
    return f"{first}/{need}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="file câu hỏi (mặc định bộ của Claude)")
    ap.add_argument("--verbose", action="store_true", help="in hạng của từng nhóm bằng chứng")
    args = ap.parse_args()

    data = load_questions(args.questions)
    questions = [q for q in data["questions"] if q["evidence"] and (not args.only or q["id"] in args.only)]
    records = read_jsonl(chunks_path(data["world_no"]))
    world = load_world(find_world_file(data["world_no"]))
    ids = [r.chunk_id for r in records]  # thứ tự đọc truyện, khớp với chỉ số của BM25

    qvecs, st = embed_with_cache(
        make_client(), [(q["id"], q["question"]) for q in questions], cache_path(world.world_no, "queries"),
        task_type=QUERY_TASK, log=lambda s: None,
    )
    chroma = get_client()
    cols = {v: get_collection(chroma, world.world_no, v) for v in ("plain", "header")}
    bm25 = BM25([r.text for r in records])  # BM25 trên text (không header)

    res: dict[str, dict[str, dict]] = {}  # qid -> method -> {"ranks": [...], "first", "need"}
    for q, qvec in zip(questions, qvecs):
        groups = gold_groups(q, records, world)
        rk = rank_all_methods(q["question"], qvec, cols, bm25, ids)
        res[q["id"]] = {}
        for m in args.methods:
            ranks = group_ranks(rk[m], groups)
            res[q["id"]][m] = {"ranks": ranks, **summarize(ranks)}

    print("Mỗi ô = 'đầu/đủ': hạng chunk đúng đầu tiên / số k tối thiểu để mọi nhóm bằng chứng đều có mặt (trong 393 chunk).")
    print("'-' = không bao giờ tìm thấy (một nhóm bằng chứng không có trong toàn bộ danh sách, không nên xảy ra).\n")
    w = 14
    print(f"{'câu':<5}{'nhóm':>5}" + "".join(f"{m:>{w}}" for m in args.methods))
    for q in questions:
        row = f"{q['id']:<5}{len(q['evidence']):>5}" + "".join(f"{cell(res[q['id']][m]):>{w}}" for m in args.methods)
        print(row)
        if args.verbose:
            for m in args.methods:
                print(f"         {m:<14} hạng từng nhóm: {res[q['id']][m]['ranks']}")

    n = len(questions)
    print("\n" + "-" * (10 + w * len(args.methods)))
    print(f"TỔNG HỢP trên {n} câu có đáp án (hit@k = chunk đúng nào đó nằm trong top-k; đủ@k = mọi nhóm bằng chứng nằm trong top-k)")
    print(f"{'':<12}" + "".join(f"{m:>{w}}" for m in args.methods))
    print(f"{'top-1 đúng':<12}" + "".join(f"{sum(res[q['id']][m]['first'] == 1 for q in questions):>{w - 3}}/{n} " for m in args.methods))
    for k in args.ks:
        print(f"{f'hit@{k}':<12}" + "".join(
            f"{sum(bool(res[q['id']][m]['first']) and res[q['id']][m]['first'] <= k for q in questions):>{w - 3}}/{n} " for m in args.methods))
    for k in args.ks:
        print(f"{f'đủ@{k}':<12}" + "".join(
            f"{sum(bool(res[q['id']][m]['need']) and res[q['id']][m]['need'] <= k for q in questions):>{w - 3}}/{n} " for m in args.methods))
    print(f"{'MRR':<12}" + "".join(
        f"{sum(1 / res[q['id']][m]['first'] for q in questions if res[q['id']][m]['first']) / n:>{w}.2f}" for m in args.methods))


if __name__ == "__main__":
    main()
