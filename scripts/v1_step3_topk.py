"""V1 - Bước 3: Kiểm tra TÌM KIẾM (retrieval), chưa dùng LLM trả lời.

Quy trình (đúng thứ tự):
  1. ĐÁP ÁN NGUỒN được ghi TRƯỚC trong data/eval/questions_w01.json (chương + số đoạn chứa đáp án), không dựa
     vào kết quả embedding. Xem lại bằng:   python scripts/v1_step3_topk.py --show-gold
  2. Embed câu hỏi bằng task RETRIEVAL_QUERY (chunk đã embed bằng RETRIEVAL_DOCUMENT ở Bước 2).
  3. Lấy top-k (mặc định 5) ở MỖI collection (w01_plain và w01_header) với CÙNG một bộ câu hỏi.
  4. Với mỗi câu hỏi và mỗi collection, kiểm tra:
       - hạng của chunk đúng đầu tiên (1 = đứng đầu; "-" = không có trong top-k)
       - top-1 có đúng không? đúng có nằm trong top-k không? (hit@k)
       - bằng chứng ĐỦ chưa? (mọi nhóm bằng chứng đều được phủ bởi ít nhất một chunk trong top-k)
     Câu "không có trong truyện" không có đáp án để trúng; ta chỉ xem điểm similarity cao nhất của chúng so với
     các câu có đáp án (để biết có thể đặt ngưỡng "không đủ thông tin" được không).

Chạy từ thư mục gốc:
    python scripts/v1_step3_topk.py --show-gold               # in đoạn nguồn của đáp án, không gọi API
    python scripts/v1_step3_topk.py                           # chạy toàn bộ bộ câu hỏi, top-5, cả 2 collection
    python scripts/v1_step3_topk.py --only Q01 Q02 --text     # xem nội dung chunk (không chỉ id) của vài câu
    python scripts/v1_step3_topk.py --ask "Ai là cha nuôi của Lãnh Tử Việt?"   # hỏi tự do, không có đáp án để chấm
    python scripts/v1_step3_topk.py --k 10
Điểm hiển thị là cosine similarity = 1 - distance của Chroma (cao hơn = gần hơn). Điểm của Gemini dồn trong
khoảng ~0.5-0.85 nên chỉ so sánh thứ hạng và chênh lệch, đừng đặt ngưỡng tuyệt đối.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_store import ChunkRecord, chunks_path, read_jsonl  # noqa: E402
from src.embedder import QUERY_TASK, cache_path, embed_texts, embed_with_cache, make_client  # noqa: E402
from src.evalset import DEFAULT_QUESTIONS, covering_chunks, gold_groups  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.vector_store import collection_name, get_client, get_collection  # noqa: E402

VARIANTS = ["plain", "header"]

# Đáp án nguồn (chương, đoạn) -> chunk phủ đoạn đó: xem src/evalset.py (covering_chunks, gold_groups).


def show_gold(questions: list[dict], records: list[ChunkRecord], world) -> None:
    """In đáp án nguồn (văn bản gốc) để người dùng kiểm tra bằng mắt. Không gọi API, không chạy retrieval."""
    for q in questions:
        print(f"\n{q['id']} [{q['type']}] {q['question']}")
        print(f"   đáp án mong đợi: {q['expected_answer']}")
        if not q["evidence"]:
            print("   (không có trong truyện, không có đoạn nguồn)")
            continue
        for gi, group in enumerate(q["evidence"], 1):
            print(f"   nhóm bằng chứng {gi} (chỉ cần MỘT trong các đoạn sau):")
            for chapter, para in group:
                text = world.chapters[chapter - 1].paragraphs[para - 1]
                print(f"     ch{chapter} đ{para}: {text[:200]}")
                print(f"       -> nằm trong chunk: {', '.join(covering_chunks(records, chapter, para))}")
        if q.get("note"):
            print(f"   ghi chú: {q['note']}")


# ---------------------------------------------------------------------------
# Truy vấn
# ---------------------------------------------------------------------------


def search(col, qvec, k: int) -> list[dict]:
    """Top-k của một collection: danh sách {id, sim, ch, ps, pe, text}, đã sắp theo độ gần giảm dần."""
    res = col.query(query_embeddings=[qvec.tolist()], n_results=k, include=["distances", "metadatas", "documents"])
    out = []
    for cid, dist, meta, doc in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]):
        out.append({"id": cid, "sim": 1.0 - dist, "ch": meta["chapter_no"], "ps": meta["para_start"],
                    "pe": meta["para_end"], "text": doc})
    return out


def print_hits(hits: list[dict], gold_ids: set[str], show_text: bool) -> None:
    for rank, h in enumerate(hits, 1):
        mark = "✓" if h["id"] in gold_ids else " "
        line = f"      {rank} {mark} {h['id']}  {h['sim']:.3f}  ch{h['ch']} đ{h['ps']}-{h['pe']}"
        print(line)
        if show_text:
            print(f"          {h['text'][:160].replace(chr(10), ' ')}...")


def judge(hits: list[dict], groups: list[set[str]]) -> dict:
    """Chấm một danh sách top-k: hạng chunk đúng đầu tiên, top-1 đúng?, trong top-k?, bằng chứng đủ chưa?"""
    gold_ids = set().union(*groups) if groups else set()
    ids = [h["id"] for h in hits]
    rank = next((i for i, cid in enumerate(ids, 1) if cid in gold_ids), None)
    covered = all(g & set(ids) for g in groups)
    return {"rank": rank, "top1": rank == 1, "intopk": rank is not None, "enough": covered, "sim1": hits[0]["sim"]}


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------


def run_ask(question: str, k: int, show_text: bool) -> None:
    """Hỏi tự do: in top-k của cả 2 collection, không có đáp án để chấm."""
    qvec = embed_texts(make_client(), [question], QUERY_TASK)[0]
    chroma = get_client()
    print(f"Câu hỏi: {question}")
    for v in VARIANTS:
        print(f"\n  [{collection_name(1, v)}]")
        print_hits(search(get_collection(chroma, 1, v), qvec, k), set(), show_text)


def run_eval(questions: list[dict], records: list[ChunkRecord], world, k: int, show_text: bool) -> None:
    # Embed câu hỏi một lần (có cache), dùng chung cho cả hai collection.
    items = [(q["id"], q["question"]) for q in questions]
    qvecs, st = embed_with_cache(make_client(), items, cache_path(world.world_no, "queries"), task_type=QUERY_TASK)
    print(f"Embed {len(items)} câu hỏi (RETRIEVAL_QUERY): {st['cached']} từ cache, {st['api']} gọi API\n")

    chroma = get_client()
    cols = {v: get_collection(chroma, world.world_no, v) for v in VARIANTS}
    results: dict[str, dict[str, dict]] = {}  # qid -> variant -> judge

    for q, qvec in zip(questions, qvecs):
        groups = gold_groups(q, records, world)
        gold_ids = set().union(*groups) if groups else set()
        print(f"{q['id']} [{q['type']}] {q['question']}")
        if groups:
            print(f"   chunk đúng: {', '.join(sorted(gold_ids))}  ({len(groups)} nhóm bằng chứng)")
        else:
            print("   (không có trong truyện)")
        results[q["id"]] = {}
        for v in VARIANTS:
            hits = search(cols[v], qvec, k)
            results[q["id"]][v] = judge(hits, groups)
            print(f"    [{v}]")
            print_hits(hits, gold_ids, show_text)
        print()

    # ---------------- bảng tóm tắt ----------------
    ans = [q for q in questions if q["evidence"]]
    unans = [q for q in questions if not q["evidence"]]
    print("=" * 78)
    print(f"TÓM TẮT (top-{k}). hạng = vị trí chunk đúng đầu tiên; 'đủ' = mọi nhóm bằng chứng đều nằm trong top-{k}")
    print(f"{'':<5}{'plain':^30}{'header':^30}")
    print(f"{'câu':<5}{'hạng':>6}{'đủ':>6}{'sim#1':>9}{'':>9}{'hạng':>6}{'đủ':>6}{'sim#1':>9}")
    for q in ans:
        cells = []
        for v in VARIANTS:
            r = results[q["id"]][v]
            cells.append(f"{r['rank'] if r['rank'] else '-':>6}{'✓' if r['enough'] else '✗':>6}{r['sim1']:>9.3f}")
        print(f"{q['id']:<5}{cells[0]}{'':>9}{cells[1]}")
    print("-" * 78)
    n = len(ans)
    for v in VARIANTS:
        rs = [results[q["id"]][v] for q in ans]
        mrr = sum(1 / r["rank"] for r in rs if r["rank"]) / n
        print(f"[{v:<6}] top-1 đúng {sum(r['top1'] for r in rs)}/{n} | hit@{k} {sum(r['intopk'] for r in rs)}/{n} "
              f"| đủ bằng chứng@{k} {sum(r['enough'] for r in rs)}/{n} | MRR {mrr:.2f}")

    if unans:
        print("\nCâu KHÔNG có trong truyện: điểm similarity của chunk đứng đầu so với các câu có đáp án")
        for v in VARIANTS:
            a = [results[q["id"]][v]["sim1"] for q in ans]
            u = [(q["id"], results[q["id"]][v]["sim1"]) for q in unans]
            print(f"  [{v:<6}] có đáp án: min {min(a):.3f} | trung bình {sum(a) / len(a):.3f} | max {max(a):.3f}"
                  f"   ||   không có: " + ", ".join(f"{i} {s:.3f}" for i, s in u))
        print("  (nếu điểm 'không có' nằm lẫn trong khoảng 'có đáp án' thì KHÔNG thể dùng điểm để từ chối trả lời)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--only", nargs="+", metavar="ID", help="chỉ chạy các câu này, vd Q01 Q03")
    ap.add_argument("--show-gold", action="store_true", help="chỉ in đoạn nguồn của đáp án (không gọi API)")
    ap.add_argument("--text", action="store_true", help="in kèm nội dung chunk")
    ap.add_argument("--ask", metavar="CÂU_HỎI", help="hỏi tự do (không chấm)")
    args = ap.parse_args()

    if args.ask:
        run_ask(args.ask, args.k, args.text)
        return

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = data["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
        if not questions:
            sys.exit(f"Không có câu nào khớp {args.only}")
    records = read_jsonl(chunks_path(data["world_no"]))
    world = load_world(find_world_file(data["world_no"]))

    if args.show_gold:
        show_gold(questions, records, world)
        return
    run_eval(questions, records, world, args.k, args.text)


if __name__ == "__main__":
    main()
