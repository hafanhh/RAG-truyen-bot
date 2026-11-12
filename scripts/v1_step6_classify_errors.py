"""V1 - Bước 6: PHÂN LOẠI LỖI của mốc (hybrid-header, k=5, không mở rộng) để chọn đúng cách sửa.

Quy trình quyết định (do người dùng chốt):
    mốc  ->  phân loại lỗi  ->  THIẾU bằng chứng       : thử viết lại / tách câu hỏi
                                ĐỦ bằng chứng nhưng sai : thử kiểm tra bằng LLM
                                chỉ thử chunk nhỏ hơn khi chẩn đoán cho thấy RANH GIỚI hoặc ĐỘ DÀI chunk thật sự là vấn đề

Đầu vào: thư mục trace của một lần chạy `v1_step5_answer.py --eval` (data/traces/...) + đáp án nguồn (data/eval/*.json)
+ điểm chấm tay (data/eval/manual_grades_w01.json: người đọc đối chiếu câu trả lời với expected_answer).

Với mỗi câu:
  * bằng chứng ĐÃ GỬI đủ chưa? (mọi nhóm bằng chứng đều có một chunk nằm trong tập gửi cho LLM)
  * verdict của mô hình và điểm chấm tay -> phân loại.
Với mỗi nhóm bằng chứng THIẾU, chẩn đoán nguyên nhân (cần vài lần gọi API embedding, tắt bằng --no-diagnose):
  * hạng thật của nhóm đó trong từng cách tìm (dense-header, bm25, hybrid-header) trên toàn bộ 393 chunk;
  * RANH GIỚI: có chunk liền kề (prev/next) của chunk chứa bằng chứng ĐÃ được tìm không?
    (có = bằng chứng bị cắt đôi ở ranh giới chunk; mở rộng lân cận hoặc đổi cách cắt sẽ cứu được)
  * LOÃNG: đoạn bằng chứng embed RIÊNG có gần câu hỏi hơn chunk chứa nó không? (cosine; Δ = đoạn - chunk; Δ lớn = chunk
    dài làm loãng chi tiết)
  * còn lại: không do chunk, mà do tìm theo ý/từ khóa không khớp câu hỏi -> thuộc nhóm viết lại/tách câu hỏi.

Chạy từ thư mục gốc:
    python scripts/v1_step6_classify_errors.py                       # lần chạy mốc mới nhất (prompt v3, hybrid-header, k5, e0)
    python scripts/v1_step6_classify_errors.py --traces data/traces/20261112-090852-hybrid-header-k5-e0
    python scripts/v1_step6_classify_errors.py --no-diagnose         # chỉ phân loại, không gọi API
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk_store import chunks_path, read_jsonl  # noqa: E402
from src.config import EVAL_DIR  # noqa: E402
from src.embedder import DOC_TASK, cache_path, embed_query_cached, embed_with_cache  # noqa: E402
from src.evalset import gold_groups, group_ranks, load_questions  # noqa: E402
from src.loader import find_world_file, load_world  # noqa: E402
from src.rag import RagPipeline  # noqa: E402
from src.retriever import dense_ranking, rank_all_methods  # noqa: E402

SETS = [EVAL_DIR / "questions_w01.json", EVAL_DIR / "questions_w01_user.json"]
GRADES = EVAL_DIR / "manual_grades_w01.json"
DILUTION_HINT = 0.05  # Δ cosine (đoạn riêng - chunk) từ mức này trở lên thì ghi nhận "gợi ý loãng"; chỉ là mốc để đọc, không phải ngưỡng đã kiểm định


def latest_baseline() -> Path:
    for d in sorted(glob.glob("data/traces/*-hybrid-header-k5-e0"), reverse=True):
        f = Path(d) / "Q02.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8"))["config"].get("prompt_version") == "v3":
            return Path(d)
    sys.exit("Không tìm thấy lần chạy mốc (prompt v3, hybrid-header, k5, e0) trong data/traces/")


def classify(answerable: bool, evidence_ok: bool, verdict: str, grade: str | None) -> str:
    if not answerable:
        return "NGOÀI TRUYỆN: từ chối đúng" if verdict == "không đủ thông tin" else "NGOÀI TRUYỆN: BỊA câu trả lời"
    if evidence_ok:
        if verdict != "đủ":
            return "ĐỦ bằng chứng nhưng mô hình từ chối/chỉ trả một phần"
        if grade in (None, "đúng"):
            return "OK (đủ bằng chứng, trả lời đúng)" if grade else "đủ bằng chứng, trả lời chưa chấm"
        return "ĐỦ bằng chứng nhưng trả lời thiếu/sai"
    if verdict == "đủ":
        return "THIẾU bằng chứng nhưng tự tin trả lời (RỦI RO)"
    return "THIẾU bằng chứng, từ chối/một phần (an toàn)"


def diagnose(pipe: RagPipeline, world, q: dict, groups: list[set[str]], missing: list[int], retrieved: set[str]) -> list[str]:
    """Chẩn đoán nguyên nhân cho từng nhóm bằng chứng thiếu. Trả về các dòng văn bản."""
    out = []
    qvec = embed_query_cached(pipe.client, q["question"])  # có cache: câu hỏi của bộ eval không tốn hạn mức API
    rk = rank_all_methods(q["question"], qvec, pipe.cols, pipe.bm25, pipe.ids)
    plain_sim = dict(dense_ranking(pipe.cols["plain"], qvec))
    for gi in missing:
        g = groups[gi]
        ranks = {m: group_ranks(rk[m], [g])[0] for m in ("dense-header", "bm25", "hybrid-header")}
        # RANH GIỚI: chunk liền kề của chunk chứa bằng chứng đã nằm trong tập tìm được?
        adj = sorted({n for c in g for n in (pipe.by_id[c].prev_id, pipe.by_id[c].next_id) if n and n in retrieved})
        # LOÃNG: đoạn bằng chứng embed riêng so với chunk chứa nó
        paras = [(ch, p) for ch, p in q["evidence"][gi]]
        out.append(f"      nhóm {gi + 1}: hạng thật trong 393 chunk: dense-header #{ranks['dense-header']}, "
                   f"bm25 #{ranks['bm25']}, hybrid-header #{ranks['hybrid-header']}")
        # Đo độ loãng cần embed RIÊNG các đoạn bằng chứng (tốn hạn mức API, có cache vào data/embeddings/w01_paragraphs_*.npz).
        # Nếu hết hạn mức thì bỏ qua phép đo này, không làm hỏng phần còn lại.
        delta = None
        try:
            items = [(f"p{ch:03d}-{p:03d}", world.chapters[ch - 1].paragraphs[p - 1]) for ch, p in paras]
            pvecs, _ = embed_with_cache(pipe.client, items, cache_path(1, "paragraphs"), task_type=DOC_TASK, log=lambda s: None)
            best = max(range(len(paras)), key=lambda i: float(qvec @ pvecs[i]))
            p_sim = float(qvec @ pvecs[best])
            c_sim = max(plain_sim[c] for c in g)
            delta = p_sim - c_sim
            out.append(f"        cosine: đoạn riêng ch{paras[best][0]} đ{paras[best][1]} {p_sim:.3f} | chunk chứa nó {c_sim:.3f} | Δ {delta:+.3f}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"        cosine: CHƯA ĐO ĐƯỢC độ loãng ({type(exc).__name__}: hết hạn mức embedding trong ngày hoặc lỗi API)")
        # Cả ba kết luận dưới đây chỉ là DẤU HIỆU để chọn thí nghiệm kế tiếp, không phải bằng chứng nhân quả.
        if adj:
            cause = (f"dấu hiệu RANH GIỚI: chunk liền kề ({', '.join(adj)}) đã được tìm, bằng chứng nằm ở chunk bên cạnh; "
                     "cần thử lấy thêm chunk lân cận để xác nhận, chưa chứng minh phải đổi kích thước chunk")
        elif delta is None:
            cause = "chưa kết luận: chưa thấy chunk liền kề được tìm và chưa đo được độ loãng (chưa loại trừ ranh giới/độ dài)"
        elif delta >= DILUTION_HINT:
            cause = f"gợi ý LOÃNG (đoạn riêng gần câu hỏi hơn chunk chứa nó, Δ={delta:+.3f}); chỉ là gợi ý, chưa chứng minh độ dài gây lỗi"
        else:
            cause = ("chưa thấy dấu hiệu do chunk (đoạn riêng không gần hơn rõ rệt, không có chunk liền kề được tìm); "
                     "đây KHÔNG phải bằng chứng loại trừ ranh giới/độ dài")
        out.append(f"        => {cause}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path)
    ap.add_argument("--questions", nargs="+", type=Path, default=SETS)
    ap.add_argument("--grades", type=Path, default=GRADES)
    ap.add_argument("--no-diagnose", action="store_true")
    args = ap.parse_args()

    tdir = args.traces or latest_baseline()
    cfg = json.loads((tdir / "Q02.json").read_text(encoding="utf-8"))["config"]
    grades = json.loads(args.grades.read_text(encoding="utf-8")) if args.grades.exists() else {"grades": {}}
    print(f"Lần chạy: {tdir} | {cfg['method']} k={cfg['k']} mở rộng ±{cfg['expand']} prompt {cfg['prompt_version']}")
    print(f"Điểm chấm tay: {grades.get('run', '(không có)')}\n")

    pipe = None if args.no_diagnose else RagPipeline(1)
    world = load_world(find_world_file(1))
    records = pipe.records if pipe else read_jsonl(chunks_path(1))
    qs = [q for p in args.questions for q in load_questions(p)["questions"] if "TRÙNG" not in q["type"]]

    rows, by_cat = [], {}
    for q in qs:
        f = tdir / f"{q['id']}.json"
        if not f.exists():
            continue
        t = json.loads(f.read_text(encoding="utf-8"))
        groups = gold_groups(q, records, world) if q["evidence"] else []
        sent, retrieved = set(t["sent_ids"]), {r["chunk_id"] for r in t["retrieved"]}
        missing = [i for i, g in enumerate(groups) if not (g & sent)]
        g = grades["grades"].get(q["id"], {})
        cat = classify(bool(groups), not missing, t["llm"]["verdict"], g.get("grade"))
        rows.append((q, t, groups, missing, retrieved, g, cat))
        by_cat.setdefault(cat, []).append(q["id"])

    for q, t, groups, missing, retrieved, g, cat in rows:
        amb = " [CÂU HỎI MƠ HỒ, báo cáo riêng]" if g.get("ambiguous_question") else ""
        print(f"{q['id']:<4} [{t['llm']['verdict']:<18}] chấm tay: {g.get('grade', '-'):<9} | {cat}{amb}")
        if g.get("note"):
            print(f"      ghi chú: {g['note']}")
        for name, text in g.get("aspects", {}).items():
            print(f"      - {name}: {text}")
        if missing and pipe:
            for line in diagnose(pipe, world, q, groups, missing, retrieved):
                print(line)

    print("\n" + "=" * 78 + "\nTỔNG HỢP THEO LOẠI LỖI")
    for cat, ids in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(ids):>2}  {cat}: {', '.join(ids)}")
    print("\nHướng xử lý theo quy trình đã chốt:")
    print("  THIẾU bằng chứng (nguyên nhân KHÔNG do chunk)   -> viết lại / tách câu hỏi")
    print("  ĐỦ bằng chứng nhưng trả lời thiếu/sai/tự tin sai -> kiểm tra bằng LLM")
    print("  RANH GIỚI hoặc LOÃNG được chẩn đoán ở trên        -> mới là lý do để thử chunk khác")


if __name__ == "__main__":
    main()
