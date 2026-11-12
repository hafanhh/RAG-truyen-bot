"""Các hàm retrieval dùng chung: xếp hạng đầy đủ bằng dense (Chroma), hợp nhất hạng (RRF), mở rộng chunk lân cận.

  * dense_ranking: hỏi Chroma với n_results = toàn bộ collection để có HẠNG ĐẦY ĐỦ của mọi chunk (dùng cho chẩn đoán:
    chunk đúng đứng thứ mấy trong 393?). Có thể kèm `where` (lọc metadata) để chỉ xếp hạng trong phần được lọc.
  * rrf: Reciprocal Rank Fusion, cách đơn giản để trộn nhiều danh sách xếp hạng (vd dense + BM25) mà không cần so
    sánh điểm của hai hệ (điểm cosine và điểm BM25 khác thang đo, không cộng trực tiếp được).
        điểm(chunk) = tổng trên các danh sách của  1 / (60 + hạng của chunk trong danh sách đó)
    60 là hằng số thông dụng. Chunk đứng cao ở CẢ HAI danh sách thì lên đầu.
  * expand_neighbors: thêm chunk liền trước/liền sau theo prev_id/next_id (xem chunk_store.ChunkRecord).
"""
from __future__ import annotations

import re

from src.bm25 import BM25
from src.chunk_store import ChunkRecord

# Các cách xếp hạng để so sánh (xem rank_all_methods)
METHODS = ["dense-plain", "dense-header", "bm25", "hybrid-plain", "hybrid-header"]


def dense_ranking(col, qvec, where: dict | None = None) -> list[tuple[str, float]]:
    """Xếp hạng toàn bộ chunk (hoặc phần khớp `where`) theo độ gần với câu hỏi. Trả về [(chunk_id, similarity)]."""
    n = col.count()
    res = col.query(query_embeddings=[qvec.tolist()], n_results=n, where=where, include=["distances"])
    ids, dists = res["ids"][0], res["distances"][0]
    if where is None and len(ids) != n:  # chốt chặn: HNSW không được trả thiếu khi xin toàn bộ
        raise RuntimeError(f"Chroma chỉ trả {len(ids)}/{n} chunk")
    return [(i, 1.0 - d) for i, d in zip(ids, dists)]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Hợp nhất nhiều danh sách chunk_id đã xếp hạng bằng Reciprocal Rank Fusion; trả về danh sách hợp nhất."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda c: -score[c])


def parse_chapter_set(text: str) -> list[int] | None:
    """Nhận ra chương trong văn bản: 'chương 58' | 'chương 55-62' | 'chương 33 và chương 58' -> danh sách số chương."""
    m = re.search(r"chương\s+(\d+)\s*(?:-|–|đến)\s*(?:chương\s*)?(\d+)", text, re.I)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return list(range(lo, hi + 1))
    nums = sorted({int(x) for x in re.findall(r"chương\s+(\d+)", text, re.I)})
    return nums or None


def chapter_where(chapters: list[int] | None) -> dict | None:
    """Danh sách chương -> điều kiện `where` của Chroma (None nếu không lọc)."""
    if not chapters:
        return None
    return {"chapter_no": chapters[0]} if len(chapters) == 1 else {"chapter_no": {"$in": chapters}}


def rank_all_methods(
    question: str, qvec, cols: dict, bm25: BM25, ids: list[str],
    where: dict | None = None, allowed: set[str] | None = None,
) -> dict[str, list[str]]:
    """Xếp hạng đầy đủ (list chunk_id từ gần nhất tới xa nhất) theo từng phương pháp trong METHODS.

    cols: {"plain": collection, "header": collection}; ids: chunk_id theo thứ tự đọc truyện, khớp chỉ số của bm25.
    Lọc chương (tùy chọn): `where` lọc phía Chroma (dense), `allowed` (tập chunk_id được phép) lọc phía BM25; hai thứ
    phải cùng một ý nghĩa, vì thế dùng chapter_where() và tập chunk của các chương đó.
    """
    r = {
        "dense-plain": [i for i, _ in dense_ranking(cols["plain"], qvec, where=where)],
        "dense-header": [i for i, _ in dense_ranking(cols["header"], qvec, where=where)],
        "bm25": [c for c in (ids[i] for i in bm25.ranking(question)) if allowed is None or c in allowed],
    }
    r["hybrid-plain"] = rrf([r["dense-plain"], r["bm25"]])
    r["hybrid-header"] = rrf([r["dense-header"], r["bm25"]])
    return r


def expand_neighbors(
    hit_ids: list[str], by_id: dict[str, ChunkRecord], n: int = 1, same_chapter: bool = True
) -> list[str]:
    """Thêm tối đa n chunk liền trước và n chunk liền sau mỗi chunk trúng, đi theo prev_id/next_id.

    same_chapter=True: không bước qua ranh giới chương (chunk đầu chương kế không liên quan chunk cuối chương trước).
    Trả về danh sách chunk_id KHÔNG trùng, theo thứ tự ĐỌC TRUYỆN (tiện đưa cho LLM đọc liền mạch).
    Lưu ý: các chunk liền kề có overlap ~200 ký tự nên nội dung ghép lại sẽ lặp một vài đoạn.
    """
    keep: set[str] = set()
    for hid in hit_ids:
        keep.add(hid)
        for link in ("prev_id", "next_id"):
            cur = by_id[hid]
            for _ in range(n):
                nxt = getattr(cur, link)
                if not nxt:
                    break
                cand = by_id[nxt]
                if same_chapter and cand.chapter_no != by_id[hid].chapter_no:
                    break
                keep.add(nxt)
                cur = cand
    order = {cid: i for i, cid in enumerate(by_id)}  # by_id giữ thứ tự đọc truyện
    return sorted(keep, key=lambda c: order[c])
