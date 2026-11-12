"""Bộ câu hỏi đánh giá retrieval: đọc đáp án nguồn và chấm điểm một danh sách kết quả.

File câu hỏi: data/eval/questions_w01.json (xem trường "evidence_rule" trong file để hiểu cách ghi đáp án).

Khái niệm:
  * Đáp án nguồn được ghi theo (chương, số đoạn). Một "NHÓM bằng chứng" là tập các đoạn thay thế nhau: CHỈ CẦN
    MỘT đoạn trong nhóm xuất hiện là nhóm đó được phủ. Muốn "đủ bằng chứng" thì MỌI nhóm phải được phủ.
  * Một chunk "trúng" một đoạn nếu khoảng para_start..para_end của nó (cùng chương) chứa đoạn đó. Vì có overlap
    nên một đoạn có thể nằm trong 2 chunk.
  * Với một danh sách chunk đã xếp hạng, mỗi nhóm có một HẠNG = vị trí tốt nhất của một chunk phủ nhóm đó.
        hạng chunk đúng đầu tiên = min(hạng các nhóm)         -> câu hỏi này "tìm ra được gì đó đúng" ở hạng nào
        k tối thiểu để đủ bằng chứng = max(hạng các nhóm)     -> cần lấy top-k bằng bao nhiêu mới đủ mọi nhóm
"""
from __future__ import annotations

import json
from pathlib import Path

from src.chunk_store import ChunkRecord
from src.config import EVAL_DIR

DEFAULT_QUESTIONS = EVAL_DIR / "questions_w01.json"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def covering_chunks(records: list[ChunkRecord], chapter: int, para: int) -> list[str]:
    """Các chunk chứa đoạn `para` của `chapter` (có thể 2 chunk do overlap)."""
    return [r.chunk_id for r in records if r.chapter_no == chapter and r.para_start <= para <= r.para_end]


def gold_groups(q: dict, records: list[ChunkRecord], world) -> list[set[str]]:
    """Với mỗi nhóm bằng chứng của câu hỏi, trả về tập chunk_id chứa nó. Báo lỗi nếu đoạn không tồn tại."""
    groups = []
    for group in q["evidence"]:
        ids: set[str] = set()
        for chapter, para in group:
            ch = world.chapters[chapter - 1]
            if not (1 <= para <= len(ch.paragraphs)):
                raise ValueError(f"{q['id']}: chương {chapter} không có đoạn {para} (chỉ có {len(ch.paragraphs)})")
            cover = covering_chunks(records, chapter, para)
            if not cover:
                raise ValueError(f"{q['id']}: không chunk nào phủ chương {chapter} đoạn {para}")
            ids.update(cover)
        groups.append(ids)
    return groups


def group_ranks(ranked_ids: list[str], groups: list[set[str]]) -> list[int | None]:
    """Hạng (1-based) của từng nhóm bằng chứng trong danh sách đã xếp hạng; None nếu nhóm không xuất hiện."""
    pos = {cid: i for i, cid in enumerate(ranked_ids, 1)}
    return [min((pos[c] for c in g if c in pos), default=None) for g in groups]


def summarize(ranks: list[int | None]) -> dict:
    """Từ hạng các nhóm -> {first: hạng chunk đúng đầu tiên, need: k tối thiểu để đủ bằng chứng} (None nếu thiếu)."""
    found = [r for r in ranks if r is not None]
    return {
        "first": min(found) if found else None,
        "need": max(ranks) if ranks and all(r is not None for r in ranks) else None,
    }
