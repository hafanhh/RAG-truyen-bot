"""Chroma: VectorDB local lưu trên đĩa (data/chroma/).

Mỗi "biến thể" embedding là một collection riêng để A/B so sánh (V1 Step 2):
    w01_plain   : vector của `text` (không header)
    w01_header  : vector của `context_header + "\\n\\n" + text` (xem chunk_store.embed_input)
Document lưu trong Chroma luôn là `text` gốc (KHÔNG gồm header) ở cả hai collection: header chỉ ảnh hưởng
vector, còn thứ đưa cho LLM/citation sau này là text + metadata.

Các điểm cần nhớ về Chroma:
  * embedding_function=None: ta TỰ đưa vector vào (đã gọi Gemini). Nếu không đặt, Chroma sẽ tự tải một model
    embedding mặc định khi bạn dùng query_texts, rất dễ nhầm.
  * space="cosine": distance = 1 - cosine similarity (0 = giống hệt; càng nhỏ càng gần).
  * Metadata chỉ nhận str/int/float/bool, KHÔNG nhận None. Chunk đầu/cuối không có prev/next nên lưu chuỗi rỗng "".
  * Chroma từ chối vector sai số chiều so với các vector đã có trong collection.
"""
from __future__ import annotations

import chromadb
import numpy as np

from src.chunk_store import ChunkRecord
from src.config import CHROMA_DIR


def get_client(path=CHROMA_DIR) -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=str(path))


def collection_name(world_no: int, variant: str) -> str:
    return f"w{world_no:02d}_{variant}"


def get_collection(client, world_no: int, variant: str):
    """Lấy (hoặc tạo) collection dùng khoảng cách cosine, không có embedding function."""
    return client.get_or_create_collection(
        collection_name(world_no, variant),
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def record_metadata(r: ChunkRecord) -> dict:
    """Metadata lưu kèm vector: dùng cho lọc (where), citation và mở rộng chunk lân cận."""
    return {
        "world_no": r.world_no,
        "chapter_no": r.chapter_no,
        "chunk_index": r.chunk_index,
        "para_start": r.para_start,
        "para_end": r.para_end,
        "prev_id": r.prev_id or "",
        "next_id": r.next_id or "",
        "context_header": r.context_header,
        "token_count": r.token_count,
    }


def upsert_records(collection, records: list[ChunkRecord], vectors: np.ndarray, batch: int = 100) -> None:
    """Ghi chunk + vector vào collection. upsert = chạy lại nhiều lần không bị trùng (cùng id thì ghi đè)."""
    assert len(records) == len(vectors), "số chunk và số vector phải bằng nhau"
    for s in range(0, len(records), batch):
        rs = records[s : s + batch]
        collection.upsert(
            ids=[r.chunk_id for r in rs],
            embeddings=vectors[s : s + batch].tolist(),
            documents=[r.text for r in rs],
            metadatas=[record_metadata(r) for r in rs],
        )
