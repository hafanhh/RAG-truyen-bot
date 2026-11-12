"""V1 - Bước 1: Embedding là gì? Tự thấy "độ gần về nghĩa" bằng cosine similarity.

Ý tưởng: embedding model biến MỘT câu/đoạn thành MỘT vector (bge-m3: 1024 số). Hai văn bản gần nghĩa
thì vector của chúng "chỉ cùng hướng" -> cosine similarity cao (gần 1). RAG dùng đúng điều này: embed câu
hỏi, embed các chunk, rồi lấy các chunk có vector gần câu hỏi nhất.

Script làm 3 thí nghiệm:
  A) Ma trận similarity giữa các câu hỏi và các đoạn truyện: câu hỏi có "gần" đúng đoạn chứa đáp án không?
  B) Từ vựng "convert" so với tiếng Việt tự nhiên: cùng một ý nhưng viết khác từ ("nữ hài" / "cô gái")
     thì điểm thay đổi ra sao?
  C) (--compare) Bạn tự đưa hai câu bất kỳ và xem điểm.

Chạy từ thư mục gốc:
    python scripts/v1_step1_embedding_basics.py
    python scripts/v1_step1_embedding_basics.py --compare "Ai cứu nữ chính?" "Chàng lao vào biển lửa"

LƯU Ý: lần chạy đầu tiên sẽ tải model BAAI/bge-m3 (~2.2GB) từ Hugging Face, vào ~/.cache/huggingface.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.chunk_store import TOKENIZER_NAME as MODEL_NAME  # noqa: E402  (cùng tên "BAAI/bge-m3" với tokenizer đã dùng)
from src.loader import paragraph_text  # noqa: E402

# ---- Dữ liệu cho thí nghiệm A: câu hỏi và đoạn văn. Các đoạn truyện (Chương 1 đ5, đ26, Chương 4 đ8) được nạp từ Books/
# lúc chạy qua paragraph_text, không chép nguyên văn vào file vì repo public. ----
QUERIES = {
    "Q1 cầm gì": "Đường Quả đang cầm vật gì trong tay?",
    "Q2 bút": "Lãnh Tử Việt đưa cho Đường Quả loại bút gì?",
}
PASSAGES = {
    "P1 trà sữa": paragraph_text(1, 5),
    "P2 bút máy": paragraph_text(1, 26, through="không thiêm"),
    "P3 ca hát": paragraph_text(4, 8),
    "P4 ngoài đề": "Hôm nay trời nắng đẹp, tôi đi chợ mua rau và cá về nấu canh.",
}

# ---- Dữ liệu cho thí nghiệm B: cùng ý, khác cách viết ----
QUERY_NATURAL = "Cô gái đang uống gì?"
SAME_MEANING = {
    "văn bản gốc (convert)": paragraph_text(1, 19, through="uống một ngụm"),
    "viết lại tự nhiên": "Đường Quả nâng ly trà sữa lên, từ từ uống một ngụm.",
}


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed một list văn bản thành ma trận (n, 1024).

    normalize_embeddings=True: chuẩn hóa mỗi vector về độ dài 1. Khi đó
    cosine similarity = tích vô hướng (dot product), tính bằng một phép nhân ma trận.
    bge-m3 không cần "câu chỉ dẫn" đặt trước câu hỏi, nên đưa văn bản vào thẳng.
    """
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def print_matrix(rows: dict[str, str], cols: dict[str, str], model: SentenceTransformer) -> None:
    """In ma trận cosine similarity: mỗi hàng là một câu hỏi, mỗi cột là một đoạn văn. Đánh dấu * ô cao nhất."""
    sims = encode(model, list(rows.values())) @ encode(model, list(cols.values())).T  # (hàng, cột)
    names = list(cols)
    print(f"{'':<14}" + "".join(f"{n:>14}" for n in names))
    for i, rname in enumerate(rows):
        best = int(np.argmax(sims[i]))
        cells = "".join(f"{sims[i][j]:>13.3f}{'*' if j == best else ' '}" for j in range(len(names)))
        print(f"{rname:<14}{cells}")
    print("  (* = đoạn gần nhất với câu hỏi ở hàng đó)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("CÂU_1", "CÂU_2"), help="so sánh hai câu bất kỳ của bạn")
    args = ap.parse_args()

    # Dùng GPU Apple (MPS) nếu có, nếu không thì CPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Đang nạp model {MODEL_NAME} trên {device} (lần đầu sẽ tải ~2.2GB)...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Nạp xong sau {time.time() - t0:.1f}s | số chiều vector: {model.get_sentence_embedding_dimension()} "
          f"| độ dài tối đa: {model.max_seq_length} token\n")

    if args.compare:
        a, b = encode(model, list(args.compare))
        print(f"cosine similarity = {float(a @ b):.3f}")
        return

    # Một vector trông như thế nào?
    v = encode(model, [QUERIES["Q1 cầm gì"]])[0]
    print(f"Vector của câu hỏi Q1: {v.shape[0]} số, độ dài (norm) = {np.linalg.norm(v):.3f}, 5 số đầu = {np.round(v[:5], 3)}\n")

    print("=== Thí nghiệm A: câu hỏi nào gần đoạn nào? ===")
    print_matrix(QUERIES, PASSAGES, model)

    print("\n=== Thí nghiệm B: cùng ý nhưng khác từ vựng ===")
    print(f"Câu hỏi (tiếng Việt tự nhiên): {QUERY_NATURAL!r}")
    q = encode(model, [QUERY_NATURAL])[0]
    for label, text in SAME_MEANING.items():
        print(f"  {label:<24} similarity = {float(q @ encode(model, [text])[0]):.3f}   | {text}")


if __name__ == "__main__":
    main()
