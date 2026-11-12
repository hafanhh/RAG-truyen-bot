"""V1 - Bước 1 (bản Gemini): Embedding là gì? Tự thấy "độ gần về nghĩa" bằng cosine similarity.

Cùng các thí nghiệm với scripts/v1_step1_embedding_basics.py (bge-m3 local), nhưng embedding do Gemini API
tính -> KHÔNG tải model nào về máy. Đổi lại, văn bản được gửi lên Google. Ở Bước 1 chỉ có vài câu mẫu ngắn;
toàn bộ chunk của World 01 chỉ bị gửi đi ở Bước 2 (khi embed 393 chunk), và sẽ hỏi lại trước khi làm.

Ý tưởng: embedding model biến MỘT câu/đoạn thành MỘT vector (Gemini: 3072 số). Hai văn bản gần nghĩa thì
vector "chỉ cùng hướng" -> cosine similarity cao (gần 1). RAG dùng đúng điều này: embed câu hỏi, embed các
chunk, rồi lấy các chunk có vector gần câu hỏi nhất.

Script làm 3 thí nghiệm:
  A) Ma trận similarity giữa các câu hỏi và các đoạn truyện: câu hỏi có "gần" đúng đoạn chứa đáp án không?
  B) Từ vựng "convert" so với tiếng Việt tự nhiên: cùng một ý nhưng viết khác từ ("nữ hài" / "cô gái")
     thì điểm thay đổi ra sao?
  C) (--compare) Bạn tự đưa hai câu bất kỳ và xem điểm.

Chạy từ thư mục gốc:
    python scripts/v1_step1_gemini_embedding.py
    python scripts/v1_step1_gemini_embedding.py --model gemini-embedding-2
    python scripts/v1_step1_gemini_embedding.py --compare "Ai cứu nữ chính?" "Chàng lao vào biển lửa"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from src.config import require_api_key  # noqa: E402
from src.loader import paragraph_text  # noqa: E402

DEFAULT_MODEL = "gemini-embedding-001"

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
    # Đối chứng: một câu KHÔNG liên quan. Điểm của Gemini dồn vào khoảng 0.5-0.85 nên phải có mốc "xa" để
    # biết 0.74 là gần hay chưa: điểm các câu liên quan phải cao hơn rõ rệt so với dòng này.
    "đối chứng (ngoài đề)": "Hôm nay trời nắng đẹp, tôi đi chợ mua rau và cá về nấu canh.",
}


def make_client() -> genai.Client:
    # timeout ngắn (30s): SDK retry rất chậm khi API quá tải, nên không để treo.
    return genai.Client(api_key=require_api_key(), http_options=types.HttpOptions(timeout=30_000))


def embed(client: genai.Client, model: str, texts: list[str], task_type: str) -> np.ndarray:
    """Embed một list văn bản thành ma trận (n, 3072), mỗi hàng đã chuẩn hoá về độ dài 1.

    task_type cho model biết vector này dùng để làm gì:
      - "RETRIEVAL_QUERY"    : cho CÂU HỎI của người dùng
      - "RETRIEVAL_DOCUMENT" : cho ĐOẠN VĂN (chunk) được lưu để tìm
    Hai phía dùng hai task_type khác nhau để model đặt câu hỏi và đoạn trả lời gần nhau hơn (bài toán bất đối
    xứng: câu hỏi ngắn, đoạn trả lời dài). Khác bge-m3: bge-m3 không cần phân biệt.

    Hai khác biệt giữa các model (đã kiểm tra thực tế):
      - gemini-embedding-001: nhận cả list, trả đúng 1 vector cho MỖI văn bản; có dùng task_type.
      - gemini-embedding-2  : coi list `contents` là các phần của MỘT đầu vào (đa phương tiện) và gộp thành
        MỘT vector duy nhất; và BỎ QUA task_type (cosine có/không task_type = 1.0). Vì vậy phải gọi từng văn bản
        một, không truyền task_type.

    Chuẩn hoá: Gemini trả vector 3072 chiều đã có norm = 1 (đã kiểm tra), nhưng ta vẫn chuẩn hoá lại cho chắc,
    để cosine similarity luôn = tích vô hướng (dot product).
    """
    if model == "gemini-embedding-2":
        vecs = [client.models.embed_content(model=model, contents=[t]).embeddings[0].values for t in texts]
    else:
        resp = client.models.embed_content(
            model=model, contents=texts, config=types.EmbedContentConfig(task_type=task_type)
        )
        # Chốt chặn: số vector phải bằng số văn bản, nếu không sẽ lưu nhầm vector mà không biết.
        if len(resp.embeddings) != len(texts):
            raise RuntimeError(f"{model}: gửi {len(texts)} văn bản nhưng nhận {len(resp.embeddings)} vector")
        vecs = [e.values for e in resp.embeddings]
    mat = np.array(vecs, dtype=np.float64)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def print_matrix(rows: dict[str, str], cols: dict[str, str], client: genai.Client, model: str) -> None:
    """In ma trận cosine similarity: mỗi hàng là một câu hỏi, mỗi cột là một đoạn văn. Đánh dấu * ô cao nhất."""
    q = embed(client, model, list(rows.values()), "RETRIEVAL_QUERY")
    p = embed(client, model, list(cols.values()), "RETRIEVAL_DOCUMENT")
    sims = q @ p.T  # (hàng, cột)
    names = list(cols)
    print(f"{'':<14}" + "".join(f"{n:>14}" for n in names))
    for i, rname in enumerate(rows):
        best = int(np.argmax(sims[i]))
        cells = "".join(f"{sims[i][j]:>13.3f}{'*' if j == best else ' '}" for j in range(len(names)))
        print(f"{rname:<14}{cells}")
    print("  (* = đoạn gần nhất với câu hỏi ở hàng đó)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model embedding (mặc định {DEFAULT_MODEL})")
    ap.add_argument("--compare", nargs=2, metavar=("CÂU_1", "CÂU_2"), help="so sánh hai câu bất kỳ của bạn")
    args = ap.parse_args()

    client = make_client()
    print(f"Model embedding: {args.model} (gọi qua Gemini API, không tải gì về máy)\n")

    if args.compare:
        # Hai câu so sánh với nhau như hai văn bản cùng loại -> dùng SEMANTIC_SIMILARITY.
        a, b = embed(client, args.model, list(args.compare), "SEMANTIC_SIMILARITY")
        print(f"cosine similarity = {float(a @ b):.3f}")
        return

    # Một vector trông như thế nào?
    t0 = time.time()
    v = embed(client, args.model, [QUERIES["Q1 cầm gì"]], "RETRIEVAL_QUERY")[0]
    print(f"Vector của câu hỏi Q1: {v.shape[0]} số, độ dài (norm) = {np.linalg.norm(v):.3f}, "
          f"5 số đầu = {np.round(v[:5], 3)}  ({time.time() - t0:.1f}s cho một lần gọi API)\n")

    print("=== Thí nghiệm A: câu hỏi nào gần đoạn nào? ===")
    print_matrix(QUERIES, PASSAGES, client, args.model)

    print("\n=== Thí nghiệm B: cùng ý nhưng khác từ vựng ===")
    print(f"Câu hỏi (tiếng Việt tự nhiên): {QUERY_NATURAL!r}")
    q = embed(client, args.model, [QUERY_NATURAL], "RETRIEVAL_QUERY")[0]
    docs = embed(client, args.model, list(SAME_MEANING.values()), "RETRIEVAL_DOCUMENT")
    for (label, text), d in zip(SAME_MEANING.items(), docs):
        print(f"  {label:<24} similarity = {float(q @ d):.3f}   | {text}")


if __name__ == "__main__":
    main()
