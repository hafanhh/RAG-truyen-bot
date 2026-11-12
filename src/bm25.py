"""BM25: tìm kiếm theo TỪ KHÓA (lexical), viết tay để thấy rõ cách nó hoạt động.

Ý tưởng: chunk nào chứa nhiều từ của câu hỏi, đặc biệt là các từ HIẾM trong cả truyện, thì điểm cao.
    điểm(chunk) = tổng, trên từng từ t của câu hỏi, của:  IDF(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl/avgdl))
  - f   = số lần t xuất hiện trong chunk
  - IDF = log(1 + (N - df + 0.5) / (df + 0.5)): từ xuất hiện ở ít chunk (df nhỏ) thì quan trọng hơn (vd "danh ngạch"
          quan trọng hơn "hắn")
  - k1  = độ bão hòa tần suất (từ lặp 10 lần không đáng gấp 10 lần), b = mức phạt chunk dài (dl = độ dài chunk)
Tham số mặc định k1=1.5, b=0.75 là giá trị thông dụng; KHÔNG chỉnh theo bộ câu hỏi (chỉ 10 câu, chỉnh sẽ overfit).

Tách từ cho tiếng Việt: mỗi tiếng (âm tiết) là một token, cách nhau bằng khoảng trắng nên cắt theo ký tự chữ là đủ.
Thêm BIGRAM (cặp tiếng liền nhau, vd "lãnh_tử", "tử_việt") để các tên nhiều tiếng và cụm "danh_ngạch", "gắt_gao"
khớp như một đơn vị. Điểm mạnh của BM25: tên riêng và từ hiếm (embedding hay yếu ở đây). Điểm yếu: không hiểu
đồng nghĩa ("cô gái" khác "nữ hài" là hai từ hoàn toàn khác nhau).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    """Chữ thường, chuẩn hóa NFC (để dấu tiếng Việt nhất quán), tách thành các tiếng, thêm bigram."""
    toks = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    if bigrams:
        toks = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75, bigrams: bool = True):
        self.k1, self.b, self.bigrams = k1, b, bigrams
        self.tf = [Counter(tokenize(d, bigrams)) for d in docs]  # tần suất từ trong từng chunk
        self.dl = [sum(c.values()) for c in self.tf]  # độ dài từng chunk (số token)
        self.avgdl = sum(self.dl) / len(self.dl)
        df = Counter(t for c in self.tf for t in c)  # df[t] = số chunk chứa t
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = set(tokenize(query, self.bigrams))  # mỗi từ của câu hỏi chỉ tính một lần
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def ranking(self, query: str) -> list[int]:
        """Chỉ số các chunk sắp theo điểm giảm dần (điểm bằng nhau thì theo thứ tự đọc truyện)."""
        sc = self.scores(query)
        return sorted(range(len(sc)), key=lambda i: (-sc[i], i))
