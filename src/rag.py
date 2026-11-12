"""Pipeline RAG hoàn chỉnh: câu hỏi -> tìm (retrieve) -> dựng ngữ cảnh -> LLM trả lời có dẫn nguồn -> kiểm chứng nguồn.

    câu hỏi ──► [1] retrieve ──► [2] build_context ──► [3] prompt + LLM ──► [4] verify ──► trace
                 top-k chunk       chunk THỰC SỰ gửi     JSON có nguồn dẫn     nguồn có thật?

Mỗi lần gọi trả về một TRACE (dict) ghi lại mọi bước để nhìn thấy pipeline (thay cho giao diện trực quan của
Langflow). Điều QUAN TRỌNG nhất của trace là phân biệt 3 tập chunk:
    retrieved : top-k do bước tìm trả về                          (ĐÃ TÌM ĐƯỢC)
    context   : retrieved + chunk lân cận (nếu bật), mỗi chunk có cờ sent / drop_reason
    sent      : phần của context thực sự nằm trong prompt          (ĐÃ GỬI CHO LLM)
Giữa "tìm được" và "gửi" có thể mất chunk vì ngân sách token; khi đó bằng chứng có thể bị loại âm thầm, nên mỗi chunk
bị loại đều được ghi kèm lý do.

Quy tắc ngân sách token: duyệt chunk theo ĐỘ ƯU TIÊN (retrieved theo hạng, rồi lân cận theo hạng của chunk cha),
cộng dồn token_count (đo bằng tokenizer bge-m3, chỉ là ước lượng so với token của Gemini); chunk đầu tiên làm vượt
ngân sách và mọi chunk sau nó bị loại ("vượt ngân sách token"). Số token thật của prompt lấy từ phản hồi của API.

Phiên bản prompt (ghi trong trace, trường config.prompt_version):
  v1: chunk gửi nguyên văn, nguồn dẫn ở mức CHUNK ([mã_chunk, chương X đoạn A-B] chép từ tiêu đề). Quan sát: mô hình
      hay tự thu hẹp "đoạn" thành số không có thật (vd chunk đoạn 1-7 mà ghi "đoạn 8-10").
  v2: mỗi đoạn trong chunk được đánh số [đN] (số đoạn thật trong chương) để mô hình dẫn đúng đoạn; mỗi citation ghi
      số đoạn và đoạn trích nguyên văn lấy từ đúng đoạn đó, nên kiểm chứng được tới mức ĐOẠN.
      Quan sát: Q01 (bằng chứng không được tìm ra) bị trả lời tự tin bằng một cảnh khác (chương 59) dù v1 từ chối đúng.
  v3: thêm quy tắc 6: đoạn dẫn phải khớp TẤT CẢ điều kiện của câu hỏi (nhân vật, sự việc, vị trí trong truyện).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from src.bm25 import BM25
from src.chunk_store import ChunkRecord, chunks_path, read_jsonl
from src.config import GEMINI_MODEL
from src.embedder import MAX_RETRIES, embed_query_cached, is_retryable, make_client
from src.retriever import chapter_where, expand_neighbors, parse_chapter_set, rank_all_methods
from src.vector_store import get_client, get_collection

PROMPT_VERSION = "v3"
DEFAULT_METHOD = "hybrid-header"  # cách tìm đã chọn từ kết quả Step 3b/4 (xem CLAUDE.md)
DEFAULT_K = 5
DEFAULT_MAX_CONTEXT_TOKENS = 6000  # ngân sách token cho phần chunk trong prompt (ước lượng bge-m3)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Bạn là trợ lý đọc truyện. Bạn nhận MỘT câu hỏi và một số đoạn truyện (chunk) đã được tìm ra. \
Hãy trả lời câu hỏi chỉ dựa trên các đoạn đó.

QUY TẮC (bắt buộc):
1. Chỉ dùng các đoạn được cung cấp làm bằng chứng. Không dùng kiến thức bên ngoài, không bịa thêm.
2. Mỗi nhận định về truyện phải kèm nguồn ngay sau nhận định, theo dạng [mã_chunk, chương X đoạn N], trong đó N là \
số đoạn ghi ở đầu đoạn chứa bằng chứng (ký hiệu [đN]); nếu bằng chứng nằm ở nhiều đoạn liên tiếp thì viết đoạn N-M. \
Chép chính xác mã chunk, số chương và số đoạn, không đoán. Ví dụ: [w01-c001-k00, chương 1 đoạn 5].
3. Không tự đoán nguồn hoặc động cơ nhân vật. Chỉ nêu nguyên nhân hay động cơ khi đoạn văn nói rõ ra; nếu không, \
đừng suy diễn.
4. Nếu bằng chứng chỉ trả lời được MỘT PHẦN câu hỏi, hãy trả lời phần đó và nêu rõ phần còn thiếu.
5. Nếu không thể xác định đáp án từ các đoạn được cung cấp, hãy trả lời đúng cụm "không đủ thông tin".
6. Trước khi trả lời, kiểm tra đoạn định dẫn có khớp TẤT CẢ điều kiện của câu hỏi hay không (nhân vật, sự việc, thời \
điểm hoặc vị trí trong truyện, vd "ở đầu truyện", "ở chương N"). Nếu chỉ có đoạn tương tự nhưng thuộc bối cảnh khác \
hoặc không rõ có thuộc bối cảnh được hỏi, đừng gán nó cho bối cảnh đó: hãy trả lời "một phần" hoặc "không đủ thông tin".

Lưu ý về văn bản: đây là truyện dịch máy kiểu convert nên dùng nhiều từ Hán-Việt ("nữ hài" là cô gái, "hắn/nàng", \
"ngươi/ta"); hãy hiểu nghĩa nhưng không thêm chi tiết không có trong văn bản. Các chunk liền kề có thể lặp lại \
vài đoạn do chồng lấn. Trả lời bằng tiếng Việt, ngắn gọn.

ĐỊNH DẠNG ĐẦU RA (JSON):
- verdict: "đủ" nếu trả lời trọn vẹn; "một phần" nếu chỉ trả lời được một phần; "không đủ thông tin" nếu không xác định được.
- answer: câu trả lời, có nguồn dẫn theo quy tắc 2 (nếu verdict là "không đủ thông tin" thì viết "không đủ thông tin").
- missing: phần còn thiếu nếu verdict là "một phần"; ngược lại để chuỗi rỗng.
- citations: danh sách {chunk_id, paragraph, quote}. paragraph là SỐ đoạn (số trong [đN]) chứa đoạn trích; quote là \
đoạn trích NGUYÊN VĂN ngắn (dưới 200 ký tự) lấy từ ĐÚNG MỘT đoạn đó. Cần nhiều đoạn thì tạo nhiều citation.
"""


class Citation(BaseModel):
    chunk_id: str
    paragraph: int
    quote: str


class LLMAnswer(BaseModel):
    verdict: Literal["đủ", "một phần", "không đủ thông tin"]
    answer: str
    missing: str
    citations: list[Citation]


def chunk_label(r: ChunkRecord) -> str:
    """Tiêu đề của chunk trong prompt: 'w01-c001-k00 | chương 1, đoạn 1-7'."""
    return f"{r.chunk_id} | chương {r.chapter_no}, đoạn {r.para_start}-{r.para_end}"


def format_citation(c: Citation | dict, by_id: dict[str, ChunkRecord]) -> str:
    """V2 (Module 06, source attribution): đổi citation nội bộ thành dòng nguồn người đọc hiểu được, theo định
    dạng đã chốt từ V0 — dùng SỐ chương vì tiêu đề chương trùng tên truyện. Nhận cả Citation lẫn dict (trace đã lưu).

    vd: Thế giới 1 – Phú thiếu bạn gái cũ, chương 75, đoạn 29: «Lãnh Duệ đưa cho nàng...»
    LLM có thể bịa chunk_id: khi đó nói thẳng là không tra được thay vì ném lỗi."""
    d = c if isinstance(c, dict) else c.model_dump()
    r = by_id.get(d["chunk_id"])
    if r is None:
        return f"[nguồn không tra được: {d['chunk_id']}]"
    quote = " ".join(str(d.get("quote") or "").split())
    if len(quote) > 120:
        quote = quote[:120].rsplit(" ", 1)[0] + "…"
    return (f"Thế giới {r.world_no} – {r.world_title}, chương {r.chapter_no}, đoạn {d['paragraph']}"
            + (f": «{quote}»" if quote else ""))


def format_citations(citations: list, by_id: dict[str, ChunkRecord]) -> list[str]:
    """Các dòng nguồn cho một câu trả lời, khử trùng lặp nhưng giữ thứ tự xuất hiện."""
    seen: set[str] = set()
    out: list[str] = []
    for c in citations:
        line = format_citation(c, by_id)
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def paragraphs(r: ChunkRecord) -> list[str]:
    """Các đoạn của chunk. Đã kiểm tra: ở cả 393 chunk, số đoạn tách bằng dòng trống == para_end - para_start + 1."""
    return r.text.split("\n\n")


def build_prompt(question: str, sent: list[ChunkRecord]) -> str:
    """Nội dung người dùng: câu hỏi + các chunk đã gửi theo THỨ TỰ ĐỌC TRUYỆN, mỗi đoạn đánh số [đN] (số thật trong chương)."""
    blocks = []
    for r in sent:
        body = "\n".join(f"[đ{r.para_start + i}] {p}" for i, p in enumerate(paragraphs(r)))
        blocks.append(f"=== [{chunk_label(r)}] ===\n{body}")
    return f"CÂU HỎI: {question}\n\nCÁC ĐOẠN TRUYỆN ({len(sent)} chunk):\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Kiểm chứng nguồn dẫn
# ---------------------------------------------------------------------------

_CITE = re.compile(r"\[(w\d+-c\d+-k\d+),\s*chương\s*(\d+)\s*đoạn\s*(\d+)(?:\s*-\s*(\d+))?\s*\]")
_ID = re.compile(r"w\d+-c\d+-k\d+")
_WORD = re.compile(r"\w+", re.UNICODE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def quote_match(quote: str, text: str) -> str:
    """So đoạn trích mô hình đưa ra với văn bản gốc. Trả về 'exact' | 'close' | 'none'.

    exact: đoạn trích là chuỗi con nguyên văn (cho phép dấu lược '…'/'...': mỗi mảnh dài >= 6 ký tự đều phải khớp).
    close: không nguyên văn nhưng >= 80% cụm 4 từ liên tiếp của đoạn trích có trong văn bản gốc (khác dấu câu, bỏ ngoặc
           kép, ghép câu...): nội dung vẫn là từ của truyện, không phải bịa.
    none : còn lại (nghi ngờ bịa hoặc diễn đạt lại).
    """
    t = _norm(text)
    frags = [_norm(f.strip(" \"'“”")) for f in re.split(r"…+|\.{3,}", quote)]
    frags = [f for f in frags if len(f) >= 6]
    if frags and all(f in t for f in frags):
        return "exact"
    qw, tw = _WORD.findall(_norm(quote).lower()), _WORD.findall(t.lower())
    if len(qw) >= 4:
        tg = {tuple(tw[i:i + 4]) for i in range(len(tw) - 3)}
        qg = [tuple(qw[i:i + 4]) for i in range(len(qw) - 3)]
        if sum(g in tg for g in qg) / len(qg) >= 0.8:
            return "close"
    return "none"


def verify(ans: LLMAnswer, sent_ids: set[str], by_id: dict[str, ChunkRecord]) -> dict:
    """Kiểm tra nguồn mô hình dẫn có thật không. Không đánh giá câu trả lời đúng/sai (việc đó cần đối chiếu đáp án)."""
    cited = {c.chunk_id for c in ans.citations} | set(_ID.findall(ans.answer))

    # (a) nhãn trong câu trả lời [mã, chương X đoạn N(-M)]: chunk có thật, chương đúng, số đoạn nằm trong khoảng của chunk
    inline, bad_label = _CITE.findall(ans.answer), []
    for cid, ch, a, b in inline:
        r = by_id.get(cid)
        lo, hi = int(a), int(b or a)
        if not r or r.chapter_no != int(ch) or not (r.para_start <= lo <= hi <= r.para_end):
            bad_label.append(f"{cid} chương {ch} đoạn {a}" + (f"-{b}" if b else "")
                             + (f" (chunk thật: chương {r.chapter_no} đoạn {r.para_start}-{r.para_end})" if r else " (chunk không tồn tại)"))

    # (b) từng citation có cấu trúc: chunk đã gửi? số đoạn trong chunk? đoạn trích có trong ĐÚNG đoạn đó không?
    checks = []
    for c in ans.citations:
        r = by_id.get(c.chunk_id)
        if r is None or c.chunk_id not in sent_ids:
            checks.append({"chunk_id": c.chunk_id, "paragraph": c.paragraph, "in_range": False, "match": "none"})
            continue
        in_range = r.para_start <= c.paragraph <= r.para_end
        match = quote_match(c.quote, paragraphs(r)[c.paragraph - r.para_start]) if in_range else "none"
        if match == "none" and quote_match(c.quote, r.text) != "none":
            match = "other_para"  # đoạn trích có trong chunk nhưng KHÔNG ở đoạn được ghi (ghép nhiều đoạn / ghi sai số đoạn)
        checks.append({"chunk_id": c.chunk_id, "paragraph": c.paragraph, "in_range": in_range, "match": match})
    n = len(checks)
    return {
        "cited_ids": sorted(cited),
        "cited_not_sent": sorted(cited - sent_ids),  # dẫn một chunk KHÔNG nằm trong tập đã gửi = bịa nguồn
        "inline_citations": len(inline),
        "inline_bad_label": bad_label,  # chương/đoạn ghi sai so với metadata của chunk
        "checks": checks,
        "quotes_exact": sum(c["match"] == "exact" for c in checks),
        "quotes_close": sum(c["match"] == "close" for c in checks),
        "quotes_other_para": sum(c["match"] == "other_para" for c in checks),
        "quotes_none": sum(c["match"] == "none" for c in checks),
        "paragraphs_in_range": sum(c["in_range"] for c in checks),
        "n_citations": n,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RagPipeline:
    """Nạp một lần (chunk, Chroma, BM25, client Gemini) rồi trả lời nhiều câu hỏi."""

    def __init__(self, world_no: int = 1):
        self.world_no = world_no
        self.records = read_jsonl(chunks_path(world_no))
        self.by_id = {r.chunk_id: r for r in self.records}  # giữ thứ tự đọc truyện
        self.ids = [r.chunk_id for r in self.records]
        chroma = get_client()
        self.cols = {v: get_collection(chroma, world_no, v) for v in ("plain", "header")}
        self.bm25 = BM25([r.text for r in self.records])
        self.client = make_client()

    # -- [1] tìm --------------------------------------------------------------
    def retrieve(self, question: str, chapters: list[int] | None) -> dict[str, list[str]]:
        qvec = embed_query_cached(self.client, question, self.world_no)  # có cache: không tốn hạn mức embedding/ngày
        allowed = {r.chunk_id for r in self.records if r.chapter_no in chapters} if chapters else None
        return rank_all_methods(question, qvec, self.cols, self.bm25, self.ids,
                                where=chapter_where(chapters), allowed=allowed)

    # -- [2] dựng ngữ cảnh ----------------------------------------------------
    def build_context(self, retrieved: list[str], expand: int, max_tokens: int) -> list[dict]:
        """Danh sách chunk ứng viên theo độ ưu tiên, mỗi chunk có cờ sent và drop_reason."""
        items = [{"chunk_id": cid, "role": f"top-{i}", "priority_rank": i} for i, cid in enumerate(retrieved, 1)]
        seen = set(retrieved)
        if expand:
            for i, cid in enumerate(retrieved, 1):
                for nb in expand_neighbors([cid], self.by_id, n=expand):
                    if nb not in seen:
                        seen.add(nb)
                        items.append({"chunk_id": nb, "role": f"lân cận của top-{i}", "priority_rank": i})
        used, cut = 0, False
        for it in items:
            r = self.by_id[it["chunk_id"]]
            it.update(chapter=r.chapter_no, para=f"{r.para_start}-{r.para_end}", est_tokens=r.token_count)
            if cut or used + r.token_count > max_tokens:
                cut = True  # từ chunk đầu tiên vượt ngân sách trở đi, mọi chunk đều bị loại (giữ đúng thứ tự ưu tiên)
                it.update(sent=False, drop_reason=f"vượt ngân sách token ({max_tokens})")
            else:
                used += r.token_count
                it.update(sent=True, drop_reason=None)
        return items

    # -- [3] gọi LLM ----------------------------------------------------------
    def _generate(self, model: str, prompt: str) -> tuple[LLMAnswer | None, dict, str | None]:
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.0,
            response_mime_type="application/json", response_schema=LLMAnswer,
        )
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == MAX_RETRIES or not is_retryable(exc):
                    return None, {}, f"{type(exc).__name__}: {str(exc)[:300]}"
                time.sleep(min(10.0 * 2**attempt, 90.0))
        u = resp.usage_metadata
        usage = {
            "api_prompt_tokens": getattr(u, "prompt_token_count", None),
            "api_output_tokens": getattr(u, "candidates_token_count", None),
            "api_thoughts_tokens": getattr(u, "thoughts_token_count", None),
        }
        try:
            return LLMAnswer.model_validate(json.loads(resp.text)), usage, None
        except Exception as exc:  # noqa: BLE001 - JSON hỏng hoặc sai schema
            return None, usage, f"đầu ra không hợp lệ ({type(exc).__name__}): {(resp.text or '')[:200]!r}"

    # -- toàn bộ --------------------------------------------------------------
    def answer(
        self, question: str, method: str = DEFAULT_METHOD, k: int = DEFAULT_K, expand: int = 0,
        chapters: list[int] | None = None, auto_chapters: bool = True,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS, model: str = GEMINI_MODEL, call_llm: bool = True,
    ) -> dict:
        t0 = time.time()
        if chapters is None and auto_chapters:
            chapters = parse_chapter_set(question)  # câu hỏi nói rõ "chương N" -> lọc theo chương
        rankings = self.retrieve(question, chapters)
        ranked = rankings[method]
        retrieved = ranked[:k]

        dense_key = method.replace("hybrid", "dense") if method.startswith("hybrid") else method
        dpos = {c: i for i, c in enumerate(rankings.get(dense_key, []), 1)}
        bpos = {c: i for i, c in enumerate(rankings["bm25"], 1)}
        retrieved_info = []
        for i, cid in enumerate(retrieved, 1):
            r = self.by_id[cid]
            retrieved_info.append({
                "rank": i, "chunk_id": cid, "chapter": r.chapter_no, "para": f"{r.para_start}-{r.para_end}",
                "dense_rank": dpos.get(cid) if dense_key != "bm25" else None,
                "bm25_rank": bpos.get(cid) if method != dense_key or method == "bm25" else None,
            })

        context = self.build_context(retrieved, expand, max_context_tokens)
        order = {cid: i for i, cid in enumerate(self.by_id)}
        sent_ids = sorted((c["chunk_id"] for c in context if c["sent"]), key=order.get)
        sent = [self.by_id[c] for c in sent_ids]
        prompt = build_prompt(question, sent)

        trace = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "config": {"model": model, "method": method, "k": k, "expand": expand,
                       "chapter_filter": chapters, "max_context_tokens": max_context_tokens, "temperature": 0.0,
                       "prompt_version": PROMPT_VERSION},
            "retrieved": retrieved_info,
            "context": context,
            "sent_ids": sent_ids,
            "dropped_ids": [c["chunk_id"] for c in context if not c["sent"]],
            "tokens": {"est_context_bge": sum(c["est_tokens"] for c in context if c["sent"]),
                       "budget": max_context_tokens, "prompt_chars": len(prompt)},
            "prompt": prompt,
            "llm": None, "error": None, "verification": None,
        }
        if call_llm:
            ans, usage, err = self._generate(model, prompt)
            trace["tokens"].update(usage)
            trace["error"] = err
            if ans:
                trace["llm"] = ans.model_dump()
                trace["verification"] = verify(ans, set(sent_ids), self.by_id)
        trace["latency_s"] = round(time.time() - t0, 1)
        return trace

    def answer_from_chunks(self, question: str, chunk_ids: list[str], feedback: str | None = None,
                           model: str = GEMINI_MODEL) -> dict:
        """Trả lời lại từ MỘT TẬP CHUNK CHO TRƯỚC theo mã đã lưu: không tìm kiếm, KHÔNG embedding mới (chỉ tốn hạn mức LLM).

        feedback (tùy chọn): ghi chú của kiểm tra viên nối vào sau các chunk. Trả về trace rút gọn; trace['prompt'] là phần
        câu hỏi + chunk (KHÔNG gồm ghi chú) để lượt kiểm tra sau dùng lại, trace['feedback'] ghi lại ghi chú đã dùng.
        """
        t0 = time.time()
        order = {cid: i for i, cid in enumerate(self.by_id)}
        ids = sorted(set(chunk_ids), key=order.get)
        prompt = build_prompt(question, [self.by_id[c] for c in ids])
        full = prompt + ("\n\n" + feedback if feedback else "")
        ans, usage, err = self._generate(model, full)
        trace = {"question": question, "sent_ids": ids, "prompt": prompt, "feedback": feedback,
                 "tokens": {"prompt_chars": len(full), **usage}, "llm": None, "error": err, "verification": None}
        if ans:
            trace["llm"] = ans.model_dump()
            trace["verification"] = verify(ans, set(ids), self.by_id)
        trace["latency_s"] = round(time.time() - t0, 1)
        return trace


# ---------------------------------------------------------------------------
# Hiển thị trace
# ---------------------------------------------------------------------------


def format_trace(t: dict, show_prompt: bool = False) -> str:
    c = t["config"]
    L = ["=" * 78, f"CÂU HỎI: {t['question']}", "=" * 78]
    L.append(f"[1] CẤU HÌNH: model={c['model']} | cách tìm={c['method']} | k={c['k']} | mở rộng lân cận ±{c['expand']} "
             f"| ngân sách chunk={c['max_context_tokens']} token | temperature={c['temperature']} | prompt {c['prompt_version']}")
    L.append(f"[2] BỘ LỌC CHƯƠNG: {c['chapter_filter'] if c['chapter_filter'] else 'không'}")
    L.append("[3] ĐÃ TÌM ĐƯỢC (top-k):")
    for r in t["retrieved"]:
        extra = []
        if r["dense_rank"]:
            extra.append(f"dense #{r['dense_rank']}")
        if r["bm25_rank"]:
            extra.append(f"bm25 #{r['bm25_rank']}")
        L.append(f"      #{r['rank']} {r['chunk_id']}  chương {r['chapter']} đoạn {r['para']}"
                 + (f"   ({', '.join(extra)})" if extra else ""))
    L.append("[4] ĐÃ GỬI CHO LLM / BỊ LOẠI:")
    for it in t["context"]:
        mark = "GỬI " if it["sent"] else "LOẠI "
        why = "" if it["sent"] else f"  <- {it['drop_reason']}"
        L.append(f"      {mark} {it['chunk_id']}  chương {it['chapter']} đoạn {it['para']}  ~{it['est_tokens']} tok  [{it['role']}]{why}")
    n_ret, n_sent = len(t["retrieved"]), len(t["sent_ids"])
    ret_lost = [i for i in t["dropped_ids"] if i in {r["chunk_id"] for r in t["retrieved"]}]
    L.append(f"      => tìm được {n_ret} chunk, gửi {n_sent} chunk (thứ tự đọc truyện), loại {len(t['dropped_ids'])}"
             + (f"  !!! TRONG ĐÓ CÓ {len(ret_lost)} CHUNK TOP-K BỊ LOẠI: {ret_lost}" if ret_lost else ""))
    tk = t["tokens"]
    L.append(f"[5] TOKEN: ước lượng phần chunk (bge-m3) {tk['est_context_bge']}/{tk['budget']} | "
             f"prompt thực tế theo API: {tk.get('api_prompt_tokens')} | đầu ra: {tk.get('api_output_tokens')} "
             f"| suy nghĩ: {tk.get('api_thoughts_tokens')} | {tk['prompt_chars']} ký tự")
    if show_prompt:
        L.append("--- PROMPT GỬI ĐI (nội dung người dùng) ---\n" + t["prompt"] + "\n--- hết prompt ---")
    if t["error"]:
        L.append(f"[6] LỖI LLM: {t['error']}")
    elif t["llm"]:
        a, v = t["llm"], t["verification"]
        L.append(f"[6] TRẢ LỜI (verdict: {a['verdict']}):\n      {a['answer']}")
        if a["missing"]:
            L.append(f"      phần còn thiếu: {a['missing']}")
        L.append("[7] NGUỒN ĐƯỢC DẪN:")
        names = {"exact": "NGUYÊN VĂN", "close": "gần nguyên văn", "other_para": "đúng chunk nhưng SAI ĐOẠN",
                 "none": "KHÔNG KHỚP"}
        for q, ck in zip(a["citations"], v["checks"]):
            rng = "" if ck["in_range"] else "  (số đoạn NGOÀI chunk!)"
            L.append(f"      {q['chunk_id']} đoạn {q['paragraph']}: {names[ck['match']]}{rng} | “{q['quote'][:100]}”")
        warn = []
        if v["cited_not_sent"]:
            warn.append(f"dẫn chunk KHÔNG được gửi: {v['cited_not_sent']}")
        if v["inline_bad_label"]:
            warn.append(f"nhãn chương/đoạn sai: {v['inline_bad_label']}")
        L.append(f"      kiểm chứng: {v['quotes_exact']} nguyên văn + {v['quotes_close']} gần nguyên văn + {v['quotes_other_para']} "
                 f"sai đoạn + {v['quotes_none']} không khớp / {v['n_citations']} trích dẫn | {v['inline_citations']} nhãn trong câu trả lời"
                 + (f" | CẢNH BÁO: {'; '.join(warn)}" if warn else " | không có cảnh báo"))
    L.append(f"(thời gian: {t['latency_s']}s)")
    return "\n".join(L)
