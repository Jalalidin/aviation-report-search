"""Search API + the single-page interface in web/index.html.

Start with `avsearch serve`, then open http://127.0.0.1:8000

The app uses the same retrievers as the experiments, held in memory. Which methods are
offered depends on which embedding files exist under data/index/dense/.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import DATA, ROOT
from .corpus import Corpus
from .retrieve import dense
from .retrieve.base import to_run
from .retrieve.bm25 import BM25Retriever, _stemmer
from .retrieve.dense import DenseRetriever
from .retrieve.fusion import rrf

app = FastAPI(title="Aviation report search")

POOL = 300  # results fetched per method before filters are applied
BM25_VIEW = "merged"  # keyword search over table + narrative text (best in the experiments)
METHOD_LABELS = {
    "bm25": "Keyword (BM25)",
    "qwen3-0.6b": "AI embedding (Qwen3)",
    "bge-small": "AI embedding (BGE small)",
    "bge-m3": "AI embedding (BGE-M3)",
}


@lru_cache(maxsize=1)
def corpus() -> Corpus:
    return Corpus.load()


@lru_cache(maxsize=1)
def bm25() -> BM25Retriever:
    path = DATA / "index" / "bm25" / BM25_VIEW
    if not (path / "doc_ids.txt").exists():
        BM25Retriever.build(corpus().doc_ids, corpus().texts(BM25_VIEW)).save(path)
    return BM25Retriever.open(path)


@lru_cache(maxsize=4)
def dense_retriever(model: str) -> DenseRetriever:
    return DenseRetriever.open(model, "narr", corpus().doc_ids)


def dense_models() -> list[str]:
    return [m for m in dense.MODELS if dense.embedding_path(m, "narr").exists()]


def methods() -> dict[str, str]:
    out = {"bm25": METHOD_LABELS["bm25"]}
    for m in dense_models():
        out[m] = METHOD_LABELS.get(m, m)
        out[f"hybrid:{m}"] = f"Hybrid: keyword + {METHOD_LABELS.get(m, m).split('(')[-1].rstrip(')')}"
    return out


def ranked(query: str, method: str) -> list[tuple[str, float]]:
    """Ranked (ev_id, score) for one query with one method."""
    if method == "bm25":
        return bm25().search([query], POOL)[0]
    if method in dense_models():
        return dense_retriever(method).search([query], POOL)[0]
    if method.startswith("hybrid:") and method[7:] in dense_models():
        runs = [
            to_run(["q"], bm25().search([query], POOL)),
            to_run(["q"], dense_retriever(method[7:]).search([query], POOL)),
        ]
        return list(rrf(runs, depth=POOL)["q"].items())
    raise HTTPException(400, f"unknown method: {method}")


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/']*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


# Common words that should not drive snippet choice or be highlighted.
_STOP = set(
    "the and for was were that with had has have this from are not his her him she but did any "
    "which there been after when what who how its into than then they their out because while "
    "where does about been being would could should also only both each over under during".split()
)


def _stems(text: str) -> set[str]:
    return {
        _stemmer.stemWord(w.lower())
        for w in _WORD.findall(text)
        if len(w) > 2 and w.lower() not in _STOP
    }


def snippet(text: str, query: str, max_chars: int = 340) -> str:
    """The stretch of the narrative that shares most words with the query, as safe HTML with
    the matching words wrapped in <mark>."""
    q = _stems(query)
    sentences = _SENTENCE.split(text.strip())
    if not sentences:
        return ""
    best = max(range(len(sentences)), key=lambda i: (len(q & _stems(sentences[i])), -i))
    chunk = " ".join(sentences[best : best + 2])[:max_chars]
    out, pos = [], 0
    for m in _WORD.finditer(chunk):
        out.append(html.escape(chunk[pos : m.start()]))
        word = html.escape(m.group())
        hit = bool(_stems(m.group()) & q)
        out.append(f"<mark>{word}</mark>" if hit else word)
        pos = m.end()
    out.append(html.escape(chunk[pos:]))
    return "".join(out) + ("…" if len(chunk) == max_chars else "")


def card(row, score: float, query: str = "") -> dict:
    return {
        "ev_id": row.ev_id,
        "ntsb_no": row.ntsb_no,
        "date": row.ev_date,
        "place": ", ".join(x for x in (row.city, row.state, row.country) if x),
        "aircraft": row.aircraft,
        "category": row.category,
        "phase": row.defining_phase,
        "event": row.defining_event,
        "injury": row.injury,
        "damage": row.damage,
        "cause": row.narr_cause,
        "snippet": snippet(row.text_narr, query) if query else "",
        "score": round(float(score), 4),
    }


@app.get("/api/meta")
def meta() -> dict:
    df = corpus().events
    return {
        "methods": methods(),
        "documents": len(df),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "categories": sorted(c for c in df["category"].unique() if c),
        "injuries": ["Fatal", "Serious", "Minor", "None"],
    }


@app.get("/api/search")
def search(
    q: str = Query(min_length=2),
    method: str = "bm25",
    k: int = Query(20, ge=1, le=50),
    year_from: int | None = None,
    year_to: int | None = None,
    category: str = "",
    injury: str = "",
) -> dict:
    c = corpus()
    results = []
    for doc_id, score in ranked(q, method):
        row = c.events.iloc[c.position[doc_id]]
        if year_from and row.year < year_from:
            continue
        if year_to and row.year > year_to:
            continue
        if category and row.category != category:
            continue
        if injury and row.injury != injury:
            continue
        results.append(card(row, score, q))
        if len(results) == k:
            break
    return {"query": q, "method": method, "results": results}


@app.get("/api/event/{ev_id}")
def event(ev_id: str, similar_by: str = "") -> dict:
    c = corpus()
    if ev_id not in c.position:
        raise HTTPException(404, "event not found")
    pos = c.position[ev_id]
    row = c.events.iloc[pos]
    models = dense_models()
    model = similar_by if similar_by in models else (
        "qwen3-0.6b" if "qwen3-0.6b" in models else (models[0] if models else "")
    )
    if model:
        hits = dense_retriever(model).search_by_doc([pos], 9)[0]
        basis = METHOD_LABELS.get(model, model)
    else:
        hits = bm25().search([row.text_narr], 9)[0]
        basis = METHOD_LABELS["bm25"]
    similar = [
        card(c.events.iloc[c.position[d]], s) for d, s in hits if d != ev_id
    ][:8]
    detail = card(row, 0.0)
    detail.update(
        {
            "registration": row.registration,
            "operation": ", ".join(x for x in (row.operation, row.purpose) if x),
            "conditions": ", ".join(x for x in (row.weather, row.light) if x),
            "fatalities": int(row.fatalities),
            "occurrences": [x for x in row.occurrences.split("; ") if x],
            "findings": [x for x in row.findings.split("; ") if x],
            "analysis": row.narr_analysis,
            "factual": row.narr_factual,
            "similar": similar,
            "similar_basis": basis,
        }
    )
    return detail


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")
