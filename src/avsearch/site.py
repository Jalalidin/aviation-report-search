"""Build the data files for the static demo site in docs/ (served by GitHub Pages).

The site has no server. Everything it shows is written here as plain files:

  docs/data/docs.json       a sample of reports: card fields + text to search in the browser
  docs/data/emb.bin         small-model embeddings of those reports (int8), for in-browser AI search
  docs/data/detail/N.json   full narratives, split into small files loaded on demand
  docs/data/results.json    the result tables, and what every method returned for each test query

The sample always contains every report that any method ranked in its top 10 for a test
query, so the side-by-side comparison shows the real system's output.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import ROOT
from .corpus import Corpus
from .eval import tasks
from .retrieve import dense

SITE = ROOT / "docs"
RESULTS = ROOT / "eval" / "results"
RUNS = ROOT / "eval" / "runs"
SHARD = 200           # reports per detail file
SEARCH_CHARS = 3000   # narrative text shipped per report for in-browser search
BROWSER_MODEL = "bge-small"  # the embedding model small enough to run in a web page

ARM_LABELS = {
    "bm25": "Keyword (BM25), narrative",
    "bm25[table]": "Keyword, table text",
    "bm25[merged]": "Keyword, table + narrative as one text",
    "bm25[table⊕narr]": "Keyword, table and narrative fused",
    "bge-small": "Embedding: BGE small",
    "bge-small[table]": "Embedding: BGE small, table text",
    "bge-small[merged]": "Embedding: BGE small, table + narrative",
    "qwen3-0.6b": "Embedding: Qwen3",
    "qwen3-0.6b[table]": "Embedding: Qwen3, table text",
    "qwen3-0.6b[merged]": "Embedding: Qwen3, table + narrative as one text",
    "qwen3-0.6b[table⊕narr]": "Embedding: Qwen3, table and narrative fused",
    "bge-m3": "Embedding: BGE-M3",
    "bm25+bge-small": "Hybrid: keyword + BGE small",
    "bm25+qwen3-0.6b": "Hybrid: keyword + Qwen3",
    "bm25+bge-m3": "Hybrid: keyword + BGE-M3",
    "bm25+bge-small+rerank": "Hybrid (BGE small) + reranker",
    "bm25+qwen3-0.6b+rerank": "Hybrid (Qwen3) + reranker",
    "bm25[merged]+qwen3-0.6b": "Hybrid: keyword on table + narrative, + Qwen3",
    "bm25[merged]+qwen3-0.6b+rerank": "Hybrid on table + narrative, + reranker",
}


def _table(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.reader(path.open()))
    return {"columns": rows[0], "rows": rows[1:]}


def _ranked(run: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(run.items(), key=lambda kv: (-kv[1], kv[0]))]


def build(n_docs: int = 6000, seed: int = 7, out: Path = SITE) -> dict:
    corpus = Corpus.load()
    df = corpus.events
    task = tasks.task_a(corpus)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((RUNS / "A").glob("*.json"))}
    # Keep the order the evaluation used (results CSV), not alphabetical.
    table_a = _table(RESULTS / "A.csv")
    order = [r[0] for r in table_a["rows"]] if table_a else list(runs)
    arms = [a for a in order if a in runs]

    # --- which reports go into the sample
    queries, needed = [], set()
    for qid in task.qids:
        target = next(iter(task.qrels[qid]))
        needed.add(target)
        tops, ranks = {}, {}
        for arm in arms:
            ranked = _ranked(runs[arm][qid])
            tops[arm] = ranked[:10]
            needed.update(ranked[:10])
            ranks[arm] = ranked.index(target) + 1 if target in ranked else None
        queries.append({"id": qid, "style": tasks.STYLES[qid.split("-")[1]],
                        "text": task.query_text[qid], "target": target, "top": tops, "rank": ranks})

    rng = np.random.default_rng(seed)
    rest = [d for d in corpus.doc_ids if d not in needed]
    extra = rng.choice(rest, size=max(0, n_docs - len(needed)), replace=False).tolist()
    ids = sorted(needed | set(extra))
    pos = np.array([corpus.position[d] for d in ids])
    local = {d: i for i, d in enumerate(ids)}
    sample = df.iloc[pos]

    # --- similar reports, precomputed with the best embedding model available
    available = [m for m in ("qwen3-0.6b", "bge-m3", "bge-small") if dense.embedding_path(m, "narr").exists()]
    similar: list[list[int]] = [[] for _ in ids]
    similar_by = ""
    if available:
        similar_by = available[0]
        e = np.load(dense.embedding_path(similar_by, "narr"))[pos].astype(np.float32)
        sims = e @ e.T
        np.fill_diagonal(sims, -1)
        similar = np.argsort(-sims, axis=1)[:, :6].tolist()

    # --- docs.json
    def clip(text: str) -> str:
        return text if len(text) <= SEARCH_CHARS else text[:SEARCH_CHARS].rsplit(" ", 1)[0]

    docs = []
    for i, r in enumerate(sample.itertuples()):
        docs.append({
            "id": r.ev_id, "n": r.ntsb_no, "d": r.ev_date, "y": int(r.year),
            "p": ", ".join(x for x in (r.city, r.state, r.country) if x),
            "a": r.aircraft, "c": r.category, "ph": r.defining_phase, "e": r.defining_event,
            "i": r.injury, "cause": r.narr_cause, "t": clip(r.text_narr), "s": similar[i],
        })
    data = out / "data"
    (data / "detail").mkdir(parents=True, exist_ok=True)
    for old in (data / "detail").glob("*.json"):
        old.unlink()
    (data / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")))

    # --- detail shards
    for start in range(0, len(ids), SHARD):
        shard = {}
        for r in sample.iloc[start : start + SHARD].itertuples():
            shard[r.ev_id] = {
                "reg": r.registration, "dmg": r.damage, "fat": int(r.fatalities),
                "op": ", ".join(x for x in (r.operation, r.purpose) if x),
                "cond": ", ".join(x for x in (r.weather, r.light) if x),
                "occ": [x for x in r.occurrences.split("; ") if x],
                "find": [x for x in r.findings.split("; ") if x],
                "an": r.narr_analysis, "fa": r.narr_factual,
            }
        (data / "detail" / f"{start // SHARD}.json").write_text(
            json.dumps(shard, ensure_ascii=False, separators=(",", ":")))

    # --- embeddings for in-browser AI search
    has_emb = dense.embedding_path(BROWSER_MODEL, "narr").exists()
    if has_emb:
        e = np.load(dense.embedding_path(BROWSER_MODEL, "narr"))[pos].astype(np.float32)
        np.clip(np.round(e * 127), -127, 127).astype(np.int8).tofile(data / "emb.bin")

    # --- results.json
    findings = RESULTS / "FINDINGS.md"
    results = {
        "built": dt.date.today().isoformat(),
        "corpus": {"events": 31453, "with_narrative": len(corpus), "sample": len(ids),
                   "years": [int(df["year"].min()), int(df["year"].max())]},
        "arms": [{"id": a, "label": ARM_LABELS.get(a, a)} for a in arms],
        "tables": {"A": table_a, "A_by_style": _table(RESULTS / "A_by_style.csv"),
                   "B": _table(RESULTS / "B.csv")},
        "findings": [ln[2:].strip() for ln in findings.read_text().splitlines() if ln.startswith("- ")]
        if findings.exists() else [],
        "queries": queries,
        "similar_by": ARM_LABELS.get(similar_by, similar_by),
        "browser_model": {"dim": 384, "available": has_emb},
        "shard": SHARD,
    }
    (data / "results.json").write_text(json.dumps(results, ensure_ascii=False, separators=(",", ":")))
    (out / ".nojekyll").write_text("")

    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in data.glob("*.*")}
    return {"reports": len(ids), "needed_by_test_queries": len(needed), "arms": arms,
            "similar_by": similar_by, "MB": sizes}
