"""Run every retrieval arm on the evaluation tasks and write the results tables.

Usage:  avsearch eval            (both tasks)
        avsearch eval --task A

Outputs, under eval/:
  runs/<task>/<arm>.json      the ranked lists each arm returned (so results can be re-scored)
  results/<task>.md           the comparison table with significance marks
  results/<task>.csv          the same numbers, for plotting
  results/A_by_style.csv      task A broken down by how the query was phrased

Arms that need files which do not exist yet (an embedding model that was never run) are
skipped with a note, so the command always works with whatever has been built so far.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from .. import DATA, ROOT
from ..corpus import Corpus
from ..retrieve import dense
from ..retrieve.base import Run, to_run
from ..retrieve.bm25 import BM25Retriever
from ..retrieve.dense import DenseRetriever
from ..retrieve.fusion import rrf
from . import tasks

warnings.filterwarnings("ignore", message="unsafe cast")

BM25_DIR = DATA / "index" / "bm25"
OUT = ROOT / "eval"
DEPTH = 100          # how many results each arm returns per query
RERANK_DEPTH = 50    # how many of those the reranker rescores
MAIN_DENSE = "qwen3-0.6b"

METRICS_A = ["mrr@10", "recall@1", "recall@10", "recall@100", "ndcg@10"]
METRICS_B = ["ndcg@10", "precision@10", "precision@10-l2", "recall@100-l2", "map@100-l2"]


def _bm25(view: str, corpus: Corpus) -> BM25Retriever:
    path = BM25_DIR / view
    if not (path / "doc_ids.txt").exists():
        BM25Retriever.build(corpus.doc_ids, corpus.texts(view)).save(path)
    return BM25Retriever.open(path)


def _dense_models(view: str) -> list[str]:
    return [m for m in dense.MODELS if dense.embedding_path(m, view).exists()]


def _timed(fn, n_queries: int) -> tuple[list, float]:
    t = time.perf_counter()
    out = fn()
    return out, 1000 * (time.perf_counter() - t) / n_queries


def _save_runs(task: str, runs: dict[str, Run]) -> None:
    folder = OUT / "runs" / task
    folder.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        (folder / f"{name}.json").write_text(json.dumps(run))


def _report(task: str, qrels: dict, runs: dict[str, Run], metrics: list[str],
            latency: dict[str, float], notes: list[str]) -> pd.DataFrame:
    """Score all arms, test differences for significance, write the .md and .csv files."""
    from ranx import Qrels, compare
    from ranx import Run as RanxRun

    ranx_runs = [RanxRun(run, name=name) for name, run in runs.items()]
    report = compare(Qrels(qrels), ranx_runs, metrics, stat_test="fisher", max_p=0.05,
                     n_permutations=2000, random_seed=42)
    table = pd.DataFrame(
        {name: report.results[name] for name in runs}
    ).T[metrics].round(3)
    table.insert(0, "ms/query", [round(latency.get(n, float("nan")), 1) for n in table.index])
    table.index.name = "arm"

    results = OUT / "results"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / f"{task}.csv")
    lines = [f"# Task {task} results", "", f"{len(qrels)} queries, top {DEPTH} results per arm.", ""]
    lines += ["```", str(report), "```", ""]
    lines += ["Letters after a score list the arms it beats with p < 0.05 "
              "(paired randomization test, 2000 permutations).", ""]
    lines += [f"- {n}" for n in notes]
    (results / f"{task}.md").write_text("\n".join(lines) + "\n")
    return table


def _maybe_rerank(runs: dict[str, Run], base: str, task: tasks.Task, corpus: Corpus,
                  notes: list[str], latency: dict[str, float]) -> None:
    try:
        from ..retrieve.rerank import Reranker

        reranker = Reranker("ettin-150m")
    except Exception as exc:  # model not downloadable / not enough memory
        notes.append(f"reranker skipped: {type(exc).__name__}: {exc}")
        return
    doc_text = dict(zip(corpus.doc_ids, corpus.texts("narr"), strict=True))
    name = f"{base}+rerank"
    runs[name], ms = _timed(
        lambda: reranker.rerank(runs[base], task.query_text, doc_text, RERANK_DEPTH), len(task.qids)
    )
    latency[name] = latency.get(base, 0) + ms


def run_a(corpus: Corpus) -> pd.DataFrame:
    """Ad-hoc search. Covers: keyword vs embedding vs both; and table vs narrative vs both."""
    task = tasks.task_a(corpus)
    queries = [task.query_text[q] for q in task.qids]
    runs: dict[str, Run] = {}
    latency: dict[str, float] = {}
    notes = ["Known-item task: only the target report counts as relevant, so absolute "
             "scores understate quality. Compare arms with each other."]

    for view in ("narr", "table", "merged"):
        bm25 = _bm25(view, corpus)  # built or loaded before timing starts
        hits, ms = _timed(lambda r=bm25: r.search(queries, DEPTH), len(queries))
        name = "bm25" if view == "narr" else f"bm25[{view}]"
        runs[name], latency[name] = to_run(task.qids, hits), ms
        for model in _dense_models(view):
            r = DenseRetriever.open(model, view, corpus.doc_ids)
            r.search(queries[:2], 5)  # load the model before timing
            hits, ms = _timed(lambda r=r: r.search(queries, DEPTH), len(queries))
            name = model if view == "narr" else f"{model}[{view}]"
            runs[name], latency[name] = to_run(task.qids, hits), ms

    # Keyword + embedding, fused (narrative view).
    for model in _dense_models("narr"):
        name = f"bm25+{model}"
        runs[name] = rrf([runs["bm25"], runs[model]], depth=DEPTH)
        latency[name] = latency["bm25"] + latency[model]

    # Two ways to use table and narrative together:
    #   early fusion = one index over the concatenated text  -> the "[merged]" arms above
    #   late fusion  = separate table and narrative indexes, results fused afterwards
    runs["bm25[table⊕narr]"] = rrf([runs["bm25[table]"], runs["bm25"]], depth=DEPTH)
    latency["bm25[table⊕narr]"] = latency["bm25[table]"] + latency["bm25"]
    if f"{MAIN_DENSE}[table]" in runs and MAIN_DENSE in runs:
        name = f"{MAIN_DENSE}[table⊕narr]"
        runs[name] = rrf([runs[f"{MAIN_DENSE}[table]"], runs[MAIN_DENSE]], depth=DEPTH)
        latency[name] = latency[f"{MAIN_DENSE}[table]"] + latency[MAIN_DENSE]

    # Rerank the main hybrid; if the main embedding model has not been run yet, rerank
    # whichever hybrid exists so the pipeline can still be checked end to end.
    hybrids = [f"bm25+{m}" for m in _dense_models("narr")]
    if f"bm25+{MAIN_DENSE}" in runs:
        _maybe_rerank(runs, f"bm25+{MAIN_DENSE}", task, corpus, notes, latency)
    elif hybrids:
        notes.append(f"{MAIN_DENSE} embeddings not built yet; reranking {hybrids[0]} instead.")
        _maybe_rerank(runs, hybrids[0], task, corpus, notes, latency)

    _save_runs("A", runs)
    table = _report("A", task.qrels, runs, METRICS_A, latency, notes)

    # Same arms, split by how the query was phrased.
    from ranx import evaluate

    rows = []
    for style, qids in task.groups.items():
        qrels = {q: task.qrels[q] for q in qids}
        for name, run in runs.items():
            scores = evaluate(qrels, {q: run[q] for q in qids}, ["mrr@10", "recall@10"])
            rows.append({"arm": name, "style": style, **{k: round(float(v), 3) for k, v in scores.items()}})
    by_style = pd.DataFrame(rows).pivot(index="arm", columns="style", values="mrr@10").loc[list(runs)]
    by_style.to_csv(OUT / "results" / "A_by_style.csv")
    with (OUT / "results" / "A.md").open("a") as f:
        f.write("\n## MRR@10 by query style\n\n```\n" + by_style.to_string() + "\n```\n")
    return table


def run_b(corpus: Corpus) -> pd.DataFrame:
    """Similar-incident search: the query is a whole report. Narrative view only, because the
    table view contains the codes that define relevance here."""
    task = tasks.task_b(corpus)
    positions = [corpus.position[q] for q in task.qids]
    queries = [task.query_text[q] for q in task.qids]
    runs: dict[str, Run] = {}
    latency: dict[str, float] = {}
    notes = [
        "Silver truth from NTSB coding: 2 = same defining event and a shared cause finding, "
        "1 = one of the two. Metrics ending in -l2 count only grade 2 as relevant.",
        "Narrative text only; finding/occurrence codes and the probable-cause statement are "
        "not in the indexed text.",
    ]

    k = DEPTH + 1  # the query report itself is retrieved, then dropped
    bm25 = _bm25("narr", corpus)
    hits, ms = _timed(lambda: bm25.search(queries, k), len(queries))
    runs["bm25"], latency["bm25"] = to_run(task.qids, hits, exclude_self=True), ms
    for model in _dense_models("narr"):
        r = DenseRetriever.open(model, "narr", corpus.doc_ids)
        hits, ms = _timed(lambda r=r: r.search_by_doc(positions, k), len(queries))
        runs[model], latency[model] = to_run(task.qids, hits, exclude_self=True), ms
        name = f"bm25+{model}"
        runs[name] = rrf([runs["bm25"], runs[model]], depth=DEPTH)
        latency[name] = latency["bm25"] + latency[model]

    _save_runs("B", runs)
    return _report("B", task.qrels, runs, METRICS_B, latency, notes)


def main(task: str = "all") -> None:
    corpus = Corpus.load()
    pd.set_option("display.width", 200)
    for name, fn in (("A", run_a), ("B", run_b)):
        if task.upper() in (name, "ALL"):
            print(f"\n=== Task {name} ===")
            print(fn(corpus).to_string())
    print(f"\nTables written to {Path(OUT / 'results')}")
