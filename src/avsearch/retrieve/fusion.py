"""Rank fusion. Reciprocal rank fusion needs no score calibration between arms,
which is why it is the default way to combine keyword and embedding results."""

from __future__ import annotations

from collections import defaultdict

from .base import Run


def rrf(runs: list[Run], k: int = 60, weights: list[float] | None = None, depth: int = 100) -> Run:
    """score(d) = sum over arms of weight / (k + rank of d in that arm), rank starting at 1."""
    weights = weights or [1.0] * len(runs)
    fused: Run = {}
    qids = set().union(*(r.keys() for r in runs))
    for qid in qids:
        acc: dict[str, float] = defaultdict(float)
        for run, w in zip(runs, weights, strict=True):
            ranked = sorted(run.get(qid, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                acc[doc_id] += w / (k + rank)
        best = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]
        fused[qid] = dict(best)
    return fused
