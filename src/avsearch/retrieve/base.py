"""Common shape for every retrieval arm.

An arm takes query strings and returns, per query, a ranked list of (doc_id, score).
A "run" is the same thing keyed by query id: {qid: {doc_id: score}} - the format ranx scores.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

Hit = tuple[str, float]
Run = dict[str, dict[str, float]]


class Retriever(Protocol):
    name: str

    def search(self, queries: list[str], k: int = 100) -> list[list[Hit]]: ...


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a (queries x docs) score matrix. Returns (indices, scores), best first."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(part_scores, order, axis=1)


def to_run(qids: list[str], hits: list[list[Hit]], exclude_self: bool = False) -> Run:
    """Turn per-query hit lists into a run. exclude_self drops a doc retrieved for its own id
    (query-by-document tasks, where the query document is trivially its own best match)."""
    run: Run = {}
    for qid, row in zip(qids, hits, strict=True):
        run[qid] = {d: float(s) for d, s in row if not (exclude_self and d == qid)}
    return run
