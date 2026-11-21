"""Second-stage reranking: a cross-encoder reads query and document together and rescores
the top candidates from a cheaper first stage."""

from __future__ import annotations

import os

from .. import DATA
from .base import Run
from .dense import device

os.environ.setdefault("HF_HOME", str(DATA / "models"))

RERANKERS = {
    "ettin-150m": "cross-encoder/ettin-reranker-150m-v1",
    "ettin-32m": "cross-encoder/ettin-reranker-32m-v1",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}


class Reranker:
    def __init__(self, model: str = "ettin-150m", max_tokens: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = f"rerank-{model}"
        self.ce = CrossEncoder(RERANKERS[model], device=device(), max_length=max_tokens)

    def rerank(
        self,
        run: Run,
        query_text: dict[str, str],
        doc_text: dict[str, str],
        depth: int = 50,
        batch_size: int = 16,
    ) -> Run:
        """Rescore the top `depth` documents of each query; documents below that keep their
        first-stage order underneath the reranked ones."""
        pairs, owners = [], []
        tails: dict[str, list[str]] = {}
        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
            head, tail = ranked[:depth], ranked[depth:]
            tails[qid] = [d for d, _ in tail]
            for doc_id, _ in head:
                pairs.append((query_text[qid], doc_text[doc_id]))
                owners.append((qid, doc_id))
        scores = self.ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        out: Run = {qid: {} for qid in run}
        for (qid, doc_id), s in zip(owners, scores, strict=True):
            out[qid][doc_id] = float(s)
        for qid, tail in tails.items():
            floor = min(out[qid].values(), default=0.0)
            for i, doc_id in enumerate(tail, start=1):
                out[qid][doc_id] = floor - i
        return out
