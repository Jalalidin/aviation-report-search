"""Keyword baseline: BM25 with English stemming and stop-word removal (bm25s)."""

from __future__ import annotations

from pathlib import Path

import bm25s
import Stemmer

from .base import Hit

_stemmer = Stemmer.Stemmer("english")


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(
        texts, stopwords="en", stemmer=_stemmer, return_ids=False, show_progress=False
    )


class BM25Retriever:
    def __init__(self, doc_ids: list[str], index: bm25s.BM25, name: str = "bm25"):
        self.doc_ids = doc_ids
        self.index = index
        self.name = name

    @classmethod
    def build(cls, doc_ids: list[str], texts: list[str], name: str = "bm25") -> BM25Retriever:
        index = bm25s.BM25(method="lucene")
        index.index(tokenize(texts), show_progress=False)
        return cls(doc_ids, index, name)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(str(path))
        (path / "doc_ids.txt").write_text("\n".join(self.doc_ids))

    @classmethod
    def open(cls, path: Path, name: str = "bm25") -> BM25Retriever:
        doc_ids = (path / "doc_ids.txt").read_text().split("\n")
        return cls(doc_ids, bm25s.BM25.load(str(path)), name)

    def search(self, queries: list[str], k: int = 100) -> list[list[Hit]]:
        k = min(k, len(self.doc_ids))
        idx, scores = self.index.retrieve(tokenize(queries), k=k, show_progress=False)
        return [
            [(self.doc_ids[i], float(s)) for i, s in zip(row_i, row_s, strict=True)]
            for row_i, row_s in zip(idx, scores, strict=True)
        ]
