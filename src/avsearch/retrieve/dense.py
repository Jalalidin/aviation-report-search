"""Dense retrieval: embed documents once, then exact cosine search with numpy.

Exact search (no approximate index) is deliberate - at this corpus size it is instant and
it means no index parameter can influence the experiment results.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import DATA
from .base import Hit, top_k

# Keep downloaded models inside the project's git-ignored data folder.
os.environ.setdefault("HF_HOME", str(DATA / "models"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# name -> Hugging Face id and how the model wants search queries marked.
MODELS: dict[str, dict] = {
    "bge-small": {
        "hf": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    # half_precision: measured on an M1 Pro to give the same vectors (cosine 1.000 against
    # full precision) about 45% faster.
    "qwen3-0.6b": {
        "hf": "Qwen/Qwen3-Embedding-0.6B",
        "query_prompt_name": "query",
        "half_precision": True,
    },
    "bge-m3": {"hf": "BAAI/bge-m3"},
}
MAX_TOKENS = 512


def device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model: str):
    import torch
    from sentence_transformers import SentenceTransformer

    spec = MODELS[model]
    dev = device()
    kwargs = {}
    if spec.get("half_precision") and dev != "cpu":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    st = SentenceTransformer(spec["hf"], device=dev, **kwargs)
    st.max_seq_length = MAX_TOKENS
    return st


def embedding_path(model: str, view: str) -> Path:
    return DATA / "index" / "dense" / f"{model}__{view}.npy"


def encode_corpus(model: str, texts: list[str], out: Path, batch_size: int = 16) -> np.ndarray:
    """Embed every document (truncated to MAX_TOKENS tokens), L2-normalised, stored as float16."""
    st = load_model(model)
    emb = st.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    return emb


class DenseRetriever:
    def __init__(self, model: str, doc_ids: list[str], embeddings: np.ndarray, name: str | None = None):
        self.model = model
        self.doc_ids = doc_ids
        self.emb = embeddings.astype(np.float32)
        self.name = name or f"dense-{model}"
        self._st = None

    @classmethod
    def open(cls, model: str, view: str, doc_ids: list[str]) -> DenseRetriever:
        emb = np.load(embedding_path(model, view))
        if len(emb) != len(doc_ids):
            raise ValueError(f"{model}/{view}: {len(emb)} vectors for {len(doc_ids)} documents")
        return cls(model, doc_ids, emb)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if self._st is None:
            self._st = load_model(self.model)
        spec = MODELS[self.model]
        kwargs = {}
        if "query_prompt_name" in spec:
            kwargs["prompt_name"] = spec["query_prompt_name"]
        if "query_prefix" in spec:
            queries = [spec["query_prefix"] + q for q in queries]
        return self._st.encode(
            queries, batch_size=16, normalize_embeddings=True, show_progress_bar=False, **kwargs
        ).astype(np.float32)

    def _rank(self, q: np.ndarray, k: int) -> list[list[Hit]]:
        idx, scores = top_k(q @ self.emb.T, k)
        return [
            [(self.doc_ids[i], float(s)) for i, s in zip(row_i, row_s, strict=True)]
            for row_i, row_s in zip(idx, scores, strict=True)
        ]

    def search(self, queries: list[str], k: int = 100) -> list[list[Hit]]:
        return self._rank(self.encode_queries(queries), k)

    def search_by_doc(self, positions: list[int], k: int = 100) -> list[list[Hit]]:
        """Similar-document search: use the stored document vectors as the queries."""
        return self._rank(self.emb[positions], k)
