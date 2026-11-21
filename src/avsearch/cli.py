"""Command line entry point: `avsearch <command>`.

Pipeline order: snapshot -> build -> index -> embed -> eval / serve.
"""

from __future__ import annotations

import time

import typer

from . import DATA

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

BM25_DIR = DATA / "index" / "bm25"


@app.command()
def snapshot() -> None:
    """Export the NTSB Access database to Parquet (needs mdbtools)."""
    from .sources import ntsb

    for table, n in ntsb.snapshot().items():
        typer.echo(f"{table:28s} {n:>8,d} rows")


@app.command()
def build() -> None:
    """Build one unified record per event from the exported tables."""
    from .records import build as b

    df = b.build()
    typer.echo(f"{len(df):,d} events, {int(df.has_narrative.sum()):,d} with a narrative -> {b.EVENTS_PATH}")


@app.command()
def index(view: str = "narr") -> None:
    """Build the BM25 keyword index for one text view (narr | table | merged)."""
    from .corpus import Corpus
    from .retrieve.bm25 import BM25Retriever

    corpus = Corpus.load()
    t = time.time()
    BM25Retriever.build(corpus.doc_ids, corpus.texts(view)).save(BM25_DIR / view)
    typer.echo(f"bm25/{view}: {len(corpus):,d} documents in {time.time() - t:.1f}s")


@app.command()
def embed(model: str, view: str = "narr", batch_size: int = 16) -> None:
    """Embed every document with one dense model (see retrieve/dense.py MODELS)."""
    from .corpus import Corpus
    from .retrieve import dense

    corpus = Corpus.load()
    out = dense.embedding_path(model, view)
    t = time.time()
    emb = dense.encode_corpus(model, corpus.texts(view), out, batch_size)
    typer.echo(f"{model}/{view}: {emb.shape} in {time.time() - t:.0f}s -> {out}")


@app.command()
def search(query: str, arm: str = "bm25", view: str = "narr", k: int = 5) -> None:
    """Run one query from the terminal. arm: bm25 | a dense model name."""
    from .corpus import Corpus
    from .retrieve.bm25 import BM25Retriever
    from .retrieve.dense import DenseRetriever

    corpus = Corpus.load()
    if arm == "bm25":
        retriever = BM25Retriever.open(BM25_DIR / view)
    else:
        retriever = DenseRetriever.open(arm, view, corpus.doc_ids)
    for doc_id, score in retriever.search([query], k)[0]:
        r = corpus.events.iloc[corpus.position[doc_id]]
        typer.echo(f"{score:8.3f}  {r.ntsb_no:11s} {r.ev_date}  {r.aircraft[:28]:28s}  {r.defining_event}")
        typer.echo(f"          {r.narr_cause[:160]}")


@app.command(name="eval")
def eval_(task: str = "all") -> None:
    """Run the retrieval experiments and write the results table. task: A | B | all"""
    from .eval import run as r

    r.main(task)


@app.command()
def site(n_docs: int = 6000) -> None:
    """Write the data files for the static demo site in docs/."""
    from . import site as s

    for key, value in s.build(n_docs).items():
        typer.echo(f"{key}: {value}")


@app.command()
def serve(port: int = 8000) -> None:
    """Start the search API and web interface."""
    import uvicorn

    uvicorn.run("avsearch.api:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
