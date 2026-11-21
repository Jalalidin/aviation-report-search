"""Does the Task B result depend on which 300 reports were drawn as queries?

Re-draws the 300 query reports with five different random seeds and scores the same methods
each time. Writes eval/results/B_seeds.csv (nDCG@10 per seed and method).

    uv run python scripts/task_b_seeds.py
"""

import pandas as pd
from ranx import Qrels, evaluate
from ranx import Run as RanxRun

from avsearch import ROOT
from avsearch.corpus import Corpus
from avsearch.eval import tasks
from avsearch.eval.run import DEPTH, _bm25, _dense_models
from avsearch.retrieve.base import to_run
from avsearch.retrieve.dense import DenseRetriever
from avsearch.retrieve.fusion import rrf

SEEDS = (13, 1, 2, 3, 4)  # 13 is the seed behind eval/results/B.md

corpus = Corpus.load()
bm25 = _bm25("narr", corpus)
dense = {m: DenseRetriever.open(m, "narr", corpus.doc_ids) for m in _dense_models("narr")}
k = DEPTH + 1  # the query report itself is retrieved, then dropped

rows = []
for seed in SEEDS:
    task = tasks.task_b(corpus, seed=seed)
    positions = [corpus.position[q] for q in task.qids]
    queries = [task.query_text[q] for q in task.qids]
    runs = {"bm25": to_run(task.qids, bm25.search(queries, k), exclude_self=True)}
    for name, retriever in dense.items():
        runs[name] = to_run(task.qids, retriever.search_by_doc(positions, k), exclude_self=True)
        runs[f"bm25+{name}"] = rrf([runs["bm25"], runs[name]], depth=DEPTH)
    qrels = Qrels(task.qrels)
    rows.append({"seed": seed} | {
        name: round(float(evaluate(qrels, RanxRun(run), "ndcg@10")), 3) for name, run in runs.items()
    })

table = pd.DataFrame(rows).set_index("seed")
table.loc["mean"] = table.mean().round(3)
table.to_csv(ROOT / "eval" / "results" / "B_seeds.csv")
print(table.to_string())
