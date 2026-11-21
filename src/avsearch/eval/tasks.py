"""The evaluation tasks: which queries, and which documents count as relevant.

Task A  ad-hoc search, known-item: hand-written queries, the target report is the one answer.
Task B  similar incident: the query is a whole report; relevance comes from NTSB's own coding
        ("silver" truth - approximate, but free and large):
          2 = same defining event AND at least one shared cause finding
          1 = exactly one of the two
        Finding codes are compared at subsection level (first 8 of 10 digits).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .. import ROOT
from ..corpus import Corpus

QUERIES_PATH = ROOT / "eval" / "queries.yaml"
STYLES = {"kw": "keywords", "nl": "question", "mm": "mismatch"}
# Defining events that say nothing about what happened.
UNINFORMATIVE = {"", "000", "900"}  # unknown/undetermined, miscellaneous/other


@dataclass
class Task:
    name: str
    qids: list[str]
    query_text: dict[str, str]
    qrels: dict[str, dict[str, int]]
    groups: dict[str, list[str]]  # named subsets of qids, for per-style breakdowns


def task_a(corpus: Corpus, path: Path = QUERIES_PATH) -> Task:
    entries = yaml.safe_load(path.read_text())
    qids, text, qrels = [], {}, {}
    groups: dict[str, list[str]] = {label: [] for label in STYLES.values()}
    for n, e in enumerate(entries, start=1):
        target = str(e["target"])  # newer event ids are all digits, which YAML reads as numbers
        if target not in corpus.position:
            raise ValueError(f"query target {target} is not in the corpus")
        for style, label in STYLES.items():
            qid = f"A{n:02d}-{style}"
            qids.append(qid)
            text[qid] = e[style]
            qrels[qid] = {target: 1}
            groups[label].append(qid)
    return Task("A", qids, text, qrels, groups)


def task_b(corpus: Corpus, n_queries: int = 300, seed: int = 13) -> Task:
    df = corpus.events
    defining = df["defining_code"].to_numpy()
    causes = [set(c) for c in df["cause_codes"]]

    eligible = np.array(
        [i for i in range(len(df)) if causes[i] and defining[i] not in UNINFORMATIVE]
    )
    rng = np.random.default_rng(seed)

    by_code: dict[str, list[int]] = {}
    for i, codes in enumerate(causes):
        for c in codes:
            by_code.setdefault(c, []).append(i)

    doc_ids = corpus.doc_ids
    qids, text, qrels = [], {}, {}
    # Walk the eligible reports in random order, keeping those that have at least one
    # grade-2 answer, until there are n_queries of them.
    for i in rng.permutation(eligible):
        same_event = defining == defining[i]
        shares_cause = np.zeros(len(df), dtype=bool)
        for c in causes[i]:
            shares_cause[by_code[c]] = True
        grade = same_event.astype(int) + shares_cause.astype(int)
        grade[i] = 0  # a report is not its own answer
        if not (grade == 2).any():
            continue
        qid = doc_ids[i]
        qids.append(qid)
        text[qid] = df["text_narr"].iat[i]
        qrels[qid] = {doc_ids[j]: int(grade[j]) for j in np.flatnonzero(grade)}
        if len(qids) == n_queries:
            break
    return Task("B", qids, text, qrels, {})
