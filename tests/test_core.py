"""Small, fast checks of the pieces the results depend on. No models, no downloaded data."""

import numpy as np
import pandas as pd
import pytest

from avsearch.corpus import Corpus
from avsearch.eval import tasks
from avsearch.records import build
from avsearch.retrieve.base import to_run, top_k
from avsearch.retrieve.bm25 import BM25Retriever
from avsearch.retrieve.fusion import rrf


def test_top_k_orders_best_first():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.2, 0.3]])
    idx, vals = top_k(scores, 2)
    assert idx.tolist() == [[1, 3], [0, 3]]
    assert vals.tolist() == [[0.9, 0.7], [1.0, 0.3]]


def test_top_k_caps_at_collection_size():
    idx, _ = top_k(np.array([[0.2, 0.1]]), 10)
    assert idx.tolist() == [[0, 1]]


def test_to_run_can_drop_the_query_document():
    run = to_run(["d1"], [[("d1", 1.0), ("d2", 0.5)]], exclude_self=True)
    assert run == {"d1": {"d2": 0.5}}


def test_rrf_scores_by_reciprocal_rank():
    a = {"q": {"x": 10.0, "y": 5.0}}
    b = {"q": {"y": 0.9, "z": 0.1}}
    fused = rrf([a, b], k=60)["q"]
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)
    assert list(fused) == ["y", "x", "z"]  # best first


def test_rrf_matches_ranx():
    from ranx import Run, fuse

    a = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}, "q2": {"a": 1.0, "d": 0.5}}
    b = {"q1": {"c": 9.0, "a": 1.0}, "q2": {"d": 2.0, "e": 1.0}}
    ours = rrf([a, b], k=60)
    theirs = fuse([Run(a), Run(b)], norm=None, method="rrf", params={"k": 60}).to_dict()
    for qid in ours:
        for doc_id, score in ours[qid].items():
            assert theirs[qid][doc_id] == pytest.approx(score)


def test_placeholder_narratives_are_treated_as_empty():
    assert build._clean("Not required for RA.") == ""
    assert build._clean(None) == ""
    assert build._clean("The  pilot\r\nreported") == "The pilot\nreported"


def test_occurrence_description_is_split_into_phase_and_event():
    es = pd.DataFrame(
        {
            "Occurrence_Description": [
                "Landing Loss of control on ground",
                "Takeoff Loss of control on ground",
                "Landing Hard landing",
            ],
            "phase_no": ["550", "300", "550"],
            "eventsoe_no": ["230", "230", "092"],
        }
    )
    out = build.split_occurrences(es)
    assert out["event"].tolist() == ["Loss of control on ground"] * 2 + ["Hard landing"]
    assert out["phase"].tolist() == ["Landing", "Takeoff", "Landing"]


def _toy_corpus() -> Corpus:
    return Corpus(
        pd.DataFrame(
            {
                "ev_id": ["e1", "e2", "e3", "e4", "e5"],
                "defining_code": ["230", "230", "230", "341", "000"],
                "cause_codes": [["01062020"], ["01062020"], ["02041510"], ["01062020"], ["01062020"]],
                "text_narr": ["n1", "n2", "n3", "n4", "n5"],
            }
        )
    )


def test_task_b_grades_by_shared_event_and_cause():
    task = tasks.task_b(_toy_corpus(), n_queries=10, seed=0)
    # Only e1 and e2 have a grade-2 answer (each other). e3 and e4 match others on one
    # criterion only; e5 has an uninformative defining event, so it is never a query -
    # but it can still be an answer.
    assert set(task.qids) == {"e1", "e2"}
    assert task.qrels["e1"] == {"e2": 2, "e3": 1, "e4": 1, "e5": 1}
    assert "e1" not in task.qrels["e1"]


def test_bm25_finds_the_matching_document(tmp_path):
    r = BM25Retriever.build(
        ["a", "b", "c"],
        [
            "The engine lost power because of water in the fuel.",
            "The pilot lost directional control during the landing roll.",
            "The helicopter tail rotor blade separated in flight.",
        ],
    )
    r.save(tmp_path)
    reopened = BM25Retriever.open(tmp_path)
    assert reopened.search(["tail rotor blades"], k=1)[0][0][0] == "c"
    assert reopened.search(["fuel contamination water"], k=1)[0][0][0] == "a"
