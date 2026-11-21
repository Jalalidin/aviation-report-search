"""The document collection the experiments run on: events that have a narrative,
in a fixed order, with a choice of text view (see records/build.py)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from .records import build

VIEWS = ("narr", "table", "merged")


@dataclass
class Corpus:
    events: pd.DataFrame  # one row per document, index 0..n-1

    @classmethod
    def load(cls) -> Corpus:
        df = build.load()
        df = df[df["has_narrative"]].sort_values("ev_id").reset_index(drop=True)
        return cls(df)

    @cached_property
    def doc_ids(self) -> list[str]:
        return self.events["ev_id"].tolist()

    @cached_property
    def position(self) -> dict[str, int]:
        return {d: i for i, d in enumerate(self.doc_ids)}

    def texts(self, view: str = "narr") -> list[str]:
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        return self.events[f"text_{view}"].tolist()

    def __len__(self) -> int:
        return len(self.events)
