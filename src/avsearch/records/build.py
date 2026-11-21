"""Build one unified record per accident from the NTSB tables.

Each event gets its coded facts (the "table" side), its narrative texts, and three text
views used by the retrieval experiments:

  text_table   coded facts rendered as a short description + the probable-cause statement
  text_narr    analysis + factual narrative only (no codes, no cause statement)
  text_merged  both

`text_narr` deliberately contains nothing derived from the finding/occurrence codes, because
those codes are the ground truth of the similar-incident task.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from .. import DATA
from ..sources import ntsb

EVENTS_PATH = DATA / "parquet" / "events.parquet"

# Text NTSB stores in a narrative field when there is no narrative.
_PLACEHOLDER = re.compile(
    r"^\s*(not required for ra\.?|none\.?|n/?a\.?|see (the )?(attached|full|public).{0,120})\s*$",
    re.I | re.S,
)
MIN_NARR_CHARS = 200

FAR_PART = {
    "091": "Part 91 general aviation",
    "121": "Part 121 air carrier",
    "135": "Part 135 air taxi and commuter",
    "137": "Part 137 agricultural",
    "133": "Part 133 rotorcraft external load",
    "129": "Part 129 foreign air carrier",
    "125": "Part 125 large aircraft",
    "NUSN": "non-U.S. non-commercial",
    "NUSC": "non-U.S. commercial",
    "PUBU": "public aircraft",
}


def _clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "" if _PLACEHOLDER.match(text) else text


def _decoders(dd: pd.DataFrame) -> dict[str, dict[str, str]]:
    """column name -> {code: meaning} from the NTSB data dictionary."""
    dd = dd.dropna(subset=["Column", "code_iaids", "meaning"])
    return {
        col: dict(zip(g["code_iaids"].str.strip(), g["meaning"].str.strip(), strict=False))
        for col, g in dd.groupby("Column")
    }


def _common_affix(texts: list[str], suffix: bool) -> str:
    """Longest run of words shared at the start (or end) of every text."""
    rows = [t.split()[::-1] if suffix else t.split() for t in texts]
    shared = os.path.commonprefix(rows)
    return " ".join(shared[::-1] if suffix else shared)


def split_occurrences(es: pd.DataFrame) -> pd.DataFrame:
    """Occurrence_Description is '<phase of flight> <event>' in one string. Recover the two
    parts: the event text is whatever all descriptions sharing an event code end with."""
    es = es.dropna(subset=["Occurrence_Description"]).copy()
    event_text = {}
    for code, g in es.groupby("eventsoe_no"):
        descs = g["Occurrence_Description"].unique().tolist()
        event_text[code] = _common_affix(descs, suffix=True) if len(descs) > 1 else ""
    phase_text = {}
    for code, g in es.groupby("phase_no"):
        descs = g["Occurrence_Description"].unique().tolist()
        phase_text[code] = _common_affix(descs, suffix=False) if len(descs) > 1 else ""

    def split(row: pd.Series) -> tuple[str, str]:
        desc = row["Occurrence_Description"]
        ev = event_text.get(row["eventsoe_no"], "")
        if ev and desc.endswith(ev):
            return desc[: -len(ev)].strip(), ev
        ph = phase_text.get(row["phase_no"], "")
        if ph and desc.startswith(ph):
            return ph, desc[len(ph) :].strip()
        return "", desc

    parts = es.apply(split, axis=1, result_type="expand")
    es["phase"], es["event"] = parts[0], parts[1]
    return es


def _join(values: pd.Series, sep: str = "; ") -> str:
    seen: list[str] = []
    for v in values:
        if isinstance(v, str) and v.strip() and v.strip() not in seen:
            seen.append(v.strip())
    return sep.join(seen)


def _table_text(r: pd.Series) -> str:
    parts = []
    where = ", ".join(x for x in [r.city, r.state, r.country] if x)
    kind = "Incident" if r.ev_type == "INC" else "Accident"
    parts.append(f"{kind} on {r.ev_date}{' at ' + where if where else ''}.")
    if r.aircraft:
        bits = [r.aircraft]
        if r.category:
            bits.append(r.category.lower())
        if r.registration:
            bits.append(f"registration {r.registration}")
        parts.append("Aircraft: " + ", ".join(bits) + ".")
    ops = ", ".join(x for x in [r.operation, r.purpose] if x)
    if ops:
        parts.append(f"Operation: {ops}.")
    if r.damage:
        parts.append(f"Damage: {r.damage.lower()}.")
    if r.injury:
        parts.append(f"Highest injury: {r.injury.lower()}.")
    cond = ", ".join(x for x in [r.weather, r.light] if x)
    if cond:
        parts.append(f"Conditions: {cond}.")
    if r.occurrences:
        parts.append(f"Sequence of events: {r.occurrences}.")
    if r.findings:
        parts.append(f"Findings: {r.findings}.")
    if r.narr_cause:
        parts.append(f"Probable cause: {r.narr_cause}")
    return " ".join(parts)


def build(out_path: Path = EVENTS_PATH) -> pd.DataFrame:
    ev = ntsb.load("events")
    ac = ntsb.load("aircraft")
    na = ntsb.load("narratives")
    fi = ntsb.load("findings")
    es = split_occurrences(ntsb.load("events_sequence"))
    dec = _decoders(ntsb.load("eadmspub_datadictionary"))

    def decode(series: pd.Series, column: str) -> pd.Series:
        return series.str.strip().map(dec.get(column, {})).fillna("")

    out = pd.DataFrame(
        {
            "ev_id": ev["ev_id"],
            "ntsb_no": ev["ntsb_no"].fillna(""),
            "ev_type": ev["ev_type"].fillna(""),
            "ev_date": ev["ev_date"].str.slice(0, 10).fillna(""),
            "year": pd.to_numeric(ev["ev_year"], errors="coerce").astype("Int64"),
            "city": ev["ev_city"].fillna("").str.title(),
            "state": ev["ev_state"].fillna(""),
            "country": ev["ev_country"].fillna(""),
            "injury": decode(ev["ev_highest_injury"], "ev_highest_injury"),
            "fatalities": pd.to_numeric(ev["inj_tot_f"], errors="coerce").fillna(0).astype(int),
            "weather": ev["wx_cond_basic"].fillna("").str.strip(),
            "light": decode(ev["light_cond"], "light_cond"),
        }
    )

    # Aircraft: an event can involve several (mid-air collisions). Keep all, lead with the first.
    ac = ac.sort_values(["ev_id", "Aircraft_Key"]).copy()
    ac["make"] = ac["acft_make"].fillna("").str.strip().str.title()
    ac["model"] = ac["acft_model"].fillna("").str.strip().str.upper()
    ac["aircraft"] = (ac["make"] + " " + ac["model"]).str.strip()
    ac["category"] = decode(ac["acft_category"], "acft_category")
    ac["damage"] = decode(ac["damage"], "damage")
    ac["operation"] = ac["far_part"].str.strip().map(FAR_PART).fillna("")
    ac["purpose"] = decode(ac["type_fly"], "type_fly")
    agg = ac.groupby("ev_id").agg(
        make=("make", "first"),
        model=("model", "first"),
        aircraft=("aircraft", _join),
        category=("category", "first"),
        damage=("damage", "first"),
        operation=("operation", "first"),
        purpose=("purpose", "first"),
        registration=("regis_no", _join),
        serial=("acft_serial_no", _join),
        n_aircraft=("Aircraft_Key", "size"),
    )
    out = out.merge(agg, on="ev_id", how="left")

    # Sequence of events: phase + event per occurrence, and the one flagged as defining.
    es = es.sort_values(["ev_id", "Aircraft_Key", "Occurrence_No"])
    seq = es.groupby("ev_id").agg(
        occurrences=("Occurrence_Description", _join),
        phases=("phase", _join),
        occurrence_codes=("eventsoe_no", lambda s: sorted(set(s.dropna()))),
    )
    defining = (
        es[es["Defining_ev"] == "1"]
        .groupby("ev_id")
        .agg(
            defining_code=("eventsoe_no", "first"),
            defining_event=("event", "first"),
            defining_phase=("phase", "first"),
        )
    )
    out = out.merge(seq, on="ev_id", how="left").merge(defining, on="ev_id", how="left")

    # Findings: full description for display, and cause codes at subsection level (8 digits,
    # i.e. without the final "modifier" pair) as labels for the similar-incident task.
    fi = fi.sort_values(["ev_id", "Aircraft_Key", "finding_no"]).copy()
    fi["cause8"] = fi["finding_code"].str.slice(0, 8).where(fi["Cause_Factor"] == "C")
    find = fi.groupby("ev_id").agg(
        findings=("finding_description", _join),
        cause_codes=("cause8", lambda s: sorted(set(s.dropna()))),
        finding_categories=("category_no", lambda s: sorted(set(s.dropna()))),
    )
    out = out.merge(find, on="ev_id", how="left")

    # Narratives: usually one row per event, occasionally one per aircraft.
    na = na.copy()
    for col in ["narr_accp", "narr_accf", "narr_cause"]:
        na[col] = na[col].map(_clean)
    narr = na.groupby("ev_id").agg(
        narr_factual=("narr_accp", lambda s: _join(s, "\n\n")),
        narr_analysis=("narr_accf", lambda s: _join(s, "\n\n")),
        narr_cause=("narr_cause", lambda s: _join(s, " ")),
    )
    out = out.merge(narr, on="ev_id", how="left")

    # Events missing from a joined table (no findings coded, no narrative row, ...) come out
    # of the merges as NaN. Every text column must be an empty string instead, or "nan"
    # leaks into the rendered text.
    list_cols = ("occurrence_codes", "cause_codes", "finding_categories")
    numeric_cols = ("year", "fatalities", "n_aircraft")
    text_cols = [c for c in out.columns if c not in list_cols + numeric_cols]
    out[text_cols] = out[text_cols].astype(object).where(out[text_cols].notna(), "")
    for col in list_cols:
        out[col] = out[col].map(lambda v: list(v) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else [])
    out["n_aircraft"] = out["n_aircraft"].fillna(0).astype(int)

    out["text_narr"] = (out["narr_analysis"] + "\n\n" + out["narr_factual"]).str.strip()
    out["text_table"] = out.apply(_table_text, axis=1)
    out["text_merged"] = (out["text_table"] + "\n\n" + out["text_narr"]).str.strip()
    out["has_narrative"] = out["text_narr"].str.len() >= MIN_NARR_CHARS

    out = out.sort_values("ev_id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def load(path: Path = EVENTS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
