"""NTSB aviation accident database (avall.mdb) -> Parquet, one file per table.

The database is a Microsoft Access file read with the `mdbtools` command line programs
(`brew install mdbtools`). Everything is exported as text; typing happens when the unified
event records are built, so this step stays a faithful copy of the source.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .. import DATA

RAW_MDB = DATA / "raw" / "ntsb" / "extracted" / "avall.mdb"
OUT_DIR = DATA / "parquet" / "ntsb"

# Tables the prototype uses. `eADMSPUB_DataDictionary` decodes the coded columns.
TABLES = [
    "events",
    "aircraft",
    "narratives",
    "Findings",
    "Events_Sequence",
    "engines",
    "eADMSPUB_DataDictionary",
]


def export_table(mdb: Path, table: str) -> pd.DataFrame:
    if shutil.which("mdb-export") is None:
        raise RuntimeError("mdb-export not found. Install it with: brew install mdbtools")
    proc = subprocess.run(
        ["mdb-export", "-D", "%Y-%m-%d", "-T", "%Y-%m-%d %H:%M:%S", str(mdb), table],
        check=True,
        capture_output=True,
    )
    text = proc.stdout.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    return df


def snapshot(mdb: Path = RAW_MDB, out_dir: Path = OUT_DIR) -> dict[str, int]:
    """Export every table in TABLES to Parquet. Returns row counts per table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table in TABLES:
        df = export_table(mdb, table)
        df.to_parquet(out_dir / f"{table.lower()}.parquet", index=False)
        counts[table] = len(df)
    return counts


def load(table: str, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / f"{table.lower()}.parquet")
