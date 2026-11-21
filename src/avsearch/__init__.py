"""avsearch: retrieval experiments and search over aviation accident reports."""

from pathlib import Path

# Repository root (src/avsearch/__init__.py -> repo). Data lives beside the code, git-ignored.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
