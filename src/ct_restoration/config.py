"""Project paths and the YAML configuration convention.

Configuration convention for this project:

* Experiment settings live in plain YAML files under ``configs/``.
* ``load_config`` returns a plain ``dict``. No schema classes, no merging,
  no inheritance - the loaded dict is passed explicitly to the code that
  needs it, and is copied verbatim into each run directory so that every
  recorded result can be traced back to the exact settings that produced it.

Path constants assume the repository layout ``<root>/src/ct_restoration/``,
which holds for an editable install (``uv sync``). A non-editable install into
site-packages would not resolve ``PROJECT_ROOT`` to the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
RUNS_DIR = OUTPUTS_DIR / "runs"
FINAL_DIR = OUTPUTS_DIR / "final"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    A bare filename or relative path that does not exist in the current
    working directory is also looked up inside ``configs/``.

    Raises:
        FileNotFoundError: the file does not exist.
        TypeError: the YAML document is not a mapping.
    """
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute():
        in_configs = CONFIGS_DIR / candidate
        if in_configs.exists():
            candidate = in_configs

    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with candidate.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Config file must contain a YAML mapping, got {type(loaded).__name__}: {path}"
        )
    return loaded


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
