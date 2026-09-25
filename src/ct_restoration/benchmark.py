"""Running one restoration method over one split, identically for every method.

The evaluation policy lives in :mod:`ct_restoration.evaluation` and the metric
formulas in :mod:`ct_restoration.metrics`. This module is the harness that
drives them: read the frozen manifest, rebuild each slice's clean reference and
degraded input, hand the degraded image to a method, score the result.

Why a shared harness
--------------------
Every method in this benchmark - the degraded baseline, CLAHE, the residual
CNN, the U-Net - must see the same clean targets, the same degraded inputs
generated from the same sample keys, the same evaluation masks, and the same
metric code. The surest way to guarantee that is for there to be exactly one
implementation of the loop, with the method supplied as a callable. A second
copy of this loop inside a method's own script is a second place for the
benchmark definition to drift.

The method contract
-------------------
A method is a :data:`RestoreFn`: it receives the degraded image and returns a
restored image of the same shape. That signature is the contract. A method does
**not** receive the clean reference, the body mask, the HU slice, the sample
key, the acquisition group, the source archive or the patient identity. Those
exist for evaluation and provenance only, and a method that could see any of
them would be scoring itself against information a real deployment would not
have.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ct_restoration.config import ensure_dir
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like
from ct_restoration.evaluation import (
    METRIC_COLUMNS,
    EvaluationConfig,
    HoldoutAccess,
    prepare_evaluation_slice,
    require_rows_split_access,
    require_split_access,
)
from ct_restoration.metrics import slice_metrics

#: A restoration method. Takes the degraded image, returns the restored image.
#: Deliberately nothing else: see the module docstring.
RestoreFn = Callable[[np.ndarray], np.ndarray]

#: Directory holding the tracked metric tables.
METRICS_DIR = Path("outputs/metrics")

#: Column order of every per-slice metric table, whatever the method.
SLICE_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "source_archive",
    "acquisition_group",
    "relative_dicom_path",
    "geometric_slice_index",
    *METRIC_COLUMNS,
    "body_pixel_fraction",
    "body_ssim_interior_fraction",
)

#: Decimal places for floats in tracked outputs. Enough for an MSE around
#: 1e-4 to keep every significant digit.
ROUND = 10


def identity_restoration(degraded: np.ndarray) -> np.ndarray:
    """The no-restoration method: return the degraded image unchanged."""
    return degraded


def round_floats(value: Any, places: int = ROUND) -> Any:
    """Round floats for a tracked summary, recursively through containers."""
    if isinstance(value, dict):
        return {key: round_floats(item, places) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, places) for item in value]
    if isinstance(value, float | np.floating):
        return round(float(value), places)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with a fixed line ending so its checksum is portable."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a tracked JSON summary: rounded, strict, newline-terminated.

    ``allow_nan=False`` on purpose. An infinite PSNR means a method reproduced
    the reference exactly, which is worth failing loudly over rather than
    emitting a token no strict JSON reader accepts.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(round_floats(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def manifest_rows(
    manifest_path: Path, split: str, access: HoldoutAccess | None = None
) -> pd.DataFrame:
    """Rows of one development split, sorted deterministically.

    The sort - numeric subject, then geometric slice index - is what makes two
    methods' per-slice tables line up row for row, so a paired comparison can
    be audited by reading the two files side by side.

    ``access`` is for the Milestone 11 held-out protocol alone: with a
    :class:`~ct_restoration.evaluation.HoldoutAccess` it returns the test rows,
    in the same order, and nothing else. Every development command calls this
    without one, which is exactly the Milestone 5-10 behaviour.

    Raises:
        HeldOutSplitError: the split is sealed to this caller.
        ValueError: the manifest holds no rows for that split.
    """
    require_split_access(split, access)
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No split == {split} rows in {manifest_path}")
    rows["subject_order"] = rows["subject_id"].astype(int)
    rows = rows.sort_values(["subject_order", "geometric_slice_index"])
    return rows.drop(columns="subject_order").reset_index(drop=True)


def evaluate_slices(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    restore: RestoreFn = identity_restoration,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Score one method on every row. One manifest row in, one metric row out.

    Args:
        rows: manifest rows from :func:`manifest_rows`.
        root: extracted CHAOS directory.
        preprocessing: the frozen Milestone 1 settings.
        evaluation: the frozen Milestone 5 policy.
        degradation: the frozen Milestone 4 corruption.
        restore: the method under test. Defaults to no restoration.

    Raises:
        HeldOutSplitError: any row is labelled test or stress, or carries no
            split label. This loop is for development splits only, however
            the rows reached it; the held-out protocol has its own.
    """
    require_rows_split_access(rows)
    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows.itertuples(), start=1):
        key = row.relative_dicom_path
        clean, body, interior = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation)
        restored = restore(degraded)

        measured = slice_metrics(clean, restored, body, interior, evaluation.ssim)
        records.append(
            {
                "subject_id": row.subject_id,
                "source_archive": row.source_archive,
                "acquisition_group": row.acquisition_group,
                "relative_dicom_path": key,
                "geometric_slice_index": int(row.geometric_slice_index),
                **measured,
                "body_pixel_fraction": float(body.mean()),
                "body_ssim_interior_fraction": float(interior.mean()),
            }
        )
        if progress_every and (position % progress_every == 0 or position == len(rows)):
            print(f"  processed {position:>5} / {len(rows)}")

    return finalise_slice_frame(records)


def finalise_slice_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Impose the canonical column order and float rounding on metric rows."""
    frame = pd.DataFrame(records)[list(SLICE_COLUMNS)]
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(ROUND)
    return frame


def check_metrics_output_policy(limit: int, slices_path: Path) -> None:
    """Refuse to write a partial evaluation over a canonical tracked table.

    ``--limit`` exists for debugging. A truncated run looks exactly like a full
    one once written to disk, and these tables are the record every later
    comparison is built on.

    Raises:
        ValueError: ``limit`` is set and the output is a canonical path.
    """
    if limit and Path(slices_path).resolve().parent == METRICS_DIR.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite canonical tracked metrics in "
            f"{METRICS_DIR.as_posix()} with a partial evaluation. "
            "Re-run without --limit, or pass a noncanonical --output-dir."
        )
