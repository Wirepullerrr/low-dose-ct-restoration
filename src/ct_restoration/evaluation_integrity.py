"""Preconditions a canonical evaluation must satisfy before it writes anything.

Two learned methods are now scored through the same frozen benchmark, and
both must clear the same bar. Keeping these gates in one module is the point:
two subtly different definitions of "integrity verified" would be worse than
one, because the weaker one would be the one that never complained.

Three things are enforced here, each because of a specific way a canonical
artifact can be wrong while looking entirely well-formed on disk.

**Checkpoint provenance**, checked before a single image is opened. A
checkpoint's own metadata proves nothing - a file can say anything about
itself - so every claim is checked against something computed independently:
the frozen config's bytes are re-hashed, the checkpoint file's bytes are
re-hashed, and the tracked training history is re-run through the predeclared
selection rule to re-derive which epoch should have been chosen.

**Ordered sample alignment**, checked before anything is written. A paired
per-patient delta is only paired if both sides ran on the same slice in the
same position. Reordered keys pass every set-based check and are still wrong
row by row, so order is compared explicitly.

**Path hygiene**, so a tracked artifact never carries an absolute filesystem
path: it differs between machines, which would end byte-identical
regeneration, and it would commit a local username.

Everything here raises rather than returning a report nobody reads. That was
a real defect once - the alignment report was computed after the CSV files
had already been written - and this module exists so it cannot recur for the
next method.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ct_restoration.config import PROJECT_ROOT
from ct_restoration.training import SELECTION_METRIC, select_best_epoch

#: The canonical training seed for every learned method in this benchmark. A
#: checkpoint carrying any other seed is not the run a report may describe.
CANONICAL_SEED = 2026

#: Tolerance when comparing one selection value against another. The tracked
#: run summary rounds every float to 10 decimal places, so half a unit in the
#: tenth place is the largest disagreement rounding alone can produce. It is a
#: representation tolerance, not a tolerance for disagreeing about which epoch
#: was selected.
SELECTION_VALUE_TOLERANCE = 5e-10

#: Where a run summary records the selected epoch's metric value. Derived from
#: the predeclared selection metric so the two can never drift apart.
SUMMARY_BEST_VALUE_KEY = f"best_{SELECTION_METRIC}"


class EvaluationIntegrityError(RuntimeError):
    """A precondition for canonical evaluation was not met.

    Raised instead of returning a report nobody reads. Every condition these
    gates check is one that would leave the written artifacts wrong in a way
    no later reader could detect from the files themselves.
    """


def _require_integer(label: str, value: Any, minimum: int | None = None) -> int:
    """A genuine integer, or refuse. Never coerced.

    ``int()`` would silently accept ``29.0``, ``"29"`` and ``True``, and a
    provenance gate that repairs malformed metadata ends up checking that the
    repair worked rather than that the metadata was right. A checkpoint whose
    epoch is a string was not written by the training command, and that is
    exactly the thing worth knowing.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise EvaluationIntegrityError(
            f"{label} must be an integer, got {type(value).__name__} {value!r}. Provenance "
            "metadata is not rounded, truncated or parsed from a string; a value of the "
            "wrong type means the artifact was not written by the canonical command."
        )
    number = int(value)
    if minimum is not None and number < minimum:
        raise EvaluationIntegrityError(f"{label} must be >= {minimum}, got {number}")
    return number


def _require_finite_real(label: str, value: Any) -> float:
    """A genuine finite real number, or refuse. Never coerced.

    ``float()`` would accept ``"0.009"`` and ``True``. NaN matters
    separately: it compares unequal to everything, so a NaN selection value
    would fail the tolerance comparison for the wrong reason and be reported
    as a mismatched checkpoint rather than as corrupt metadata.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise EvaluationIntegrityError(
            f"{label} must be a real number, got {type(value).__name__} {value!r}. "
            "Provenance metadata is not parsed from a string."
        )
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationIntegrityError(f"{label} must be finite, got {value!r}")
    return number


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """A path as written into a tracked artifact: relative to the repository.

    Tracked outputs in this project carry no absolute filesystem path. An
    absolute one would differ between machines, so two runs of an unchanged
    definition would stop producing byte-identical files, and it would put a
    local username into a committed artifact.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def check_sample_alignment(slice_frame: pd.DataFrame, others: dict[str, Path]) -> dict[str, Any]:
    """Did this method score exactly the slices the other methods scored?

    Not a formality: a paired per-patient comparison is only paired if both
    sides ran on the same images in the same order. Returns the report;
    :func:`require_sample_alignment` is what acts on it.
    """
    keys = list(slice_frame["relative_dicom_path"])
    report: dict[str, Any] = {
        "rows": len(keys),
        "unique_rows": len(set(keys)),
        "duplicates": len(keys) - len(set(keys)),
        "references": sorted(others),
    }
    for name, path in others.items():
        other = list(pd.read_csv(path)["relative_dicom_path"])
        paired = sum(1 for a, b in zip(keys, other, strict=False) if a != b)
        report[name] = {
            "reference_rows": len(other),
            "missing": len(set(other) - set(keys)),
            "extra": len(set(keys) - set(other)),
            # zip() stops at the shorter list, so a length difference would
            # otherwise hide every unpaired row. Count those as mismatches.
            "order_mismatches": paired + abs(len(keys) - len(other)),
            "identical_order": keys == other,
        }
    return report


def require_sample_alignment(
    report: dict[str, Any], expected_rows: int, method: str = "this method"
) -> None:
    """Refuse the evaluation unless every paired comparison is genuinely paired.

    A report nobody acts on is not a check. Each condition below, left
    unenforced, produces per-patient deltas that subtract one slice's metric
    from a different slice's metric while looking perfectly well-formed on
    disk.

    Raises:
        EvaluationIntegrityError: any alignment condition failed.
    """
    failures: list[str] = []
    if report["rows"] != expected_rows:
        failures.append(f"scored {report['rows']} rows, expected {expected_rows}")
    if report["unique_rows"] != expected_rows:
        failures.append(f"{report['unique_rows']} unique sample keys, expected {expected_rows}")
    if report["duplicates"]:
        failures.append(f"{report['duplicates']} duplicate sample keys")

    for name in report["references"]:
        entry = report[name]
        for field in ("missing", "extra", "order_mismatches"):
            if entry[field]:
                failures.append(f"{name}: {entry[field]} {field.replace('_', ' ')}")
        if not entry["identical_order"]:
            failures.append(f"{name}: sample keys are not in identical order")

    if failures:
        raise EvaluationIntegrityError(
            f"canonical evaluation refused - {method} did not score the same slices, in the "
            "same order, as the methods it would be compared against: " + "; ".join(failures)
        )


def verify_checkpoint_provenance(
    payload: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    run_dir: Path,
    method: str = "model",
    trainer: str = "the canonical training command",
    expected_seed: int = CANONICAL_SEED,
) -> dict[str, Any]:
    """Refuse to score a checkpoint that is not the frozen canonical run's.

    Reading a checkpoint's own metadata and copying it into the report proves
    nothing: a file can say anything about itself. Every field here is checked
    against something computed independently - the config file's bytes, the
    checkpoint file's bytes, the tracked training history re-run through the
    predeclared selection rule, and the tracked run summary.

    The gate runs before a single validation image is opened, so a wrong
    checkpoint costs an error message rather than a plausible-looking table.

    Architecture-agnostic by construction: it reads files, not models.
    ``method`` and ``trainer`` only name things in the messages and the
    record, so a reader can tell which method was verified.

    Raises:
        EvaluationIntegrityError: the checkpoint is not the canonical one, or
            the run it claims to come from is not internally consistent.
    """
    required = ("config_sha256", "seed", "selection_metric", "selection_value", "epoch")
    absent = [field for field in required if field not in payload]
    if absent:
        raise EvaluationIntegrityError(
            "checkpoint carries no provenance for " + ", ".join(absent) + f". It was not "
            f"written by {trainer} and cannot be identified as the canonical run."
        )

    summary_path = run_dir / "run_summary.json"
    history_path = run_dir / "training_history.csv"
    for path in (summary_path, history_path):
        if not path.exists():
            raise EvaluationIntegrityError(
                f"{path.as_posix()} is missing; checkpoint provenance cannot be verified "
                "without the tracked record of the run it came from."
            )

    # The checkpoint's own numeric metadata, type-checked rather than coerced.
    checkpoint_seed = _require_integer("checkpoint seed", payload["seed"])
    checkpoint_epoch = _require_integer("checkpoint epoch", payload["epoch"], minimum=0)
    checkpoint_value = _require_finite_real(
        "checkpoint selection_value", payload["selection_value"]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)
    try:
        summary_config_sha = str(summary["config"]["sha256"])
        raw_summary_seed = summary["training"]["seed"]
        selection = summary["checkpoint_selection"]
        summary_metric = str(selection["primary_metric"])
        raw_summary_epoch = selection["best_epoch"]
        raw_summary_value = selection[SUMMARY_BEST_VALUE_KEY]
        summary_checkpoint_sha = str(summary["checkpoint"]["sha256"])
        round_trip = summary["checkpoint"]["round_trip"]
    except (KeyError, TypeError) as error:
        raise EvaluationIntegrityError(
            f"{summary_path.as_posix()} is missing the provenance fields this gate needs "
            f"({error}); it was not written by the canonical training command."
        ) from error

    summary_seed = _require_integer("run summary training seed", raw_summary_seed)
    summary_epoch = _require_integer("run summary best_epoch", raw_summary_epoch, minimum=0)
    summary_value = _require_finite_real(f"run summary {SUMMARY_BEST_VALUE_KEY}", raw_summary_value)

    config_sha = file_sha256(config_path)
    checkpoint_sha = file_sha256(checkpoint_path)

    # Re-derive the selected epoch from the tracked history with the same
    # predeclared rule the training run used, rather than trusting the epoch
    # either the checkpoint or the summary claims.
    recomputed_epoch = select_best_epoch(history, SELECTION_METRIC, epoch_zero_eligible=False)
    recomputed_value = float(
        history.loc[history["epoch"] == recomputed_epoch, SELECTION_METRIC].iloc[0]
    )

    def close(left: float, right: float) -> bool:
        # Both sides are already validated finite reals, so this can never be
        # comparing a parsed string or quietly returning False for a NaN.
        return abs(left - right) <= SELECTION_VALUE_TOLERANCE

    checks = {
        "checkpoint_config_sha_matches_frozen_config": (
            str(payload["config_sha256"]) == config_sha
        ),
        "checkpoint_config_sha_matches_run_summary": (
            str(payload["config_sha256"]) == summary_config_sha
        ),
        "checkpoint_seed_is_canonical": checkpoint_seed == expected_seed,
        "checkpoint_seed_matches_run_summary": checkpoint_seed == summary_seed,
        "run_summary_seed_is_canonical": summary_seed == expected_seed,
        "checkpoint_selection_metric_is_predeclared": (
            str(payload["selection_metric"]) == SELECTION_METRIC
        ),
        "run_summary_selection_metric_is_predeclared": summary_metric == SELECTION_METRIC,
        "checkpoint_epoch_matches_run_summary": checkpoint_epoch == summary_epoch,
        "checkpoint_epoch_matches_recomputed_selection": (
            checkpoint_epoch == int(recomputed_epoch)
        ),
        "checkpoint_selection_value_matches_history": close(checkpoint_value, recomputed_value),
        "checkpoint_selection_value_matches_run_summary": close(checkpoint_value, summary_value),
        "checkpoint_file_sha_matches_run_summary": checkpoint_sha == summary_checkpoint_sha,
        "round_trip_state_dict_clean": (round_trip.get("state_dict_tensor_mismatches") == 0),
        "round_trip_predictions_clean": round_trip.get("probe_prediction_mismatches") == 0,
        "round_trip_reproduces_saved_model": round_trip.get("reproduces_saved_model") is True,
    }

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise EvaluationIntegrityError(
            f"canonical evaluation refused - this checkpoint is not the frozen {method} "
            "run, or that run is not internally consistent. Failed: " + ", ".join(failed)
        )

    return {
        "verified": True,
        "verified_before_any_image_was_scored": True,
        "checks_passed": sorted(checks),
        "checks_failed": failed,
        "recomputed_from_files": {
            "config_sha256": config_sha,
            "checkpoint_file_sha256": checkpoint_sha,
            "selected_epoch_from_training_history": int(recomputed_epoch),
            "selection_value_from_training_history": recomputed_value,
        },
        "selection_rule_reapplied": (
            f"lowest {SELECTION_METRIC}, ties to the earlier epoch, epoch 0 ineligible"
        ),
        "selection_value_tolerance": SELECTION_VALUE_TOLERANCE,
        "sources": {
            "frozen_config": repository_path(config_path),
            "run_summary": summary_path.as_posix(),
            "training_history": history_path.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
        },
        "note": (
            "Each field is checked against something computed independently of the "
            "checkpoint - the config bytes, the checkpoint bytes, and the tracked history "
            "re-run through the predeclared selection rule - not copied from the payload."
        ),
    }
