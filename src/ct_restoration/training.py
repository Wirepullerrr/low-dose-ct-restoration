"""Training-side policy: the loss, the validation metric, and how a
checkpoint is chosen.

Kept here, in tested library code rather than inside a script, because these
three decisions are where a learned result quietly stops being comparable
with the rest of the benchmark. All of them are pure functions of their
inputs, so they can be exercised on synthetic tables where the right answer is
known by construction.

The loss
--------
Plain full-frame L1 on the **raw** restoration, ``degraded + correction``,
against the clean target. Clamping before the loss would put a zero-gradient
region wherever a prediction left [0, 1], so a pixel that overshot would stop
receiving any signal to come back.

Full-frame and unweighted is a deliberate choice, not a claim that it is
optimal. A body-weighted loss would need the evaluation body mask, which is
derived from the clean reference, so the training objective would start
consuming an evaluation-only quantity. An SSIM or composite loss would
optimize directly for a reported metric. Both are legitimate experiments and
both are different experiments; this is the first learned baseline and it uses
the simplest direct supervised objective there is.

The checkpoint metric
---------------------
One predeclared criterion: **patient-weighted validation full-frame MAE**,
computed from the clamped prediction, lower is better. Not a scan across
PSNR, SSIM, body and full - choosing the checkpoint on whichever metric looks
best is the same error as choosing the metric after seeing the table.

It is patient-weighted because patient is this project's unit of analysis
everywhere else, and it is full-frame MAE because that is what training
optimizes. Aligned with the objective, and decided in advance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from ct_restoration.data.splits import subject_sort_key

#: The predeclared checkpoint-selection metric column.
SELECTION_METRIC = "validation_patient_weighted_full_mae"

#: Epoch 0 is the zero-initialized identity sanity check, recorded so the
#: evaluation path can be verified against the frozen degraded baseline. It is
#: not a trained model and cannot be selected as one.
EPOCH_ZERO_ELIGIBLE = False


class TrainingError(ValueError):
    """A training-side table or argument is not usable as given."""


def l1_training_loss(restored_raw: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over every pixel of the raw restoration.

    Deliberately takes the unclamped restoration. See the module docstring.
    """
    if restored_raw.shape != clean.shape:
        raise TrainingError(
            f"restored and clean must have the same shape, got {tuple(restored_raw.shape)} "
            f"and {tuple(clean.shape)}"
        )
    return torch.mean(torch.abs(restored_raw - clean))


def slice_absolute_errors(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Per-slice full-frame MAE for a batch: one scalar per item.

    Takes the **clamped** restoration: this feeds the checkpoint metric, which
    has to describe the image the benchmark would actually score.
    """
    if restored.shape != clean.shape:
        raise TrainingError(
            f"restored and clean must have the same shape, got {tuple(restored.shape)} "
            f"and {tuple(clean.shape)}"
        )
    if restored.dim() != 4:
        raise TrainingError(f"expected [B, C, H, W], got {tuple(restored.shape)}")
    return torch.abs(restored - clean).flatten(start_dim=1).mean(dim=1)


def patient_weighted_mean(values: Sequence[float], subject_ids: Sequence[str]) -> float:
    """Average within each patient first, then across patients equally.

    This is the whole reason the function exists. Pooling slices would weight
    a 294-slice scan more than a 78-slice one, which is the unit the rest of
    this project deliberately does not use.

    Raises:
        TrainingError: the inputs are empty or of different lengths.
    """
    if len(values) != len(subject_ids):
        raise TrainingError(
            f"got {len(values)} values for {len(subject_ids)} subject ids; they must align"
        )
    if not len(values):
        raise TrainingError("cannot average an empty set of slices")

    totals: dict[str, list[float]] = {}
    for value, subject_id in zip(values, subject_ids, strict=True):
        totals.setdefault(str(subject_id), []).append(float(value))
    per_patient = [float(np.mean(scores)) for _, scores in sorted(totals.items())]
    return float(np.mean(per_patient))


def slice_weighted_mean(values: Sequence[float]) -> float:
    """Pool every slice equally. Descriptive only; never the selection metric."""
    if not len(values):
        raise TrainingError("cannot average an empty set of slices")
    return float(np.mean([float(value) for value in values]))


def patient_means(values: Sequence[float], subject_ids: Sequence[str]) -> dict[str, float]:
    """Each patient's mean, in canonical natural subject order."""
    totals: dict[str, list[float]] = {}
    for value, subject_id in zip(values, subject_ids, strict=True):
        totals.setdefault(str(subject_id), []).append(float(value))
    return {
        subject_id: float(np.mean(totals[subject_id]))
        for subject_id in sorted(totals, key=subject_sort_key)
    }


def select_best_epoch(
    history: pd.DataFrame | Sequence[Mapping[str, Any]],
    metric: str = SELECTION_METRIC,
    epoch_zero_eligible: bool = EPOCH_ZERO_ELIGIBLE,
) -> int:
    """The predeclared checkpoint rule: lowest metric, earliest epoch on a tie.

    Reads only values, never row position, so a reordered history file cannot
    change which checkpoint was selected.

    Raises:
        TrainingError: the history is empty, missing the epoch or metric
            column, repeats or skips an epoch, carries a non-finite metric
            among the eligible rows, or leaves no eligible epoch at all.
    """
    frame = pd.DataFrame(list(history)) if not isinstance(history, pd.DataFrame) else history
    if frame.empty:
        raise TrainingError("cannot select a checkpoint from an empty training history")
    for column in ("epoch", metric):
        if column not in frame.columns:
            raise TrainingError(f"training history is missing the {column!r} column")

    epochs = [_require_epoch(value) for value in frame["epoch"]]
    if len(set(epochs)) != len(epochs):
        repeats = sorted({value for value in epochs if epochs.count(value) > 1})
        raise TrainingError(f"training history repeats epoch(s): {repeats}")
    if sorted(epochs) != list(range(min(epochs), max(epochs) + 1)):
        gaps = sorted(set(range(min(epochs), max(epochs) + 1)) - set(epochs))
        raise TrainingError(f"training history skips epoch(s): {gaps}")

    eligible = [
        (epoch, value)
        for epoch, value in zip(epochs, frame[metric], strict=True)
        if epoch_zero_eligible or epoch != 0
    ]
    if not eligible:
        raise TrainingError(
            "no epoch is eligible for selection; epoch 0 is the identity sanity check and "
            "cannot be chosen as a trained checkpoint."
        )

    scores: list[tuple[int, float]] = []
    for epoch, value in eligible:
        score = _require_finite(metric, epoch, value)
        scores.append((epoch, score))

    # Sort on (metric, epoch): the second key is the predeclared tie-break, so
    # an exact tie deterministically takes the earlier epoch.
    best_epoch, _ = min(scores, key=lambda item: (item[1], item[0]))
    return int(best_epoch)


def _require_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TrainingError(f"epoch must be an integer, got {type(value).__name__} {value!r}")
    if int(value) < 0:
        raise TrainingError(f"epoch must be non-negative, got {value!r}")
    return int(value)


def _require_finite(metric: str, epoch: int, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TrainingError(
            f"{metric} at epoch {epoch} must be a real number, got {type(value).__name__} {value!r}"
        )
    score = float(value)
    if not np.isfinite(score):
        raise TrainingError(
            f"{metric} at epoch {epoch} is {value!r}. A non-finite validation metric means "
            "the run diverged; it is not silently skipped over in favour of another epoch."
        )
    return score
