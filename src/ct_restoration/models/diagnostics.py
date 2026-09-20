"""What a learned model's raw output looks like before the clamp.

Purely descriptive, and shared by every learned method so that the residual
CNN and the lightweight U-Net are described by one definition rather than two
that drift. Nothing computed here is an optimization objective, and no model
or config decision may be made from it.

The distinction this module exists to keep straight
---------------------------------------------------
Both architectures are defined as ``raw = degraded + correction`` with
``restored = clamp(raw, 0, 1)``, so there are two different "corrections":

* the **predicted correction**, ``raw - degraded`` - what the network asked
  for, measured on the unclamped output;
* the **post-clamp change**, ``restored - degraded`` - the part of that
  request which survived into the image the metrics were computed on.

They differ wherever the raw output leaves [0, 1]. Reporting either under the
other's name misstates how hard the model is pushing, which is exactly the
defect this module was extracted to stop recurring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like
from ct_restoration.evaluation import EvaluationConfig, prepare_evaluation_slice
from ct_restoration.models.adapter import raw_restore_array, restore_array

#: Quantiles reported for the per-slice diagnostics.
QUANTILES = (0.05, 0.5, 0.95)

#: The only label the raw ``raw - degraded`` image may be displayed under.
PREDICTED_CORRECTION_LABEL = "predicted correction (raw - degraded)"

#: The only label the clamped ``restored - degraded`` image may be shown under.
POST_CLAMP_CHANGE_LABEL = "post-clamp change (clamped - degraded)"


def predicted_correction_panel(
    model, degraded: np.ndarray, device: torch.device | str = "cpu"
) -> tuple[str, np.ndarray]:
    """The RAW predicted correction, together with the label it must carry.

    Returned as a pair on purpose. The two correction quantities differ only
    where the clamp engages, so a figure showing one under the other's name
    looks entirely plausible and is simply wrong - which is exactly what
    happened to the first CNN and U-Net QC panels. Binding the image to its
    label means a caller cannot pick the wrong one without saying so.
    """
    raw = raw_restore_array(model, degraded, device)
    return PREDICTED_CORRECTION_LABEL, raw - degraded


def post_clamp_change_panel(
    model, degraded: np.ndarray, device: torch.device | str = "cpu"
) -> tuple[str, np.ndarray]:
    """The part of the correction that survived the clamp, and its label.

    This is what the reported metrics are computed from; the predicted
    correction is what the network asked for.
    """
    restored = restore_array(model, degraded, device)
    return POST_CLAMP_CHANGE_LABEL, restored - degraded


def raw_output_diagnostics(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    model,
    device: torch.device,
) -> dict[str, Any]:
    """How much work the final clamp is doing, and how large the correction is.

    Two different quantities, deliberately not merged. The model is defined as
    ``raw = degraded + predicted_correction`` and ``restored = clamp(raw)``,
    so:

    * the **predicted correction** is ``raw - degraded`` - what the network
      actually asked for, which is what says how hard it is pushing;
    * the **post-clamp change** is ``restored - degraded`` - what survived
      into the reported image.

    Wherever the raw output leaves [0, 1] these differ, and on this model that
    is around half the frame, so reporting one under the other's name would
    understate the correction by a wide margin.

    Descriptive only. None of this becomes an objective: the point is to know
    whether clamping is a rare safety boundary or a structural part of the
    method, which changes how the reported metrics should be read.
    """
    pixels = 0
    below = above = clamp_changed = 0
    predicted_sum = 0.0
    post_clamp_sum = 0.0
    per_slice_predicted: list[float] = []
    per_slice_post_clamp: list[float] = []
    raw_minimum = float("inf")
    raw_maximum = float("-inf")
    finite_failures = 0

    for row in rows.itertuples():
        key = row.relative_dicom_path
        clean, _, _ = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation).astype(np.float64)

        raw = raw_restore_array(model, degraded, device).astype(np.float64)
        restored = restore_array(model, degraded, device).astype(np.float64)
        if not np.all(np.isfinite(raw)):
            finite_failures += 1

        pixels += raw.size
        below += int((raw < 0.0).sum())
        above += int((raw > 1.0).sum())
        clamp_changed += int((raw != restored).sum())
        raw_minimum = min(raw_minimum, float(raw.min()))
        raw_maximum = max(raw_maximum, float(raw.max()))

        # The predicted correction comes from the RAW output. Taking it from
        # the clamped one would silently report the post-clamp change instead.
        predicted = np.abs(raw - degraded)
        post_clamp = np.abs(restored - degraded)
        predicted_sum += float(predicted.sum())
        post_clamp_sum += float(post_clamp.sum())
        per_slice_predicted.append(float(predicted.mean()))
        per_slice_post_clamp.append(float(post_clamp.mean()))

    def quantiles(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}

    return {
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
        "fraction_raw_below_zero": below / pixels,
        "fraction_raw_above_one": above / pixels,
        "fraction_changed_by_clamp": clamp_changed / pixels,
        "mean_absolute_predicted_correction": predicted_sum / pixels,
        "per_slice_mean_absolute_predicted_correction": quantiles(per_slice_predicted),
        "mean_absolute_post_clamp_change_vs_degraded": post_clamp_sum / pixels,
        "per_slice_mean_absolute_post_clamp_change": quantiles(per_slice_post_clamp),
        "finite_value_failures": finite_failures,
        "definitions": {
            "predicted_correction": "abs(raw - degraded), from the UNCLAMPED model output",
            "post_clamp_change": "abs(clamp(raw, 0, 1) - degraded), what reached the metrics",
            "why_both": (
                "They differ wherever the raw output leaves [0, 1]. Reporting either "
                "one under the other's name misstates how hard the network is pushing."
            ),
        },
        "note": (
            "Descriptive diagnostics of the clamp and the predicted correction. Reported "
            "metrics use the clamped image; nothing here is an optimization objective."
        ),
    }
