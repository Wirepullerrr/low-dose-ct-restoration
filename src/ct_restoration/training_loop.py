"""The model-agnostic parts of running a training experiment.

Two architectures are trained by this project under a deliberately identical
policy: same data, same sampler, same loss, same optimizer, same seed, same
epoch budget, same checkpoint-selection rule. The only intended difference is
the model itself.

This module is how that stays true. Every helper here takes a model and calls
only ``model(...)``, ``model.restore(...)`` and ``model.state_dict()``, so
there is one epoch loop, one validation pass, one checkpoint format and one
round-trip check rather than a copy per architecture. Two copies would drift,
and a drifted training pipeline turns an architecture comparison into a
comparison of two slightly different experiments.

What is *not* here: anything that decides science. The loss, the aggregation
and the epoch-selection rule live in :mod:`ct_restoration.training`; the
frozen data policy lives in the Milestone 7 Dataset, sampler and loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ct_restoration.evaluation_integrity import file_sha256
from ct_restoration.training import (
    SELECTION_METRIC,
    TrainingError,
    l1_training_loss,
    patient_means,
    patient_weighted_mean,
    slice_absolute_errors,
    slice_weighted_mean,
)

#: Seed for the synthetic tensors ``checkpoint_round_trip`` predicts on.
#:
#: NOT a training seed, despite sharing the number with the canonical one.
#: It controls only deterministic verification probes: random images used to
#: check that a reloaded checkpoint predicts what the in-memory model
#: predicts. It is deliberately independent of the statistical training seed,
#: and must stay fixed when that seed changes - a multi-seed comparison needs
#: every run's round-trip check to use identical probe inputs, or a
#: difference in the check would say nothing about the checkpoint.
CHECKPOINT_ROUND_TRIP_PROBE_SEED = 2026

__all__ = [
    "CHECKPOINT_ROUND_TRIP_PROBE_SEED",
    "baseline_patient_weighted_full_mae",
    "checkpoint_round_trip",
    "evaluate_validation",
    "file_sha256",
    "save_checkpoint",
    "train_one_epoch",
]


def baseline_patient_weighted_full_mae(patients_csv: Path) -> float:
    """The committed degraded-baseline figure the identity check compares to.

    Loaded from the tracked Milestone 5 output rather than hard-coded: a
    rounded literal in this file would be a second source of truth, and the
    check would start passing against the wrong number the moment either
    drifted.
    """
    if not patients_csv.exists():
        raise TrainingError(
            f"{patients_csv.as_posix()} is missing; the epoch-0 identity check compares "
            "against the committed degraded baseline. Run evaluate_degraded_baseline first."
        )
    frame = pd.read_csv(patients_csv, dtype={"subject_id": str})
    return float(frame["mean_full_mae"].mean())


@torch.inference_mode()
def evaluate_validation(model, loader, device) -> dict[str, Any]:
    """Patient-weighted validation full-frame MAE from the clamped prediction.

    The only quantity computed per epoch. The full eight-metric benchmark runs
    once, later, on the selected checkpoint: running SSIM 31 times to decide
    something that was predeclared as MAE would be wasted work and an
    invitation to peek at metrics the rule does not use.
    """
    model.eval()
    errors: list[float] = []
    subject_ids: list[str] = []
    for batch in loader:
        degraded = batch["degraded"].to(device=device, non_blocking=False)
        clean = batch["clean"].to(device=device, non_blocking=False)
        restored = model.restore(degraded)
        errors.extend(float(value) for value in slice_absolute_errors(restored, clean).cpu())
        subject_ids.extend(str(value) for value in batch["subject_id"])

    return {
        "slices": len(errors),
        "patient_weighted_full_mae": patient_weighted_mean(errors, subject_ids),
        "slice_weighted_full_mae": slice_weighted_mean(errors),
        "per_patient_full_mae": patient_means(errors, subject_ids),
    }


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> dict[str, Any]:
    """One patient-balanced pass. Returns the mean raw L1 over the epoch."""
    # The sampler owns ordering, and it must be told the epoch or every epoch
    # would draw the identical sequence.
    loader.sampler.set_epoch(epoch)

    model.train()
    total = 0.0
    seen = 0
    for batch in loader:
        degraded = batch["degraded"].to(device=device)
        clean = batch["clean"].to(device=device)

        optimizer.zero_grad(set_to_none=True)
        restored_raw = model(degraded)
        loss = l1_training_loss(restored_raw, clean)
        loss.backward()
        optimizer.step()

        count = degraded.shape[0]
        total += float(loss.detach()) * count
        seen += count

    if seen == 0:
        raise TrainingError(f"epoch {epoch} drew no samples at all")
    return {"train_raw_l1": total / seen, "draws": seen}


def save_checkpoint(
    model, path: Path, epoch: int, seed: int, config_sha: str, value: float
) -> Path:
    """A plain dictionary of tensors, not a pickled class instance.

    Keeps the file loadable with ``weights_only=True``, which means restoring
    it never executes arbitrary code from the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()
            },
            "epoch": int(epoch),
            "seed": int(seed),
            "config_sha256": str(config_sha),
            "selection_metric": SELECTION_METRIC,
            "selection_value": float(value),
        },
        path,
    )
    return path


@torch.inference_mode()
def checkpoint_round_trip(model, path: Path, device, rebuild) -> dict[str, Any]:
    """Reload the saved checkpoint and confirm it reproduces the saved model.

    A checkpoint that cannot reproduce the model it was written from is a
    failed run, however good the validation curve looked: every number
    reported later comes from reloading this file, not from the process that
    trained it.

    Checked two ways - tensor equality of the state dict, and prediction
    equality on deterministic probes - because a missing buffer or a silently
    downcast tensor can pass one and fail the other.

    ``rebuild`` is the architecture's own ``build_model``: this helper is
    shared between the CNN and the U-Net and never needs to know which it
    holds.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    revived = rebuild(model.config, device)
    revived.load_state_dict(payload["model_state_dict"])
    revived.eval()
    model.eval()

    saved = model.state_dict()
    tensor_mismatches = sum(
        0 if torch.equal(saved[name].to("cpu"), tensor.to("cpu")) else 1
        for name, tensor in revived.state_dict().items()
    )

    generator = torch.Generator().manual_seed(CHECKPOINT_ROUND_TRIP_PROBE_SEED)
    probes = torch.rand(4, 1, 256, 256, generator=generator, dtype=torch.float32).to(device)
    prediction_mismatches = int((model.restore(probes) != revived.restore(probes)).sum())

    return {
        "state_dict_keys": len(saved),
        "state_dict_tensor_mismatches": tensor_mismatches,
        "probe_tensors": int(probes.shape[0]),
        "probe_prediction_mismatches": prediction_mismatches,
        "reproduces_saved_model": bool(tensor_mismatches == 0 and prediction_mismatches == 0),
        "loaded_with_weights_only": True,
    }
