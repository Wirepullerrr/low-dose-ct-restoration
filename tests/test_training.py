"""Tests for the training loss, the validation metric and checkpoint selection.

Fully synthetic and CPU-only; nothing here needs CHAOS or a GPU.

These three decisions are where a learned result quietly stops being
comparable with the rest of the benchmark. A checkpoint chosen on a
slice-weighted metric, a selection that depends on CSV row order, or a loss
that clamps before measuring would all still train, still produce a number,
and still look entirely healthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ct_restoration.models.cnn import build_model
from ct_restoration.training import (
    EPOCH_ZERO_ELIGIBLE,
    SELECTION_METRIC,
    TrainingError,
    l1_training_loss,
    patient_means,
    patient_weighted_mean,
    select_best_epoch,
    slice_absolute_errors,
    slice_weighted_mean,
)


def history(values: dict[int, float], metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """A training history with one row per epoch, in epoch order."""
    return pd.DataFrame([{"epoch": epoch, metric: value} for epoch, value in values.items()])


# --------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------


def test_the_loss_is_mean_absolute_error() -> None:
    restored = torch.tensor([[[[0.5, 0.2]]]])
    clean = torch.tensor([[[[0.1, 0.4]]]])

    assert float(l1_training_loss(restored, clean)) == pytest.approx((0.4 + 0.2) / 2)


def test_the_loss_does_not_clamp_its_input() -> None:
    """Clamping in the loss would zero the gradient outside [0, 1].

    A pixel predicted at 1.4 against a target of 0.9 must still be charged
    0.5, or nothing pushes it back into range.
    """
    overshooting = torch.tensor([[[[1.4]]]])
    clean = torch.tensor([[[[0.9]]]])

    assert float(l1_training_loss(overshooting, clean)) == pytest.approx(0.5)


def test_the_loss_keeps_a_gradient_outside_the_range() -> None:
    overshooting = torch.tensor([[[[1.4]]]], requires_grad=True)
    l1_training_loss(overshooting, torch.tensor([[[[0.9]]]])).backward()

    assert float(overshooting.grad.abs().sum()) > 0


def test_the_loss_is_zero_for_a_perfect_restoration() -> None:
    image = torch.rand(2, 1, 8, 8)

    assert float(l1_training_loss(image, image)) == 0.0


def test_mismatched_shapes_are_refused() -> None:
    with pytest.raises(TrainingError, match="same shape"):
        l1_training_loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))


# --------------------------------------------------------------------------
# Per-slice validation error
# --------------------------------------------------------------------------


def test_slice_errors_are_one_scalar_per_item() -> None:
    restored = torch.zeros(3, 1, 4, 4)
    clean = torch.ones(3, 1, 4, 4)
    clean[1] *= 0.5

    errors = slice_absolute_errors(restored, clean)

    assert errors.shape == (3,)
    assert errors.tolist() == pytest.approx([1.0, 0.5, 1.0])


def test_slice_errors_refuse_a_non_batched_tensor() -> None:
    with pytest.raises(TrainingError, match=r"B, C, H, W"):
        slice_absolute_errors(torch.zeros(4, 4), torch.zeros(4, 4))


# --------------------------------------------------------------------------
# Patient weighting: the reason the aggregation exists
# --------------------------------------------------------------------------


def test_patient_weighting_differs_from_slice_weighting() -> None:
    """The example the checkpoint rule is pinned to.

    Patient A contributes one slice with MAE 0. Patient B contributes nine
    slices with MAE 1. Pooling slices gives 0.9; weighting patients equally
    gives 0.5. Checkpoint selection must use 0.5.
    """
    values = [0.0] + [1.0] * 9
    subject_ids = ["A"] + ["B"] * 9

    assert slice_weighted_mean(values) == pytest.approx(0.9)
    assert patient_weighted_mean(values, subject_ids) == pytest.approx(0.5)


def test_patient_weighting_is_independent_of_slice_order() -> None:
    values = [0.0] + [1.0] * 9
    subject_ids = ["A"] + ["B"] * 9
    shuffled = list(zip(values, subject_ids, strict=True))
    shuffled.reverse()

    assert patient_weighted_mean(
        [value for value, _ in shuffled], [name for _, name in shuffled]
    ) == pytest.approx(0.5)


def test_patient_means_are_in_canonical_natural_order() -> None:
    means = patient_means([1.0, 2.0, 3.0, 4.0], ["21", "3", "21", "2"])

    assert list(means) == ["2", "3", "21"]
    assert means == pytest.approx({"2": 4.0, "3": 2.0, "21": 2.0})


def test_a_single_patient_averages_to_its_own_mean() -> None:
    assert patient_weighted_mean([0.2, 0.4], ["7", "7"]) == pytest.approx(0.3)


def test_misaligned_values_and_subjects_are_refused() -> None:
    with pytest.raises(TrainingError, match="must align"):
        patient_weighted_mean([0.1, 0.2], ["a"])


def test_an_empty_validation_pass_is_refused() -> None:
    with pytest.raises(TrainingError, match="empty"):
        patient_weighted_mean([], [])


# --------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------


def test_the_lowest_metric_wins() -> None:
    assert select_best_epoch(history({0: 0.02, 1: 0.019, 2: 0.011, 3: 0.015})) == 2


def test_epoch_zero_cannot_be_selected() -> None:
    """The identity sanity check is not a trained model."""
    assert EPOCH_ZERO_ELIGIBLE is False
    assert select_best_epoch(history({0: 0.001, 1: 0.02, 2: 0.03})) == 1


def test_epoch_zero_can_be_selected_when_explicitly_allowed() -> None:
    assert select_best_epoch(history({0: 0.001, 1: 0.02}), epoch_zero_eligible=True) == 0


def test_an_exact_tie_takes_the_earlier_epoch() -> None:
    assert select_best_epoch(history({0: 0.9, 1: 0.05, 2: 0.04, 3: 0.04, 4: 0.06})) == 2


def test_a_tie_at_the_last_epoch_still_takes_the_earlier() -> None:
    assert select_best_epoch(history({0: 0.9, 1: 0.01, 2: 0.05, 3: 0.01})) == 1


@pytest.mark.parametrize("permutation", ["reversed", "shuffled"])
def test_row_order_cannot_change_the_winner(permutation) -> None:
    frame = history({0: 0.9, 1: 0.05, 2: 0.03, 3: 0.04, 4: 0.03})
    reordered = (
        frame.iloc[::-1] if permutation == "reversed" else frame.sample(frac=1, random_state=3)
    )

    assert select_best_epoch(frame) == select_best_epoch(reordered.reset_index(drop=True)) == 2


def test_a_repeated_epoch_is_refused() -> None:
    frame = pd.concat([history({0: 0.9, 1: 0.05}), history({1: 0.01})], ignore_index=True)

    with pytest.raises(TrainingError, match="repeats epoch"):
        select_best_epoch(frame)


def test_a_missing_epoch_is_refused() -> None:
    with pytest.raises(TrainingError, match="skips epoch"):
        select_best_epoch(history({0: 0.9, 1: 0.05, 3: 0.01}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_metric_is_refused(bad) -> None:
    """A diverged epoch is not silently skipped over in favour of another."""
    with pytest.raises(TrainingError, match="non-finite|is nan|is inf|is -inf"):
        select_best_epoch(history({0: 0.9, 1: 0.05, 2: bad}))


def test_a_non_finite_metric_at_epoch_zero_is_ignored_when_ineligible() -> None:
    """Epoch 0 is not a candidate, so its value cannot block selection."""
    assert select_best_epoch(history({0: float("nan"), 1: 0.05, 2: 0.04})) == 2


def test_an_empty_history_is_refused() -> None:
    with pytest.raises(TrainingError, match="empty training history"):
        select_best_epoch(pd.DataFrame())


def test_a_history_with_only_epoch_zero_is_refused() -> None:
    with pytest.raises(TrainingError, match="no epoch is eligible"):
        select_best_epoch(history({0: 0.02}))


def test_a_missing_metric_column_is_refused() -> None:
    with pytest.raises(TrainingError, match="missing the"):
        select_best_epoch(history({0: 0.9, 1: 0.05}).drop(columns=[SELECTION_METRIC]))


@pytest.mark.parametrize("bad", [1.5, "2", True, -1])
def test_a_malformed_epoch_is_refused(bad) -> None:
    frame = pd.DataFrame([{"epoch": bad, SELECTION_METRIC: 0.05}])

    with pytest.raises(TrainingError, match="epoch must be"):
        select_best_epoch(frame)


def test_selection_accepts_a_list_of_records() -> None:
    records = [
        {"epoch": 0, SELECTION_METRIC: 0.9},
        {"epoch": 1, SELECTION_METRIC: 0.05},
        {"epoch": 2, SELECTION_METRIC: 0.04},
    ]

    assert select_best_epoch(records) == 2


def test_the_selection_metric_name_is_pinned() -> None:
    assert SELECTION_METRIC == "validation_patient_weighted_full_mae"


# --------------------------------------------------------------------------
# No evaluation-only quantity reaches the training side
# --------------------------------------------------------------------------


def test_the_training_module_never_mentions_the_body_mask() -> None:
    """A regression guard, not a style check.

    The body mask and the clean HU slice are derived from the clean
    reference. A loss or a checkpoint rule that started consuming either
    would be optimizing against evaluation-only information, and the guard
    that catches that has to be mechanical.
    """
    from pathlib import Path

    import ct_restoration.training as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    # The module docstring explains why these are absent, so only look at code.
    code = source.split('"""', 2)[-1]
    for forbidden in ("body_mask", "evaluation_mask", "clean_hu", "ssim_interior"):
        assert forbidden not in code


def test_the_loss_signature_takes_only_two_images() -> None:
    import inspect

    assert list(inspect.signature(l1_training_loss).parameters) == ["restored_raw", "clean"]
    assert list(inspect.signature(slice_absolute_errors).parameters) == ["restored", "clean"]


# --------------------------------------------------------------------------
# A tiny end-to-end optimization, as an engineering check only
# --------------------------------------------------------------------------


def synthetic_pairs(count: int = 16, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean images and a fixed deterministic perturbation of them.

    Not a scientific result and not evidence about the benchmark: a toy
    problem that a working training loop must be able to reduce the loss on.
    """
    generator = torch.Generator().manual_seed(17)
    clean = torch.rand(count, 1, size, size, generator=generator, dtype=torch.float32)
    perturbation = 0.15 * torch.sin(
        torch.linspace(0, 6.0, size * size, dtype=torch.float32)
    ).reshape(1, 1, size, size)
    degraded = torch.clamp(clean + perturbation, 0.0, 1.0)
    return clean, degraded


def test_a_tiny_training_run_reduces_the_loss() -> None:
    clean, degraded = synthetic_pairs()
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    first = float(l1_training_loss(model(degraded), clean).detach())

    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = l1_training_loss(model(degraded), clean)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(value) for value in losses)
    assert losses[-1] < first * 0.8
    assert any(not torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_an_optimizer_step_changes_the_parameters() -> None:
    clean, degraded = synthetic_pairs(count=2, size=8)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    optimizer.zero_grad(set_to_none=True)
    l1_training_loss(model(degraded), clean).backward()
    optimizer.step()

    changed = [
        name for name, tensor in model.state_dict().items() if not torch.equal(before[name], tensor)
    ]
    assert changed


def test_the_loss_stays_finite_on_degenerate_input() -> None:
    model = build_model()
    for image in (torch.zeros(2, 1, 8, 8), torch.ones(2, 1, 8, 8)):
        assert np.isfinite(float(l1_training_loss(model(image), image).detach()))
