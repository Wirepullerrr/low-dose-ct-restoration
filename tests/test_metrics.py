"""Tests for the shared image-quality metrics.

Fully synthetic. The metrics are pure functions of two arrays and a mask, so
nothing here needs the CHAOS download.

The formulas are simple enough that the real risk is not arithmetic but
contract: an input silently resized, an array normalized behind the caller's
back, or a masked metric quietly influenced by pixels outside its mask. Most of
what follows tests those.
"""

from __future__ import annotations

import numpy as np
import pytest

from ct_restoration.metrics import (
    DATA_RANGE,
    MetricError,
    SsimSettings,
    masked_ssim_map_mean,
    mean_absolute_error,
    mean_squared_error,
    peak_signal_noise_ratio,
    slice_metrics,
    structural_similarity_map,
    validate_mask,
    validate_pair,
)


@pytest.fixture
def settings() -> SsimSettings:
    return SsimSettings()


@pytest.fixture
def reference() -> np.ndarray:
    """A deterministic textured image in [0, 1], built without global RNG."""
    rows, columns = np.meshgrid(np.linspace(0.1, 0.9, 64), np.linspace(0.0, 0.5, 64), indexing="ij")
    image = (rows + columns) / 2.0
    centre = (np.arange(64) - 32.0) ** 2
    image[centre[:, None] + centre[None, :] < 150] = 0.85
    return image.astype(np.float32)


@pytest.fixture
def mask() -> np.ndarray:
    """A central square, comfortably larger than an 11x11 SSIM window."""
    selected = np.zeros((64, 64), dtype=bool)
    selected[16:48, 16:48] = True
    return selected


# --------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------


def test_identical_images_have_zero_error(reference) -> None:
    assert mean_absolute_error(reference, reference) == 0.0
    assert mean_squared_error(reference, reference) == 0.0
    assert peak_signal_noise_ratio(reference, reference) == float("inf")


def test_identical_images_have_unit_ssim(reference, settings) -> None:
    mean_ssim, ssim_map = structural_similarity_map(reference, reference, settings)

    assert mean_ssim == pytest.approx(1.0)
    assert ssim_map.min() == pytest.approx(1.0)


def test_a_different_image_has_ssim_below_one(reference, settings) -> None:
    other = np.clip(reference.astype(np.float64) + 0.05, 0.0, 1.0)

    mean_ssim, _ = structural_similarity_map(reference, other, settings)

    assert mean_ssim < 1.0


def test_a_constant_offset_gives_the_expected_mae_and_mse() -> None:
    base = np.full((32, 32), 0.40, dtype=np.float64)
    shifted = np.full((32, 32), 0.55, dtype=np.float64)

    assert mean_absolute_error(base, shifted) == pytest.approx(0.15)
    assert mean_squared_error(base, shifted) == pytest.approx(0.15**2)


def test_psnr_matches_the_explicit_formula() -> None:
    base = np.full((16, 16), 0.2, dtype=np.float64)
    shifted = np.full((16, 16), 0.3, dtype=np.float64)
    expected = 10.0 * np.log10(DATA_RANGE**2 / 0.1**2)

    assert peak_signal_noise_ratio(base, shifted) == pytest.approx(expected)


def test_psnr_rises_as_error_falls() -> None:
    base = np.full((16, 16), 0.5, dtype=np.float64)
    near = np.full((16, 16), 0.51, dtype=np.float64)
    far = np.full((16, 16), 0.60, dtype=np.float64)

    assert peak_signal_noise_ratio(base, near) > peak_signal_noise_ratio(base, far)


def test_mae_and_mse_are_symmetric(reference) -> None:
    other = np.clip(reference.astype(np.float64) * 0.9, 0.0, 1.0)

    assert mean_absolute_error(reference, other) == mean_absolute_error(other, reference)
    assert mean_squared_error(reference, other) == mean_squared_error(other, reference)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_pixels_outside_the_mask_cannot_affect_masked_metrics(reference, mask) -> None:
    """The point of a region metric: only the region is measured."""
    prediction = np.clip(reference.astype(np.float64) + 0.02, 0.0, 1.0)
    before = (
        mean_absolute_error(reference, prediction, mask),
        mean_squared_error(reference, prediction, mask),
        peak_signal_noise_ratio(reference, prediction, mask),
    )

    vandalised = prediction.copy()
    vandalised[~mask] = 1.0 - vandalised[~mask]
    after = (
        mean_absolute_error(reference, vandalised, mask),
        mean_squared_error(reference, vandalised, mask),
        peak_signal_noise_ratio(reference, vandalised, mask),
    )

    assert before == after


def test_a_full_true_mask_reproduces_the_unmasked_metric(reference) -> None:
    prediction = np.clip(reference.astype(np.float64) + 0.03, 0.0, 1.0)
    everything = np.ones_like(reference, dtype=bool)

    assert mean_absolute_error(reference, prediction, everything) == pytest.approx(
        mean_absolute_error(reference, prediction)
    )
    assert mean_squared_error(reference, prediction, everything) == pytest.approx(
        mean_squared_error(reference, prediction)
    )


def test_masked_metrics_measure_only_the_selected_pixels() -> None:
    base = np.zeros((10, 10), dtype=np.float64)
    prediction = np.zeros((10, 10), dtype=np.float64)
    prediction[:5] = 0.4  # error only in the top half
    top_half = np.zeros((10, 10), dtype=bool)
    top_half[:5] = True

    assert mean_absolute_error(base, prediction, top_half) == pytest.approx(0.4)
    assert mean_absolute_error(base, prediction, ~top_half) == pytest.approx(0.0)
    assert mean_absolute_error(base, prediction) == pytest.approx(0.2)


def test_body_ssim_averages_the_map_over_the_interior(reference, mask, settings) -> None:
    prediction = np.clip(reference.astype(np.float64) + 0.04, 0.0, 1.0)
    _, ssim_map = structural_similarity_map(reference, prediction, settings)

    from ct_restoration.evaluation import ssim_interior_mask

    interior = ssim_interior_mask(mask, settings.win_size)
    body_ssim = masked_ssim_map_mean(ssim_map, interior)

    assert body_ssim == pytest.approx(float(ssim_map[interior].mean()))


def test_masked_ssim_rejects_an_empty_interior(reference, settings) -> None:
    _, ssim_map = structural_similarity_map(reference, reference, settings)

    with pytest.raises(MetricError, match="selects no pixels"):
        masked_ssim_map_mean(ssim_map, np.zeros_like(ssim_map, dtype=bool))


# --------------------------------------------------------------------------
# SSIM settings
# --------------------------------------------------------------------------


def test_default_ssim_settings_match_the_frozen_policy(settings) -> None:
    assert settings.win_size == 11
    assert settings.gaussian_weights is True
    assert settings.sigma == 1.5
    assert settings.use_sample_covariance is False
    assert settings.data_range == 1.0
    assert settings.border == 5


def test_the_pinned_ssim_settings_are_actually_used(reference) -> None:
    """A different window must give a different number, proving it is passed."""
    prediction = np.clip(reference.astype(np.float64) + 0.05, 0.0, 1.0)

    pinned, _ = structural_similarity_map(reference, prediction, SsimSettings())
    wider, _ = structural_similarity_map(reference, prediction, SsimSettings(win_size=21))

    assert pinned != wider


@pytest.mark.parametrize("win_size", [2, 10, 1, 0, -3])
def test_invalid_ssim_windows_are_rejected(win_size) -> None:
    with pytest.raises(MetricError, match="odd integer"):
        SsimSettings(win_size=win_size)


@pytest.mark.parametrize("value", [0.0, -1.5, float("nan"), float("inf")])
def test_invalid_ssim_sigma_is_rejected(value) -> None:
    with pytest.raises(MetricError, match="finite and positive"):
        SsimSettings(sigma=value)


def test_non_boolean_ssim_flags_are_rejected() -> None:
    with pytest.raises(MetricError, match="gaussian_weights must be a bool"):
        SsimSettings(gaussian_weights=1)
    with pytest.raises(MetricError, match="use_sample_covariance must be a bool"):
        SsimSettings(use_sample_covariance="no")


def test_an_image_smaller_than_the_ssim_window_is_rejected(settings) -> None:
    tiny = np.full((5, 5), 0.5, dtype=np.float64)

    with pytest.raises(MetricError, match="smaller than the SSIM window"):
        structural_similarity_map(tiny, tiny, settings)


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(MetricError, match="same shape"):
        mean_absolute_error(np.zeros((8, 8)), np.zeros((8, 9)))


@pytest.mark.parametrize("shape", [(16,), (4, 4, 3), (1, 8, 8), ()])
def test_non_2d_images_are_rejected(shape) -> None:
    image = np.full(shape, 0.5)

    with pytest.raises(MetricError, match="2D image"):
        mean_absolute_error(image, image)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_images_are_rejected(bad, reference) -> None:
    broken = reference.astype(np.float64)
    broken[2, 2] = bad

    with pytest.raises(MetricError, match="must be finite"):
        mean_absolute_error(reference, broken)


@pytest.mark.parametrize("value", [-0.5, 1.5, 255.0])
def test_grossly_out_of_range_images_are_rejected(value, reference) -> None:
    """Metrics do not normalize; an unnormalized array is an upstream bug."""
    broken = reference.astype(np.float64)
    broken[0, 0] = value

    with pytest.raises(MetricError, match=r"must lie in \[0.0, 1.0\]"):
        mean_absolute_error(reference, broken)


def test_tiny_numerical_overshoot_is_tolerated(reference) -> None:
    nudged = reference.astype(np.float64)
    nudged[0, 0] = -1e-9
    nudged[0, 1] = 1.0 + 1e-9

    assert mean_absolute_error(reference, nudged) >= 0.0


def test_an_empty_image_is_rejected() -> None:
    with pytest.raises(MetricError, match="non-empty"):
        mean_absolute_error(np.zeros((0, 4)), np.zeros((0, 4)))


def test_complex_images_are_rejected() -> None:
    image = np.full((8, 8), 0.5, dtype=np.complex128)

    with pytest.raises(MetricError, match="real numeric"):
        mean_absolute_error(image, image)


def test_a_mask_of_the_wrong_shape_is_rejected(reference) -> None:
    with pytest.raises(MetricError, match="does not match image shape"):
        mean_absolute_error(reference, reference, np.ones((8, 8), dtype=bool))


def test_an_empty_mask_is_rejected(reference) -> None:
    with pytest.raises(MetricError, match="selects no pixels"):
        mean_absolute_error(reference, reference, np.zeros_like(reference, dtype=bool))


def test_a_non_binary_mask_is_rejected(reference) -> None:
    with pytest.raises(MetricError, match="must be boolean or 0/1"):
        validate_mask(np.full(reference.shape, 2), reference.shape)


def test_an_integer_zero_one_mask_is_accepted(reference, mask) -> None:
    integer_mask = mask.astype(np.uint8)

    assert mean_absolute_error(reference, reference, integer_mask) == 0.0


# --------------------------------------------------------------------------
# Inputs are never modified
# --------------------------------------------------------------------------


def test_metrics_do_not_modify_their_inputs(reference, mask, settings) -> None:
    prediction = np.clip(reference.astype(np.float64) + 0.05, 0.0, 1.0)
    reference_before = reference.copy()
    prediction_before = prediction.copy()
    mask_before = mask.copy()

    from ct_restoration.evaluation import ssim_interior_mask

    interior = ssim_interior_mask(mask, settings.win_size)
    slice_metrics(reference, prediction, mask, interior, settings)

    assert np.array_equal(reference, reference_before)
    assert np.array_equal(prediction, prediction_before)
    assert np.array_equal(mask, mask_before)


def test_validate_pair_returns_copies_not_views(reference) -> None:
    validated_reference, validated_prediction = validate_pair(reference, reference)

    assert not np.shares_memory(validated_reference, reference)
    assert not np.shares_memory(validated_prediction, reference)


# --------------------------------------------------------------------------
# The combined per-slice call
# --------------------------------------------------------------------------


def test_slice_metrics_returns_all_eight_numbers(reference, mask, settings) -> None:
    from ct_restoration.evaluation import ssim_interior_mask

    prediction = np.clip(reference.astype(np.float64) + 0.03, 0.0, 1.0)
    interior = ssim_interior_mask(mask, settings.win_size)

    measured = slice_metrics(reference, prediction, mask, interior, settings)

    assert set(measured) == {
        "full_mae",
        "full_mse",
        "full_psnr",
        "full_ssim",
        "body_mae",
        "body_mse",
        "body_psnr",
        "body_ssim",
    }
    assert all(isinstance(value, float) for value in measured.values())


def test_slice_metrics_agrees_with_the_individual_functions(reference, mask, settings) -> None:
    from ct_restoration.evaluation import ssim_interior_mask

    prediction = np.clip(reference.astype(np.float64) + 0.03, 0.0, 1.0)
    interior = ssim_interior_mask(mask, settings.win_size)

    measured = slice_metrics(reference, prediction, mask, interior, settings)
    full_ssim, ssim_map = structural_similarity_map(reference, prediction, settings)

    assert measured["full_mae"] == pytest.approx(mean_absolute_error(reference, prediction))
    assert measured["body_mae"] == pytest.approx(mean_absolute_error(reference, prediction, mask))
    assert measured["full_psnr"] == pytest.approx(peak_signal_noise_ratio(reference, prediction))
    assert measured["full_ssim"] == pytest.approx(full_ssim)
    assert measured["body_ssim"] == pytest.approx(masked_ssim_map_mean(ssim_map, interior))


def test_slice_metrics_on_an_identical_pair_is_perfect(reference, mask, settings) -> None:
    from ct_restoration.evaluation import ssim_interior_mask

    interior = ssim_interior_mask(mask, settings.win_size)
    measured = slice_metrics(reference, reference, mask, interior, settings)

    assert measured["full_mae"] == 0.0
    assert measured["body_mse"] == 0.0
    assert measured["full_psnr"] == float("inf")
    assert measured["full_ssim"] == pytest.approx(1.0)
    assert measured["body_ssim"] == pytest.approx(1.0)
