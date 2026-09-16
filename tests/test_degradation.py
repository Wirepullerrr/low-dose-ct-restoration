"""Tests for the synthetic low-dose-like degradation.

Every test here is dataset-free. The degradation is a pure function of an
array, a key and a config, so nothing in this file needs the CHAOS download or
network access.

The emphasis is reproducibility. A restoration benchmark is only meaningful if
every method sees exactly the same corrupted input, which means the corruption
must not depend on call order, global random state, platform path conventions,
or anything else outside its three arguments.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ct_restoration.config import load_config
from ct_restoration.data.degradation import (
    ALGORITHM_VERSION,
    RANGE_TOLERANCE,
    DegradationConfig,
    DegradationError,
    canonical_sample_key,
    correlated_gaussian_field,
    degrade_low_dose_like,
    derive_sample_seed,
    noise_scale,
)

KEY = "Train_Sets/CT/2/DICOM_anon/i0001.dcm"
OTHER_KEY = "Train_Sets/CT/2/DICOM_anon/i0002.dcm"


@pytest.fixture
def config() -> DegradationConfig:
    return DegradationConfig()


@pytest.fixture
def clean() -> np.ndarray:
    """A deterministic stand-in for a preprocessed slice, in [0, 1].

    Built with a local generator so the fixture itself cannot be disturbed by
    global random state either. A smooth ramp plus a bright disc gives it both
    low- and high-intensity regions, which the signal-dependent sigma needs.
    """
    rows, columns = np.meshgrid(np.linspace(0.0, 1.0, 64), np.linspace(0.0, 0.6, 64), indexing="ij")
    image = (rows + columns) / 2.0
    centre = (np.arange(64) - 32.0) ** 2
    disc = centre[:, None] + centre[None, :] < 150
    image[disc] = 0.95
    return image.astype(np.float32)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_default_config_matches_the_frozen_values(config) -> None:
    assert config.algorithm == ALGORITHM_VERSION == "correlated_heteroscedastic_gaussian_v1"
    assert config.global_seed == 2026
    assert config.sigma_floor == 0.015
    assert config.sigma_signal == 0.035
    assert config.correlation_sigma_px == 0.6
    assert (config.clip_min, config.clip_max) == (0.0, 1.0)


def test_committed_config_file_matches_the_defaults(config) -> None:
    """The tracked YAML is the canonical definition; code defaults must agree."""
    loaded = DegradationConfig.from_mapping(load_config("degradation.yaml"))

    assert loaded == config


def test_config_accepts_the_inner_section_alone(config) -> None:
    assert DegradationConfig.from_mapping(config.as_dict()) == config


@pytest.mark.parametrize("field", ["sigma_floor", "sigma_signal", "correlation_sigma_px"])
def test_negative_sigma_parameters_are_rejected(field) -> None:
    with pytest.raises(DegradationError, match="must be non-negative"):
        DegradationConfig(**{field: -0.001})


@pytest.mark.parametrize("field", ["sigma_floor", "sigma_signal", "correlation_sigma_px"])
def test_non_finite_parameters_are_rejected(field) -> None:
    with pytest.raises(DegradationError, match="must be finite"):
        DegradationConfig(**{field: float("nan")})


@pytest.mark.parametrize("bounds", [(1.0, 0.0), (0.5, 0.5)])
def test_clip_bounds_must_be_ordered(bounds) -> None:
    with pytest.raises(DegradationError, match="strictly less than"):
        DegradationConfig(clip_min=bounds[0], clip_max=bounds[1])


def test_an_unknown_algorithm_version_is_refused() -> None:
    """A future v2 must be implemented, never run through v1 by accident."""
    with pytest.raises(DegradationError, match="Unsupported degradation algorithm"):
        DegradationConfig(algorithm="correlated_heteroscedastic_gaussian_v2")


def test_config_refuses_missing_keys() -> None:
    with pytest.raises(DegradationError, match="missing key"):
        DegradationConfig.from_mapping({"algorithm": ALGORITHM_VERSION, "global_seed": 1})


def test_config_refuses_unknown_keys(config) -> None:
    """A silently dropped key would mean the file no longer describes the run."""
    with pytest.raises(DegradationError, match="unrecognised key"):
        DegradationConfig.from_mapping({**config.as_dict(), "sigma_strong": 0.2})


def test_config_refuses_a_non_mapping() -> None:
    with pytest.raises(DegradationError, match="must be a mapping"):
        DegradationConfig.from_mapping([("sigma_floor", 0.1)])  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------


def test_seed_is_stable_for_the_same_key() -> None:
    assert derive_sample_seed(2026, KEY) == derive_sample_seed(2026, KEY)


def test_windows_and_posix_separators_give_the_same_seed() -> None:
    windows = r"Train_Sets\CT\2\DICOM_anon\i0001.dcm"

    assert canonical_sample_key(windows) == KEY
    assert derive_sample_seed(2026, windows) == derive_sample_seed(2026, KEY)


def test_path_objects_and_strings_give_the_same_seed() -> None:
    from pathlib import PurePosixPath, PureWindowsPath

    assert derive_sample_seed(2026, PurePosixPath(KEY)) == derive_sample_seed(2026, KEY)
    assert derive_sample_seed(
        2026, PureWindowsPath(r"Train_Sets\CT\2\DICOM_anon\i0001.dcm")
    ) == derive_sample_seed(2026, KEY)


def test_a_comma_in_the_filename_survives_canonicalization() -> None:
    """The real CHAOS manifest contains filenames like ``i0095,0000b.dcm``."""
    key = r"Train_Sets\CT\1\DICOM_anon\i0095,0000b.dcm"

    assert canonical_sample_key(key) == "Train_Sets/CT/1/DICOM_anon/i0095,0000b.dcm"


def test_different_keys_give_different_seeds() -> None:
    assert derive_sample_seed(2026, KEY) != derive_sample_seed(2026, OTHER_KEY)


def test_different_global_seeds_give_different_seeds() -> None:
    assert derive_sample_seed(2026, KEY) != derive_sample_seed(2027, KEY)


def test_different_algorithm_versions_give_different_seeds() -> None:
    assert derive_sample_seed(2026, KEY, "v1") != derive_sample_seed(2026, KEY, "v2")


def test_seed_is_a_known_constant() -> None:
    """Pins the derivation itself, so a refactor cannot silently reseed the benchmark."""
    assert derive_sample_seed(2026, KEY) == 9605225287292335134


@pytest.mark.parametrize("key", ["", "   ", "\t\n"])
def test_empty_sample_keys_are_rejected(key) -> None:
    with pytest.raises(DegradationError, match="non-empty"):
        canonical_sample_key(key)


def test_degradation_rejects_an_empty_sample_key(clean, config) -> None:
    with pytest.raises(DegradationError, match="non-empty"):
        degrade_low_dose_like(clean, "", config)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_repeated_calls_are_byte_identical(clean, config) -> None:
    first = degrade_low_dose_like(clean, KEY, config)
    second = degrade_low_dose_like(clean, KEY, config)

    assert first.tobytes() == second.tobytes()


def test_call_order_cannot_change_results(clean, config) -> None:
    """Forward, reverse and shuffled traversal must all give the same arrays."""
    keys = [f"Train_Sets/CT/2/DICOM_anon/i{index:04d}.dcm" for index in range(8)]
    forward = {key: degrade_low_dose_like(clean, key, config) for key in keys}

    reverse = {key: degrade_low_dose_like(clean, key, config) for key in reversed(keys)}
    shuffled_keys = list(keys)
    random.Random(7).shuffle(shuffled_keys)
    shuffled = {key: degrade_low_dose_like(clean, key, config) for key in shuffled_keys}

    for key in keys:
        assert np.array_equal(forward[key], reverse[key])
        assert np.array_equal(forward[key], shuffled[key])


def test_an_unrelated_slice_processed_first_changes_nothing(clean, config) -> None:
    alone = degrade_low_dose_like(clean, KEY, config)

    degrade_low_dose_like(clean * 0.5, OTHER_KEY, config)
    after = degrade_low_dose_like(clean, KEY, config)

    assert np.array_equal(alone, after)


def test_different_keys_give_different_realizations(clean, config) -> None:
    assert not np.array_equal(
        degrade_low_dose_like(clean, KEY, config),
        degrade_low_dose_like(clean, OTHER_KEY, config),
    )


def test_a_different_global_seed_gives_a_different_realization(clean, config) -> None:
    other = DegradationConfig(global_seed=config.global_seed + 1)

    assert not np.array_equal(
        degrade_low_dose_like(clean, KEY, config),
        degrade_low_dose_like(clean, KEY, other),
    )


# --------------------------------------------------------------------------
# RNG isolation
# --------------------------------------------------------------------------


def test_global_numpy_random_state_is_not_consumed(clean, config) -> None:
    """Degradation must own its RNG; it may not draw from the global stream.

    If it did, the noise on a slice would depend on how many random numbers the
    rest of the program happened to have drawn first.
    """
    np.random.seed(12345)
    before = np.random.random(5)

    np.random.seed(12345)
    _ = np.random.random(5)
    degrade_low_dose_like(clean, KEY, config)
    after = np.random.random(5)

    np.random.seed(12345)
    _ = np.random.random(5)
    expected = np.random.random(5)

    assert np.array_equal(after, expected)
    assert not np.array_equal(before, after)


def test_global_numpy_random_state_is_not_mutated(clean, config) -> None:
    np.random.seed(999)
    before = np.random.get_state()

    degrade_low_dose_like(clean, KEY, config)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_global_random_activity_cannot_change_results(clean, config) -> None:
    np.random.seed(1)
    first = degrade_low_dose_like(clean, KEY, config)

    np.random.seed(2)
    np.random.random(1000)
    second = degrade_low_dose_like(clean, KEY, config)

    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_clean_input_is_not_modified(clean, config) -> None:
    original = clean.copy()

    degraded = degrade_low_dose_like(clean, KEY, config)

    assert np.array_equal(clean, original)
    assert not np.shares_memory(degraded, clean)


@pytest.mark.parametrize("shape", [(16, 16), (32, 64), (256, 256)])
def test_output_shape_matches_the_input(shape, config) -> None:
    image = np.full(shape, 0.5, dtype=np.float32)

    assert degrade_low_dose_like(image, KEY, config).shape == shape


def test_output_is_float32(clean, config) -> None:
    assert degrade_low_dose_like(clean, KEY, config).dtype == np.float32


def test_output_is_finite_and_in_range(clean, config) -> None:
    degraded = degrade_low_dose_like(clean, KEY, config)

    assert np.all(np.isfinite(degraded))
    assert degraded.min() >= config.clip_min
    assert degraded.max() <= config.clip_max


def test_output_respects_custom_clip_bounds(clean) -> None:
    config = DegradationConfig(clip_min=0.0, clip_max=0.5)
    degraded = degrade_low_dose_like(np.clip(clean, 0.0, 0.5), KEY, config)

    assert degraded.max() <= 0.5


def test_a_saturated_image_stays_in_range(config) -> None:
    """Clipped background is the common case; noise must not push past 1."""
    for value in (0.0, 1.0):
        image = np.full((64, 64), value, dtype=np.float32)
        degraded = degrade_low_dose_like(image, KEY, config)

        assert degraded.min() >= 0.0
        assert degraded.max() <= 1.0
        # Half of a zero-mean field is clipped away, so a saturated region is
        # visibly perturbed in one direction only.
        assert not np.array_equal(degraded, image)


def test_zero_noise_configuration_returns_the_clean_image(clean) -> None:
    config = DegradationConfig(sigma_floor=0.0, sigma_signal=0.0)

    assert np.array_equal(degrade_low_dose_like(clean, KEY, config), clean)


def test_a_single_pixel_image_is_handled(config) -> None:
    """The unit-variance guard: one sample has zero spread to rescale by."""
    degraded = degrade_low_dose_like(np.full((1, 1), 0.5, dtype=np.float32), KEY, config)

    assert degraded.shape == (1, 1)
    assert np.all(np.isfinite(degraded))


# --------------------------------------------------------------------------
# Signal dependence
# --------------------------------------------------------------------------


def test_sigma_map_follows_the_documented_equation(config) -> None:
    image = np.array([[0.0, 0.25], [0.81, 1.0]], dtype=np.float32)

    sigma = noise_scale(image, config)
    expected = config.sigma_floor + config.sigma_signal * np.sqrt(image.astype(np.float64))

    assert np.allclose(sigma, expected)


def test_sigma_map_increases_with_intensity(config) -> None:
    ramp = np.linspace(0.0, 1.0, 50, dtype=np.float32).reshape(1, 50)

    sigma = noise_scale(ramp, config)

    assert np.all(np.diff(sigma[0]) > 0)
    assert sigma[0, 0] == pytest.approx(config.sigma_floor)
    assert sigma[0, -1] == pytest.approx(config.sigma_floor + config.sigma_signal)


def test_sigma_map_is_never_negative(config) -> None:
    """A hair below zero, within tolerance, must not produce NaN."""
    image = np.array([[-RANGE_TOLERANCE / 2, 0.0, 1.0]], dtype=np.float64)

    sigma = noise_scale(image, config)

    assert np.all(sigma >= 0)
    assert np.all(np.isfinite(sigma))


def test_brighter_regions_receive_more_noise(config) -> None:
    """The point of heteroscedasticity, measured on the realized perturbation."""
    dark = np.full((128, 128), 0.05, dtype=np.float32)
    bright = np.full((128, 128), 0.95, dtype=np.float32)

    # Measure away from the clip bounds so clipping cannot explain the result.
    dark_spread = float((degrade_low_dose_like(dark, KEY, config) - dark).std())
    bright_spread = float((degrade_low_dose_like(bright, KEY, config) - bright).std())

    assert bright_spread > dark_spread


# --------------------------------------------------------------------------
# Spatial correlation
# --------------------------------------------------------------------------


def test_uncorrelated_field_is_supported() -> None:
    field = correlated_gaussian_field((64, 64), seed=5, correlation_sigma_px=0.0)

    assert field.shape == (64, 64)
    assert np.all(np.isfinite(field))
    assert float(field.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(field.std()) == pytest.approx(1.0, abs=1e-12)


def test_correlated_field_is_normalized_to_unit_variance() -> None:
    """Smoothing lowers variance; renormalizing keeps sigma the only amplitude."""
    field = correlated_gaussian_field((64, 64), seed=5, correlation_sigma_px=0.6)

    assert float(field.mean()) == pytest.approx(0.0, abs=1e-12)
    assert float(field.std()) == pytest.approx(1.0, abs=1e-12)


def test_correlation_changes_the_realization_at_the_same_seed() -> None:
    uncorrelated = correlated_gaussian_field((64, 64), seed=5, correlation_sigma_px=0.0)
    correlated = correlated_gaussian_field((64, 64), seed=5, correlation_sigma_px=0.6)

    assert not np.array_equal(uncorrelated, correlated)


def test_correlation_introduces_neighbour_dependence() -> None:
    """A correlated field agrees with its neighbour more than white noise does."""

    def neighbour_correlation(field: np.ndarray) -> float:
        left, right = field[:, :-1].ravel(), field[:, 1:].ravel()
        return float(np.corrcoef(left, right)[0, 1])

    white = correlated_gaussian_field((256, 256), seed=11, correlation_sigma_px=0.0)
    smooth = correlated_gaussian_field((256, 256), seed=11, correlation_sigma_px=0.6)

    assert abs(neighbour_correlation(white)) < 0.05
    assert neighbour_correlation(smooth) > 0.3


def test_correlated_field_is_deterministic_for_a_seed() -> None:
    first = correlated_gaussian_field((32, 32), seed=3, correlation_sigma_px=0.6)
    second = correlated_gaussian_field((32, 32), seed=3, correlation_sigma_px=0.6)

    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64,), (4, 8, 3), (1, 64, 64), ()])
def test_non_2d_images_are_rejected(shape, config) -> None:
    image = np.full(shape, 0.5, dtype=np.float32)

    with pytest.raises(DegradationError, match="2D image"):
        degrade_low_dose_like(image, KEY, config)


def test_an_empty_image_is_rejected(config) -> None:
    with pytest.raises(DegradationError, match="non-empty image"):
        degrade_low_dose_like(np.zeros((0, 8), dtype=np.float32), KEY, config)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(bad, config, clean) -> None:
    broken = clean.copy()
    broken[3, 3] = bad

    with pytest.raises(DegradationError, match="finite values"):
        degrade_low_dose_like(broken, KEY, config)


@pytest.mark.parametrize("value", [-0.05, 1.05, 255.0])
def test_out_of_range_input_is_rejected(value, config, clean) -> None:
    """Unnormalized input is a preprocessing bug and must not be silently fixed."""
    broken = clean.copy()
    broken[0, 0] = value

    with pytest.raises(DegradationError, match="normalized image"):
        degrade_low_dose_like(broken, KEY, config)


def test_tiny_numerical_overshoot_is_tolerated(config, clean) -> None:
    nudged = clean.astype(np.float64)
    nudged[0, 0] = -RANGE_TOLERANCE / 2
    nudged[0, 1] = 1.0 + RANGE_TOLERANCE / 2

    degraded = degrade_low_dose_like(nudged, KEY, config)

    assert np.all(np.isfinite(degraded))


def test_complex_input_is_rejected(config) -> None:
    image = np.full((8, 8), 0.5, dtype=np.complex128)

    with pytest.raises(DegradationError, match="real numeric image"):
        degrade_low_dose_like(image, KEY, config)


def test_integer_input_within_range_is_accepted(config) -> None:
    """An all-zero integer image is in range; dtype alone is not a failure."""
    degraded = degrade_low_dose_like(np.zeros((8, 8), dtype=np.int16), KEY, config)

    assert degraded.dtype == np.float32


# --------------------------------------------------------------------------
# Sample-key contract: the reserved payload separator
# --------------------------------------------------------------------------

#: The byte the seed payload uses to join its fields. A POSIX filename may
#: legally contain it, so keys are forbidden from doing so by contract.
SEPARATOR = "\x1f"


@pytest.mark.parametrize(
    "key",
    [
        "a" + SEPARATOR + "b",
        "Train_Sets/CT/1/DICOM_anon/i0001" + SEPARATOR + ".dcm",
        SEPARATOR + "leading",
        "trailing" + SEPARATOR,
    ],
)
def test_sample_keys_containing_the_payload_separator_are_rejected(key) -> None:
    """The seed payload joins algorithm, seed and key with U+001F.

    A POSIX filename may legally contain that byte, so the separator is not
    intrinsically unambiguous. It is made unambiguous by forbidding it in keys.
    Without this check two different keys could serialize to one payload and
    collide onto a single seed.
    """
    with pytest.raises(DegradationError, match=r"U\+001F"):
        canonical_sample_key(key)


def test_degradation_rejects_a_sample_key_with_the_separator(clean, config) -> None:
    with pytest.raises(DegradationError, match=r"U\+001F"):
        degrade_low_dose_like(clean, "Train_Sets/CT/1/a" + SEPARATOR + "b.dcm", config)


def test_the_separator_rule_does_not_invalidate_any_real_key() -> None:
    """Every key the frozen benchmark actually uses must remain valid."""
    import csv
    from pathlib import Path

    manifest = Path(__file__).resolve().parents[1] / "data" / "splits" / "chaos_slice_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        keys = [row["relative_dicom_path"] for row in csv.DictReader(handle)]

    assert len(keys) == 6407
    # No exception, and canonicalization stays injective over the real cohort.
    assert len({canonical_sample_key(key) for key in keys}) == 6407


def test_the_pinned_seed_survives_the_separator_contract() -> None:
    """The contract must not have reseeded the frozen benchmark."""
    assert derive_sample_seed(2026, KEY) == 9605225287292335134


# --------------------------------------------------------------------------
# global_seed must be a genuine integer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [2026.5, 2026.9, 2026.0, True, False, "2026", None])
def test_non_integer_global_seeds_are_rejected(value, config) -> None:
    """A frozen seed must never be silently coerced.

    ``int(2026.9)`` is 2026, so a coercing reader would turn a typo into a
    different but plausible-looking seed and reseed every slice in the
    benchmark with nothing reporting a problem. ``bool`` is refused explicitly
    because it is an ``int`` subclass and ``True`` would otherwise read as
    seed 1. An integral float is refused too: in a frozen definition the field
    is an integer, and accepting 2026.0 would mean accepting 2026.0000001 was
    only a rounding away from valid.
    """
    with pytest.raises(DegradationError, match="global_seed must be an integer"):
        DegradationConfig.from_mapping({**config.as_dict(), "global_seed": value})

    with pytest.raises(DegradationError, match="global_seed must be an integer"):
        DegradationConfig(global_seed=value)


def test_integer_global_seeds_are_accepted() -> None:
    assert DegradationConfig(global_seed=0).global_seed == 0
    assert DegradationConfig(global_seed=-7).global_seed == -7
    assert DegradationConfig(global_seed=2026).global_seed == 2026


def test_the_canonical_global_seed_still_parses_as_an_integer() -> None:
    loaded = DegradationConfig.from_mapping(load_config("degradation.yaml"))

    assert loaded.global_seed == 2026
    assert isinstance(loaded.global_seed, int)
    assert not isinstance(loaded.global_seed, bool)


@pytest.mark.parametrize("field", ["sigma_floor", "sigma_signal", "correlation_sigma_px"])
@pytest.mark.parametrize("value", ["0.015", None, True])
def test_non_numeric_sigma_parameters_are_rejected(field, value, config) -> None:
    """A quoted YAML number is a mistake, not something to silently parse."""
    with pytest.raises(DegradationError, match="must be a real number"):
        DegradationConfig.from_mapping({**config.as_dict(), field: value})


def test_config_values_are_not_coerced_away_from_the_frozen_definition() -> None:
    """The whole point: parsing the tracked YAML yields exactly the frozen values."""
    loaded = DegradationConfig.from_mapping(load_config("degradation.yaml"))

    assert loaded.as_dict() == {
        "algorithm": "correlated_heteroscedastic_gaussian_v1",
        "global_seed": 2026,
        "sigma_floor": 0.015,
        "sigma_signal": 0.035,
        "correlation_sigma_px": 0.6,
        "clip_min": 0.0,
        "clip_max": 1.0,
    }
