"""Tests for the CLAHE classical method.

Fully synthetic. CLAHE is a pure function of an image and a config, so nothing
here needs the CHAOS download.

The risk with a wrapper around a library routine is not the algorithm but the
contract around it: an input silently rescaled, a quantization that drifts, a
bool accepted where an integer was meant, or an output that leaves [0, 1] and
breaks every metric downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from ct_restoration.classical.clahe import (
    ALGORITHM_VERSION,
    QUANTIZATION_BITS,
    QUANTIZATION_ERROR_BOUND,
    QUANTIZATION_MAX,
    ClaheConfig,
    ClaheError,
    apply_clahe,
    clahe_restorer,
    from_uint8,
    to_uint8,
)
from ct_restoration.config import PROJECT_ROOT, load_config


@pytest.fixture
def config() -> ClaheConfig:
    return ClaheConfig()


@pytest.fixture
def degraded() -> np.ndarray:
    """A deterministic textured stand-in for a degraded slice, in [0, 1]."""
    rows, columns = np.meshgrid(
        np.linspace(0.05, 0.9, 64), np.linspace(0.0, 0.4, 64), indexing="ij"
    )
    image = (rows + columns) / 2.0
    centre = (np.arange(64) - 32.0) ** 2
    image[centre[:, None] + centre[None, :] < 200] = 0.8
    grain = np.sin(np.arange(64) * 1.7)[:, None] * np.cos(np.arange(64) * 2.3)[None, :]
    return np.clip(image + 0.03 * grain, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Quantization: part of the method, not an implementation detail
# --------------------------------------------------------------------------


def test_quantization_constants_are_eight_bit() -> None:
    assert QUANTIZATION_BITS == 8
    assert QUANTIZATION_MAX == 255
    assert QUANTIZATION_ERROR_BOUND == pytest.approx(0.5 / 255)


def test_zero_maps_to_zero() -> None:
    image = np.zeros((4, 4), dtype=np.float32)

    assert int(to_uint8(image)[0, 0]) == 0
    assert float(from_uint8(to_uint8(image))[0, 0]) == 0.0


def test_one_maps_to_the_top_level_and_back() -> None:
    image = np.ones((4, 4), dtype=np.float32)

    assert int(to_uint8(image)[0, 0]) == 255
    assert float(from_uint8(to_uint8(image))[0, 0]) == 1.0


def test_round_trip_error_is_bounded_by_half_a_level() -> None:
    """The whole cost of quantizing, stated as a number rather than assumed.

    The bound is half a level, 0.5/255, plus the float32 representation error
    of the returned value: ``from_uint8`` returns float32, so a recovered
    level such as 128/255 is itself stored to about 1e-7. Measured excess over
    the exact bound is around 3e-8, well inside one float32 epsilon.
    """
    values = np.linspace(0.0, 1.0, 4001, dtype=np.float64).reshape(1, -1)
    tolerance = QUANTIZATION_ERROR_BOUND + float(np.finfo(np.float32).eps)

    recovered = from_uint8(to_uint8(values)).astype(np.float64)
    worst = float(np.abs(recovered - values).max())

    assert worst <= tolerance
    # Still essentially half a level: the float32 slack must not be doing the work.
    assert worst < QUANTIZATION_ERROR_BOUND * 1.001


def test_quantization_uses_the_fixed_range_not_per_image_minmax() -> None:
    """A dim slice must not be stretched to fill 0-255; that would break comparability."""
    dim = np.full((8, 8), 0.2, dtype=np.float32)

    assert int(to_uint8(dim)[0, 0]) == 51  # round(0.2 * 255)
    assert int(to_uint8(dim).max()) == 51


def test_quantization_is_deterministic(degraded) -> None:
    assert np.array_equal(to_uint8(degraded), to_uint8(degraded))


def test_quantization_output_dtype_is_uint8(degraded) -> None:
    assert to_uint8(degraded).dtype == np.uint8


def test_from_uint8_refuses_a_non_uint8_image() -> None:
    with pytest.raises(ClaheError, match="expects a uint8 image"):
        from_uint8(np.zeros((4, 4), dtype=np.float32))


def test_to_uint8_does_not_modify_its_input(degraded) -> None:
    before = degraded.copy()

    to_uint8(degraded)

    assert np.array_equal(degraded, before)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_committed_config_matches_the_frozen_selection() -> None:
    """The tracked YAML is the canonical runtime definition."""
    loaded = ClaheConfig.from_mapping(load_config("clahe.yaml"))

    assert loaded.algorithm == ALGORITHM_VERSION
    assert loaded.input_quantization_bits == 8
    assert loaded.clip_limit > 0
    assert len(loaded.tile_grid_size) == 2


def test_the_committed_config_carries_its_selection_provenance() -> None:
    document = load_config("clahe.yaml")["clahe"]

    assert document["selected_by"]["split"] == "validation"
    assert document["selected_by"]["primary_metric"] == "body_ssim"
    assert document["selected_by"]["aggregation"] == "patient_weighted"
    # The tracked repo-relative path, not a bare filename: provenance should
    # name a file a reader can open.
    assert document["selected_by"]["search_config"] == "configs/clahe_search.yaml"
    assert (PROJECT_ROOT / "configs/clahe_search.yaml").exists()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_clip_limits_are_rejected(value) -> None:
    with pytest.raises(ClaheError, match="clip_limit"):
        ClaheConfig(clip_limit=value)


@pytest.mark.parametrize("value", ["2.0", None])
def test_non_numeric_clip_limits_are_rejected(value) -> None:
    with pytest.raises(ClaheError, match="must be a real number"):
        ClaheConfig(clip_limit=value)


def test_a_bool_clip_limit_is_rejected() -> None:
    with pytest.raises(ClaheError, match="must be a real number"):
        ClaheConfig(clip_limit=True)


@pytest.mark.parametrize("grid", [(0, 8), (8, 0), (-1, 4), (8, -2)])
def test_non_positive_grid_dimensions_are_rejected(grid) -> None:
    with pytest.raises(ClaheError, match=">= 1"):
        ClaheConfig(tile_grid_size=grid)


@pytest.mark.parametrize("grid", [(8,), (4, 4, 4), 8, "8x8", None])
def test_malformed_grids_are_rejected(grid) -> None:
    with pytest.raises(ClaheError, match="two positive integers"):
        ClaheConfig(tile_grid_size=grid)


@pytest.mark.parametrize("grid", [(True, 8), (8, False), (True, True)])
def test_a_bool_is_not_accepted_as_a_grid_dimension(grid) -> None:
    """bool is an int subclass; True would silently become a 1-tile axis."""
    with pytest.raises(ClaheError, match="A bool is not"):
        ClaheConfig(tile_grid_size=grid)


@pytest.mark.parametrize("grid", [(4.0, 4.0), (8, 8.5)])
def test_float_grid_dimensions_are_rejected(grid) -> None:
    with pytest.raises(ClaheError, match="must be integers"):
        ClaheConfig(tile_grid_size=grid)


def test_an_unknown_algorithm_is_refused() -> None:
    with pytest.raises(ClaheError, match="Unsupported CLAHE algorithm"):
        ClaheConfig(algorithm="opencv_clahe_v2")


@pytest.mark.parametrize("bits", [16, 12, 7, True])
def test_a_different_quantization_depth_is_refused(bits) -> None:
    """A different bit depth is a different method, not a tuning knob."""
    with pytest.raises(ClaheError, match="input_quantization_bits must be 8"):
        ClaheConfig(input_quantization_bits=bits)


def test_config_refuses_missing_keys() -> None:
    with pytest.raises(ClaheError, match="missing key"):
        ClaheConfig.from_mapping({"algorithm": ALGORITHM_VERSION})


def test_config_refuses_unknown_keys(config) -> None:
    with pytest.raises(ClaheError, match="unrecognised key"):
        ClaheConfig.from_mapping({**config.as_dict(), "gamma": 1.2})


def test_config_accepts_a_list_grid_from_yaml(config) -> None:
    loaded = ClaheConfig.from_mapping({**config.as_dict(), "tile_grid_size": [4, 16]})

    assert loaded.tile_rows == 4
    assert loaded.tile_columns == 16


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_output_shape_matches_the_input(degraded, config) -> None:
    assert apply_clahe(degraded, config).shape == degraded.shape


def test_output_is_float32(degraded, config) -> None:
    assert apply_clahe(degraded, config).dtype == np.float32


def test_output_is_finite_and_in_range(degraded, config) -> None:
    restored = apply_clahe(degraded, config)

    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0
    assert restored.max() <= 1.0


def test_input_is_not_modified(degraded, config) -> None:
    before = degraded.copy()

    restored = apply_clahe(degraded, config)

    assert np.array_equal(degraded, before)
    assert not np.shares_memory(restored, degraded)


def test_a_saturated_image_stays_in_range(config) -> None:
    for value in (0.0, 1.0):
        restored = apply_clahe(np.full((64, 64), value, dtype=np.float32), config)

        assert restored.min() >= 0.0
        assert restored.max() <= 1.0
        assert np.all(np.isfinite(restored))


def test_clahe_actually_changes_a_textured_image(degraded, config) -> None:
    """Guards against a wrapper that silently returns its input."""
    assert not np.array_equal(apply_clahe(degraded, config), degraded)


def test_different_parameters_give_different_output(degraded) -> None:
    gentle = apply_clahe(degraded, ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4)))
    strong = apply_clahe(degraded, ClaheConfig(clip_limit=4.0, tile_grid_size=(4, 4)))

    assert not np.array_equal(gentle, strong)


# --------------------------------------------------------------------------
# Determinism and isolation
# --------------------------------------------------------------------------


def test_same_image_and_config_give_byte_identical_output(degraded, config) -> None:
    first = apply_clahe(degraded, config)
    second = apply_clahe(degraded, config)

    assert first.tobytes() == second.tobytes()


def test_global_numpy_rng_cannot_affect_the_output(degraded, config) -> None:
    np.random.seed(1)
    first = apply_clahe(degraded, config)

    np.random.seed(2)
    np.random.random(10_000)
    second = apply_clahe(degraded, config)

    assert np.array_equal(first, second)


def test_processing_another_image_first_changes_nothing(degraded, config) -> None:
    alone = apply_clahe(degraded, config)

    apply_clahe(np.clip(degraded * 0.4, 0, 1), config)
    after = apply_clahe(degraded, config)

    assert np.array_equal(alone, after)


# --------------------------------------------------------------------------
# Input contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64,), (4, 4, 3), (1, 32, 32), ()])
def test_non_2d_images_are_rejected(shape, config) -> None:
    with pytest.raises(ClaheError, match="2D image"):
        apply_clahe(np.full(shape, 0.5), config)


def test_an_empty_image_is_rejected(config) -> None:
    with pytest.raises(ClaheError, match="non-empty"):
        apply_clahe(np.zeros((0, 8), dtype=np.float32), config)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(bad, degraded, config) -> None:
    broken = degraded.copy()
    broken[2, 2] = bad

    with pytest.raises(ClaheError, match="finite values"):
        apply_clahe(broken, config)


@pytest.mark.parametrize("value", [-0.5, 1.5, 255.0])
def test_grossly_out_of_range_input_is_rejected(value, degraded, config) -> None:
    broken = degraded.astype(np.float64)
    broken[0, 0] = value

    with pytest.raises(ClaheError, match="normalized image in"):
        apply_clahe(broken, config)


def test_tiny_numerical_overshoot_is_tolerated(degraded, config) -> None:
    nudged = degraded.astype(np.float64)
    nudged[0, 0] = -1e-9
    nudged[0, 1] = 1.0 + 1e-9

    assert np.all(np.isfinite(apply_clahe(nudged, config)))


def test_complex_input_is_rejected(config) -> None:
    with pytest.raises(ClaheError, match="real numeric"):
        apply_clahe(np.full((8, 8), 0.5, dtype=np.complex128), config)


# --------------------------------------------------------------------------
# The restoration contract: the method sees the degraded image and nothing else
# --------------------------------------------------------------------------


def test_the_restorer_takes_only_the_degraded_image(degraded, config) -> None:
    """One argument, by design: no clean reference, mask or patient reaches it."""
    import inspect

    restore = clahe_restorer(config)

    assert list(inspect.signature(restore).parameters) == ["degraded"]
    assert np.array_equal(restore(degraded), apply_clahe(degraded, config))


def test_the_restorer_binds_its_configuration(degraded) -> None:
    gentle = clahe_restorer(ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4)))
    strong = clahe_restorer(ClaheConfig(clip_limit=4.0, tile_grid_size=(16, 16)))

    assert not np.array_equal(gentle(degraded), strong(degraded))


# --------------------------------------------------------------------------
# A documented property of the pinned OpenCV build, not a design choice
# --------------------------------------------------------------------------


@pytest.fixture
def benchmark_sized() -> np.ndarray:
    """A deterministic 256x256 image, the size the benchmark actually uses.

    The collapse below depends on the tile area, so it only shows up at the
    real image size; the small fixture above would not reproduce it.
    """
    rng = np.random.default_rng(7)
    rows, columns = np.meshgrid(
        np.linspace(0.0, 1.0, 256), np.linspace(0.0, 1.0, 256), indexing="ij"
    )
    image = 0.25 + 0.5 * rows * columns
    centre = (np.arange(256) - 128.0) ** 2
    image[centre[:, None] + centre[None, :] < 3600] = 0.75
    return np.clip(image + 0.04 * rng.standard_normal((256, 256)), 0.0, 1.0).astype(np.float32)


def test_clip_limits_half_and_one_collapse_at_a_16x16_grid(benchmark_sized) -> None:
    """Two declared candidates are one effective operator here.

    OpenCV turns ``clip_limit`` into an integer per-bin threshold,
    ``max(int(clip_limit * tile_area / 256), 1)``. At a 16x16 grid on a 256x256
    image the tile area is exactly 256, so 0.5 floors to 0 and 1.0 floors to 1,
    and both are raised to the same minimum of 1.

    This pins current benchmark behaviour under the pinned OpenCV version. It
    is a property discovered after the sweep ran, not a reason to remove either
    declared candidate: the predeclared grid stands as declared, and the
    predeclared tie-breaker resolves the pair in favour of the lower clip limit.
    """
    tile_area = (256 // 16) * (256 // 16)
    assert tile_area == 256
    assert max(int(0.5 * tile_area / 256), 1) == max(int(1.0 * tile_area / 256), 1) == 1

    gentle = apply_clahe(benchmark_sized, ClaheConfig(clip_limit=0.5, tile_grid_size=(16, 16)))
    firmer = apply_clahe(benchmark_sized, ClaheConfig(clip_limit=1.0, tile_grid_size=(16, 16)))

    assert gentle.tobytes() == firmer.tobytes()


def test_the_same_two_clip_limits_stay_distinct_at_a_4x4_grid(benchmark_sized) -> None:
    """The collapse is specific to the tile area, not to the clip limits."""
    tile_area = (256 // 4) * (256 // 4)
    assert max(int(0.5 * tile_area / 256), 1) != max(int(1.0 * tile_area / 256), 1)

    gentle = apply_clahe(benchmark_sized, ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4)))
    firmer = apply_clahe(benchmark_sized, ClaheConfig(clip_limit=1.0, tile_grid_size=(4, 4)))

    assert gentle.tobytes() != firmer.tobytes()


def test_twelve_declared_candidates_give_eleven_distinct_operators(benchmark_sized) -> None:
    """The count the CLAHE results section reports, checked rather than asserted.

    The search space still holds 12 declared parameter tuples. Under this
    OpenCV build, two of them map to the same effective operator, so the sweep
    covers 11 distinct ones.
    """
    from ct_restoration.classical.search import ClaheSearchSpace

    candidates = ClaheSearchSpace().candidates()
    outputs: dict[bytes, list[str]] = {}
    for candidate in candidates:
        outputs.setdefault(apply_clahe(benchmark_sized, candidate).tobytes(), []).append(
            candidate.label
        )

    collapsed = sorted(labels for labels in outputs.values() if len(labels) > 1)

    assert len(candidates) == 12
    assert len(outputs) == 11
    assert collapsed == [["clip0.5_grid16x16", "clip1_grid16x16"]]
