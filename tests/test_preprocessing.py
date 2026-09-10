"""Tests for CT windowing, normalization and resizing."""

from __future__ import annotations

import numpy as np
import pytest

from ct_restoration.config import load_config
from ct_restoration.data.dicom import load_ct_hu
from ct_restoration.data.preprocessing import preprocess_ct_slice, resize_image, window_ct

# center = 0, width = 400  ->  window covers [-200, 200] HU
WINDOW_EXAMPLE_HU = np.array([[-300.0, -200.0, 0.0, 200.0, 300.0]])
WINDOW_EXAMPLE_EXPECTED = np.array([[0.0, 0.0, 0.5, 1.0, 1.0]], dtype=np.float32)


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_windowing_maps_the_documented_example() -> None:
    result = window_ct(WINDOW_EXAMPLE_HU, center=0, width=400)

    np.testing.assert_array_equal(result, WINDOW_EXAMPLE_EXPECTED)


def test_windowing_clips_both_tails_to_the_boundaries() -> None:
    image = np.array([[-5000.0, 5000.0]])

    result = window_ct(image, center=40, width=400)

    np.testing.assert_array_equal(result, np.array([[0.0, 1.0]], dtype=np.float32))


def test_window_boundaries_map_exactly_to_zero_and_one() -> None:
    # center 40, width 400 -> lower = -160, upper = 240
    image = np.array([[-160.0, 40.0, 240.0]])

    result = window_ct(image, center=40, width=400)

    np.testing.assert_array_equal(result, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))


def test_windowing_output_is_float32_in_unit_range() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(-1200.0, 3000.0, size=(32, 32))

    result = window_ct(image, center=40, width=400)

    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_windowing_accepts_integer_and_float_parameters() -> None:
    integers = window_ct(WINDOW_EXAMPLE_HU, center=0, width=400)
    floats = window_ct(WINDOW_EXAMPLE_HU, center=0.0, width=400.0)

    np.testing.assert_array_equal(integers, floats)


def test_windowing_does_not_mutate_its_input() -> None:
    image = WINDOW_EXAMPLE_HU.copy()

    window_ct(image, center=0, width=400)

    np.testing.assert_array_equal(image, WINDOW_EXAMPLE_HU)


@pytest.mark.parametrize("width", [0, -1, -400.0])
def test_non_positive_window_width_is_rejected(width) -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        window_ct(WINDOW_EXAMPLE_HU, center=0, width=width)


def test_windowing_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        window_ct(np.zeros((2, 4, 4)), center=0, width=400)


def test_windowing_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    image = rng.uniform(-1000.0, 2000.0, size=(16, 16))

    first = window_ct(image, center=40, width=400)
    second = window_ct(image, center=40, width=400)

    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------


def test_resize_produces_the_requested_shape() -> None:
    image = np.zeros((512, 512), dtype=np.float32)

    assert resize_image(image, size=(256, 256)).shape == (256, 256)


def test_resize_uses_numpy_height_width_order() -> None:
    """Guards against the OpenCV (width, height) transposition."""
    image = np.zeros((512, 512), dtype=np.float32)

    assert resize_image(image, size=(128, 64)).shape == (128, 64)


def test_resize_returns_float32_within_unit_range() -> None:
    rng = np.random.default_rng(2)
    image = rng.random((100, 100)).astype(np.float32)

    result = resize_image(image, size=(256, 256))

    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_resize_averages_rather_than_copying_nearest_pixels() -> None:
    """Nearest-neighbour would return exactly 0.0 or 1.0; area averaging gives 0.5."""
    image = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 2))

    result = resize_image(image, size=(2, 2), interpolation="area")

    np.testing.assert_allclose(result, np.full((2, 2), 0.5, dtype=np.float32))


def test_linear_upsampling_creates_intermediate_values() -> None:
    image = np.array([[0.0, 1.0]], dtype=np.float32)

    result = resize_image(image, size=(1, 5), interpolation="linear")

    assert np.any((result > 0.0) & (result < 1.0))


def test_resize_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    image = rng.random((300, 300)).astype(np.float32)

    np.testing.assert_array_equal(
        resize_image(image, size=(256, 256)),
        resize_image(image, size=(256, 256)),
    )


def test_resize_does_not_mutate_its_input() -> None:
    rng = np.random.default_rng(4)
    image = rng.random((64, 64)).astype(np.float32)
    original = image.copy()

    resize_image(image, size=(32, 32))

    np.testing.assert_array_equal(image, original)


def test_resize_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="Unknown interpolation"):
        resize_image(np.zeros((8, 8), dtype=np.float32), size=(4, 4), interpolation="nearest")


@pytest.mark.parametrize("size", [(0, 256), (256, -1), (256,)])
def test_resize_rejects_invalid_sizes(size) -> None:
    with pytest.raises(ValueError, match="size must be"):
        resize_image(np.zeros((8, 8), dtype=np.float32), size=size)


def test_resize_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        resize_image(np.zeros((1, 8, 8), dtype=np.float32), size=(4, 4))


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_preprocess_ct_slice_returns_a_model_ready_image() -> None:
    rng = np.random.default_rng(5)
    image_hu = rng.uniform(-1024.0, 3000.0, size=(512, 512))

    result = preprocess_ct_slice(image_hu, window_center=40, window_width=400)

    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_ct_slice_is_deterministic() -> None:
    rng = np.random.default_rng(6)
    image_hu = rng.uniform(-1024.0, 3000.0, size=(128, 128))

    np.testing.assert_array_equal(
        preprocess_ct_slice(image_hu, window_center=40, window_width=400),
        preprocess_ct_slice(image_hu, window_center=40, window_width=400),
    )


def test_baseline_config_drives_the_preprocessing_functions() -> None:
    """The shipped config must stay usable by the code that consumes it."""
    settings = load_config("baseline.yaml")["preprocessing"]
    rng = np.random.default_rng(7)
    image_hu = rng.uniform(-1024.0, 3000.0, size=(512, 512))

    result = preprocess_ct_slice(
        image_hu,
        window_center=settings["window_center"],
        window_width=settings["window_width"],
        size=tuple(settings["image_size"]),
        interpolation=settings["interpolation"],
    )

    assert settings["window_center"] == 40
    assert settings["window_width"] == 400
    assert result.shape == (256, 256)
    assert result.dtype == np.float32


def test_full_pipeline_from_a_synthetic_dicom_file(make_ct_dataset, write_dicom) -> None:
    """DICOM on disk -> stored pixels -> HU -> window -> normalize -> resize."""
    # Stored values 0..4095 across a 512x512 gradient; with slope 1 and
    # intercept -1024 this spans -1024 HU (air) to 3071 HU (dense bone).
    gradient = np.linspace(0, 4095, 512 * 512, dtype=np.uint16).reshape(512, 512)
    path = write_dicom(make_ct_dataset(gradient, slope=1.0, intercept=-1024.0))

    image_hu = load_ct_hu(path)
    result = preprocess_ct_slice(image_hu, window_center=40, window_width=400)

    assert image_hu.shape == (512, 512)
    assert image_hu.min() == -1024.0
    assert image_hu.max() == 3071.0
    assert result.shape == (256, 256)
    assert result.dtype == np.float32
    assert result.min() == 0.0  # air is clipped to the window floor
    assert result.max() == 1.0  # bone is clipped to the window ceiling
