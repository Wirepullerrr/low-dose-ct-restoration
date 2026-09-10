"""Tests for DICOM reading and Hounsfield Unit conversion."""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut

from ct_restoration.data.dicom import (
    DicomMetadataError,
    MissingRescaleMetadataError,
    apply_rescale,
    describe_ct_dataset,
    get_stored_pixel_array,
    load_ct_hu,
    load_dicom,
    pixel_padding_mask,
    to_hounsfield_units,
)

STORED_EXAMPLE = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_synthetic_dicom_round_trips_through_disk(make_ct_dataset, write_dicom) -> None:
    path = write_dicom(make_ct_dataset(STORED_EXAMPLE))

    dataset = load_dicom(path)
    stored = get_stored_pixel_array(dataset)

    assert dataset.Modality == "CT"
    np.testing.assert_array_equal(stored, STORED_EXAMPLE)


def test_load_dicom_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dicom(tmp_path / "absent.dcm")


def test_dataset_without_pixel_data_fails_clearly(make_ct_dataset) -> None:
    dataset = make_ct_dataset(STORED_EXAMPLE, include_pixel_data=False)

    with pytest.raises(DicomMetadataError, match="no PixelData"):
        get_stored_pixel_array(dataset)


@pytest.mark.parametrize("modality", ["MR", "PT", "US", None])
def test_non_ct_modality_is_always_rejected(make_ct_dataset, write_dicom, modality) -> None:
    """HU are defined by the CT scale, so there is no opt-out for other modalities."""
    dataset = make_ct_dataset(STORED_EXAMPLE, modality=modality)
    if modality is None:
        del dataset.Modality
    path = write_dicom(dataset)

    with pytest.raises(DicomMetadataError, match="Modality"):
        load_ct_hu(path)


# --------------------------------------------------------------------------
# Modality rescaling: stored pixel values -> Hounsfield Units
# --------------------------------------------------------------------------


def test_rescale_with_unit_slope_and_negative_intercept() -> None:
    result = apply_rescale(STORED_EXAMPLE, slope=1, intercept=-1024)

    np.testing.assert_array_equal(result, np.array([[-1024.0, -24.0], [976.0, 1976.0]]))


def test_rescale_with_non_unit_slope() -> None:
    result = apply_rescale(STORED_EXAMPLE, slope=2, intercept=-1000)

    np.testing.assert_array_equal(result, np.array([[-1000.0, 1000.0], [3000.0, 5000.0]]))


def test_rescale_returns_a_floating_array() -> None:
    assert apply_rescale(STORED_EXAMPLE, slope=1, intercept=-1024).dtype.kind == "f"


def test_rescale_produces_negative_hu_without_unsigned_wraparound() -> None:
    """A uint16 array plus a negative intercept must not wrap to ~65000."""
    stored = np.array([[0, 100]], dtype=np.uint16)

    result = apply_rescale(stored, slope=1, intercept=-1024)

    np.testing.assert_array_equal(result, np.array([[-1024.0, -924.0]]))
    assert result.min() < 0


def test_rescale_does_not_mutate_its_input() -> None:
    stored = STORED_EXAMPLE.copy()

    apply_rescale(stored, slope=2, intercept=-1000)

    np.testing.assert_array_equal(stored, STORED_EXAMPLE)


def test_to_hounsfield_units_reads_slope_and_intercept_from_the_dataset(
    make_ct_dataset,
) -> None:
    dataset = make_ct_dataset(STORED_EXAMPLE, slope=1.0, intercept=-1024.0)

    result = to_hounsfield_units(get_stored_pixel_array(dataset), dataset)

    np.testing.assert_array_equal(result, np.array([[-1024.0, -24.0], [976.0, 1976.0]]))


def test_our_rescale_agrees_with_pydicoms_reference_implementation(make_ct_dataset) -> None:
    dataset = make_ct_dataset(STORED_EXAMPLE, slope=2.5, intercept=-1024.0)
    stored = get_stored_pixel_array(dataset)

    np.testing.assert_allclose(
        apply_rescale(stored, dataset.RescaleSlope, dataset.RescaleIntercept),
        apply_modality_lut(stored, dataset),
    )


# --------------------------------------------------------------------------
# Missing or alternative calibration metadata
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slope", "intercept", "expected_in_message"),
    [
        (None, -1024.0, "RescaleSlope"),
        (1.0, None, "RescaleIntercept"),
        (None, None, "RescaleSlope and RescaleIntercept"),
    ],
)
def test_missing_rescale_metadata_raises_instead_of_assuming_defaults(
    make_ct_dataset, slope, intercept, expected_in_message
) -> None:
    dataset = make_ct_dataset(STORED_EXAMPLE, slope=slope, intercept=intercept)

    with pytest.raises(MissingRescaleMetadataError, match=expected_in_message):
        to_hounsfield_units(get_stored_pixel_array(dataset), dataset)


def test_pydicom_would_silently_pass_uncalibrated_pixels_through(make_ct_dataset) -> None:
    """Documents the behaviour our wrapper exists to prevent."""
    dataset = make_ct_dataset(STORED_EXAMPLE, slope=None, intercept=None)
    stored = get_stored_pixel_array(dataset)

    np.testing.assert_array_equal(apply_modality_lut(stored, dataset), stored)


def _dataset_with_modality_lut(make_ct_dataset, lut_type: str | None) -> Dataset:
    lut = Dataset()
    lut.LUTDescriptor = [4, 0, 16]  # 4 entries, first mapped value 0, 16 bits per entry
    lut.LUTData = [0, 100, 200, 300]
    if lut_type is not None:
        lut.ModalityLUTType = lut_type

    return make_ct_dataset(
        np.array([[0, 1], [2, 3]], dtype=np.uint16),
        slope=None,
        intercept=None,
        ModalityLUTSequence=[lut],
    )


def test_modality_lut_declaring_hu_output_is_applied(make_ct_dataset) -> None:
    dataset = _dataset_with_modality_lut(make_ct_dataset, "HU")

    result = to_hounsfield_units(get_stored_pixel_array(dataset), dataset)

    np.testing.assert_array_equal(result, np.array([[0.0, 100.0], [200.0, 300.0]]))


@pytest.mark.parametrize("lut_type", ["OD", "ED", "MGML", "HU_MOD", "US", None])
def test_modality_lut_with_non_hu_output_is_rejected(make_ct_dataset, lut_type) -> None:
    """A modality LUT may emit optical or electron density; that is not HU."""
    dataset = _dataset_with_modality_lut(make_ct_dataset, lut_type)

    with pytest.raises(DicomMetadataError, match="ModalityLUTType"):
        to_hounsfield_units(get_stored_pixel_array(dataset), dataset)


# --------------------------------------------------------------------------
# Photometric interpretation and pixel padding
# --------------------------------------------------------------------------


def test_monochrome1_does_not_invert_hu(make_ct_dataset) -> None:
    """MONOCHROME1 is a display convention; quantitative HU stay untouched."""
    monochrome2 = make_ct_dataset(STORED_EXAMPLE, photometric_interpretation="MONOCHROME2")
    monochrome1 = make_ct_dataset(STORED_EXAMPLE, photometric_interpretation="MONOCHROME1")

    hu2 = to_hounsfield_units(get_stored_pixel_array(monochrome2), monochrome2)
    hu1 = to_hounsfield_units(get_stored_pixel_array(monochrome1), monochrome1)

    np.testing.assert_array_equal(hu1, hu2)
    assert describe_ct_dataset(monochrome1)["photometric_interpretation"] == "MONOCHROME1"


def test_no_padding_declared_returns_no_mask(make_ct_dataset) -> None:
    dataset = make_ct_dataset(STORED_EXAMPLE)

    assert pixel_padding_mask(dataset, get_stored_pixel_array(dataset)) is None


def test_single_pixel_padding_value_is_located(make_ct_dataset) -> None:
    stored = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    dataset = make_ct_dataset(stored, PixelPaddingValue=2000)

    mask = pixel_padding_mask(dataset, stored)

    np.testing.assert_array_equal(mask, np.array([[False, False], [True, False]]))


def test_pixel_padding_range_is_located(make_ct_dataset) -> None:
    stored = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
    dataset = make_ct_dataset(stored, PixelPaddingValue=1000, PixelPaddingRangeLimit=2000)

    mask = pixel_padding_mask(dataset, stored)

    np.testing.assert_array_equal(mask, np.array([[False, True], [True, False]]))


# --------------------------------------------------------------------------
# Metadata reporting
# --------------------------------------------------------------------------


def test_describe_ct_dataset_reports_technical_fields_and_no_patient_identifiers(
    make_ct_dataset,
) -> None:
    dataset = make_ct_dataset(
        STORED_EXAMPLE,
        WindowCenter=[40, 300],
        WindowWidth=[400, 1500],
        PatientID="SHOULD-NOT-APPEAR",
        PatientName="SHOULD^NOT^APPEAR",
    )

    description = describe_ct_dataset(dataset)

    assert description["modality"] == "CT"
    assert description["rows"] == 2
    assert description["rescale_intercept"] == -1024.0
    assert description["has_modality_lut"] is False
    # DICOM display presets are recorded for inspection, not used for windowing.
    assert description["display_window_center"] == [40.0, 300.0]
    assert "SHOULD-NOT-APPEAR" not in str(description)
    assert not any("patient" in key.lower() for key in description)


# --------------------------------------------------------------------------
# End to end: file on disk -> HU
# --------------------------------------------------------------------------


def test_load_ct_hu_reads_a_file_into_hounsfield_units(make_ct_dataset, write_dicom) -> None:
    path = write_dicom(make_ct_dataset(STORED_EXAMPLE, slope=1.0, intercept=-1024.0))

    hu = load_ct_hu(path)

    assert hu.dtype.kind == "f"
    np.testing.assert_array_equal(hu, np.array([[-1024.0, -24.0], [976.0, 1976.0]]))
