"""Shared test fixtures.

Everything here is synthetic. No test in this suite downloads data or touches
the network, so the suite runs before any CT dataset is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def build_ct_dataset(
    stored_pixels: np.ndarray,
    *,
    slope: float | None = 1.0,
    intercept: float | None = -1024.0,
    modality: str = "CT",
    photometric_interpretation: str = "MONOCHROME2",
    pixel_representation: int = 0,
    include_pixel_data: bool = True,
    **extra_elements: Any,
) -> Dataset:
    """Build a minimal, single-frame, CT-like DICOM dataset in memory.

    Args:
        stored_pixels: 2D array of stored (uncalibrated) pixel values.
        slope: *RescaleSlope*, or ``None`` to omit the element entirely.
        intercept: *RescaleIntercept*, or ``None`` to omit the element.
        modality: value for *Modality*.
        photometric_interpretation: ``MONOCHROME1`` or ``MONOCHROME2``.
        pixel_representation: 0 for unsigned stored values, 1 for signed.
        include_pixel_data: set to ``False`` to build a dataset with no image.
        **extra_elements: any further DICOM keywords to set, for example
            ``PixelPaddingValue=-2000``.
    """
    dtype = np.uint16 if pixel_representation == 0 else np.int16
    pixels = np.asarray(stored_pixels, dtype=dtype)

    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = CTImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.ImplementationClassUID = generate_uid()

    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()

    dataset.Modality = modality
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = pixel_representation

    if slope is not None:
        dataset.RescaleSlope = slope
    if intercept is not None:
        dataset.RescaleIntercept = intercept
    if include_pixel_data:
        dataset.PixelData = pixels.tobytes()

    for keyword, value in extra_elements.items():
        setattr(dataset, keyword, value)

    return dataset


@pytest.fixture
def make_ct_dataset():
    """Factory fixture returning :func:`build_ct_dataset`."""
    return build_ct_dataset


@pytest.fixture
def write_dicom(tmp_path: Path):
    """Factory fixture writing a dataset to a real DICOM file in ``tmp_path``."""

    def _write(dataset: Dataset, name: str = "slice.dcm") -> Path:
        path = tmp_path / name
        dataset.save_as(path, enforce_file_format=True)
        return path

    return _write
