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


# --------------------------------------------------------------------------
# A miniature on-disk cohort, for the Dataset / DataLoader layer
# --------------------------------------------------------------------------
#
# Real DICOM files, a real manifest, four partitions, uneven slice counts per
# patient - everything the Dataset and the patient-balanced sampler need, at a
# size that runs in well under a second. Still no CHAOS download and no
# network.

#: (subject_id, split, acquisition_group, source_archive, slice_count).
#: Slice counts are deliberately uneven: that inequality is the whole reason
#: the patient-balanced sampler exists.
SYNTHETIC_COHORT: tuple[tuple[str, str, str, str, int], ...] = (
    ("2", "train", "B", "Train_Sets", 3),
    ("3", "train", "B", "Test_Sets", 5),
    ("21", "train", "A", "Train_Sets", 2),
    ("31", "train", "A", "Test_Sets", 6),
    ("4", "validation", "B", "Test_Sets", 4),
    ("23", "validation", "A", "Train_Sets", 3),
    ("11", "test", "A", "Test_Sets", 3),
    ("1", "stress", "C", "Train_Sets", 2),
)

#: Preprocessing for the synthetic cohort. Same shape of config as
#: configs/baseline.yaml, with a small image so the tests stay fast.
SYNTHETIC_PREPROCESSING: dict[str, Any] = {
    "window_center": 40,
    "window_width": 400,
    "image_size": [16, 16],
    "interpolation": "area",
}


def _synthetic_stored_pixels(subject_id: str, index: int) -> np.ndarray:
    """A distinct, deterministic 32x32 stored-pixel pattern per slice."""
    grid = np.arange(32, dtype=np.float64)
    rows, columns = np.meshgrid(grid, grid, indexing="ij")
    offset = (int(subject_id) * 37 + index * 11) % 400
    body = 1024 + offset + rows * 6 + columns * 4
    radius = (rows - 16.0) ** 2 + (columns - 16.0) ** 2
    body[radius > 180] = 24.0  # air-like rim, so a body region exists
    return body.astype(np.uint16)


@pytest.fixture
def synthetic_cohort(tmp_path: Path):
    """Write the miniature cohort to disk and return its pieces.

    Returns:
        ``(root, manifest_frame, preprocessing)``. ``root`` holds real DICOM
        files at the relative POSIX keys the manifest names.
    """
    import pandas as pd

    root = tmp_path / "chaos"
    records = []
    for subject_id, split, group, archive, count in SYNTHETIC_COHORT:
        for index in range(count):
            key = f"{archive}/CT/{subject_id}/DICOM_anon/i{index:04d}.dcm"
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset = build_ct_dataset(_synthetic_stored_pixels(subject_id, index))
            dataset.save_as(path, enforce_file_format=True)
            records.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "source_archive": archive,
                    "acquisition_group": group,
                    "relative_dicom_path": key,
                    "geometric_slice_position": -100.0 + index,
                    "geometric_slice_index": index,
                }
            )
    return root, pd.DataFrame(records), dict(SYNTHETIC_PREPROCESSING)


@pytest.fixture
def synthetic_manifest_csv(synthetic_cohort, tmp_path: Path) -> Path:
    """The miniature manifest written to a CSV, as the real commands read it."""
    _, frame, _ = synthetic_cohort
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    return path
