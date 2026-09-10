"""DICOM reading and conversion of stored pixel values to Hounsfield Units.

Scope of this module: getting from a DICOM file on disk to a quantitative
Hounsfield Unit (HU) array. Everything that follows - windowing, normalization,
resizing - lives in :mod:`ct_restoration.data.preprocessing`.

Three decisions are worth stating explicitly, because getting them wrong is a
silent correctness bug rather than a crash:

1. **Stored pixel values are not Hounsfield Units.** DICOM stores integers that
   have to be mapped into the modality's physical units. For CT that mapping is
   ``HU = stored * RescaleSlope + RescaleIntercept``, or a Modality LUT.

2. **Missing or non-HU calibration is an error, not a default.** If the
   calibration cannot be determined this module raises rather than assuming
   ``slope = 1`` and ``intercept = 0``, because that would silently produce
   numbers that look like HU but are not. A Modality LUT is only accepted when
   the dataset declares that its output *is* Hounsfield Units, since a modality
   LUT may equally well output optical density or electron density.

3. **Photometric interpretation is a display concept, not a quantitative one.**
   ``MONOCHROME1`` means a viewer should render low values as white. It says
   nothing about the physical meaning of the values, so HU are never inverted
   here. See :func:`describe_ct_dataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.pixels import apply_modality_lut


class DicomMetadataError(ValueError):
    """A DICOM dataset lacks metadata this project requires, or is unusable."""


class MissingRescaleMetadataError(DicomMetadataError):
    """HU calibration cannot be determined from the dataset.

    Raised instead of assuming a default calibration, so that an uncalibrated
    file fails loudly rather than producing plausible-looking non-HU values.
    """


def load_dicom(path: str | Path) -> Dataset:
    """Read a single DICOM file.

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        pydicom.errors.InvalidDicomError: the file is not valid DICOM.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {file_path}")
    return pydicom.dcmread(file_path)


def get_stored_pixel_array(dataset: Dataset) -> np.ndarray:
    """Return the raw stored pixel values of a single-frame image.

    These are the values as stored by the scanner. They are *not* Hounsfield
    Units; pass the result to :func:`to_hounsfield_units`.

    Raises:
        DicomMetadataError: no pixel data, or the image is not a single 2D frame.
    """
    if "PixelData" not in dataset:
        raise DicomMetadataError("Dataset contains no PixelData element")

    array = dataset.pixel_array
    if array.ndim != 2:
        raise DicomMetadataError(
            f"Expected a single 2D frame, got an array with shape {array.shape}"
        )
    return array


def apply_rescale(
    stored_pixels: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Apply the linear modality rescale ``stored * slope + intercept``.

    The array is promoted to ``float64`` *before* the arithmetic. This matters:
    stored CT pixels are usually ``uint16``, and applying a negative intercept
    to an unsigned integer array would wrap around instead of producing
    negative HU.

    Returns:
        A new ``float64`` array. The input is never modified.
    """
    array = np.asarray(stored_pixels, dtype=np.float64)
    return array * float(slope) + float(intercept)


def to_hounsfield_units(stored_pixels: np.ndarray, dataset: Dataset) -> np.ndarray:
    """Convert stored pixel values to Hounsfield Units using dataset metadata.

    Resolution order, following the DICOM Modality LUT module (PS3.3 C.11.1),
    which states that a *Modality LUT Sequence* and *Rescale Slope* /
    *Rescale Intercept* "shall be present but not both":

    1. A *Modality LUT Sequence* is present **and** its *Modality LUT Type* is
       ``HU``: delegate to pydicom's standards-aware
       :func:`~pydicom.pixels.apply_modality_lut`, rather than re-implementing
       LUT descriptor handling here.
    2. A *Modality LUT Sequence* is present with any other declared output
       units: raise. The defined terms for *Modality LUT Type* include ``OD``
       (optical density), ``ED`` / ``EDW`` (electron density), ``MGML``,
       ``Z_EFF``, ``HU_MOD`` (modified Hounsfield Unit) and ``US``
       (unspecified). None of those are Hounsfield Units, and neither is an
       absent value, so calling the output HU would be a fabrication.
    3. Both *RescaleSlope* and *RescaleIntercept* are present: apply the linear
       rescale through :func:`apply_rescale`, which is directly unit-tested.
    4. Anything else, including exactly one of the two being present: raise.

    Note that pydicom's own helper returns the array *unchanged* when no
    calibration is present. That silent pass-through is exactly what this
    project must avoid, which is why case 4 exists.

    Raises:
        DicomMetadataError: a Modality LUT is present but does not declare
            Hounsfield Unit output.
        MissingRescaleMetadataError: HU calibration cannot be determined.
    """
    has_modality_lut = "ModalityLUTSequence" in dataset and len(dataset.ModalityLUTSequence) > 0
    if has_modality_lut:
        # The standard specifies a single item; pydicom reads the first.
        lut_type = getattr(dataset.ModalityLUTSequence[0], "ModalityLUTType", None)
        if lut_type != "HU":
            raise DicomMetadataError(
                "Modality LUT Sequence is present but its ModalityLUTType is "
                f"{lut_type!r}, not 'HU'. A modality LUT may output optical density, "
                "electron density or unspecified units, so its output cannot be "
                "treated as Hounsfield Units."
            )
        return np.asarray(apply_modality_lut(stored_pixels, dataset), dtype=np.float64)

    has_slope = "RescaleSlope" in dataset
    has_intercept = "RescaleIntercept" in dataset
    if has_slope and has_intercept:
        return apply_rescale(stored_pixels, dataset.RescaleSlope, dataset.RescaleIntercept)

    missing = [
        name
        for name, present in (("RescaleSlope", has_slope), ("RescaleIntercept", has_intercept))
        if not present
    ]
    raise MissingRescaleMetadataError(
        "Cannot convert to Hounsfield Units: no Modality LUT Sequence and missing "
        f"{' and '.join(missing)}. Refusing to assume a default calibration."
    )


def pixel_padding_mask(dataset: Dataset, stored_pixels: np.ndarray) -> np.ndarray | None:
    """Locate padded (non-anatomical) pixels, if the dataset declares any.

    Scanners pad the area outside the reconstruction circle with a sentinel
    value declared in *PixelPaddingValue*, optionally as a range together with
    *PixelPaddingRangeLimit*. Those pixels carry no tissue information.

    This function only *reports* padding; it never rewrites pixel values,
    because choosing a replacement would be inventing data. V1 relies on
    windowing to bound the effect of padding sentinels and uses this mask for
    inspection only.

    **Open question for the first real CHAOS audit.** Before any metric is
    reported on real data, that audit must establish:

    * whether *PixelPaddingValue* / *PixelPaddingRangeLimit* occur at all in
      the CHAOS CT series;
    * how large the padding and background region is as a fraction of a slice;
    * whether padding always maps safely outside the selected HU window, or
      whether some of it lands inside and is treated as anatomy;
    * whether final PSNR and SSIM should exclude padding, or use a valid-body
      mask.

    That last point matters for the honesty of the benchmark. A large region
    that is identical in the target and in every method's output contributes
    near-zero error to MAE and MSE, which raises PSNR, and contributes a
    near-perfect local score to SSIM. Metrics computed over the whole frame can
    therefore look good largely because of empty background rather than because
    of restored anatomy. The policy is deliberately left undecided until real
    CHAOS slices have been inspected.

    Returns:
        A boolean mask of padded pixels, or ``None`` if no padding is declared.
    """
    padding_value = getattr(dataset, "PixelPaddingValue", None)
    if padding_value is None:
        return None

    array = np.asarray(stored_pixels)
    range_limit = getattr(dataset, "PixelPaddingRangeLimit", None)

    try:
        if range_limit is None:
            return array == int(padding_value)
        low, high = sorted((int(padding_value), int(range_limit)))
    except (TypeError, ValueError) as error:  # e.g. ambiguous VR left as raw bytes
        raise DicomMetadataError(
            f"PixelPaddingValue/PixelPaddingRangeLimit are not integers: {error}"
        ) from error

    return (array >= low) & (array <= high)


def describe_ct_dataset(dataset: Dataset) -> dict[str, Any]:
    """Summarise the technical metadata this project cares about.

    Deliberately excludes patient-identifying elements (name, ID, birth date,
    institution). Only acquisition and encoding attributes are reported.

    ``display_window_center`` and ``display_window_width`` are recorded for
    inspection only. They are DICOM *display presets*: they may hold several
    values, may differ between series, and are not the preprocessing definition
    this project uses. The experiment window comes from the project config.
    """

    def as_float_or_list(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            return [float(item) for item in value]
        return float(value)

    return {
        "modality": getattr(dataset, "Modality", None),
        "rows": getattr(dataset, "Rows", None),
        "columns": getattr(dataset, "Columns", None),
        "photometric_interpretation": getattr(dataset, "PhotometricInterpretation", None),
        "bits_stored": getattr(dataset, "BitsStored", None),
        "pixel_representation": getattr(dataset, "PixelRepresentation", None),
        "rescale_slope": as_float_or_list(getattr(dataset, "RescaleSlope", None)),
        "rescale_intercept": as_float_or_list(getattr(dataset, "RescaleIntercept", None)),
        "has_modality_lut": "ModalityLUTSequence" in dataset,
        "pixel_padding_value": getattr(dataset, "PixelPaddingValue", None),
        "pixel_padding_range_limit": getattr(dataset, "PixelPaddingRangeLimit", None),
        "display_window_center": as_float_or_list(getattr(dataset, "WindowCenter", None)),
        "display_window_width": as_float_or_list(getattr(dataset, "WindowWidth", None)),
    }


def load_ct_hu(path: str | Path) -> np.ndarray:
    """Read one CT DICOM file and return its Hounsfield Unit array.

    This function is deliberately CT-only and offers no opt-out. Hounsfield
    Units are defined by the CT scale, so returning an array called "HU" for an
    MR or other non-CT dataset would be scientifically misleading no matter
    what calibration metadata that dataset happens to carry. Use
    :func:`load_dicom` with :func:`get_stored_pixel_array` to read another
    modality.

    Args:
        path: DICOM file to read.

    Returns:
        A 2D ``float64`` array of Hounsfield Units.

    Raises:
        DicomMetadataError: not CT, no pixel data, not a single 2D frame, or a
            Modality LUT that does not declare Hounsfield Unit output.
        MissingRescaleMetadataError: HU calibration cannot be determined.
    """
    dataset = load_dicom(path)

    modality = getattr(dataset, "Modality", None)
    if modality != "CT":
        raise DicomMetadataError(
            f"Expected Modality 'CT', got {modality!r} in {path}. "
            "Hounsfield Units are only defined for CT."
        )

    stored = get_stored_pixel_array(dataset)
    return to_hounsfield_units(stored, dataset)
