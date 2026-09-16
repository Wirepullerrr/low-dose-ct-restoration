"""Image-quality metrics shared by every method in this benchmark.

One implementation, called by all of them. The degraded baseline, CLAHE, the
residual CNN and the U-Net must be scored by exactly the same code: a metric
re-derived inside a method's own script is a metric that can drift, and a
comparison between two slightly different PSNRs is not a comparison at all.

What is measured
----------------
``MAE``
    mean absolute error, ``mean(|prediction - reference|)``. Lower is better.
    In the same units as the normalized image.
``MSE``
    mean squared error, ``mean((prediction - reference)^2)``. Lower is better.
    Penalises large deviations far more than small ones.
``PSNR``
    ``10 * log10(data_range^2 / MSE)``, in decibels. Higher is better. It is
    reported because it is the conventional unit in this literature.

    **For one image pair in one region**, PSNR is a strictly decreasing
    transform of MSE, so the two rank identically. That equivalence does *not*
    survive aggregation. This benchmark computes MSE and PSNR per slice and
    then averages each over slices and patients, and the logarithm is
    nonlinear, so ``mean(PSNR_i)`` is not a monotone function of
    ``mean(MSE_i)``. Two methods can therefore be ordered one way by mean MSE
    and the other way by mean PSNR - a method with uneven per-slice errors is
    flattered by mean PSNR, because the logarithm rewards its very good slices
    more than it punishes its very bad ones. Read an aggregate PSNR as its own
    figure, not as a restatement of aggregate MSE.
``SSIM``
    local structural similarity, via :func:`skimage.metrics.structural_similarity`.
    Higher is better, 1.0 for identical images. Unlike MSE it compares local
    luminance, contrast and structure in a sliding window, so it responds to
    texture and edges rather than to per-pixel differences alone.

Both regions, always
--------------------
Every metric is computed twice, once over the full frame and once over the
evaluation body region, because a large part of a windowed CT frame is flat
background. See :mod:`ct_restoration.evaluation` for the mask.

A note on the two SSIM figures
------------------------------
``full_ssim`` is exactly what scikit-image returns: the mean of the local SSIM
map over the frame, excluding a ``(win_size - 1) // 2`` pixel border that the
sliding window cannot cover. ``body_ssim`` is the mean of that *same* map over
interior body pixels. It is not a different definition of SSIM, and it must not
be described as one. Flattening the masked pixels into a vector and running
SSIM on that would be meaningless, because SSIM is defined on a local
neighbourhood that a flattened vector no longer has.

Strictness
----------
Every input is validated and nothing is silently repaired. Arrays are not
resized, not renormalized and not modified in place. A shape mismatch or an
out-of-range array is an upstream bug, and a metric that quietly papers over it
would report a number for something other than what was intended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

#: Dynamic range of the normalized images this project compares. Fixed, not
#: inferred per image: an image-dependent range would make PSNR incomparable
#: between slices, which is the opposite of what a benchmark needs.
DATA_RANGE = 1.0

#: How far outside [0, 1] an image may stray before it is rejected. Only
#: float32 round-trip error is tolerated.
RANGE_TOLERANCE = 1e-6

#: Metric names produced for each region, in reporting order.
METRIC_NAMES: tuple[str, ...] = ("mae", "mse", "psnr", "ssim")

#: Evaluation regions, in reporting order.
REGION_NAMES: tuple[str, ...] = ("full", "body")


class MetricError(ValueError):
    """A metric input is not usable as given."""


@dataclass(frozen=True)
class SsimSettings:
    """Pinned SSIM parameters.

    Explicit rather than defaulted, because scikit-image's defaults have
    changed across releases and a benchmark whose definition moves with a
    library upgrade is not frozen at all.

    ``sigma = 1.5`` with scikit-image's gaussian weighting produces exactly an
    11-tap window, so ``win_size = 11`` agrees with it rather than competing
    with it.
    """

    win_size: int = 11
    gaussian_weights: bool = True
    sigma: float = 1.5
    use_sample_covariance: bool = False
    data_range: float = DATA_RANGE

    def __post_init__(self) -> None:
        if isinstance(self.win_size, bool) or not isinstance(self.win_size, int):
            raise MetricError(f"win_size must be an integer, got {self.win_size!r}")
        if self.win_size < 3 or self.win_size % 2 == 0:
            raise MetricError(f"win_size must be an odd integer >= 3, got {self.win_size}")
        if not isinstance(self.gaussian_weights, bool):
            raise MetricError(f"gaussian_weights must be a bool, got {self.gaussian_weights!r}")
        if not isinstance(self.use_sample_covariance, bool):
            raise MetricError(
                f"use_sample_covariance must be a bool, got {self.use_sample_covariance!r}"
            )
        for name in ("sigma", "data_range"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise MetricError(f"{name} must be a real number, got {value!r}")
            if not np.isfinite(value) or value <= 0:
                raise MetricError(f"{name} must be finite and positive, got {value}")

    @property
    def border(self) -> int:
        """Pixels at each edge that the sliding window cannot cover."""
        return (self.win_size - 1) // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "win_size": int(self.win_size),
            "gaussian_weights": bool(self.gaussian_weights),
            "sigma": float(self.sigma),
            "use_sample_covariance": bool(self.use_sample_covariance),
            "data_range": float(self.data_range),
        }


def _validate_image(array: np.ndarray, name: str, data_range: float) -> np.ndarray:
    """Check one image and return it as float64. Never modifies the input."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise MetricError(f"{name} must be a 2D image, got shape {values.shape}")
    if values.size == 0:
        raise MetricError(f"{name} must be non-empty, got an empty array")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise MetricError(f"{name} must be a real numeric image, got dtype {values.dtype}")

    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        count = int((~np.isfinite(values)).sum())
        raise MetricError(f"{name} must be finite, found {count} NaN or infinite pixel(s)")

    low, high = 0.0, float(data_range)
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < low - RANGE_TOLERANCE or maximum > high + RANGE_TOLERANCE:
        raise MetricError(
            f"{name} must lie in [{low}, {high}], got [{minimum:.6g}, {maximum:.6g}]. "
            "Metrics do not normalize their inputs; run the established preprocessing first."
        )
    return values


def validate_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
    data_range: float = DATA_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a reference/prediction pair and return both as float64.

    Raises:
        MetricError: either image is unusable, or their shapes differ.
    """
    validated_reference = _validate_image(reference, "reference", data_range)
    validated_prediction = _validate_image(prediction, "prediction", data_range)
    if validated_reference.shape != validated_prediction.shape:
        raise MetricError(
            f"reference and prediction must have the same shape, got "
            f"{validated_reference.shape} and {validated_prediction.shape}. "
            "Metrics do not resize their inputs."
        )
    return validated_reference, validated_prediction


def validate_mask(mask: np.ndarray, shape: tuple[int, ...], name: str = "mask") -> np.ndarray:
    """Validate a boolean evaluation mask against an image shape.

    Raises:
        MetricError: wrong shape, not boolean-like, or empty.
    """
    values = np.asarray(mask)
    if values.shape != shape:
        raise MetricError(f"{name} shape {values.shape} does not match image shape {shape}")
    if values.dtype != bool:
        if not np.issubdtype(values.dtype, np.number):
            raise MetricError(f"{name} must be boolean, got dtype {values.dtype}")
        unique = set(np.unique(values).tolist())
        if not unique <= {0, 1}:
            raise MetricError(f"{name} must be boolean or 0/1, found values {sorted(unique)}")
        values = values.astype(bool)
    if not values.any():
        raise MetricError(
            f"{name} selects no pixels; a region metric over an empty region is undefined"
        )
    return values


def _difference(
    reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    if mask is None:
        return prediction - reference
    selected = validate_mask(mask, reference.shape)
    return prediction[selected] - reference[selected]


def mean_absolute_error(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = DATA_RANGE,
) -> float:
    """``mean(|prediction - reference|)``, over the mask if one is given."""
    values = validate_pair(reference, prediction, data_range)
    return float(np.abs(_difference(*values, mask)).mean())


def mean_squared_error(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = DATA_RANGE,
) -> float:
    """``mean((prediction - reference)^2)``, over the mask if one is given."""
    values = validate_pair(reference, prediction, data_range)
    return float(np.square(_difference(*values, mask)).mean())


def peak_signal_noise_ratio(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    data_range: float = DATA_RANGE,
) -> float:
    """``10 * log10(data_range^2 / MSE)`` in decibels, over the mask if given.

    Returns ``+inf`` when the images are identical, which is the limit of the
    expression rather than a sentinel. Callers that serialize the result have
    to handle it; this function does not invent a finite stand-in.
    """
    error = mean_squared_error(reference, prediction, mask, data_range)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(data_range) ** 2 / error))


def structural_similarity_map(
    reference: np.ndarray,
    prediction: np.ndarray,
    settings: SsimSettings | None = None,
) -> tuple[float, np.ndarray]:
    """Full-frame SSIM and the local SSIM map behind it.

    Returns:
        ``(mean_ssim, ssim_map)``. ``mean_ssim`` is scikit-image's own figure,
        the mean of the map excluding the ``settings.border`` pixel rim the
        sliding window cannot cover. The map has the image's shape, with those
        border values present but not meaningful.
    """
    settings = settings or SsimSettings()
    validated_reference, validated_prediction = validate_pair(
        reference, prediction, settings.data_range
    )
    if min(validated_reference.shape) < settings.win_size:
        raise MetricError(
            f"image shape {validated_reference.shape} is smaller than the SSIM window "
            f"{settings.win_size}; SSIM is undefined there"
        )

    mean_ssim, ssim_map = structural_similarity(
        validated_reference,
        validated_prediction,
        data_range=settings.data_range,
        gaussian_weights=settings.gaussian_weights,
        sigma=settings.sigma,
        use_sample_covariance=settings.use_sample_covariance,
        win_size=settings.win_size,
        channel_axis=None,
        full=True,
    )
    return float(mean_ssim), np.asarray(ssim_map, dtype=np.float64)


def masked_ssim_map_mean(ssim_map: np.ndarray, interior_mask: np.ndarray) -> float:
    """Mean of a local SSIM map over an interior mask.

    This is what ``body_ssim`` is: the standard local SSIM map, averaged over
    body pixels whose whole SSIM neighbourhood lies inside the body. It is not
    a separate SSIM definition.

    Raises:
        MetricError: the mask is the wrong shape or selects nothing.
    """
    values = np.asarray(ssim_map, dtype=np.float64)
    selected = validate_mask(interior_mask, values.shape, "ssim interior mask")
    return float(values[selected].mean())


def slice_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    body_mask: np.ndarray,
    ssim_interior_mask: np.ndarray,
    settings: SsimSettings | None = None,
) -> dict[str, float]:
    """All eight numbers for one slice: four metrics in two regions.

    The SSIM map is computed once and reused for both regions, so the
    full-frame and body figures are guaranteed to come from the same map.

    Returns:
        ``{"full_mae", "full_mse", "full_psnr", "full_ssim", "body_mae",
        "body_mse", "body_psnr", "body_ssim"}``.
    """
    settings = settings or SsimSettings()
    validated_reference, validated_prediction = validate_pair(
        reference, prediction, settings.data_range
    )
    body = validate_mask(body_mask, validated_reference.shape, "body mask")
    interior = validate_mask(ssim_interior_mask, validated_reference.shape, "ssim interior mask")

    full_ssim, ssim_map = structural_similarity_map(
        validated_reference, validated_prediction, settings
    )
    return {
        "full_mae": mean_absolute_error(
            validated_reference, validated_prediction, None, settings.data_range
        ),
        "full_mse": mean_squared_error(
            validated_reference, validated_prediction, None, settings.data_range
        ),
        "full_psnr": peak_signal_noise_ratio(
            validated_reference, validated_prediction, None, settings.data_range
        ),
        "full_ssim": full_ssim,
        "body_mae": mean_absolute_error(
            validated_reference, validated_prediction, body, settings.data_range
        ),
        "body_mse": mean_squared_error(
            validated_reference, validated_prediction, body, settings.data_range
        ),
        "body_psnr": peak_signal_noise_ratio(
            validated_reference, validated_prediction, body, settings.data_range
        ),
        "body_ssim": masked_ssim_map_mean(ssim_map, interior),
    }
