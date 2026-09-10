"""CT windowing, normalization and resizing.

Pipeline order used by this project, and why each step sits where it does::

    stored DICOM pixels
        -> modality rescale (Hounsfield Units)     # quantitative calibration
        -> window in HU space                      # select the tissue range
        -> normalize to [0, 1]                     # ML input convention
        -> resize the normalized image             # geometry, last

* **HU conversion precedes windowing** because a window is defined in
  Hounsfield Units. ``center = 40, width = 400`` only means "abdominal soft
  tissue" once the pixels are calibrated; applied to raw stored values it would
  select an arbitrary and scanner-dependent range.

* **Windowing precedes normalization** because it defines the range being
  normalized. Normalizing by each image's own min and max instead would make
  the mapping from HU to [0, 1] different for every slice, so identical tissue
  would land on different input values and metrics would not be comparable
  across images.

* **Resizing comes last**, for three reasons that are worth stating precisely.

  First, note what is *not* a reason: the modality rescale is affine, and
  linear interpolation is a weighted average whose weights sum to one, so
  ``interpolate(a * x + b) == a * interpolate(x) + b``. Resizing before or
  after the rescale would give the same result up to floating-point error.
  Calibration order is not what forces resizing to the end.

  What does force it is that **windowing is not linear**. It clips, and
  ``clip`` does not commute with interpolation: resizing first lets a value far
  outside the window pull its neighbours across the window boundary, whereas
  clipping first bounds every contribution before any averaging happens.

  That non-commutation has a concrete consequence for **padding and other
  sentinel values**. Pixels outside the reconstruction circle can sit at
  extreme HU. Windowing first pins them to the window floor, so a padded pixel
  can influence its neighbours by at most the width of the window. Resizing
  first would let a sentinel at, say, -2000 HU drag a genuine soft-tissue
  neighbour far below its true value and leave a dark rim of pixels that
  correspond to no real tissue.

  Finally, **one fixed order is what makes the benchmark reproducible.** Every
  method, every image, and every milestone uses this same sequence, so any
  difference in the final metrics comes from the restoration method rather than
  from the preprocessing path a particular image happened to take.

What windowing intentionally discards: every HU value outside
``[center - width/2, center + width/2]`` is clipped to the range boundary. Bone
and metal above the upper bound, and lung and air below the lower bound, all
collapse to a single value. Models trained on this representation restore a
windowed 2D view of the scan, not the full CT dynamic range.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Interpolation methods offered for continuous-valued CT images.
#: Nearest-neighbour is deliberately absent: it duplicates pixels instead of
#: reconstructing intensity, which produces aliasing on downsampling and would
#: bias image-quality metrics.
_INTERPOLATIONS = {
    "area": cv2.INTER_AREA,  # pixel-area averaging; the anti-aliased choice for downsampling
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def window_ct(image_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip a Hounsfield Unit image to a CT window and normalize it to [0, 1].

    Convention used throughout this project::

        lower  = center - width / 2
        upper  = center + width / 2
        output = (clip(image_hu, lower, upper) - lower) / (upper - lower)

    This is an explicit machine-learning preprocessing definition. It is a
    linear ramp between the window boundaries, and it does not attempt to
    reproduce the VOI LUT behaviour or the sigmoid presentation that a
    particular DICOM viewer may apply.

    Args:
        image_hu: 2D array of Hounsfield Units.
        center: window centre in HU, integer or float.
        width: window width in HU, must be positive.

    Returns:
        A new ``float32`` array in [0, 1] with the same shape. The input is
        never modified.

    Raises:
        ValueError: the input is not 2D, or ``width`` is not positive.
    """
    array = np.asarray(image_hu)
    if array.ndim != 2:
        raise ValueError(f"window_ct expects a 2D image, got shape {array.shape}")

    width = float(width)
    center = float(center)
    if not width > 0:
        raise ValueError(f"Window width must be positive, got {width}")

    lower = center - width / 2.0
    upper = center + width / 2.0

    clipped = np.clip(array.astype(np.float64), lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.astype(np.float32)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int] = (256, 256),
    interpolation: str = "area",
) -> np.ndarray:
    """Resize a 2D image deterministically.

    Args:
        image: 2D array. No channel dimension is added.
        size: target ``(height, width)`` in NumPy order.
        interpolation: one of ``"area"``, ``"linear"``, ``"cubic"``. The default
            ``"area"`` averages over the source pixel area, which is the
            appropriate anti-aliased method for the downsampling this project
            performs (512x512 CT slices to 256x256). ``"linear"`` or
            ``"cubic"`` are better suited to upsampling.

    Returns:
        A new ``float32`` array of shape ``size``. The input is not modified.

    Raises:
        ValueError: the input is not 2D, the target size is not two positive
            integers, or the interpolation name is unknown.
    """
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"resize_image expects a 2D image, got shape {array.shape}")

    if len(size) != 2 or any(int(dimension) <= 0 for dimension in size):
        raise ValueError(f"size must be two positive integers (height, width), got {size}")

    if interpolation not in _INTERPOLATIONS:
        raise ValueError(
            f"Unknown interpolation {interpolation!r}; choose one of {sorted(_INTERPOLATIONS)}"
        )

    height, width = int(size[0]), int(size[1])
    # OpenCV takes dsize as (width, height), the transpose of the NumPy order.
    resized = cv2.resize(
        np.ascontiguousarray(array, dtype=np.float32),
        (width, height),
        interpolation=_INTERPOLATIONS[interpolation],
    )
    return np.asarray(resized, dtype=np.float32)


def preprocess_ct_slice(
    image_hu: np.ndarray,
    window_center: float,
    window_width: float,
    size: tuple[int, int] = (256, 256),
    interpolation: str = "area",
) -> np.ndarray:
    """Window, normalize and resize one HU slice into a model-ready image.

    Composes :func:`window_ct` and :func:`resize_image`, then clips the result
    back into [0, 1]. The final clip only matters for interpolation methods
    that can overshoot (``"cubic"``); it keeps the guarantee that every image
    entering a model or a metric lies in [0, 1].

    Returns:
        A ``float32`` array of shape ``size`` with values in [0, 1].
    """
    windowed = window_ct(image_hu, window_center, window_width)
    resized = resize_image(windowed, size=size, interpolation=interpolation)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)
