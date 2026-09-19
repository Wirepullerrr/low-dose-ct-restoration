"""CLAHE: the classical comparison method.

Contrast Limited Adaptive Histogram Equalization, via OpenCV. It is **not** a
denoiser and carries no trained parameters. What it does is redistribute
intensities locally, tile by tile, so that low-contrast structure becomes more
visible.

Whether that helps on a noisy CT representation is an open question this
milestone measures rather than assumes. CLAHE spreading local intensities
apart amplifies whatever is in a tile, and in a degraded image part of what is
in a tile is noise. It is entirely possible for every CLAHE configuration to
score worse than returning the degraded image untouched, and if that is what
the measurement says, that is what gets reported.

What CLAHE sees
---------------
Exactly the degraded image, and nothing else. No clean reference, no HU slice,
no body mask, no acquisition group, no patient identity. The mask in
particular is evaluation-only: applying CLAHE inside it would feed the method
a region derived from the clean reference, which is information a deployed
method would not have.

The 8-bit conversion is part of the method
------------------------------------------
OpenCV's CLAHE builds per-tile histograms and needs an integer single-channel
image, so the [0, 1] benchmark representation is quantized to 8 bits, enhanced,
and mapped back. That quantization is a **defining part of this method**, not
an incidental implementation detail, and it is declared in the config as
``input_quantization_bits``.

The mapping is fixed: ``round(x * 255)``, using the whole 0-255 range for the
whole [0, 1] benchmark range. It is never per-image min/max normalization,
which would give a different mapping to every slice and silently destroy the
comparability that the fixed 40/400 HU window exists to provide.

The cost of quantizing: the 400 HU window spread over 256 levels is roughly
1.6 HU per level in the linear portion of the window, and the round-trip error
is at most ``0.5 / 255`` in normalized units, about 0.8 HU. That is small
relative to the degradation's own perturbation scale, which M4 measured at a
per-slice standard deviation around 0.026, but it is an approximation all the
same. It is not a claim that 8 bits is clinically lossless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

#: The only CLAHE variant this module implements. A config naming anything
#: else is refused rather than approximated.
ALGORITHM_VERSION = "opencv_clahe_v1"

#: Bit depth of the integer image handed to OpenCV.
QUANTIZATION_BITS = 8

#: Highest integer level, ``2**QUANTIZATION_BITS - 1``.
QUANTIZATION_MAX = 255

#: Largest possible round-trip error of the fixed quantization, in normalized
#: units: half a level.
QUANTIZATION_ERROR_BOUND = 0.5 / QUANTIZATION_MAX

#: How far outside [0, 1] an input may stray before it is rejected. Only
#: float32 round-trip error is tolerated.
RANGE_TOLERANCE = 1e-6


class ClaheError(ValueError):
    """The CLAHE configuration or input is not usable as given."""


@dataclass(frozen=True)
class ClaheConfig:
    """Parameters of one CLAHE configuration.

    ``clip_limit``
        How far any single histogram bin may rise before the excess is
        redistributed across the tile. This is the "contrast limited" part.
        Without it, a tile of near-uniform tissue would have its tiny
        intensity range stretched across the full output range, turning noise
        into visible texture. A lower limit means a gentler enhancement.

    ``tile_grid_size``
        ``(rows, columns)`` of tiles the image is divided into, each
        equalized on its own histogram and then bilinearly blended with its
        neighbours. On a 256x256 image, ``(4, 4)`` gives 64x64-pixel tiles,
        ``(8, 8)`` gives 32x32, and ``(16, 16)`` gives 16x16. Smaller tiles
        adapt to finer local structure but see fewer pixels, so their
        histograms are noisier and their enhancement is more easily driven by
        noise; larger tiles behave more like global equalization.
    """

    algorithm: str = ALGORITHM_VERSION
    input_quantization_bits: int = QUANTIZATION_BITS
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM_VERSION:
            raise ClaheError(
                f"Unsupported CLAHE algorithm {self.algorithm!r}; this module implements "
                f"{ALGORITHM_VERSION!r} only."
            )
        if (
            isinstance(self.input_quantization_bits, bool)
            or not isinstance(self.input_quantization_bits, int)
            or self.input_quantization_bits != QUANTIZATION_BITS
        ):
            raise ClaheError(
                f"input_quantization_bits must be {QUANTIZATION_BITS}, got "
                f"{self.input_quantization_bits!r}. The 8-bit conversion is part of this "
                "method's definition; a different depth is a different method."
            )

        if isinstance(self.clip_limit, bool) or not isinstance(self.clip_limit, int | float):
            raise ClaheError(f"clip_limit must be a real number, got {self.clip_limit!r}")
        if not np.isfinite(self.clip_limit) or self.clip_limit <= 0:
            raise ClaheError(f"clip_limit must be finite and positive, got {self.clip_limit}")

        grid = self.tile_grid_size
        if isinstance(grid, str) or not hasattr(grid, "__len__") or len(grid) != 2:
            raise ClaheError(
                f"tile_grid_size must be two positive integers (rows, columns), got {grid!r}"
            )
        for value in grid:
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise ClaheError(
                    f"tile_grid_size entries must be integers, got {grid!r}. A bool is not "
                    "accepted as a grid dimension."
                )
            if int(value) < 1:
                raise ClaheError(f"tile_grid_size entries must be >= 1, got {grid!r}")

    @property
    def tile_rows(self) -> int:
        return int(self.tile_grid_size[0])

    @property
    def tile_columns(self) -> int:
        return int(self.tile_grid_size[1])

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ClaheConfig:
        """Build a config from a loaded YAML document.

        Accepts the whole file, which nests parameters under ``clahe``, or that
        inner mapping alone. Values are passed through untouched so that
        validation, not silent coercion, decides whether they are usable.

        Raises:
            ClaheError: not a mapping, or a missing or unrecognised key.
        """
        if not isinstance(mapping, dict):
            raise ClaheError(f"CLAHE config must be a mapping, got {type(mapping).__name__}")
        section = mapping.get("clahe", mapping)
        if not isinstance(section, dict):
            raise ClaheError(f"'clahe' section must be a mapping, got {type(section).__name__}")

        fields = ("algorithm", "input_quantization_bits", "clip_limit", "tile_grid_size")
        # selected_by is provenance recorded alongside the runtime definition.
        known = {*fields, "selected_by"}
        missing = sorted(set(fields) - set(section))
        if missing:
            raise ClaheError(f"CLAHE config is missing key(s): {missing}")
        unknown = sorted(set(section) - known)
        if unknown:
            raise ClaheError(
                f"CLAHE config has unrecognised key(s): {unknown}. Known keys are {sorted(known)}."
            )

        grid = section["tile_grid_size"]
        if isinstance(grid, list):
            grid = tuple(grid)
        return cls(
            algorithm=section["algorithm"],
            input_quantization_bits=section["input_quantization_bits"],
            clip_limit=section["clip_limit"],
            tile_grid_size=grid,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "input_quantization_bits": int(self.input_quantization_bits),
            "clip_limit": float(self.clip_limit),
            "tile_grid_size": [self.tile_rows, self.tile_columns],
        }

    @property
    def label(self) -> str:
        """Short human-readable identity, for tables and figure titles."""
        return f"clip{self.clip_limit:g}_grid{self.tile_rows}x{self.tile_columns}"


def validate_image(image: np.ndarray) -> np.ndarray:
    """Check that an array is the normalized benchmark representation.

    Strict by design: CLAHE does not normalize, resize or repair its input.
    An out-of-range array means something upstream is wrong, and quietly
    rescaling it would change what the method is being measured on.

    Returns:
        A new ``float64`` array. The caller's array is never modified.

    Raises:
        ClaheError: not a 2-D finite real image within [0, 1].
    """
    values = np.asarray(image)
    if values.ndim != 2:
        raise ClaheError(f"CLAHE expects a 2D image, got shape {values.shape}")
    if values.size == 0:
        raise ClaheError("CLAHE expects a non-empty image, got an empty array")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise ClaheError(f"CLAHE expects a real numeric image, got dtype {values.dtype}")

    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        count = int((~np.isfinite(values)).sum())
        raise ClaheError(f"CLAHE expects finite values, found {count} NaN or infinite pixel(s)")

    minimum, maximum = float(values.min()), float(values.max())
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise ClaheError(
            f"CLAHE expects a normalized image in [0, 1], got [{minimum:.6g}, {maximum:.6g}]. "
            "It does not normalize arbitrary input; run the established preprocessing first."
        )
    return values


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Quantize a normalized [0, 1] image to 8 bits: ``round(x * 255)``.

    The mapping is fixed to the benchmark range, never to the image's own
    minimum and maximum. Rounding is NumPy's round-half-to-even, pinned here so
    the method is reproducible rather than dependent on a rounding default.
    """
    values = validate_image(image)
    scaled = np.rint(np.clip(values, 0.0, 1.0) * QUANTIZATION_MAX)
    return np.asarray(scaled, dtype=np.uint8)


def from_uint8(image: np.ndarray) -> np.ndarray:
    """Map an 8-bit image back to ``float32`` in [0, 1]."""
    values = np.asarray(image)
    if values.dtype != np.uint8:
        raise ClaheError(f"from_uint8 expects a uint8 image, got dtype {values.dtype}")
    return np.asarray(values, dtype=np.float32) / np.float32(QUANTIZATION_MAX)


def apply_clahe(degraded: np.ndarray, config: ClaheConfig | None = None) -> np.ndarray:
    """Enhance one degraded slice with CLAHE.

    A pure function of the image and the config: no random state, no global
    state, no dependence on what was processed before.

    Args:
        degraded: 2-D normalized image in [0, 1]. Never modified.
        config: the frozen CLAHE parameters.

    Returns:
        A new ``float32`` array of the same shape, within [0, 1].

    Raises:
        ClaheError: the image is not the expected representation.
    """
    config = config or ClaheConfig()
    quantized = to_uint8(degraded)
    operator = cv2.createCLAHE(
        clipLimit=float(config.clip_limit),
        tileGridSize=(config.tile_columns, config.tile_rows),
    )
    enhanced = operator.apply(quantized)
    return from_uint8(np.asarray(enhanced, dtype=np.uint8))


def clahe_restorer(config: ClaheConfig | None = None):
    """Bind a config into the harness's one-argument restoration contract.

    The returned callable takes only the degraded image, which is the whole
    point: a method cannot reach the clean reference, the mask or the patient.
    """
    config = config or ClaheConfig()

    def restore(degraded: np.ndarray) -> np.ndarray:
        return apply_clahe(degraded, config)

    return restore
