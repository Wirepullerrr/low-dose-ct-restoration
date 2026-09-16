"""Synthetic low-dose-like image-domain degradation.

This module defines the corruption that turns a clean reference image into the
model input for the whole benchmark. It is deliberately implemented before any
restoration method exists, so that no method can influence what it has to undo.

Scope, stated plainly
---------------------
This is **not** a physical dose simulator. CHAOS supplies reconstructed CT
DICOM images, not calibrated raw projection data and no scanner dose or noise
calibration, so nothing here corresponds to a particular mAs, a percentage dose
reduction, a photon count, a scanner-equivalent low-dose reconstruction, or a
clinically validated low-dose protocol. What it *is* is a controlled,
reproducible, image-domain corruption whose dominant characteristic -
increased, spatially varying, spatially correlated noise - is qualitatively the
kind of difference that separates a lower-dose reconstruction from a
higher-dose one. The intensity dependence is a declared heuristic, not a
physical derivation; see :func:`noise_scale`. That makes this a usable
restoration benchmark, and nothing more.

Where it applies
----------------
After the fixed Milestone 1 preprocessing, never before it::

    stored DICOM pixels
        -> HU conversion
        -> 40/400 HU window
        -> normalize to [0, 1]
        -> resize to 256x256
        -> DEGRADATION              <- this module
        -> restoration method input

So the restoration target is a windowed, normalized 2-D CT representation, not
the full original CT dynamic range.

The algorithm
-------------
For a clean image ``x`` in [0, 1]::

    sigma(x) = sigma_floor + sigma_signal * sqrt(x)
    epsilon  = unit-variance, zero-mean Gaussian field, spatially correlated
    degraded = clip(x + sigma(x) * epsilon, clip_min, clip_max)

Only noise is added. No blur, contrast compression, sharpening, streaks,
motion, gamma or histogram operation is applied. Piling several phenomena into
one corruption would make any later result much harder to attribute: a method
that improved the score could be undoing any one of them, or trading one
against another.

A caveat about the clipped background
-------------------------------------
The fixed CT window leaves large regions at exactly 0 or exactly 1, so this
degradation adds noise to pixels that preprocessing already saturated. That is
an artefact of operating on a windowed representation and does not reproduce
the physics of air, out-of-field padding or saturated dense structures. It is
one of the reasons the evaluation milestone may need both full-frame and
body-region metrics. No metric mask is defined here.

Determinism
-----------
The output is a pure function of the clean image, the config and the sample
key. It does not depend on call order, on any other slice having been
processed, on global NumPy random state, on the time, the hostname, the
absolute path or the process id. Each sample's seed is derived with SHA-256 and
drives a private ``Generator``; the global RNG is never read or written.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

#: The only algorithm this module implements. A config naming anything else is
#: refused rather than approximated, so that a future v2 definition can never be
#: silently executed by v1 code.
ALGORITHM_VERSION = "correlated_heteroscedastic_gaussian_v1"

#: Separator for the hashed seed payload. A POSIX filename may legally contain
#: any byte except "/" and NUL, this one included, so the separator is not
#: intrinsically safe. It is safe by *contract* instead: this benchmark forbids
#: it in sample keys, and :func:`canonical_sample_key` enforces that. With the
#: byte excluded from every field, the serialized payload is unambiguous and no
#: two distinct (version, seed, key) triples can produce the same payload.
_FIELD_SEPARATOR = "\x1f"

#: Bytes of the SHA-256 digest used as the seed. 64 bits is ample for the 6407
#: slices in this cohort and prints readably in the tracked summary.
_SEED_BYTES = 8

#: How far outside [0, 1] a clean image may stray before it is rejected. Only
#: float32 round-trip error is tolerated; genuinely unnormalized input is a
#: preprocessing bug and must surface as one.
RANGE_TOLERANCE = 1e-6

#: Standard deviations of the Gaussian correlation kernel retained. Pinned
#: explicitly rather than left to the library default, because the frozen
#: benchmark definition includes the exact filter.
CORRELATION_TRUNCATE = 4.0

#: Below this, the correlated field is treated as constant and left unscaled
#: instead of being divided by ~0.
_MIN_FIELD_STD = 1e-12


class DegradationError(ValueError):
    """The degradation configuration or input is not usable as given."""


def _require_integer(name: str, value: Any) -> int:
    """Accept only a genuine integer configuration value.

    Deliberately refuses to coerce. ``int(2026.9)`` is 2026, so a coercing
    reader would turn a typo into a different but plausible-looking frozen
    seed, and every slice in the benchmark would be reseeded without anything
    in the run reporting a problem. ``bool`` is refused too, since it is an
    ``int`` subclass in Python and ``True`` would otherwise read as seed 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise DegradationError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}. "
            "It is not rounded or truncated: a silently altered seed would change "
            "every slice's noise realization."
        )
    return int(value)


def _require_real(name: str, value: Any) -> float:
    """Accept only a real number, not a string that happens to parse as one."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise DegradationError(
            f"{name} must be a real number, got {type(value).__name__} {value!r}"
        )
    return float(value)


@dataclass(frozen=True)
class DegradationConfig:
    """Validated parameters of the canonical degradation.

    Raises:
        DegradationError: on construction, if any parameter is unusable.
    """

    algorithm: str = ALGORITHM_VERSION
    global_seed: int = 2026
    sigma_floor: float = 0.015
    sigma_signal: float = 0.035
    correlation_sigma_px: float = 0.6
    clip_min: float = 0.0
    clip_max: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM_VERSION:
            raise DegradationError(
                f"Unsupported degradation algorithm {self.algorithm!r}; this module implements "
                f"{ALGORITHM_VERSION!r} only. Implement the new version explicitly rather than "
                "running it through this one."
            )

        _require_integer("global_seed", self.global_seed)

        for name in ("sigma_floor", "sigma_signal", "correlation_sigma_px"):
            value = _require_real(name, getattr(self, name))
            if not np.isfinite(value):
                raise DegradationError(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise DegradationError(f"{name} must be non-negative, got {value}")

        _require_real("clip_min", self.clip_min)
        _require_real("clip_max", self.clip_max)
        if not np.isfinite(self.clip_min) or not np.isfinite(self.clip_max):
            raise DegradationError(
                f"clip bounds must be finite, got [{self.clip_min}, {self.clip_max}]"
            )
        if not self.clip_min < self.clip_max:
            raise DegradationError(
                f"clip_min must be strictly less than clip_max, got "
                f"[{self.clip_min}, {self.clip_max}]"
            )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DegradationConfig:
        """Build a config from a loaded YAML document.

        Accepts either the whole config file, which nests the parameters under
        a ``degradation`` key, or that inner mapping on its own.

        Raises:
            DegradationError: the mapping is not a mapping, is missing a
                required key, or carries an unrecognised one. Unknown keys are
                refused rather than ignored, because a silently dropped
                parameter would mean the file on disk no longer describes what
                actually ran.
        """
        if not isinstance(mapping, dict):
            raise DegradationError(
                f"Degradation config must be a mapping, got {type(mapping).__name__}"
            )
        section = mapping.get("degradation", mapping)
        if not isinstance(section, dict):
            raise DegradationError(
                f"'degradation' section must be a mapping, got {type(section).__name__}"
            )

        # Names only. Values are passed through untouched so that __post_init__
        # can judge them: casting here would silently repair malformed input,
        # which is precisely what a frozen experiment definition must not do.
        fields = (
            "algorithm",
            "global_seed",
            "sigma_floor",
            "sigma_signal",
            "correlation_sigma_px",
            "clip_min",
            "clip_max",
        )
        missing = sorted(set(fields) - set(section))
        if missing:
            raise DegradationError(f"Degradation config is missing key(s): {missing}")
        unknown = sorted(set(section) - set(fields))
        if unknown:
            raise DegradationError(
                f"Degradation config has unrecognised key(s): {unknown}. "
                f"Known keys are {sorted(fields)}."
            )

        return cls(**{name: section[name] for name in fields})

    def as_dict(self) -> dict[str, Any]:
        """Plain-data view, for embedding in tracked summaries."""
        return {
            "algorithm": self.algorithm,
            "global_seed": int(self.global_seed),
            "sigma_floor": float(self.sigma_floor),
            "sigma_signal": float(self.sigma_signal),
            "correlation_sigma_px": float(self.correlation_sigma_px),
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
        }


def canonical_sample_key(sample_key: str | PurePath) -> str:
    """Normalize a sample key so one logical slice always hashes identically.

    Path separators are the only thing canonicalized: ``Train_Sets\\CT\\1\\a.dcm``
    and ``Train_Sets/CT/1/a.dcm`` name the same slice and must derive the same
    seed, whether the caller is on Windows or POSIX and whether the key arrived
    as a string or a path object.

    Nothing else is rewritten. Surrounding whitespace, case and duplicated
    slashes are left alone deliberately: quietly repairing them would hide the
    upstream bug that produced them, and the keys this project actually uses
    come from the frozen slice manifest, which is already POSIX-style.

    One byte is forbidden. The seed payload joins the algorithm version, the
    global seed and this key with U+001F, so a key containing that byte could
    make two distinct triples serialize to the same payload. A POSIX filename
    may legally contain it, so it is excluded here by contract rather than
    assumed absent. No CHAOS sample key contains it.

    Raises:
        DegradationError: the key is empty, only whitespace, or contains the
            reserved payload separator.
    """
    text = str(sample_key).replace("\\", "/")
    if not text.strip():
        raise DegradationError(
            "sample_key must be a non-empty string; an empty key would give every "
            "slice the same noise realization."
        )
    if _FIELD_SEPARATOR in text:
        raise DegradationError(
            "sample_key must not contain U+001F, the reserved seed-payload field "
            f"separator; got {text!r}. Two different keys containing it could "
            "serialize to the same payload and collide onto a single seed."
        )
    return text


def derive_sample_seed(
    global_seed: int,
    sample_key: str | PurePath,
    algorithm_version: str = ALGORITHM_VERSION,
) -> int:
    """Derive one slice's seed from the experiment definition and its key.

    SHA-256 over ``algorithm_version``, ``global_seed`` and the canonical key.
    Python's built-in :func:`hash` is unusable here: it is randomized per
    process for strings, so the same slice would get different noise in
    different runs, and results would not be reproducible at all.

    Returns:
        A 64-bit integer seed, a deterministic function of its inputs on every
        platform, process and run.
    """
    payload = _FIELD_SEPARATOR.join(
        (str(algorithm_version), str(int(global_seed)), canonical_sample_key(sample_key))
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


def validate_clean_image(image: np.ndarray, config: DegradationConfig) -> np.ndarray:
    """Check that an array really is the clean reference representation.

    Deliberately strict. Silently coercing an out-of-range or non-finite array
    would mask a preprocessing defect and quietly change what the benchmark
    measures, so every violation raises instead.

    Returns:
        The input as ``float64``, for the arithmetic below. A new array; the
        caller's array is never touched.

    Raises:
        DegradationError: not 2-D, not numeric, not finite, or outside
            ``[clip_min, clip_max]`` by more than :data:`RANGE_TOLERANCE`.
    """
    array = np.asarray(image)
    if array.ndim != 2:
        raise DegradationError(
            f"degradation expects a 2D image, got shape {array.shape}. Degradation runs on one "
            "preprocessed slice at a time; batching belongs to a later milestone."
        )
    if array.size == 0:
        raise DegradationError("degradation expects a non-empty image, got an empty array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise DegradationError(f"degradation expects a real numeric image, got dtype {array.dtype}")

    array = array.astype(np.float64)
    if not np.all(np.isfinite(array)):
        count = int((~np.isfinite(array)).sum())
        raise DegradationError(
            f"degradation expects finite values, found {count} NaN or infinite pixel(s)"
        )

    low, high = float(config.clip_min), float(config.clip_max)
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < low - RANGE_TOLERANCE or maximum > high + RANGE_TOLERANCE:
        raise DegradationError(
            f"degradation expects a normalized image in [{low}, {high}], got "
            f"[{minimum:.6g}, {maximum:.6g}]. Run the Milestone 1 preprocessing first; "
            "this function does not normalize arbitrary input."
        )
    return array


def noise_scale(image: np.ndarray, config: DegradationConfig) -> np.ndarray:
    """Per-pixel noise standard deviation ``sigma(x) = floor + signal*sqrt(x)``.

    Heteroscedastic by design: the noise magnitude varies across the image
    rather than being one constant for the whole frame. The motivation is that
    noise in a reconstructed low-dose CT image is not generally spatially
    uniform either. The motivation stops there.

    What this function is: **a simple heuristic** that makes the corruption
    spatially varying and trivially reproducible, using the one quantity
    available at this point in the pipeline, the local normalized intensity.
    The ``sqrt`` shape is a modelling convention, chosen so the noise grows
    with intensity but sub-linearly.

    What it is **not**: it is not derived from projection-domain photon counts,
    from line-integral attenuation, from scanner calibration, or from any
    physical noise model. A voxel's intensity is not the number of photons
    detected anywhere. Real image-domain noise at a point depends on the whole
    set of ray paths crossing it and on the reconstruction algorithm, none of
    which is represented here. So this does not claim that a brighter pixel
    physically receives the amount of noise configured for it; it claims only
    that the corruption is non-uniform in a fixed, declared, reproducible way.

    ``sigma_floor`` keeps noise present in the darkest regions, where a pure
    ``sqrt`` term would vanish and leave the background implausibly clean.

    Returns:
        A ``float64`` array of the same shape, elementwise non-negative.
    """
    array = np.asarray(image, dtype=np.float64)
    # Clamp before the square root so that a value a hair below zero, within
    # the tolerated numerical slack, cannot produce NaN.
    return float(config.sigma_floor) + float(config.sigma_signal) * np.sqrt(
        np.clip(array, 0.0, None)
    )


def correlated_gaussian_field(
    shape: tuple[int, ...],
    seed: int,
    correlation_sigma_px: float,
) -> np.ndarray:
    """A zero-mean, unit-variance Gaussian field with optional spatial correlation.

    White Gaussian noise is drawn from a ``Generator`` private to this call,
    optionally smoothed with a Gaussian filter under reflect boundary handling,
    then recentred and rescaled to zero mean and unit standard deviation.

    The rescaling matters: smoothing averages neighbouring samples and so
    *reduces* variance, by a factor that depends on the correlation width. Left
    uncorrected, a wider correlation would silently mean weaker noise, and the
    correlation width would be entangled with the noise magnitude. Normalizing
    afterwards keeps ``sigma`` the only thing that sets amplitude.

    ``correlation_sigma_px = 0`` skips the filter entirely and yields
    pixel-independent noise.

    Returns:
        A ``float64`` array of the requested shape.
    """
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    field = generator.standard_normal(size=shape)

    if float(correlation_sigma_px) > 0:
        field = gaussian_filter(
            field,
            sigma=float(correlation_sigma_px),
            mode="reflect",
            truncate=CORRELATION_TRUNCATE,
        )

    field = field - field.mean()
    deviation = float(field.std())
    # A constant field has nothing to rescale; dividing would be a 0/0. This is
    # reachable for a 1-pixel image, where the single sample is its own mean.
    if deviation > _MIN_FIELD_STD:
        field = field / deviation
    return field


def degrade_low_dose_like(
    clean_image: np.ndarray,
    sample_key: str | PurePath,
    config: DegradationConfig | None = None,
) -> np.ndarray:
    """Apply the canonical synthetic low-dose-like degradation to one slice.

    A pure function of its three arguments. The same clean image, key and
    config always give a byte-identical result, on any run, in any order,
    regardless of what else has been processed or what the global NumPy random
    state contains.

    Args:
        clean_image: 2-D preprocessed reference in [0, 1]. Never modified.
        sample_key: stable identity of this slice. For CHAOS this is the
            ``relative_dicom_path`` of the frozen slice manifest.
        config: defaults to the canonical parameters.

    Returns:
        A new ``float32`` array of the same shape, within
        ``[clip_min, clip_max]``.

    Raises:
        DegradationError: the image or the key is unusable.
    """
    config = config or DegradationConfig()
    clean = validate_clean_image(clean_image, config)
    seed = derive_sample_seed(config.global_seed, sample_key, config.algorithm)

    field = correlated_gaussian_field(clean.shape, seed, config.correlation_sigma_px)
    degraded = clean + noise_scale(clean, config) * field
    degraded = np.clip(degraded, float(config.clip_min), float(config.clip_max))
    return degraded.astype(np.float32)
