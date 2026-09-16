"""Evaluation policy: the body mask, the aggregation, and the hold-out gate.

This module holds everything about *how* the benchmark is scored that is not a
metric formula. Like the split and the degradation before it, the policy is
frozen in a tracked config and applied identically to every method, so that a
later comparison reflects the methods and not the measuring apparatus.

The evaluation body mask
------------------------
Roughly 54 percent of clean training pixels are exactly 0 after the fixed
40/400 HU window, so a full-frame metric assigns that region substantial
weight and the background can materially move the score.

Not because the background is trivially correct. The frozen degradation uses
``sigma_floor = 0.015``, so clipped background pixels are perturbed too, and
about half of them leave zero. What the background does is affect the metric
*families* differently. Under the intensity-dependent degradation its
corruption is smaller than much of the body's, which makes full-frame MAE, MSE
and PSNR look easier. SSIM reacts the other way: adding noise to a
low-variance flat region changes its local luminance, contrast and variance
statistics sharply, so SSIM there is lower than inside the body.

Full-frame numbers are reported because they describe the whole output
honestly, and each is paired with a body-region number because neither region
alone tells the whole story.

The mask is a **crude deterministic silhouette, not clinical segmentation**. It
separates "inside the patient" from "air and table", nothing more, and no
anatomical claim is made about it.

Three properties matter more than its accuracy:

* it is derived from the **clean HU slice**, never from a degraded or restored
  image, so a method cannot influence the region it is judged on;
* it is **never given to a restoration method**, so it is not a hidden input;
* it is **identical for every method** on a given slice, so the comparison is
  between methods rather than between regions.

The patient is the experimental unit
------------------------------------
Slices from one patient are strongly correlated: neighbouring slices are 1 to 2
mm apart, from one acquisition, one reconstruction and one anatomy. Treating
them as independent observations would overstate the evidence and would let a
patient with 294 slices count nearly four times as much as one with 78. So
per-slice metrics are averaged within a patient first, and patients are then
averaged with equal weight. A slice-weighted figure is computed too, but only
as a secondary descriptive diagnostic.

Hold-out discipline
-------------------
:func:`require_development_split` is the gate: test and stress are not
reachable from the development commands at all.

Stated precisely, because the honest claim is narrower than "never read".
Before the patient split was frozen, the Milestone 2 cohort audit
characterized all 40 subjects, which was dataset-level technical QC rather
than model selection. Since the split was frozen, test and stress image
content has not been used for degradation design, evaluation-policy
development, model selection, or benchmark scoring, and the commands written
after the freeze refuse those splits outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from ct_restoration.metrics import METRIC_NAMES, REGION_NAMES, SsimSettings

#: Splits whose image content this stage of the project may read.
DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "validation")

#: Splits sealed until the final benchmark. Reading their image content now
#: would turn a held-out estimate into a fitted one.
SEALED_SPLITS: tuple[str, ...] = ("test", "stress")

#: Column order of the per-slice metric table.
METRIC_COLUMNS: tuple[str, ...] = tuple(
    f"{region}_{metric}" for region in REGION_NAMES for metric in METRIC_NAMES
)

#: Delta degrees of freedom for the reported standard deviation. 1 gives the
#: sample standard deviation, the convention for describing spread in a small
#: group of observations such as six patients.
STD_DDOF = 1


class EvaluationError(ValueError):
    """The evaluation configuration or request is not usable as given."""


class HeldOutSplitError(EvaluationError):
    """An attempt to read image content from a sealed split."""


@dataclass(frozen=True)
class BodyMaskSettings:
    """Parameters of the evaluation body silhouette.

    ``threshold_hu`` sits between air, around -1000 HU, and soft tissue, around
    0 to 60 HU. It is far from both, so the silhouette does not hinge on a
    finely tuned cut. It was fixed in advance and never chosen by comparing
    metrics; doing that would tune the evaluation region against the scores it
    produces.
    """

    threshold_hu: float = -500.0
    connectivity: int = 8
    keep_largest_component: bool = True
    fill_holes: bool = True
    resize_interpolation: str = "nearest"

    def __post_init__(self) -> None:
        if isinstance(self.threshold_hu, bool) or not isinstance(self.threshold_hu, int | float):
            raise EvaluationError(f"threshold_hu must be a real number, got {self.threshold_hu!r}")
        if not np.isfinite(self.threshold_hu):
            raise EvaluationError(f"threshold_hu must be finite, got {self.threshold_hu}")
        if self.connectivity != 8:
            raise EvaluationError(
                f"connectivity must be 8, got {self.connectivity}. The frozen policy uses "
                "8-connectivity; implement a new policy explicitly rather than varying this."
            )
        if not self.keep_largest_component:
            raise EvaluationError("keep_largest_component must be true under the frozen policy")
        if not self.fill_holes:
            raise EvaluationError("fill_holes must be true under the frozen policy")
        if self.resize_interpolation != "nearest":
            raise EvaluationError(
                f"resize_interpolation must be 'nearest', got {self.resize_interpolation!r}. "
                "Any interpolating method would produce fractional values in a binary mask."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold_hu": float(self.threshold_hu),
            "connectivity": int(self.connectivity),
            "keep_largest_component": bool(self.keep_largest_component),
            "fill_holes": bool(self.fill_holes),
            "resize_interpolation": str(self.resize_interpolation),
        }


@dataclass(frozen=True)
class AggregationSettings:
    """How per-slice numbers become one split-level number."""

    slice_to_patient: str = "arithmetic_mean"
    patient_to_split: str = "arithmetic_mean"
    primary_unit: str = "patient"

    def __post_init__(self) -> None:
        for name in ("slice_to_patient", "patient_to_split"):
            value = getattr(self, name)
            if value != "arithmetic_mean":
                raise EvaluationError(f"{name} must be 'arithmetic_mean', got {value!r}")
        if self.primary_unit != "patient":
            raise EvaluationError(
                f"primary_unit must be 'patient', got {self.primary_unit!r}. The slice is not "
                "an independent experimental unit in this benchmark."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice_to_patient": self.slice_to_patient,
            "patient_to_split": self.patient_to_split,
            "primary_unit": self.primary_unit,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """The whole frozen evaluation policy."""

    data_range: float = 1.0
    ssim: SsimSettings = field(default_factory=SsimSettings)
    body_mask: BodyMaskSettings = field(default_factory=BodyMaskSettings)
    aggregation: AggregationSettings = field(default_factory=AggregationSettings)

    def __post_init__(self) -> None:
        if isinstance(self.data_range, bool) or not isinstance(self.data_range, int | float):
            raise EvaluationError(f"data_range must be a real number, got {self.data_range!r}")
        if not np.isfinite(self.data_range) or self.data_range <= 0:
            raise EvaluationError(f"data_range must be finite and positive, got {self.data_range}")
        if float(self.data_range) != float(self.ssim.data_range):
            raise EvaluationError(
                f"data_range {self.data_range} disagrees with the SSIM data_range "
                f"{self.ssim.data_range}; one benchmark cannot have two dynamic ranges"
            )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> EvaluationConfig:
        """Build the policy from a loaded YAML document.

        Unknown and missing keys are refused rather than defaulted, because a
        silently ignored key would mean the tracked file no longer describes
        what actually ran.
        """
        if not isinstance(mapping, dict):
            raise EvaluationError(
                f"Evaluation config must be a mapping, got {type(mapping).__name__}"
            )
        section = mapping.get("evaluation", mapping)
        if not isinstance(section, dict):
            raise EvaluationError(
                f"'evaluation' section must be a mapping, got {type(section).__name__}"
            )

        expected = {"data_range", "ssim", "body_mask", "aggregation"}
        missing = sorted(expected - set(section))
        if missing:
            raise EvaluationError(f"Evaluation config is missing key(s): {missing}")
        unknown = sorted(set(section) - expected)
        if unknown:
            raise EvaluationError(
                f"Evaluation config has unrecognised key(s): {unknown}. "
                f"Known keys are {sorted(expected)}."
            )

        data_range = section["data_range"]
        ssim = _build(SsimSettings, section["ssim"], "ssim", extra={"data_range": data_range})
        body_mask = _build(BodyMaskSettings, section["body_mask"], "body_mask")
        aggregation = _build(AggregationSettings, section["aggregation"], "aggregation")
        return cls(data_range=data_range, ssim=ssim, body_mask=body_mask, aggregation=aggregation)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_range": float(self.data_range),
            "ssim": self.ssim.as_dict(),
            "body_mask": self.body_mask.as_dict(),
            "aggregation": self.aggregation.as_dict(),
        }


def _build(cls: type, section: Any, name: str, extra: dict[str, Any] | None = None) -> Any:
    """Construct a settings dataclass from a mapping, refusing surprises."""
    if not isinstance(section, dict):
        raise EvaluationError(f"'{name}' must be a mapping, got {type(section).__name__}")
    allowed = {f for f in cls.__dataclass_fields__ if f not in (extra or {})}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise EvaluationError(f"'{name}' has unrecognised key(s): {unknown}")
    missing = sorted(allowed - set(section))
    if missing:
        raise EvaluationError(f"'{name}' is missing key(s): {missing}")
    return cls(**section, **(extra or {}))


def require_development_split(split: str) -> str:
    """Gate every command that reads image content during development.

    Raises:
        HeldOutSplitError: ``split`` is sealed.
        EvaluationError: ``split`` is not a partition of this benchmark.
    """
    if split in SEALED_SPLITS:
        raise HeldOutSplitError(
            f"Refusing to read image content from the {split!r} split. Test and stress are "
            "sealed until the final benchmark, after every method decision is frozen. "
            f"This command accepts {list(DEVELOPMENT_SPLITS)}."
        )
    if split not in DEVELOPMENT_SPLITS:
        raise EvaluationError(
            f"Unknown split {split!r}; this command accepts {list(DEVELOPMENT_SPLITS)}"
        )
    return split


def body_mask_from_hu(
    image_hu: np.ndarray,
    size: tuple[int, int] = (256, 256),
    settings: BodyMaskSettings | None = None,
) -> np.ndarray:
    """Build the evaluation body mask from a clean HU slice.

    The frozen rule, in order:

    1. threshold ``HU > threshold_hu``;
    2. label 2-D connected components with 8-connectivity;
    3. keep the largest component, which is the patient rather than the table
       rail or a stray bright speck;
    4. fill enclosed holes, so that low-HU structures inside the body - bowel
       gas, lung base - are evaluated rather than cut out;
    5. resize to the evaluation size with nearest-neighbour interpolation;
    6. return as ``bool``.

    Nearest-neighbour is required at step 5 and is the one place this project
    uses it: any interpolating method would produce fractional values along the
    boundary of a binary mask, which is not a mask. The continuous-image
    resizing in :mod:`ct_restoration.data.preprocessing` deliberately excludes
    it for the opposite reason.

    Ties at step 3 resolve to the lowest component label, which makes the
    result deterministic even for two components of exactly equal area.

    Args:
        image_hu: 2-D Hounsfield Unit slice, before windowing. Not modified.
        size: evaluation image size as ``(height, width)``.
        settings: frozen mask parameters.

    Returns:
        A new boolean array of shape ``size``.

    Raises:
        EvaluationError: the input is not a usable 2-D HU slice, or no pixel
            exceeds the threshold, which would mean the slice contains no body.
    """
    settings = settings or BodyMaskSettings()
    values = np.asarray(image_hu)
    if values.ndim != 2:
        raise EvaluationError(f"body mask expects a 2D HU slice, got shape {values.shape}")
    if values.size == 0:
        raise EvaluationError("body mask expects a non-empty HU slice")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise EvaluationError(f"body mask expects a real numeric HU slice, got {values.dtype}")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise EvaluationError("body mask expects finite HU values")

    if len(size) != 2 or any(int(dimension) <= 0 for dimension in size):
        raise EvaluationError(f"size must be two positive integers (height, width), got {size}")

    candidate = values > float(settings.threshold_hu)
    if not candidate.any():
        raise EvaluationError(
            f"No pixel exceeds {settings.threshold_hu} HU, so the slice contains no body to "
            "evaluate. This is a data problem, not something to work around here."
        )

    # A 3x3 all-ones structuring element is 8-connectivity.
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=int))
    if count == 0:  # pragma: no cover - unreachable given the check above
        raise EvaluationError("body mask found no connected component above the threshold")

    areas = np.bincount(labels.ravel())
    areas[0] = 0  # label 0 is background
    mask = labels == int(areas.argmax())

    if settings.fill_holes:
        mask = ndimage.binary_fill_holes(mask)

    height, width = int(size[0]), int(size[1])
    # OpenCV takes dsize as (width, height), the transpose of the NumPy order.
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized, dtype=bool)


def ssim_interior_mask(body_mask: np.ndarray, win_size: int = 11) -> np.ndarray:
    """Body pixels whose whole SSIM neighbourhood lies inside the body.

    SSIM is defined on a local window, so a body pixel whose window overlaps
    background carries a value that partly describes background. Eroding the
    mask by an all-true ``win_size`` square keeps only centres whose complete
    neighbourhood is body.

    Erosion treats outside-the-array as background, so pixels within
    ``(win_size - 1) // 2`` of the image edge are removed too. That is correct
    and deliberate: scikit-image cannot compute a meaningful SSIM value there
    either, and excludes the same rim from its own mean.

    Args:
        body_mask: boolean mask from :func:`body_mask_from_hu`. Not modified.
        win_size: the frozen SSIM window size.

    Returns:
        A new boolean array of the same shape. May be empty for a slice whose
        body is thinner than the window; callers must handle that explicitly
        rather than silently averaging nothing.

    Raises:
        EvaluationError: the mask is not 2-D boolean-like, or ``win_size`` is
            not an odd integer of at least 3.
    """
    mask = np.asarray(body_mask)
    if mask.ndim != 2:
        raise EvaluationError(f"body mask must be 2D, got shape {mask.shape}")
    if mask.dtype != bool:
        unique = set(np.unique(mask).tolist())
        if not unique <= {0, 1}:
            raise EvaluationError(f"body mask must be boolean or 0/1, found {sorted(unique)}")
        mask = mask.astype(bool)
    if isinstance(win_size, bool) or not isinstance(win_size, int):
        raise EvaluationError(f"win_size must be an integer, got {win_size!r}")
    if win_size < 3 or win_size % 2 == 0:
        raise EvaluationError(f"win_size must be an odd integer >= 3, got {win_size}")

    structure = np.ones((win_size, win_size), dtype=bool)
    return np.asarray(ndimage.binary_erosion(mask, structure=structure, border_value=0), dtype=bool)


def prepare_evaluation_slice(
    dicom_path: Path | str,
    preprocessing: dict[str, Any],
    config: EvaluationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the clean reference and both evaluation masks for one slice.

    One code path, shared by the mask audit and by every method's evaluation,
    so that the reference image and the region a method is scored on cannot
    drift apart between commands.

    Note the order: the mask comes from the **HU slice before windowing**,
    while the reference comes from the full preprocessing. The mask is a
    property of the anatomy, not of the windowed representation, and deriving
    it before the window keeps it independent of that choice.

    Returns:
        ``(clean_reference, body_mask, ssim_interior_mask)``. The reference is
        ``float32`` in [0, 1]; both masks are ``bool`` of the same shape.
    """
    from ct_restoration.data.dicom import load_ct_hu
    from ct_restoration.data.preprocessing import preprocess_ct_slice

    config = config or EvaluationConfig()
    image_hu = load_ct_hu(dicom_path)
    size = tuple(preprocessing["image_size"])

    body = body_mask_from_hu(image_hu, size, config.body_mask)
    interior = ssim_interior_mask(body, config.ssim.win_size)
    clean = preprocess_ct_slice(
        image_hu,
        window_center=preprocessing["window_center"],
        window_width=preprocessing["window_width"],
        size=size,
        interpolation=preprocessing["interpolation"],
    )
    return clean, body, interior


def aggregate_slices_to_patients(
    slice_frame: pd.DataFrame,
    metric_columns: tuple[str, ...] = METRIC_COLUMNS,
    carry: tuple[str, ...] = ("source_archive", "acquisition_group"),
) -> pd.DataFrame:
    """Average per-slice metrics within each patient.

    This is the step that stops a patient with many slices from dominating.
    Each patient contributes one row afterwards, whatever its slice count.

    Raises:
        EvaluationError: the frame is empty, lacks ``subject_id``, lacks a
            metric column, or a carried attribute varies within one patient.
    """
    if slice_frame.empty:
        raise EvaluationError("cannot aggregate an empty slice table")
    if "subject_id" not in slice_frame:
        raise EvaluationError("slice table must have a subject_id column")
    missing = [name for name in metric_columns if name not in slice_frame]
    if missing:
        raise EvaluationError(f"slice table is missing metric column(s): {missing}")

    rows: list[dict[str, Any]] = []
    for subject_id, group in slice_frame.groupby("subject_id", sort=False):
        row: dict[str, Any] = {"subject_id": subject_id}
        for name in carry:
            if name in group:
                values = set(group[name].unique().tolist())
                if len(values) != 1:
                    raise EvaluationError(
                        f"subject {subject_id!r} has inconsistent {name}: {sorted(values)}"
                    )
                row[name] = group[name].iloc[0]
        row["slice_count"] = int(len(group))
        for name in metric_columns:
            row[f"mean_{name}"] = float(group[name].mean())
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["order"] = frame["subject_id"].map(lambda value: (len(str(value)), str(value)))
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def _describe(values: np.ndarray) -> dict[str, float | None]:
    """mean / std / median / min / max of one metric across patients.

    ``std`` is ``None`` rather than NaN when there are too few observations to
    define it, so the summary stays strictly serializable JSON and an undefined
    spread reads as null instead of as a number-shaped non-number.
    """
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=STD_DDOF)) if array.size > STD_DDOF else None,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarise_patients(
    patient_frame: pd.DataFrame,
    metric_columns: tuple[str, ...] = METRIC_COLUMNS,
) -> dict[str, Any]:
    """The PRIMARY split-level result: every patient weighted equally.

    Returns:
        ``{metric: {mean, std, median, min, max}}`` over the patient means,
        plus the patient count. The ``mean`` entry is the headline estimate.
    """
    if patient_frame.empty:
        raise EvaluationError("cannot summarise an empty patient table")
    summary: dict[str, Any] = {"patients": int(len(patient_frame))}
    for name in metric_columns:
        column = f"mean_{name}"
        if column not in patient_frame:
            raise EvaluationError(f"patient table is missing {column}")
        summary[name] = _describe(patient_frame[column].to_numpy())
    return summary


def slice_weighted_summary(
    slice_frame: pd.DataFrame,
    metric_columns: tuple[str, ...] = METRIC_COLUMNS,
) -> dict[str, Any]:
    """SECONDARY descriptive summary pooling every slice equally.

    Reported for transparency, to show how much unequal slice counts would move
    the result. It is **not** the benchmark figure: it treats correlated slices
    as independent observations, which they are not.
    """
    if slice_frame.empty:
        raise EvaluationError("cannot summarise an empty slice table")
    summary: dict[str, Any] = {"slices": int(len(slice_frame))}
    for name in metric_columns:
        if name not in slice_frame:
            raise EvaluationError(f"slice table is missing {name}")
        summary[name] = {"mean": float(slice_frame[name].mean())}
    return summary


def group_breakdown(
    patient_frame: pd.DataFrame,
    column: str = "acquisition_group",
    metric_columns: tuple[str, ...] = METRIC_COLUMNS,
) -> dict[str, Any]:
    """Descriptive patient-weighted means per acquisition group.

    With three patients per group this detects a glaring imbalance and nothing
    else. No significance test, no claim about acquisition settings in general,
    and no conclusion from a small difference.
    """
    if column not in patient_frame:
        return {}
    breakdown: dict[str, Any] = {}
    for value, group in patient_frame.groupby(column, sort=True):
        entry: dict[str, Any] = {"patients": int(len(group))}
        for name in metric_columns:
            entry[name] = {"mean": float(group[f"mean_{name}"].mean())}
        breakdown[str(value)] = entry
    return breakdown
