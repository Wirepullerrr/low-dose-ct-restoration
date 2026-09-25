"""The supervised restoration Dataset: one frozen manifest slice per item.

This is the layer the learned methods will train on. It defines exactly what
one supervised sample contains and nothing beyond that: no architecture, no
optimizer, no loss, no augmentation.

What one sample is
------------------
One row of the frozen slice manifest, turned into a supervised pair::

    DICOM -> HU -> frozen Milestone 1 preprocessing            -> clean
    clean + frozen Milestone 4 degradation + sample key        -> degraded

``degraded`` is the model input. ``clean`` is the supervised target. Everything
else in the sample is metadata for auditing and grouping, never model input.

The pair is a pure function of the slice
----------------------------------------
Both halves are rebuilt on demand from the DICOM by calling the existing
preprocessing and degradation code. The formulas are not reimplemented here,
and no new noise realization is created: :func:`degrade_low_dose_like` derives
that slice's seed by SHA-256 over the algorithm version, the global seed and
the sample key, so a given slice always carries the same frozen corruption.

That matters for training specifically. Sampling order may change from epoch to
epoch; the corruption may not. This benchmark defines exactly one degraded
counterpart per clean slice, and holding that pair fixed is what makes the
training inputs reproducible and lets the CNN and the U-Net inherit an
identical input-target mapping. Nothing here depends on the epoch, the batch,
the worker, the access count, the sampler seed, or the global NumPy or PyTorch
RNG.

Re-drawing the corruption every epoch would be a perfectly legitimate
denoising-training strategy - it is a standard stochastic augmentation, not a
form of cheating - but it is a *different* one: it changes the training
distribution and adds a second stochastic policy to account for. That is worth
studying on its own; it is deliberately not an experimental factor in the
first learned benchmark here.

What the sample deliberately does not contain
---------------------------------------------
No body mask, no SSIM interior mask, no clean HU array, no segmentation, no
DICOM UID, no absolute filesystem path. The evaluation body mask is derived
from the clean reference and exists for scoring only; handing it to a model
would feed the method a region computed from the target it is trying to
predict. The clean image reaches the model only as the supervised target.

No cache
--------
The pair is generated on demand rather than written out as ``.npy``, ``.pt``,
PNG, LMDB or HDF5 files. Establishing a correct contract comes first; a cache
is a second representation of the benchmark data that would itself have to be
audited for equivalence. If DICOM decoding later turns out to be a material
training bottleneck, it can be optimized then, against this contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like
from ct_restoration.data.splits import subject_sort_key
from ct_restoration.evaluation import require_development_split, require_rows_split_access

#: The pair-generation contract this module implements. A future variant -
#: a different preprocessing, a different degradation, a cached representation
#: - would be a new version rather than a silent change to this one.
PAIR_GENERATION_VERSION = "frozen_clean_plus_degradation_v1"

#: Channel count of the model tensors. CT slices here are single-channel.
IMAGE_CHANNELS = 1

#: Exactly the keys one sample carries. Asserted in tests: a sample that grew
#: an extra key would be a silent change to what a model can see.
SAMPLE_FIELDS: tuple[str, ...] = (
    "degraded",
    "clean",
    "subject_id",
    "sample_key",
    "source_archive",
    "acquisition_group",
    "geometric_slice_index",
)

#: Manifest columns the Dataset needs to build and identify a sample.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "source_archive",
    "acquisition_group",
    "relative_dicom_path",
    "geometric_slice_index",
)

#: Names that must never appear in a sample. These are evaluation-only or
#: clean-derived quantities; a model receiving any of them would be reading
#: information computed from its own target.
FORBIDDEN_SAMPLE_FIELDS: tuple[str, ...] = (
    "body_mask",
    "evaluation_mask",
    "ssim_interior_mask",
    "interior_mask",
    "clean_hu",
    "image_hu",
    "segmentation",
    "mask",
    "split",
    "dicom_path",
    "absolute_path",
    "series_instance_uid",
    "study_instance_uid",
    "sop_instance_uid",
)

#: How far outside [0, 1] a produced tensor may stray before it is rejected.
#: Only float32 round-trip error is tolerated.
RANGE_TOLERANCE = 1e-6


class DatasetError(ValueError):
    """The dataset rows, configuration or a produced sample are not usable."""


def _require_slice_index(value: Any, sample_key: str) -> int:
    """Accept only a genuine integer geometric slice index.

    ``int(3.8)`` is 3, so a coercing reader would turn a malformed manifest
    value into a different but plausible-looking position in the canonical
    order, silently renaming which slice a Dataset index refers to.

    An exactly integral float **is** accepted, unlike the hand-written YAML
    configs elsewhere in this project. The difference is where the value comes
    from: pandas promotes a whole integer column to ``float64`` if a single
    cell is missing, so ``3.0`` here can be an artifact of parsing rather than
    a typo, and converting it back is exact rather than a repair. ``3.8``,
    ``NaN`` and a string are none of those things and are refused.
    """
    if isinstance(value, bool):
        raise DatasetError(
            f"geometric_slice_index for {sample_key!r} must be an integer, got bool {value!r}"
        )
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not np.isfinite(value):
            raise DatasetError(
                f"geometric_slice_index for {sample_key!r} must be a finite integer, "
                f"got {value!r}. A missing manifest value reads as NaN here."
            )
        if float(value) != int(value):
            raise DatasetError(
                f"geometric_slice_index for {sample_key!r} must be a whole number, "
                f"got {value!r}. It is not rounded or truncated."
            )
        return int(value)
    raise DatasetError(
        f"geometric_slice_index for {sample_key!r} must be an integer, got "
        f"{type(value).__name__} {value!r}. It is not parsed from a string."
    )


def _require_image_size(value: Any) -> tuple[int, int]:
    """Accept only two genuine positive integer dimensions.

    Strict in the other direction from :func:`_require_slice_index`: this
    value comes from a hand-written YAML config, where ``256.7`` or ``"256"``
    means the file was written wrong, not that a parser promoted a dtype. A
    silently truncated image size would change every tensor the benchmark
    produces while every downstream shape check still passed.
    """
    if isinstance(value, str) or not hasattr(value, "__len__") or len(value) != 2:
        raise DatasetError(
            f"preprocessing image_size must be two positive integers (height, width), got {value!r}"
        )
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int | np.integer):
            raise DatasetError(
                f"preprocessing image_size entries must be integers, got {value!r}. "
                "They are not rounded, truncated or parsed from strings."
            )
        if int(dimension) < 1:
            raise DatasetError(f"preprocessing image_size entries must be >= 1, got {value!r}")
        dimensions.append(int(dimension))
    return (dimensions[0], dimensions[1])


def canonical_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Sort manifest rows into the benchmark's canonical sample order.

    Natural subject order, then geometric slice index. This is a definition,
    not a convenience: it is the order the evaluation tables already use, so a
    Dataset index and a metric-table row refer to the same slice. It is derived
    from the manifest's own fields, so an accidentally reordered CSV cannot
    change which slice a Dataset index names.

    Raises:
        DatasetError: a required column is missing, a sample key is not a
            portable relative POSIX path, a slice index is not a genuine
            integer, or the rows do not identify exactly one slice per sample
            key and per (subject, geometric slice index).
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise DatasetError(f"manifest rows are missing column(s): {missing}")
    if rows.empty:
        raise DatasetError("manifest rows are empty; a Dataset needs at least one slice")

    ordered = rows.copy()
    ordered["subject_id"] = ordered["subject_id"].astype(str)
    ordered["relative_dicom_path"] = ordered["relative_dicom_path"].astype(str)

    unusable = sorted(
        key for key in ordered["relative_dicom_path"] if not sample_key_is_portable(key)
    )
    if unusable:
        raise DatasetError(
            f"manifest rows carry non-portable sample key(s): {unusable[:5]}. A sample key "
            "must be a relative POSIX path with no drive letter, no backslash and no '..' "
            "segment: it identifies the slice on every machine and is one input to the "
            "derivation of that slice's degradation seed. It is not rewritten into a "
            "valid form."
        )

    ordered["geometric_slice_index"] = [
        _require_slice_index(value, key)
        for value, key in zip(
            ordered["geometric_slice_index"], ordered["relative_dicom_path"], strict=True
        )
    ]

    duplicated = ordered.duplicated(subset=["relative_dicom_path"])
    if duplicated.any():
        repeats = sorted(set(ordered.loc[duplicated, "relative_dicom_path"]))
        raise DatasetError(f"manifest rows repeat sample key(s): {repeats[:5]}")

    # The sort below is on (subject, geometric slice index), so those two
    # fields have to identify a slice on their own. If two different paths
    # claimed one position, their relative order would fall back to input row
    # order, and "canonical" would become a property of the CSV rather than of
    # the manifest's contents.
    clashing = ordered.duplicated(subset=["subject_id", "geometric_slice_index"])
    if clashing.any():
        pairs = sorted(
            {
                (subject, int(index))
                for subject, index in zip(
                    ordered.loc[clashing, "subject_id"],
                    ordered.loc[clashing, "geometric_slice_index"],
                    strict=True,
                )
            }
        )
        raise DatasetError(
            f"manifest rows give one (subject_id, geometric_slice_index) to more than one "
            f"sample key: {pairs[:5]}. The canonical order sorts on exactly those two "
            "fields, so a duplicated pair would leave the sample order depending on input "
            "row order."
        )

    ordered["_subject_order"] = ordered["subject_id"].map(subject_sort_key)
    ordered = ordered.sort_values(["_subject_order", "geometric_slice_index"], kind="mergesort")
    return ordered.drop(columns="_subject_order").reset_index(drop=True)


def development_rows(manifest: Path | str | pd.DataFrame, split: str) -> pd.DataFrame:
    """Canonically ordered rows of one development split.

    The hold-out gate lives here, so every Dataset built for development work
    passes through it. Test and stress stay sealed until the final benchmark,
    which will build its held-out datasets explicitly rather than by reaching
    through a development helper.

    Raises:
        HeldOutSplitError: the split is sealed.
        EvaluationError: the split is not a partition of this benchmark.
        DatasetError: the manifest holds no rows for that split.
    """
    require_development_split(split)
    frame = (
        manifest
        if isinstance(manifest, pd.DataFrame)
        else pd.read_csv(manifest, dtype={"subject_id": str})
    )
    if "split" not in frame.columns:
        raise DatasetError("manifest has no 'split' column; cannot select a development split")
    rows = frame[frame["split"].astype(str) == split]
    if rows.empty:
        raise DatasetError(f"manifest holds no rows for split {split!r}")
    return canonical_order(rows)


def _validated_pair_half(array: np.ndarray, name: str, sample_key: str) -> torch.Tensor:
    """Turn one half of a pair into the model tensor contract, or refuse.

    Strict on purpose. The lower-level preprocessing and degradation already
    guarantee a finite ``float32`` image in [0, 1]; this repeats the check at
    the boundary a model actually consumes, because a silent renormalization or
    a stray NaN here would poison training and every metric downstream without
    anything reporting a problem. Nothing is repaired: an out-of-contract array
    is a bug upstream and must surface as one.

    Returns:
        A ``float32`` tensor of shape ``[1, H, W]`` owning its own storage.
    """
    values = np.asarray(array)
    if values.ndim != 2:
        raise DatasetError(f"{name} for {sample_key!r} must be a 2-D image, got {values.shape}")
    if values.dtype != np.float32:
        raise DatasetError(f"{name} for {sample_key!r} must be float32, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        count = int((~np.isfinite(values)).sum())
        raise DatasetError(f"{name} for {sample_key!r} has {count} non-finite pixel(s)")

    minimum, maximum = float(values.min()), float(values.max())
    if minimum < -RANGE_TOLERANCE or maximum > 1.0 + RANGE_TOLERANCE:
        raise DatasetError(
            f"{name} for {sample_key!r} must lie in [0, 1], got "
            f"[{minimum:.6g}, {maximum:.6g}]. The Dataset does not renormalize."
        )
    # copy() so the tensor owns its buffer and can never alias the NumPy array
    # the degradation was computed from.
    return torch.from_numpy(values.copy()).unsqueeze(0)


class RestorationDataset(Dataset):
    """Supervised (degraded, clean) pairs over one split of the frozen manifest.

    Items are addressed in :func:`canonical_order`, so index ``i`` names the
    same slice however the input rows happened to be ordered.

    Picklable by construction: it holds only plain tuples, a path, a mapping
    copy and a frozen config, so a DataLoader worker process can reconstruct it
    and will produce byte-identical pairs.

    Args:
        rows: manifest rows for this split, in any order.
        root: extracted CHAOS directory. Sample keys are resolved against it.
        preprocessing: the frozen Milestone 1 settings. Copied, never mutated.
        degradation: the frozen Milestone 4 corruption. Defaults to canonical.
        split: the partition these rows came from, recorded for provenance.

    Raises:
        HeldOutSplitError: a row is labelled test or stress, or carries no
            split label. The supervised pairs are development data only, and
            this is checked here as well as in :func:`development_rows`, so
            rows that reach the constructor by another route are refused too.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        root: Path | str,
        preprocessing: Mapping[str, Any],
        degradation: DegradationConfig | None = None,
        split: str | None = None,
    ) -> None:
        require_rows_split_access(rows)
        if split is not None:
            require_development_split(str(split))
        ordered = canonical_order(rows)

        for field in ("window_center", "window_width", "image_size", "interpolation"):
            if field not in preprocessing:
                raise DatasetError(f"preprocessing config is missing {field!r}")
        # A copy, so indexing this Dataset can never write through to the
        # caller's loaded config dict.
        self._preprocessing: dict[str, Any] = dict(preprocessing)
        self._image_size = _require_image_size(preprocessing["image_size"])

        self._root = Path(root)
        self._degradation = degradation or DegradationConfig()
        self._split = None if split is None else str(split)

        self._subject_ids: tuple[str, ...] = tuple(ordered["subject_id"])
        self._sample_keys: tuple[str, ...] = tuple(ordered["relative_dicom_path"])
        self._source_archives: tuple[str, ...] = tuple(ordered["source_archive"].astype(str))
        self._acquisition_groups: tuple[str, ...] = tuple(ordered["acquisition_group"].astype(str))
        # Already validated as genuine integers by canonical_order.
        self._slice_indices: tuple[int, ...] = tuple(
            int(v) for v in ordered["geometric_slice_index"]
        )

    # -- identity and structure, all metadata-only ------------------------

    def __len__(self) -> int:
        return len(self._sample_keys)

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    @property
    def degradation(self) -> DegradationConfig:
        return self._degradation

    @property
    def subject_ids(self) -> tuple[str, ...]:
        """Subject of each index, in Dataset order."""
        return self._subject_ids

    @property
    def sample_keys(self) -> tuple[str, ...]:
        """Portable POSIX-style relative key of each index, in Dataset order."""
        return self._sample_keys

    @property
    def patients(self) -> tuple[str, ...]:
        """Distinct subjects, in canonical natural order."""
        return tuple(sorted(set(self._subject_ids), key=subject_sort_key))

    def indices_by_patient(self) -> dict[str, tuple[int, ...]]:
        """Dataset indices of each subject, ascending, in canonical patient order."""
        grouped: dict[str, list[int]] = {patient: [] for patient in self.patients}
        for index, subject_id in enumerate(self._subject_ids):
            grouped[subject_id].append(index)
        return {patient: tuple(indices) for patient, indices in grouped.items()}

    # -- the pair ---------------------------------------------------------

    def clean_array(self, index: int) -> np.ndarray:
        """The clean reference for one index, as the frozen pipeline builds it."""
        from ct_restoration.data.dicom import load_ct_hu
        from ct_restoration.data.preprocessing import preprocess_ct_slice

        key = self._sample_keys[self._checked(index)]
        image_hu = load_ct_hu(self._root / key)
        return preprocess_ct_slice(
            image_hu,
            window_center=self._preprocessing["window_center"],
            window_width=self._preprocessing["window_width"],
            size=self._image_size,
            interpolation=self._preprocessing["interpolation"],
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = self._checked(index)
        key = self._sample_keys[position]

        clean = self.clean_array(position)
        # degrade_low_dose_like is documented never to modify its input and
        # seeds itself from the key alone, so the pair is a pure function of
        # the slice and the frozen config.
        degraded = degrade_low_dose_like(clean, key, self._degradation)

        return {
            "degraded": _validated_pair_half(degraded, "degraded", key),
            "clean": _validated_pair_half(clean, "clean", key),
            "subject_id": self._subject_ids[position],
            "sample_key": key,
            "source_archive": self._source_archives[position],
            "acquisition_group": self._acquisition_groups[position],
            "geometric_slice_index": self._slice_indices[position],
        }

    def _checked(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int | np.integer):
            raise DatasetError(f"dataset index must be an integer, got {type(index).__name__}")
        position = int(index)
        if position < 0:
            position += len(self._sample_keys)
        if not 0 <= position < len(self._sample_keys):
            raise IndexError(f"dataset index {index} out of range for {len(self)} samples")
        return position

    def __repr__(self) -> str:
        return (
            f"RestorationDataset(split={self._split!r}, samples={len(self)}, "
            f"patients={len(self.patients)}, image_size={self._image_size})"
        )


def development_dataset(
    split: str,
    manifest: Path | str | pd.DataFrame,
    root: Path | str,
    preprocessing: Mapping[str, Any],
    degradation: DegradationConfig | None = None,
) -> RestorationDataset:
    """Build a Dataset for a development split, refusing the sealed ones.

    Raises:
        HeldOutSplitError: ``split`` is test or stress.
        EvaluationError: ``split`` is not a partition of this benchmark.
    """
    rows = development_rows(manifest, split)
    return RestorationDataset(rows, root, preprocessing, degradation, split=split)


def sample_key_is_portable(sample_key: str) -> bool:
    """True when a key is the relative POSIX form the benchmark requires.

    No empty string, no leading slash, no drive letter, no backslash and no
    ``..`` segment. The key has to name the same slice on any machine, because
    it is the stable portable identity of that slice and one of the inputs to
    the SHA-256 derivation of the slice's degradation seed. It is not the
    numeric seed itself, and two spellings of one path would derive two
    different seeds for the same image.
    """
    text = str(sample_key)
    if not text or "\\" in text or text.startswith("/"):
        return False
    if PurePosixPath(text).is_absolute():
        return False
    segments = text.split("/")
    if ".." in segments:
        return False
    return ":" not in segments[0]


def flatten_sample_keys(batches: Sequence[Mapping[str, Any]]) -> list[str]:
    """Concatenate the ``sample_key`` lists of collated batches, in order."""
    keys: list[str] = []
    for batch in batches:
        keys.extend(str(key) for key in batch["sample_key"])
    return keys
