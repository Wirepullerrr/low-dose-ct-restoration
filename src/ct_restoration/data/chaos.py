"""Locating CT studies inside a CHAOS dataset directory.

This module holds the only CHAOS-specific knowledge in the package: how the
downloaded archives lay out their folders. Everything it returns is plain
paths, which the generic readers in :mod:`ct_restoration.data.dicom` then
consume. No CHAOS path ever appears in the generic DICOM code, and no DICOM
parsing happens here.

The two archives extract to parallel trees::

    Train_Sets/
        CT/<subject id>/DICOM_anon/*.dcm    the CT slices
        CT/<subject id>/Ground/*.png        liver masks (unused here)
        MR/                                 not used by this project
    Test_Sets/
        CT/<subject id>/DICOM_anon/*.dcm    the CT slices, no public masks
        MR/                                 not used by this project

``Train_Sets`` and ``Test_Sets`` are the **challenge's** own split of its
segmentation task. This project does not perform that task and does not reuse
that split. Both trees are read as one pool of reference CT studies, and the
originating tree is recorded on each subject as ``source`` so it stays visible
as dataset provenance. This project's own train/validation/test split is a
separate, later decision.

Subject identifiers are the CHAOS folder names. They are the dataset's own
anonymized case labels, and this project uses them rather than any DICOM
patient field, so no patient attribute needs to be read to identify a case.
Across the two archives the folder names are disjoint, and
:func:`find_ct_subjects` refuses to continue if that ever stops being true,
because a silently merged identifier would put one patient on both sides of a
split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DICOM_SUFFIX = ".dcm"


@dataclass(frozen=True)
class ChaosCtSubject:
    """One CHAOS CT case discovered on disk.

    Attributes:
        subject_id: the CHAOS folder name, for example ``"21"``.
        source: the archive tree it came from, for example ``"Train_Sets"``.
            Recorded as provenance; it is not this project's ML split.
        dicom_dir: directory holding the slice files.
        dicom_paths: the slice files, sorted by filename.
        ground_truth_dir: segmentation mask folder if present. This project
            does not use the masks; the field exists so the audit can report
            what the download contains.
    """

    subject_id: str
    source: str
    dicom_dir: Path
    dicom_paths: tuple[Path, ...]
    ground_truth_dir: Path | None

    @property
    def slice_count(self) -> int:
        return len(self.dicom_paths)


def _natural_key(text: str) -> tuple[object, ...]:
    """Sort key that orders embedded numbers numerically ("2" before "10")."""
    return tuple(
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", text)
        if chunk
    )


def locate_ct_roots(root: str | Path) -> list[Path]:
    """Find every ``CT`` directory inside an extracted CHAOS download.

    Accepts the directory that *is* ``CT``, one that contains it, or one that
    contains several archive trees such as ``Train_Sets/CT`` and
    ``Test_Sets/CT``, so callers can point at whichever level they extracted
    to.

    Returns:
        Every CT directory found, in sorted path order. More than one is
        normal once both archives are extracted side by side.

    Raises:
        FileNotFoundError: the root does not exist, or holds no CT directory.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"CHAOS root directory not found: {base}")

    if base.name == "CT":
        return [base]
    if (base / "CT").is_dir():
        return [base / "CT"]

    candidates = sorted({path for path in base.glob("*/CT") if path.is_dir()})
    if not candidates:
        raise FileNotFoundError(
            f"No CT directory found in {base}. Expected <root>/CT, or archive "
            "trees such as <root>/Train_Sets/CT from an extracted CHAOS download."
        )
    return candidates


def _subjects_in_root(ct_root: Path) -> list[ChaosCtSubject]:
    source = ct_root.parent.name if ct_root.parent.name else ct_root.name

    subjects: list[ChaosCtSubject] = []
    for subject_dir in sorted(ct_root.iterdir(), key=lambda path: _natural_key(path.name)):
        if not subject_dir.is_dir():
            continue

        dicom_paths = tuple(
            sorted(
                (
                    path
                    for path in subject_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() == DICOM_SUFFIX
                ),
                key=lambda path: _natural_key(path.name),
            )
        )
        if not dicom_paths:
            continue

        ground_truth_dir = subject_dir / "Ground"
        subjects.append(
            ChaosCtSubject(
                subject_id=subject_dir.name,
                source=source,
                dicom_dir=dicom_paths[0].parent,
                dicom_paths=dicom_paths,
                ground_truth_dir=ground_truth_dir if ground_truth_dir.is_dir() else None,
            )
        )

    return subjects


def find_ct_subjects(root: str | Path) -> list[ChaosCtSubject]:
    """Discover every CHAOS CT subject below ``root``, across all archive trees.

    Slice files are collected recursively from each subject folder and sorted
    by filename. That ordering is a stable listing order only. It is **not** an
    anatomical ordering: filenames are not guaranteed to follow the scan
    direction, so anything that depends on slice order must sort by DICOM
    geometry instead.

    Subjects whose folder contains no DICOM file are skipped, so an unpacked
    archive that also holds notes or mask-only folders does not break
    discovery.

    Returns:
        Subjects ordered by their numeric folder name, pooled across archives.

    Raises:
        ValueError: the same subject id appears in more than one archive tree.
            Merging those would create one patient with two identities, which
            could place the same person in both a training and a test split.
    """
    subjects: list[ChaosCtSubject] = []
    for ct_root in locate_ct_roots(root):
        subjects.extend(_subjects_in_root(ct_root))

    seen: dict[str, str] = {}
    for subject in subjects:
        if subject.subject_id in seen:
            raise ValueError(
                f"Subject id {subject.subject_id!r} appears in both "
                f"{seen[subject.subject_id]!r} and {subject.source!r}. Refusing to pool "
                "them, because the same identifier in two archives cannot be assumed to "
                "be the same or a different patient."
            )
        seen[subject.subject_id] = subject.source

    return sorted(subjects, key=lambda subject: _natural_key(subject.subject_id))
