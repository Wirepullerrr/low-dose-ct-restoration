"""Generate the frozen patient-level split for the CHAOS CT cohort.

    uv run python scripts/create_splits.py --root data/raw/chaos

Reads the Milestone 2 audit, assigns every subject to exactly one partition,
and writes three artifacts:

``data/splits/chaos_patient_split.csv``
    the canonical experiment definition, one row per subject.
``data/splits/chaos_slice_manifest.csv``
    one row per slice, each inheriting its patient's partition. Slices are
    ordered by DICOM geometry, never by filename.
``outputs/audit/chaos_split_summary.json``
    counts, balance and the SHA-256 of the canonical CSV.

The assignment involves no random number generator, does not depend on input
row order, and contains no timestamp or other run-varying field. Re-running
reproduces all three files byte for byte.

Nothing here looks at image content. Subject identity supplies only canonical
ordering and tie-breaking, acquisition group constrains the split, and slice
count balances it. ``source_archive`` is carried through as provenance and
audited after the assignment; it does not drive the partitioning. So the split
cannot be tuned, even accidentally, towards a restoration outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.config import ensure_dir  # noqa: E402
from ct_restoration.data.chaos import find_ct_subjects  # noqa: E402
from ct_restoration.data.splits import (  # noqa: E402
    PRIMARY_SPLIT_SIZES,
    SPLIT_ORDER,
    assign_splits,
    classify_acquisition_group,
    subject_records_from_audit,
    subject_sort_key,
    validate_assignment,
)

PATIENT_SPLIT_COLUMNS = [
    "subject_id",
    "source_archive",
    "acquisition_group",
    "slice_count",
    "rescale_intercept",
    "slice_thickness",
    "pixel_representation",
    "split",
]

MANIFEST_COLUMNS = [
    "subject_id",
    "split",
    "source_archive",
    "acquisition_group",
    "relative_dicom_path",
    "geometric_slice_position",
    "geometric_slice_index",
]


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with a fixed line ending so its checksum is portable."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_patient_split(audit_rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = subject_records_from_audit(audit_rows)
    assignment = assign_splits(records)
    validate_assignment(records, assignment)

    frame = pd.DataFrame(
        [
            {
                "subject_id": record.subject_id,
                "source_archive": record.source_archive,
                "acquisition_group": classify_acquisition_group(record),
                "slice_count": record.slice_count,
                "rescale_intercept": record.rescale_intercept,
                "slice_thickness": record.slice_thickness,
                "pixel_representation": record.pixel_representation,
                "split": assignment[record.subject_id],
            }
            for record in records
        ]
    )
    frame["order"] = frame["subject_id"].map(subject_sort_key)
    return frame.sort_values("order").drop(columns="order")[PATIENT_SPLIT_COLUMNS]


def build_slice_manifest(
    audit: pd.DataFrame, patient_split: pd.DataFrame, root: Path
) -> pd.DataFrame:
    """One row per slice, inheriting the patient partition and ordered by geometry."""
    paths: dict[tuple[str, str], Path] = {}
    for subject in find_ct_subjects(root):
        for path in subject.dicom_paths:
            key = (subject.subject_id, path.name)
            if key in paths:
                raise ValueError(f"Duplicate slice filename {path.name} for subject {key[0]}")
            paths[key] = path

    lookup = patient_split.set_index("subject_id")
    rows: list[dict[str, Any]] = []
    for record in audit.to_dict("records"):
        subject_id = str(record["subject_id"])
        key = (subject_id, str(record["file"]))
        if key not in paths:
            raise ValueError(f"Audit row has no file on disk: {key}")
        patient = lookup.loc[subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "split": patient["split"],
                "source_archive": patient["source_archive"],
                "acquisition_group": patient["acquisition_group"],
                "relative_dicom_path": paths[key].relative_to(root).as_posix(),
                "geometric_slice_position": float(record["projected_position"]),
            }
        )

    frame = pd.DataFrame(rows)
    # Order by DICOM geometry, ascending along the slice normal. InstanceNumber
    # runs the other way in this cohort; filenames are never used as truth.
    frame["subject_order"] = frame["subject_id"].map(subject_sort_key)
    frame = frame.sort_values(["subject_order", "geometric_slice_position"])
    frame["geometric_slice_index"] = frame.groupby("subject_id").cumcount()
    return frame.reset_index(drop=True)[MANIFEST_COLUMNS]


def summarise(
    patient_split: pd.DataFrame, manifest: pd.DataFrame, split_csv: Path
) -> dict[str, Any]:
    groups = sorted(patient_split["acquisition_group"].unique())
    archives = sorted(patient_split["source_archive"].unique())

    per_split: dict[str, Any] = {}
    for name in SPLIT_ORDER:
        subset = patient_split[patient_split["split"] == name]
        if subset.empty:
            continue
        counts = subset["slice_count"]
        per_split[name] = {
            "subjects": int(len(subset)),
            "slices": int(counts.sum()),
            "subject_ids": sorted(subset["subject_id"].tolist(), key=subject_sort_key),
            "acquisition_groups": {
                group: int((subset["acquisition_group"] == group).sum()) for group in groups
            },
            "source_archives": {
                archive: int((subset["source_archive"] == archive).sum()) for archive in archives
            },
            "slices_per_subject": {
                "min": int(counts.min()),
                "median": float(counts.median()),
                "max": int(counts.max()),
            },
        }

    members = {
        name: set(patient_split.loc[patient_split["split"] == name, "subject_id"])
        for name in per_split
    }
    overlap = 0
    for index, left in enumerate(sorted(members)):
        for right in sorted(members)[index + 1 :]:
            overlap += len(members[left] & members[right])

    manifest_matches = manifest.merge(
        patient_split[["subject_id", "split"]], on="subject_id", suffixes=("", "_patient")
    )
    mismatched = int((manifest_matches["split"] != manifest_matches["split_patient"]).sum())

    return {
        "cohort": {
            "subjects": int(len(patient_split)),
            "slices": int(patient_split["slice_count"].sum()),
            "acquisition_groups": {
                group: int((patient_split["acquisition_group"] == group).sum()) for group in groups
            },
        },
        "primary_split_sizes": PRIMARY_SPLIT_SIZES,
        "splits": per_split,
        "patient_overlap": overlap,
        "stress_subject_ids": per_split.get("stress", {}).get("subject_ids", []),
        "split_csv": split_csv.as_posix(),
        "split_csv_sha256": sha256_of(split_csv),
        "slice_manifest_rows": int(len(manifest)),
        "slice_manifest_rows_with_wrong_split": mismatched,
        "slice_manifest_duplicate_rows": int(manifest.duplicated().sum()),
        "notes": (
            "Subject identity supplies canonical ordering only, acquisition group constrains "
            "the split, and slice count balances it. source_archive is provenance carried "
            "through the outputs and audited afterwards; it does not drive the assignment, so "
            "the archive counts reported here are an observed property of the result. The "
            "stress set holds every subject from the rare acquisition groups; it is a small "
            "robustness stress test, not a powered benchmark. This file contains no timestamp "
            "so that regenerating an unchanged experiment definition is a byte-identical "
            "no-op."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--audit", default="outputs/audit/chaos_slice_metrics.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--summary", default="outputs/audit/chaos_split_summary.json")
    parser.add_argument(
        "--no-manifest", action="store_true", help="write only the canonical patient split"
    )
    arguments = parser.parse_args()

    audit = pd.read_csv(arguments.audit)
    audit = audit[audit["error"].isna()] if "error" in audit else audit
    print(f"audit rows      : {len(audit)}")

    patient_split = build_patient_split(audit.to_dict("records"))
    splits_dir = ensure_dir(arguments.splits_dir)
    split_csv = write_csv(patient_split, splits_dir / "chaos_patient_split.csv")
    print(f"wrote           : {split_csv}")

    manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if not arguments.no_manifest:
        manifest = build_slice_manifest(audit, patient_split, Path(arguments.root))
        manifest_csv = write_csv(manifest, splits_dir / "chaos_slice_manifest.csv")
        print(f"wrote           : {manifest_csv} ({len(manifest)} rows)")

    summary = summarise(patient_split, manifest, split_csv)
    summary_path = Path(arguments.summary)
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"wrote           : {summary_path}")

    print()
    header = f"{'split':<11}{'patients':>9}{'slices':>8}"
    groups = sorted(patient_split["acquisition_group"].unique())
    archives = sorted(patient_split["source_archive"].unique())
    header += "".join(f"{group:>4}" for group in groups)
    header += "".join(f"{archive:>12}" for archive in archives)
    print(header)
    print("-" * len(header))
    for name, values in summary["splits"].items():
        line = f"{name:<11}{values['subjects']:>9}{values['slices']:>8}"
        line += "".join(f"{values['acquisition_groups'][g]:>4}" for g in groups)
        line += "".join(f"{values['source_archives'][a]:>12}" for a in archives)
        print(line)
    total = summary["cohort"]
    line = f"{'TOTAL':<11}{total['subjects']:>9}{total['slices']:>8}"
    line += "".join(f"{total['acquisition_groups'][g]:>4}" for g in groups)
    print("-" * len(header))
    print(line)

    print()
    print(f"patient_overlap : {summary['patient_overlap']}")
    print(f"split_csv_sha256: {summary['split_csv_sha256']}")
    print(f"stress subjects : {summary['stress_subject_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
