"""Tests for the patient-level split.

Algorithm tests run on a synthetic cohort and never need the CHAOS download.
A separate group of tests checks the committed split artifact itself, which is
a small text file tracked in the repository.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from ct_restoration.data.splits import (
    ACQUISITION_GROUPS,
    COHORT_SPLIT_SIZES,
    PRIMARY_GROUPS,
    PRIMARY_SPLIT_SIZES,
    STRESS_GROUPS,
    STRESS_SPLIT,
    SplitError,
    SubjectRecord,
    assign_splits,
    choose_group_allocation,
    classify_acquisition_group,
    subject_records_from_audit,
    validate_assignment,
)

REPOSITORY = Path(__file__).resolve().parents[1]
COMMITTED_SPLIT = REPOSITORY / "data" / "splits" / "chaos_patient_split.csv"

SIGNATURES = {label: signature for signature, label in ACQUISITION_GROUPS.items()}

#: Mirrors the real cohort shape: 16 group A, 21 group B, 2 group C, 1 group D.
SYNTHETIC_GROUP_SIZES = {"A": 16, "B": 21, "C": 2, "D": 1}


def make_record(subject_id: str, group: str, slice_count: int, archive: str) -> SubjectRecord:
    intercept, thickness, representation = SIGNATURES[group]
    return SubjectRecord(
        subject_id=subject_id,
        source_archive=archive,
        slice_count=slice_count,
        rescale_intercept=intercept,
        slice_thickness=thickness,
        pixel_representation=representation,
    )


def synthetic_cohort() -> list[SubjectRecord]:
    """A 40-subject cohort with the real group counts and varied slice counts."""
    records: list[SubjectRecord] = []
    number = 1
    for group, size in SYNTHETIC_GROUP_SIZES.items():
        for index in range(size):
            # Group A subjects are large, group B small, as in the real cohort.
            base = {"A": 210, "B": 80, "C": 90, "D": 240}[group]
            records.append(
                make_record(
                    str(number),
                    group,
                    base + 7 * index,
                    "Train_Sets" if number % 2 else "Test_Sets",
                )
            )
            number += 1
    return records


@pytest.fixture
def cohort() -> list[SubjectRecord]:
    return synthetic_cohort()


@pytest.fixture
def assignment(cohort) -> dict[str, str]:
    return assign_splits(cohort)


def members(assignment: dict[str, str], split: str) -> set[str]:
    return {subject for subject, name in assignment.items() if name == split}


# --------------------------------------------------------------------------
# Acquisition group classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("group", sorted(SIGNATURES))
def test_each_known_signature_classifies_to_its_group(group) -> None:
    assert classify_acquisition_group(make_record("1", group, 100, "Train_Sets")) == group


def test_an_unknown_acquisition_signature_is_refused() -> None:
    record = SubjectRecord("1", "Train_Sets", 100, -512.0, 5.0, 0)

    with pytest.raises(SplitError, match="unrecognised acquisition signature"):
        classify_acquisition_group(record)


# --------------------------------------------------------------------------
# Group allocation across partitions
# --------------------------------------------------------------------------


def test_allocation_keeps_both_primary_groups_in_every_evaluation_partition() -> None:
    allocation = choose_group_allocation({"A": 16, "B": 21}, PRIMARY_SPLIT_SIZES)

    for split in ("validation", "test"):
        for group in PRIMARY_GROUPS:
            assert allocation[split][group] >= 1


def test_allocation_fills_every_partition_exactly() -> None:
    allocation = choose_group_allocation({"A": 16, "B": 21}, PRIMARY_SPLIT_SIZES)

    for split, size in PRIMARY_SPLIT_SIZES.items():
        assert sum(allocation[split].values()) == size
    for group, total in {"A": 16, "B": 21}.items():
        assert sum(allocation[split][group] for split in PRIMARY_SPLIT_SIZES) == total


def test_allocation_gives_validation_and_test_the_same_composition() -> None:
    """Matching A/B counts removes one known source of compositional mismatch.

    It is not why this allocation wins -- several feasible allocations are
    symmetric -- and it does not make six validation patients representative of
    six different test patients.
    """
    allocation = choose_group_allocation({"A": 16, "B": 21}, PRIMARY_SPLIT_SIZES)

    assert allocation["validation"] == allocation["test"]


def test_symmetry_alone_does_not_select_the_allocation() -> None:
    """Guards the documented claim: symmetry is a tie-break, not the reason."""
    sizes = PRIMARY_SPLIT_SIZES
    symmetric = [
        {
            "validation": {"A": count, "B": sizes["validation"] - count},
            "test": {"A": count, "B": sizes["test"] - count},
        }
        for count in range(1, sizes["validation"])
    ]
    feasible = [
        option for option in symmetric if 0 <= 16 - 2 * option["validation"]["A"] <= sizes["train"]
    ]
    assert len(feasible) > 1, "more than one symmetric allocation must exist"

    chosen = choose_group_allocation({"A": 16, "B": 21}, sizes)
    share = 16 / 37

    def deviation(option: dict) -> float:
        option = dict(option)
        option["train"] = {"A": 16 - 2 * option["validation"]["A"]}
        return sum(abs(option[name]["A"] / sizes[name] - share) for name in sizes)

    assert deviation(chosen) == min(deviation(option) for option in feasible)


def test_allocation_is_refused_when_the_cohort_does_not_fit() -> None:
    with pytest.raises(SplitError, match="cannot fill partitions"):
        choose_group_allocation({"A": 5, "B": 5}, PRIMARY_SPLIT_SIZES)


# --------------------------------------------------------------------------
# Assignment invariants
# --------------------------------------------------------------------------


def test_every_subject_is_assigned_exactly_once(cohort, assignment) -> None:
    assert len(assignment) == len(cohort) == 40
    assert set(assignment) == {record.subject_id for record in cohort}


def test_partitions_are_mutually_exclusive(assignment) -> None:
    names = ["train", "validation", "test", STRESS_SPLIT]
    seen: set[str] = set()
    for name in names:
        current = members(assignment, name)
        assert not (current & seen), f"{name} overlaps an earlier partition"
        seen |= current
    assert len(seen) == 40


def test_patient_counts_match_the_configured_targets(assignment) -> None:
    for name, size in PRIMARY_SPLIT_SIZES.items():
        assert len(members(assignment, name)) == size
    assert len(members(assignment, STRESS_SPLIT)) == 3


def test_stress_holds_only_rare_groups(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}

    assert {groups[s] for s in members(assignment, STRESS_SPLIT)} <= set(STRESS_GROUPS)


def test_stress_holds_every_rare_group_subject(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    rare = {s for s, group in groups.items() if group in STRESS_GROUPS}

    assert members(assignment, STRESS_SPLIT) == rare


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_primary_partitions_hold_only_primary_groups(cohort, assignment, split) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}

    assert {groups[s] for s in members(assignment, split)} <= set(PRIMARY_GROUPS)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_partitions_contain_both_primary_groups(cohort, assignment, split) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}

    assert {groups[s] for s in members(assignment, split)} == set(PRIMARY_GROUPS)


def test_validation_and_test_have_comparable_slice_totals(cohort, assignment) -> None:
    slices = {record.subject_id: record.slice_count for record in cohort}
    validation = sum(slices[s] for s in members(assignment, "validation"))
    test = sum(slices[s] for s in members(assignment, "test"))

    assert abs(validation - test) / max(validation, test) < 0.15


def test_duplicate_subject_ids_are_refused() -> None:
    cohort = synthetic_cohort()
    cohort.append(make_record(cohort[0].subject_id, "B", 100, "Train_Sets"))

    with pytest.raises(SplitError, match="Duplicate subject ids"):
        assign_splits(cohort)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_assignment_is_identical_across_repeated_runs(cohort) -> None:
    assert assign_splits(cohort) == assign_splits(cohort)


@pytest.mark.parametrize("rotation", [1, 7, 19, 39])
def test_assignment_does_not_depend_on_input_row_order(cohort, rotation) -> None:
    rotated = cohort[rotation:] + cohort[:rotation]

    assert assign_splits(rotated) == assign_splits(cohort)


def test_assignment_does_not_depend_on_a_reversed_cohort(cohort) -> None:
    assert assign_splits(list(reversed(cohort))) == assign_splits(cohort)


def test_source_archive_is_not_an_input_to_the_assignment(cohort, assignment) -> None:
    """Relabelling the archives changes nothing, because nothing reads them.

    CHAOS Train_Sets / Test_Sets membership is segmentation-challenge
    provenance, not a restoration-task label. Every split-relevant fact is held
    fixed here and only the archive label moves, so any difference in the
    result would mean provenance had leaked into the partitioning.
    """
    swapped = [
        SubjectRecord(
            subject_id=record.subject_id,
            source_archive=("Test_Sets" if record.source_archive == "Train_Sets" else "Train_Sets"),
            slice_count=record.slice_count,
            rescale_intercept=record.rescale_intercept,
            slice_thickness=record.slice_thickness,
            pixel_representation=record.pixel_representation,
        )
        for record in cohort
    ]
    assert assign_splits(swapped) == assignment

    uniform = [
        SubjectRecord(
            subject_id=record.subject_id,
            source_archive="Train_Sets",
            slice_count=record.slice_count,
            rescale_intercept=record.rescale_intercept,
            slice_thickness=record.slice_thickness,
            pixel_representation=record.pixel_representation,
        )
        for record in cohort
    ]
    assert assign_splits(uniform) == assignment


def test_archive_mixing_is_an_observed_property_not_a_criterion(cohort, assignment) -> None:
    """The resulting split happens to mix both archives; it was not asked to."""
    archives = {record.subject_id: record.source_archive for record in cohort}

    for split in ("train", "validation", "test"):
        present = {archives[s] for s in members(assignment, split)}
        assert len(present) > 1, f"{split} drew from a single source archive"


# --------------------------------------------------------------------------
# Validation of an assignment
# --------------------------------------------------------------------------


def test_validate_accepts_a_generated_assignment(cohort, assignment) -> None:
    validate_assignment(cohort, assignment)


def test_validate_rejects_a_rare_group_subject_in_a_primary_partition(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    rare = next(s for s, group in groups.items() if group in STRESS_GROUPS)
    broken = dict(assignment)
    broken[rare] = "train"

    with pytest.raises(SplitError, match="rare group"):
        validate_assignment(cohort, broken)


def test_validate_rejects_a_partition_missing_a_primary_group(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    broken = dict(assignment)

    # Empty group A out of validation entirely, backfilling from train with B.
    leaving = [s for s in members(assignment, "validation") if groups[s] == "A"]
    arriving = [s for s in sorted(members(assignment, "train")) if groups[s] == "B"][: len(leaving)]
    for subject in leaving:
        broken[subject] = "train"
    for subject in arriving:
        broken[subject] = "validation"

    with pytest.raises(SplitError, match="missing primary group"):
        validate_assignment(cohort, broken)


def test_validate_rejects_an_incomplete_assignment(cohort, assignment) -> None:
    broken = dict(assignment)
    broken.pop(cohort[0].subject_id)

    with pytest.raises(SplitError, match="does not cover the cohort"):
        validate_assignment(cohort, broken)


def test_validate_rejects_a_subject_that_is_not_in_the_cohort(cohort, assignment) -> None:
    broken = dict(assignment)
    broken["not-a-subject"] = "train"

    with pytest.raises(SplitError, match="does not cover the cohort"):
        validate_assignment(cohort, broken)


def test_validate_rejects_an_unknown_split_label(cohort, assignment) -> None:
    """An unrecognised partition name must be refused, never silently ignored."""
    broken = dict(assignment)
    broken[sorted(members(assignment, "train"))[0]] = "other"

    with pytest.raises(SplitError, match="unknown split label"):
        validate_assignment(cohort, broken)


@pytest.mark.parametrize("label", ["", "Train", "TEST", "holdout"])
def test_validate_rejects_every_label_outside_the_four_partitions(
    cohort, assignment, label
) -> None:
    broken = dict(assignment)
    broken[sorted(members(assignment, "train"))[0]] = label

    with pytest.raises(SplitError, match="unknown split label"):
        validate_assignment(cohort, broken)


def test_validate_rejects_the_wrong_number_of_patients_in_a_partition(cohort, assignment) -> None:
    """Moving a primary subject between primary partitions breaks only counts."""
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    broken = dict(assignment)
    # Pick a group the validation set already holds twice, so validation still
    # contains both A and B afterwards and only the counts are wrong.
    validation_groups = [groups[s] for s in members(assignment, "validation")]
    duplicated = next(g for g in PRIMARY_GROUPS if validation_groups.count(g) > 1)
    moving = next(s for s in sorted(members(assignment, "train")) if groups[s] == duplicated)
    broken[moving] = "validation"

    # Both counts are now wrong; train is reported first by SPLIT_ORDER.
    with pytest.raises(SplitError, match="holds 24 patients, expected 25"):
        validate_assignment(cohort, broken)
    assert len(members(broken, "validation")) == 7


def test_validate_rejects_a_rare_group_subject_moved_out_of_stress(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    rare = sorted(s for s, group in groups.items() if group in STRESS_GROUPS)[0]
    broken = dict(assignment)
    broken[rare] = "validation"

    with pytest.raises(SplitError, match="reserved for the stress set"):
        validate_assignment(cohort, broken)


def test_validate_rejects_a_primary_subject_moved_into_stress(cohort, assignment) -> None:
    groups = {record.subject_id: classify_acquisition_group(record) for record in cohort}
    primary = sorted(members(assignment, "train"))[0]
    assert groups[primary] in PRIMARY_GROUPS
    broken = dict(assignment)
    broken[primary] = STRESS_SPLIT

    with pytest.raises(SplitError, match="from primary group"):
        validate_assignment(cohort, broken)


def test_validate_rejects_a_duplicated_subject_in_the_cohort(cohort, assignment) -> None:
    with pytest.raises(SplitError, match="Duplicate subject ids"):
        validate_assignment([*cohort, cohort[0]], assignment)


def test_validate_enforces_the_frozen_cohort_counts(cohort, assignment) -> None:
    """The default policy is the real cohort's 25 / 6 / 6 / 3."""
    assert COHORT_SPLIT_SIZES == {"train": 25, "validation": 6, "test": 6, "stress": 3}
    validate_assignment(cohort, assignment, COHORT_SPLIT_SIZES)

    with pytest.raises(SplitError, match="holds 25 patients, expected 24"):
        validate_assignment(cohort, assignment, {**COHORT_SPLIT_SIZES, "train": 24})


def test_validate_rejects_expected_counts_naming_an_unknown_partition(cohort, assignment) -> None:
    with pytest.raises(SplitError, match="unknown partition"):
        validate_assignment(cohort, assignment, {**COHORT_SPLIT_SIZES, "holdout": 0})


# --------------------------------------------------------------------------
# Building subject records from audit rows
# --------------------------------------------------------------------------


def audit_rows(subject_id: str, count: int, **overrides) -> list[dict]:
    base = {
        "subject_id": subject_id,
        "source": "Train_Sets",
        "rescale_intercept": -1024.0,
        "slice_thickness": 2.0,
        "pixel_representation": 0,
    }
    base.update(overrides)
    return [dict(base, file=f"i{index:04d}.dcm") for index in range(count)]


def test_subject_records_count_slices_and_carry_attributes() -> None:
    records = subject_records_from_audit(audit_rows("7", 12))

    assert len(records) == 1
    assert records[0].subject_id == "7"
    assert records[0].slice_count == 12
    assert classify_acquisition_group(records[0]) == "A"


def test_subject_records_are_ordered_numerically() -> None:
    rows = audit_rows("10", 2) + audit_rows("2", 2) + audit_rows("1", 2)

    assert [record.subject_id for record in subject_records_from_audit(rows)] == ["1", "2", "10"]


def test_subject_records_refuse_inconsistent_acquisition_attributes() -> None:
    rows = audit_rows("1", 2)
    rows[1]["rescale_intercept"] = -1000.0

    with pytest.raises(SplitError, match="inconsistent rescale_intercept"):
        subject_records_from_audit(rows)


# --------------------------------------------------------------------------
# The committed split artifact
# --------------------------------------------------------------------------


def read_committed_split() -> list[dict[str, str]]:
    with COMMITTED_SPLIT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_committed_split_file_exists() -> None:
    assert COMMITTED_SPLIT.is_file(), (
        "The frozen patient split must be committed; it is the experiment definition."
    )


def test_committed_split_covers_forty_unique_subjects() -> None:
    rows = read_committed_split()
    identifiers = [row["subject_id"] for row in rows]

    assert len(rows) == 40
    assert len(set(identifiers)) == 40


def test_committed_split_uses_only_the_four_partition_names() -> None:
    names = {row["split"] for row in read_committed_split()}

    assert names == {"train", "validation", "test", STRESS_SPLIT}


def test_committed_split_has_the_expected_patient_counts() -> None:
    rows = read_committed_split()
    counts = {
        name: sum(1 for row in rows if row["split"] == name) for name in {r["split"] for r in rows}
    }

    assert counts == {"train": 25, "validation": 6, "test": 6, "stress": 3}


def test_committed_split_keeps_rare_groups_out_of_the_primary_partitions() -> None:
    for row in read_committed_split():
        if row["split"] == STRESS_SPLIT:
            assert row["acquisition_group"] in STRESS_GROUPS
        else:
            assert row["acquisition_group"] in PRIMARY_GROUPS


@pytest.mark.parametrize("split", ["validation", "test"])
def test_committed_evaluation_partitions_contain_both_primary_groups(split) -> None:
    groups = {row["acquisition_group"] for row in read_committed_split() if row["split"] == split}

    assert groups == set(PRIMARY_GROUPS)


def test_committed_split_can_be_reproduced_by_the_generator() -> None:
    """The committed assignment must be what the algorithm produces, not hand-edited."""
    rows = read_committed_split()
    records = [
        SubjectRecord(
            subject_id=row["subject_id"],
            source_archive=row["source_archive"],
            slice_count=int(row["slice_count"]),
            rescale_intercept=float(row["rescale_intercept"]),
            slice_thickness=float(row["slice_thickness"]),
            pixel_representation=int(row["pixel_representation"]),
        )
        for row in rows
    ]

    regenerated = assign_splits(records)

    assert regenerated == {row["subject_id"]: row["split"] for row in rows}
    validate_assignment(records, regenerated)


def test_committed_split_checksum_is_frozen() -> None:
    """Guards the frozen experiment definition.

    If this fails, the patient split changed. That is allowed only as a
    deliberate methodological decision, and it breaks comparability with any
    result already reported against the previous split. Updating this constant
    is the explicit act that records such a decision.
    """
    expected = "4a11d080c80e4096c052436751ba948eb450539e5f83470b5ab2cf49722b3866"

    assert hashlib.sha256(COMMITTED_SPLIT.read_bytes()).hexdigest() == expected


# --------------------------------------------------------------------------
# Slice manifest
# --------------------------------------------------------------------------


def load_create_splits():
    path = REPOSITORY / "scripts" / "create_splits.py"
    spec = importlib.util.spec_from_file_location("create_splits", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["create_splits"] = module
    spec.loader.exec_module(module)
    return module


def test_slice_manifest_inherits_the_patient_partition(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    create_splits = load_create_splits()
    from test_chaos import build_chaos_tree

    build_chaos_tree(tmp_path, {"1": 3, "2": 2}, prefix="Train_Sets", with_mr=False)

    audit = pd.DataFrame(
        [
            {
                "subject_id": subject,
                "file": f"i{index:04d},0000b.dcm",
                "projected_position": float(10 - index),
            }
            for subject, count in (("1", 3), ("2", 2))
            for index in range(count)
        ]
    )
    patient_split = pd.DataFrame(
        [
            {
                "subject_id": "1",
                "source_archive": "Train_Sets",
                "acquisition_group": "A",
                "split": "train",
            },
            {
                "subject_id": "2",
                "source_archive": "Train_Sets",
                "acquisition_group": "B",
                "split": "test",
            },
        ]
    )

    manifest = create_splits.build_slice_manifest(audit, patient_split, tmp_path)

    assert len(manifest) == 5
    assert set(manifest[manifest.subject_id == "1"]["split"]) == {"train"}
    assert set(manifest[manifest.subject_id == "2"]["split"]) == {"test"}
    assert not manifest.duplicated().any()


def test_slice_manifest_orders_by_geometry_not_filename(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    create_splits = load_create_splits()
    from test_chaos import build_chaos_tree

    build_chaos_tree(tmp_path, {"1": 3}, prefix="Train_Sets", with_mr=False)

    # Filename order is the reverse of geometric order, as in the real cohort.
    audit = pd.DataFrame(
        [
            {"subject_id": "1", "file": "i0000,0000b.dcm", "projected_position": 30.0},
            {"subject_id": "1", "file": "i0001,0000b.dcm", "projected_position": 20.0},
            {"subject_id": "1", "file": "i0002,0000b.dcm", "projected_position": 10.0},
        ]
    )
    patient_split = pd.DataFrame(
        [
            {
                "subject_id": "1",
                "source_archive": "Train_Sets",
                "acquisition_group": "A",
                "split": "train",
            }
        ]
    )

    manifest = create_splits.build_slice_manifest(audit, patient_split, tmp_path)

    assert list(manifest["geometric_slice_position"]) == [10.0, 20.0, 30.0]
    assert list(manifest["geometric_slice_index"]) == [0, 1, 2]
    assert manifest.iloc[0]["relative_dicom_path"].endswith("i0002,0000b.dcm")
