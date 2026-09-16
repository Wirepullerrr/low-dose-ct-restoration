"""Patient-level experimental partitioning.

The independent experimental unit in this project is the **patient**, never the
slice. All slices of one CHAOS subject always land in exactly one partition, so
that no anatomy from a test patient can ever have been seen during training.

Four partitions are produced:

``train``
    parameter fitting.
``validation``
    every empirical decision: hyperparameters, degradation settings, CLAHE
    parameters, checkpoint and model selection.
``test``
    the final primary benchmark, evaluated once decisions are frozen.
``stress``
    a small held-out rare-acquisition robustness stress set, evaluated only
    after the primary decisions are frozen.

Why a stress set exists
-----------------------
The cohort contains four observed acquisition and reconstruction parameter
groups. Two of them hold only two subjects and one subject. A group that small
cannot be represented in train *and* validation *and* test at the same time,
and reporting a one-patient slice of a partition as an estimate of performance
on that acquisition type would be misleading. Rather than pretend to balanced
acquisition coverage, every subject from the rare groups is withheld entirely,
and the primary benchmark states plainly that it covers the two adequately
represented groups.

Determinism
-----------
No random number generator is involved. Subjects are canonically ordered before
anything is decided, the allocation of acquisition groups across partitions is
chosen by exhaustive scoring, and assignment is a largest-first quota fill
followed by an exhaustive improving-swap refinement. The same inputs always
produce the same assignment, in any input row order.

What actually drives the assignment
-----------------------------------
``subject_id``
    deterministic canonical ordering and tie-breaking only. It never makes a
    subject more or less likely to land in a particular partition.
``acquisition group``
    a hard constraint: rare groups are withheld, and the primary groups are
    allocated across partitions by the scoring below.
``slice_count``
    balance, so that partitions holding equal patient counts also hold
    comparable image counts.

``source_archive`` is **not** an input to any decision here. CHAOS
``Train_Sets`` / ``Test_Sets`` membership is segmentation-challenge provenance,
not a restoration-task label, so it is carried through to the outputs and
audited after the fact rather than optimised for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Any

#: Observed acquisition and reconstruction parameter signatures, measured in
#: Milestone 2 and keyed by (RescaleIntercept, SliceThickness,
#: PixelRepresentation). Labels are stable so that documentation, outputs and
#: code agree. Subject *membership* is never hard-coded; it is derived by
#: matching each subject's measured attributes against these signatures.
ACQUISITION_GROUPS: dict[tuple[float, float, int], str] = {
    (-1024.0, 2.0, 0): "A",
    (-1000.0, 3.2, 0): "B",
    (-1200.0, 3.0, 0): "C",
    (0.0, 2.0, 1): "D",
}

#: Groups with enough independent patients to appear in every partition.
PRIMARY_GROUPS: tuple[str, ...] = ("A", "B")
#: Groups too small to stratify; withheld whole as the stress set.
STRESS_GROUPS: tuple[str, ...] = ("C", "D")

#: Target patient counts for the primary partitions.
PRIMARY_SPLIT_SIZES: dict[str, int] = {"train": 25, "validation": 6, "test": 6}

#: Expected patient counts for every partition under the configured policy,
#: applied to the frozen 40-subject CHAOS cohort (16 A, 21 B, 2 C, 1 D). The
#: stress count is the size of the rare groups, which is a property of the
#: cohort rather than a free choice. :func:`validate_assignment` enforces these
#: counts exactly unless a caller passes different ones.
COHORT_SPLIT_SIZES: dict[str, int] = {"train": 25, "validation": 6, "test": 6, "stress": 3}

#: Fixed partition order, used for deterministic tie-breaking.
SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test", "stress")

STRESS_SPLIT = "stress"


class SplitError(ValueError):
    """The cohort does not support the configured partitioning."""


@dataclass(frozen=True)
class SubjectRecord:
    """One subject's split-relevant facts, all measured in the audit."""

    subject_id: str
    source_archive: str
    slice_count: int
    rescale_intercept: float
    slice_thickness: float
    pixel_representation: int

    @property
    def acquisition_signature(self) -> tuple[float, float, int]:
        return (
            float(self.rescale_intercept),
            float(self.slice_thickness),
            int(self.pixel_representation),
        )


def subject_sort_key(subject_id: str) -> tuple[object, ...]:
    """Canonical subject ordering: numeric where possible, then textual."""
    return tuple(
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", subject_id)
        if chunk
    )


def classify_acquisition_group(record: SubjectRecord) -> str:
    """Label a subject by its measured acquisition signature.

    Raises:
        SplitError: the subject's attributes match no known signature. Failing
            here is deliberate: an unrecognised acquisition group must be an
            explicit design decision, not silently folded into an existing one.
    """
    group = ACQUISITION_GROUPS.get(record.acquisition_signature)
    if group is None:
        raise SplitError(
            f"Subject {record.subject_id!r} has an unrecognised acquisition signature "
            f"{record.acquisition_signature}. Known signatures: {sorted(ACQUISITION_GROUPS)}. "
            "Classify it explicitly before generating a split."
        )
    return group


def choose_group_allocation(
    group_totals: dict[str, int],
    split_sizes: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Decide how many subjects of each primary group each partition receives.

    Every feasible integer allocation is enumerated and scored. The score is the
    summed absolute deviation of each partition's group-A share from the share
    across the whole primary cohort, so an allocation is preferred when all
    three partitions look like the cohort they were drawn from. Ties are broken
    first towards validation and test having identical composition, and then by
    a fixed ordering so the result never depends on iteration order.

    Several feasible allocations give validation and test identical
    composition, so that property alone does not select one. It only decides
    between allocations that already score equally on deviation. Matching the
    two evaluation partitions removes one known source of compositional
    mismatch between them; it does not make six validation patients
    representative of six different test patients.

    Both evaluation partitions are required to contain both primary groups.

    Returns:
        ``{split: {group: subject count}}``.
    """
    if len(PRIMARY_GROUPS) != 2:
        raise SplitError("choose_group_allocation assumes exactly two primary groups")

    first, second = PRIMARY_GROUPS
    total_first, total_second = group_totals[first], group_totals[second]
    total_subjects = total_first + total_second
    if total_subjects != sum(split_sizes.values()):
        raise SplitError(
            f"{total_subjects} primary subjects cannot fill partitions of sizes {split_sizes}."
        )

    cohort_share = total_first / total_subjects
    evaluation_splits = [name for name in SPLIT_ORDER if name in split_sizes and name != "train"]

    candidates: list[tuple[tuple[float, int, tuple[int, ...]], dict[str, dict[str, int]]]] = []
    ranges = [range(1, split_sizes[name]) for name in evaluation_splits]
    for counts in product(*ranges):
        allocation: dict[str, dict[str, int]] = {}
        used_first = 0
        for name, count in zip(evaluation_splits, counts, strict=True):
            allocation[name] = {first: count, second: split_sizes[name] - count}
            used_first += count

        train_first = total_first - used_first
        train_second = split_sizes["train"] - train_first
        if train_first < 0 or train_second < 0:
            continue
        if train_second != total_second - sum(allocation[n][second] for n in evaluation_splits):
            continue
        allocation["train"] = {first: train_first, second: train_second}

        deviation = sum(
            abs(allocation[name][first] / split_sizes[name] - cohort_share) for name in split_sizes
        )
        asymmetry = len({allocation[name][first] for name in evaluation_splits})
        candidates.append(((round(deviation, 12), asymmetry, counts), allocation))

    if not candidates:
        raise SplitError(
            f"No allocation of groups {PRIMARY_GROUPS} satisfies partition sizes {split_sizes} "
            "while keeping both groups in every evaluation partition."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _quota_fill(
    records: list[SubjectRecord],
    groups: dict[str, str],
    allocation: dict[str, dict[str, int]],
    split_sizes: dict[str, int],
) -> dict[str, str]:
    """Largest-first fill: each subject joins the partition furthest below quota.

    Subjects are placed heaviest first, because a large subject placed late can
    no longer be compensated for. Each partition carries a slice quota
    proportional to its patient count, and a subject goes to whichever partition
    still has room for its acquisition group and is furthest below that quota.
    """
    total_slices = sum(record.slice_count for record in records)
    total_patients = sum(split_sizes.values())
    quota = {name: total_slices * size / total_patients for name, size in split_sizes.items()}

    remaining = {name: dict(counts) for name, counts in allocation.items()}
    assigned_slices = dict.fromkeys(split_sizes, 0)
    assignment: dict[str, str] = {}

    ordered = sorted(
        records,
        key=lambda record: (-record.slice_count, subject_sort_key(record.subject_id)),
    )
    for record in ordered:
        group = groups[record.subject_id]
        options = [name for name in SPLIT_ORDER if remaining.get(name, {}).get(group, 0) > 0]
        if not options:
            raise SplitError(
                f"No partition can still accept a group {group} subject "
                f"({record.subject_id}); allocation and cohort disagree."
            )
        chosen = min(options, key=lambda name: (assigned_slices[name] - quota[name], name))
        assignment[record.subject_id] = chosen
        remaining[chosen][group] -= 1
        assigned_slices[chosen] += record.slice_count

    return assignment


def _balance_evaluation_partitions(
    records: list[SubjectRecord],
    groups: dict[str, str],
    assignment: dict[str, str],
) -> dict[str, str]:
    """Swap same-group subjects between validation and test to even slice totals.

    Patient counts alone do not balance image counts, because subjects hold
    between 78 and 294 slices. Only swaps within one acquisition group are
    considered, so the allocation chosen earlier is preserved exactly. Every
    candidate swap is evaluated and the best improving one is applied, repeatedly
    until none improves, which makes the outcome independent of ordering.
    """
    slices = {record.subject_id: record.slice_count for record in records}
    assignment = dict(assignment)

    def total(split: str) -> int:
        return sum(slices[s] for s, name in assignment.items() if name == split)

    while True:
        current = abs(total("validation") - total("test"))
        best_gap, best_swap = current, None

        validation = sorted(
            (s for s, name in assignment.items() if name == "validation"), key=subject_sort_key
        )
        test = sorted((s for s, name in assignment.items() if name == "test"), key=subject_sort_key)
        difference = total("validation") - total("test")

        for left in validation:
            for right in test:
                if groups[left] != groups[right]:
                    continue
                gap = abs(difference - 2 * (slices[left] - slices[right]))
                if gap < best_gap:
                    best_gap, best_swap = gap, (left, right)

        if best_swap is None:
            return assignment
        left, right = best_swap
        assignment[left], assignment[right] = "test", "validation"


def assign_splits(
    records: list[SubjectRecord],
    split_sizes: dict[str, int] | None = None,
) -> dict[str, str]:
    """Assign every subject to exactly one partition, deterministically.

    Args:
        records: one entry per subject. Order is irrelevant.
        split_sizes: patient counts for the primary partitions.

    Returns:
        ``{subject_id: split}`` covering every input subject.

    Raises:
        SplitError: duplicate subjects, an unknown acquisition signature, or a
            cohort that cannot fill the configured partitions.
    """
    split_sizes = dict(split_sizes or PRIMARY_SPLIT_SIZES)

    identifiers = [record.subject_id for record in records]
    duplicates = sorted({s for s in identifiers if identifiers.count(s) > 1}, key=subject_sort_key)
    if duplicates:
        raise SplitError(f"Duplicate subject ids in the cohort: {duplicates}")

    ordered = sorted(records, key=lambda record: subject_sort_key(record.subject_id))
    groups = {record.subject_id: classify_acquisition_group(record) for record in ordered}

    stress = [record for record in ordered if groups[record.subject_id] in STRESS_GROUPS]
    primary = [record for record in ordered if groups[record.subject_id] in PRIMARY_GROUPS]

    group_totals = {
        group: sum(1 for record in primary if groups[record.subject_id] == group)
        for group in PRIMARY_GROUPS
    }
    allocation = choose_group_allocation(group_totals, split_sizes)

    assignment = _quota_fill(primary, groups, allocation, split_sizes)
    assignment = _balance_evaluation_partitions(primary, groups, assignment)
    for record in stress:
        assignment[record.subject_id] = STRESS_SPLIT

    return assignment


def validate_assignment(
    records: list[SubjectRecord],
    assignment: dict[str, str],
    expected_counts: dict[str, int] | None = None,
) -> None:
    """Assert every invariant the experiment depends on.

    This is the gate the generated split has to pass before it is written, and
    the same gate the committed file is re-checked against. It is deliberately
    stricter than the generator: it re-derives acquisition groups from the
    measured records rather than trusting any label handed to it, and it
    accepts nothing it was not configured to accept.

    Checked here:

    1. the cohort contains no duplicate subject ids;
    2. the assignment covers every subject exactly once, and nothing else;
    3. every assigned label is one of ``train``, ``validation``, ``test``,
       ``stress`` -- an unrecognised label is refused, never ignored;
    4. partitions are mutually exclusive;
    5. every rare-group subject is in the stress set;
    6. every stress subject is from a rare group;
    7. every primary-group subject is in train, validation or test;
    8. train, validation and test contain no rare-group subject;
    9. validation and test each contain both primary groups;
    10. patient counts match ``expected_counts`` exactly.

    Counts are checked last on purpose. A misplaced patient disturbs a count as
    well as the invariant it actually broke, and the invariant is the more
    useful thing to be told about.

    Args:
        records: the measured cohort; acquisition groups are re-derived from it.
        assignment: ``{subject_id: split}``.
        expected_counts: required patient count per partition. Defaults to
            :data:`COHORT_SPLIT_SIZES`, the configured real-cohort policy.

    Raises:
        SplitError: any invariant is violated.
    """
    expected_counts = dict(expected_counts or COHORT_SPLIT_SIZES)
    unknown_expected = sorted(set(expected_counts) - set(SPLIT_ORDER))
    if unknown_expected:
        raise SplitError(
            f"expected_counts names unknown partition(s) {unknown_expected}; "
            f"known partitions are {list(SPLIT_ORDER)}"
        )

    identifier_list = [record.subject_id for record in records]
    duplicates = sorted(
        {s for s in identifier_list if identifier_list.count(s) > 1}, key=subject_sort_key
    )
    if duplicates:
        raise SplitError(f"Duplicate subject ids in the cohort: {duplicates}")

    groups = {record.subject_id: classify_acquisition_group(record) for record in records}
    identifiers = set(identifier_list)

    if set(assignment) != identifiers:
        missing = sorted(identifiers - set(assignment), key=subject_sort_key)
        extra = sorted(set(assignment) - identifiers, key=subject_sort_key)
        raise SplitError(f"Assignment does not cover the cohort; missing {missing}, extra {extra}")

    unknown_labels = sorted(set(assignment.values()) - set(SPLIT_ORDER))
    if unknown_labels:
        raise SplitError(
            f"Assignment uses unknown split label(s) {unknown_labels}; "
            f"permitted labels are {list(SPLIT_ORDER)}"
        )

    members = {name: {s for s, v in assignment.items() if v == name} for name in SPLIT_ORDER}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = members[left] & members[right]
            if overlap:
                raise SplitError(
                    f"Patient overlap between {left} and {right}: "
                    f"{sorted(overlap, key=subject_sort_key)}"
                )

    for subject, group in sorted(groups.items(), key=lambda item: subject_sort_key(item[0])):
        if group in STRESS_GROUPS and assignment[subject] != STRESS_SPLIT:
            raise SplitError(
                f"Subject {subject} is from rare group {group} but sits in "
                f"{assignment[subject]!r}; rare groups are reserved for the stress set."
            )
        if group in PRIMARY_GROUPS and assignment[subject] == STRESS_SPLIT:
            raise SplitError(f"Stress subject {subject} is from primary group {group}")

    for subject in sorted(members[STRESS_SPLIT], key=subject_sort_key):
        if groups[subject] not in STRESS_GROUPS:
            raise SplitError(f"Stress subject {subject} is from primary group {groups[subject]}")

    for name in ("train", "validation", "test"):
        for subject in sorted(members[name], key=subject_sort_key):
            if groups[subject] not in PRIMARY_GROUPS:
                raise SplitError(
                    f"{name} contains subject {subject} from rare group {groups[subject]}; "
                    "rare groups are reserved for the stress set."
                )

    for name in ("validation", "test"):
        present = {groups[subject] for subject in members[name]}
        missing = set(PRIMARY_GROUPS) - present
        if missing:
            raise SplitError(f"{name} is missing primary group(s) {sorted(missing)}")

    for name in SPLIT_ORDER:
        expected = expected_counts.get(name, 0)
        actual = len(members[name])
        if actual != expected:
            raise SplitError(
                f"Partition {name!r} holds {actual} patients, expected {expected}. "
                "Patient counts are part of the experiment definition, not an outcome."
            )


def subject_records_from_audit(rows: list[dict[str, Any]]) -> list[SubjectRecord]:
    """Build subject records from per-slice audit rows.

    Each subject's acquisition attributes must be constant across its slices;
    disagreement means the cohort is not what the audit described.

    Raises:
        SplitError: a subject's slices disagree about an acquisition attribute.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)

    records: list[SubjectRecord] = []
    for subject_id in sorted(grouped, key=subject_sort_key):
        slices = grouped[subject_id]
        for field in ("source", "rescale_intercept", "slice_thickness", "pixel_representation"):
            values = {row[field] for row in slices}
            if len(values) != 1:
                raise SplitError(
                    f"Subject {subject_id!r} has inconsistent {field}: {sorted(values)}"
                )
        first = slices[0]
        records.append(
            SubjectRecord(
                subject_id=subject_id,
                source_archive=str(first["source"]),
                slice_count=len(slices),
                rescale_intercept=float(first["rescale_intercept"]),
                slice_thickness=float(first["slice_thickness"]),
                pixel_representation=int(first["pixel_representation"]),
            )
        )
    return records
