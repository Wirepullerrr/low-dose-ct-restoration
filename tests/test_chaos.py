"""Tests for CHAOS directory discovery.

These build synthetic folder trees in a temporary directory. They never touch a
real CHAOS download, so the suite still runs on a fresh clone with no data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ct_restoration.data.chaos import find_ct_subjects, locate_ct_roots


def build_chaos_tree(
    root: Path,
    subjects: dict[str, int],
    *,
    prefix: str = "Train_Sets",
    with_ground_truth: bool = True,
    with_mr: bool = True,
) -> Path:
    """Create a CHAOS-shaped tree with `subjects` mapping id -> slice count."""
    ct_root = root / prefix / "CT" if prefix else root / "CT"
    for subject_id, slice_count in subjects.items():
        dicom_dir = ct_root / subject_id / "DICOM_anon"
        dicom_dir.mkdir(parents=True)
        for index in range(slice_count):
            (dicom_dir / f"i{index:04d},0000b.dcm").write_bytes(b"not-real-dicom")
        if with_ground_truth:
            ground = ct_root / subject_id / "Ground"
            ground.mkdir()
            (ground / f"liver_GT_{subject_id}.png").write_bytes(b"not-a-real-png")

    if with_mr:
        mr_dir = root / prefix / "MR" / "1" / "T1DUAL" if prefix else root / "MR" / "1"
        mr_dir.mkdir(parents=True)
        (mr_dir / "mr.dcm").write_bytes(b"not-real-dicom")

    return ct_root


# --------------------------------------------------------------------------
# Locating the CT roots
# --------------------------------------------------------------------------


def test_locate_ct_roots_from_the_extraction_root(tmp_path: Path) -> None:
    ct_root = build_chaos_tree(tmp_path, {"1": 2})

    assert locate_ct_roots(tmp_path) == [ct_root]


def test_locate_ct_roots_when_ct_is_directly_below_root(tmp_path: Path) -> None:
    ct_root = build_chaos_tree(tmp_path, {"1": 2}, prefix="")

    assert locate_ct_roots(tmp_path) == [ct_root]


def test_locate_ct_roots_when_pointed_straight_at_one(tmp_path: Path) -> None:
    ct_root = build_chaos_tree(tmp_path, {"1": 2})

    assert locate_ct_roots(ct_root) == [ct_root]


def test_locate_ct_roots_finds_both_archive_trees(tmp_path: Path) -> None:
    train = build_chaos_tree(tmp_path, {"1": 2}, prefix="Train_Sets", with_mr=False)
    test = build_chaos_tree(tmp_path, {"3": 2}, prefix="Test_Sets", with_mr=False)

    assert locate_ct_roots(tmp_path) == sorted([train, test])


def test_locate_ct_roots_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        locate_ct_roots(tmp_path / "nowhere")


def test_locate_ct_roots_reports_an_empty_download(tmp_path: Path) -> None:
    (tmp_path / "Train_Sets" / "MR").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No CT directory"):
        locate_ct_roots(tmp_path)


# --------------------------------------------------------------------------
# Discovering subjects
# --------------------------------------------------------------------------


def test_subjects_are_discovered_with_their_slice_files(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 3, "2": 5})

    subjects = find_ct_subjects(tmp_path)

    assert [subject.subject_id for subject in subjects] == ["1", "2"]
    assert [subject.slice_count for subject in subjects] == [3, 5]
    assert all(path.suffix == ".dcm" for subject in subjects for path in subject.dicom_paths)


def test_subject_ids_sort_numerically_not_lexicographically(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 1, "2": 1, "10": 1, "21": 1})

    subjects = find_ct_subjects(tmp_path)

    assert [subject.subject_id for subject in subjects] == ["1", "2", "10", "21"]


def test_slice_filenames_sort_numerically(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 12})

    names = [path.name for path in find_ct_subjects(tmp_path)[0].dicom_paths]

    assert names[:3] == ["i0000,0000b.dcm", "i0001,0000b.dcm", "i0002,0000b.dcm"]
    assert names[-1] == "i0011,0000b.dcm"


def test_mr_series_are_not_reported_as_ct_subjects(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 2}, with_mr=True)

    subjects = find_ct_subjects(tmp_path)

    assert len(subjects) == 1
    assert all("MR" not in path.parts for path in subjects[0].dicom_paths)


def test_ground_truth_folder_is_reported_when_present(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 2}, with_ground_truth=True)

    assert find_ct_subjects(tmp_path)[0].ground_truth_dir is not None


def test_ground_truth_folder_is_none_when_absent(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 2}, with_ground_truth=False)

    assert find_ct_subjects(tmp_path)[0].ground_truth_dir is None


def test_folders_without_dicom_files_are_skipped(tmp_path: Path) -> None:
    ct_root = build_chaos_tree(tmp_path, {"1": 2})
    (ct_root / "9" / "Ground").mkdir(parents=True)
    (ct_root / "9" / "Ground" / "mask.png").write_bytes(b"not-a-real-png")

    assert [subject.subject_id for subject in find_ct_subjects(tmp_path)] == ["1"]


def test_discovery_finds_slices_nested_below_the_subject_folder(tmp_path: Path) -> None:
    ct_root = build_chaos_tree(tmp_path, {"1": 0}, with_ground_truth=False)
    nested = ct_root / "1" / "DICOM_anon" / "extra"
    nested.mkdir(parents=True)
    (nested / "a.dcm").write_bytes(b"not-real-dicom")

    subjects = find_ct_subjects(tmp_path)

    assert len(subjects) == 1
    assert subjects[0].slice_count == 1


# --------------------------------------------------------------------------
# Pooling the two archive trees
# --------------------------------------------------------------------------


def test_subjects_from_both_archives_are_pooled_and_ordered(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 2, "10": 2}, prefix="Train_Sets", with_mr=False)
    build_chaos_tree(tmp_path, {"3": 2, "40": 2}, prefix="Test_Sets", with_mr=False)

    subjects = find_ct_subjects(tmp_path)

    assert [subject.subject_id for subject in subjects] == ["1", "3", "10", "40"]


def test_each_subject_records_the_archive_it_came_from(tmp_path: Path) -> None:
    build_chaos_tree(tmp_path, {"1": 2}, prefix="Train_Sets", with_mr=False)
    build_chaos_tree(tmp_path, {"3": 2}, prefix="Test_Sets", with_mr=False)

    sources = {subject.subject_id: subject.source for subject in find_ct_subjects(tmp_path)}

    assert sources == {"1": "Train_Sets", "3": "Test_Sets"}


def test_a_subject_id_appearing_in_both_archives_is_refused(tmp_path: Path) -> None:
    """Pooling a repeated id could put one patient on both sides of a split."""
    build_chaos_tree(tmp_path, {"7": 2}, prefix="Train_Sets", with_mr=False)
    build_chaos_tree(tmp_path, {"7": 2}, prefix="Test_Sets", with_mr=False)

    with pytest.raises(ValueError, match="appears in both"):
        find_ct_subjects(tmp_path)
