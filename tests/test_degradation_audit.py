"""Tests for the pure helpers inside the degradation audit script.

The audit itself needs the real CHAOS download, so it is a command rather than
a test. These cover its policy and selection logic on synthetic frames, so a
mistake cannot quietly corrupt the tracked record.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_degradation.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_degradation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_degradation"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


# --------------------------------------------------------------------------
# --limit must never overwrite the canonical tracked summary
# --------------------------------------------------------------------------


def test_canonical_summary_is_the_parser_default() -> None:
    """The guard only protects the real path if that is the path used."""
    defaults = {action.dest: action.default for action in audit.build_parser()._actions}

    assert defaults["summary"] == audit.CANONICAL_SUMMARY.as_posix()
    assert defaults["limit"] == 0


def test_a_full_audit_to_the_canonical_summary_is_allowed() -> None:
    audit.check_output_policy(0, audit.CANONICAL_SUMMARY)


def test_limit_is_refused_when_writing_the_canonical_summary() -> None:
    """A truncated run looks exactly like a full one once written to disk."""
    with pytest.raises(ValueError, match="debug-only"):
        audit.check_output_policy(40, audit.CANONICAL_SUMMARY)


def test_limit_is_refused_through_an_equivalent_path_spelling(tmp_path) -> None:
    """Resolving defeats a different spelling of the same file."""
    roundabout = audit.CANONICAL_SUMMARY.parent / ".." / "audit" / audit.CANONICAL_SUMMARY.name

    with pytest.raises(ValueError, match="debug-only"):
        audit.check_output_policy(1, roundabout)


def test_limit_is_refused_through_an_absolute_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="debug-only"):
        audit.check_output_policy(1, audit.CANONICAL_SUMMARY.resolve())


def test_limit_is_allowed_to_a_noncanonical_summary(tmp_path) -> None:
    audit.check_output_policy(40, tmp_path / "debug_summary.json")


def test_the_refusal_names_the_way_out() -> None:
    with pytest.raises(ValueError) as caught:
        audit.check_output_policy(40, audit.CANONICAL_SUMMARY)

    message = str(caught.value)
    assert "--summary" in message
    assert audit.CANONICAL_SUMMARY.as_posix() in message


# --------------------------------------------------------------------------
# Manifest selection
# --------------------------------------------------------------------------


def _manifest_frame() -> pd.DataFrame:
    rows = []
    for subject, split, group, archive in [
        ("2", "train", "B", "Train_Sets"),
        ("21", "train", "A", "Train_Sets"),
        ("3", "train", "B", "Test_Sets"),
        ("31", "train", "A", "Test_Sets"),
        ("4", "validation", "A", "Test_Sets"),
        ("11", "test", "A", "Test_Sets"),
        ("1", "stress", "C", "Train_Sets"),
    ]:
        for index in range(5):
            rows.append(
                {
                    "subject_id": subject,
                    "split": split,
                    "source_archive": archive,
                    "acquisition_group": group,
                    "relative_dicom_path": f"{archive}/CT/{subject}/DICOM_anon/i{index:04d}.dcm",
                    "geometric_slice_position": -100.0 + index,
                    "geometric_slice_index": index,
                }
            )
    return pd.DataFrame(rows)


def test_training_slices_selects_only_the_train_split(tmp_path) -> None:
    """Held-out image content must not be reachable from this audit."""
    path = tmp_path / "manifest.csv"
    _manifest_frame().to_csv(path, index=False)

    training = audit.training_slices(path)

    assert set(training["split"]) == {"train"}
    assert len(training) == 20
    assert sorted(training["subject_id"].unique(), key=int) == ["2", "3", "21", "31"]


def test_training_slices_refuses_a_manifest_without_training_rows(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    frame = _manifest_frame()
    frame[frame["split"] != "train"].to_csv(path, index=False)

    with pytest.raises(ValueError, match="No split == train rows"):
        audit.training_slices(path)


def test_qc_selection_is_deterministic_and_covers_groups_and_archives() -> None:
    training = _manifest_frame()
    training = training[training["split"] == "train"].reset_index(drop=True)

    first = audit.select_qc_slices(training)
    second = audit.select_qc_slices(training.iloc[::-1].reset_index(drop=True))

    assert list(first["relative_dicom_path"]) == list(second["relative_dicom_path"])
    assert set(first["acquisition_group"]) == {"A", "B"}
    assert set(first["source_archive"]) == {"Train_Sets", "Test_Sets"}
    assert len(first) == 4


def test_qc_selection_takes_the_middle_slice_by_geometry() -> None:
    training = _manifest_frame()
    training = training[training["split"] == "train"].reset_index(drop=True)

    selection = audit.select_qc_slices(training)

    assert set(selection["geometric_slice_index"]) == {2}
