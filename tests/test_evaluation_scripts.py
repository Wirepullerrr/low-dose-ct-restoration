"""Tests for the helpers inside the Milestone 5 commands.

The commands themselves need the real CHAOS download, so they are commands
rather than tests. These cover their policy logic on synthetic frames, so the
hold-out gate and the canonical-output guard cannot quietly stop working.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from ct_restoration.evaluation import EvaluationError, HeldOutSplitError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mask_audit = _load("audit_body_mask")
baseline = _load("evaluate_degraded_baseline")


def manifest_frame() -> pd.DataFrame:
    """A miniature manifest covering all four partitions."""
    rows = []
    for subject, split, group, archive in [
        ("2", "train", "B", "Train_Sets"),
        ("21", "train", "A", "Train_Sets"),
        ("3", "train", "B", "Test_Sets"),
        ("31", "train", "A", "Test_Sets"),
        ("4", "validation", "B", "Test_Sets"),
        ("23", "validation", "A", "Train_Sets"),
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


@pytest.fixture
def manifest(tmp_path) -> Path:
    path = tmp_path / "manifest.csv"
    manifest_frame().to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# The M5 commands cannot reach test or stress
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module", [mask_audit, baseline], ids=["mask_audit", "baseline"])
@pytest.mark.parametrize("split", ["test", "stress"])
def test_commands_refuse_sealed_splits(module, split, manifest) -> None:
    """No accidental path to spending held-out data during development."""
    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        module.manifest_rows(manifest, split)


@pytest.mark.parametrize("module", [mask_audit, baseline], ids=["mask_audit", "baseline"])
@pytest.mark.parametrize(("split", "expected_rows"), [("train", 20), ("validation", 10)])
def test_commands_accept_development_splits(module, split, expected_rows, manifest) -> None:
    rows = module.manifest_rows(manifest, split)

    assert set(rows["split"]) == {split}
    assert len(rows) == expected_rows


@pytest.mark.parametrize("module", [mask_audit, baseline], ids=["mask_audit", "baseline"])
def test_commands_refuse_an_unknown_split(module, manifest) -> None:
    with pytest.raises(EvaluationError, match="Unknown split"):
        module.manifest_rows(manifest, "holdout")


def test_the_baseline_split_option_defaults_to_validation() -> None:
    defaults = {action.dest: action.default for action in baseline.build_parser()._actions}

    assert defaults["split"] == "validation"
    assert defaults["limit"] == 0


def test_the_baseline_help_names_the_refused_splits() -> None:
    options = {action.dest: action for action in baseline.build_parser()._actions}

    assert "refused" in options["split"].help


def test_the_mask_audit_reads_train_only() -> None:
    """It takes no --split option at all; train is hard-coded."""
    destinations = {action.dest for action in mask_audit.build_parser()._actions}

    assert "split" not in destinations


def test_manifest_rows_refuses_a_split_with_no_rows(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    frame = manifest_frame()
    frame[frame["split"] != "validation"].to_csv(path, index=False)

    with pytest.raises(ValueError, match="No split == validation rows"):
        baseline.manifest_rows(path, "validation")


def test_baseline_rows_are_sorted_deterministically(manifest) -> None:
    rows = baseline.manifest_rows(manifest, "train")
    shuffled = manifest.parent / "shuffled.csv"
    manifest_frame().iloc[::-1].to_csv(shuffled, index=False)

    assert list(rows["relative_dicom_path"]) == list(
        baseline.manifest_rows(shuffled, "train")["relative_dicom_path"]
    )
    # Numeric subject order, then geometric slice index.
    assert list(rows["subject_id"])[:5] == ["2"] * 5
    assert list(rows["geometric_slice_index"])[:5] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# --limit must never overwrite a canonical tracked output
# --------------------------------------------------------------------------


def test_mask_audit_refuses_limit_to_the_canonical_summary() -> None:
    with pytest.raises(ValueError, match="debug-only"):
        mask_audit.check_output_policy(30, mask_audit.CANONICAL_SUMMARY)


def test_mask_audit_allows_a_full_run_to_the_canonical_summary() -> None:
    mask_audit.check_output_policy(0, mask_audit.CANONICAL_SUMMARY)


def test_mask_audit_allows_limit_to_a_noncanonical_summary(tmp_path) -> None:
    mask_audit.check_output_policy(30, tmp_path / "debug.json")


def test_baseline_refuses_limit_to_the_canonical_metrics_directory() -> None:
    canonical = baseline.METRICS_DIR / "degraded_baseline_validation_slices.csv"

    with pytest.raises(ValueError, match="debug-only"):
        baseline.check_output_policy(30, canonical)


def test_baseline_allows_a_full_run_to_the_canonical_metrics_directory() -> None:
    baseline.check_output_policy(0, baseline.METRICS_DIR / "x_slices.csv")


def test_baseline_allows_limit_to_a_noncanonical_directory(tmp_path) -> None:
    baseline.check_output_policy(30, tmp_path / "debug_slices.csv")


def test_the_refusals_name_the_way_out() -> None:
    with pytest.raises(ValueError) as caught:
        baseline.check_output_policy(30, baseline.METRICS_DIR / "degraded_baseline_x_slices.csv")

    assert "--output-dir" in str(caught.value)


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------


def test_the_slice_table_columns_are_the_documented_ones() -> None:
    assert baseline.SLICE_COLUMNS == (
        "subject_id",
        "source_archive",
        "acquisition_group",
        "relative_dicom_path",
        "geometric_slice_index",
        "full_mae",
        "full_mse",
        "full_psnr",
        "full_ssim",
        "body_mae",
        "body_mse",
        "body_psnr",
        "body_ssim",
        "body_pixel_fraction",
        "body_ssim_interior_fraction",
    )


def test_the_slice_table_carries_no_identifying_or_varying_fields() -> None:
    """No DICOM UID, no timestamp, no pixel data."""
    forbidden = ("uid", "instance", "study", "series", "timestamp", "generated", "patient_name")

    for column in baseline.SLICE_COLUMNS:
        assert not any(token in column.lower() for token in forbidden), column


def test_qc_selection_is_deterministic_and_covers_groups_and_archives() -> None:
    rows = manifest_frame()
    rows = rows[rows["split"] == "train"].reset_index(drop=True)

    first = mask_audit.select_qc_slices(rows)
    second = mask_audit.select_qc_slices(rows.iloc[::-1].reset_index(drop=True))

    assert list(first["relative_dicom_path"]) == list(second["relative_dicom_path"])
    assert set(first["acquisition_group"]) == {"A", "B"}
    assert set(first["source_archive"]) == {"Train_Sets", "Test_Sets"}
    assert set(first["geometric_slice_index"]) == {2}
