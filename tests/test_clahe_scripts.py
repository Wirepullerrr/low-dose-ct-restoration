"""Tests for the policy logic inside the Milestone 6 commands.

The commands need the real CHAOS download, so they are commands rather than
tests. What is testable without it is exactly what must not quietly break: the
hold-out gate, the freeze guard, the canonical-output guard, and above all
that a candidate is chosen on patient-weighted rather than slice-weighted
numbers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ct_restoration.classical.clahe import ClaheConfig
from ct_restoration.classical.search import PATIENT_WEIGHTED_PREFIX, rank_candidates
from ct_restoration.evaluation import (
    METRIC_COLUMNS,
    EvaluationError,
    HeldOutSplitError,
    aggregate_slices_to_patients,
    slice_weighted_summary,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tune = _load("tune_clahe")
evaluate = _load("evaluate_clahe")
qc = _load("qc_clahe")


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
# Hold-out discipline
# --------------------------------------------------------------------------


def test_tuning_is_pinned_to_validation() -> None:
    """No --split option at all: there is no way to tune against test."""
    destinations = {action.dest for action in tune.build_parser()._actions}

    assert tune.TUNING_SPLIT == "validation"
    assert "split" not in destinations


def test_qc_is_pinned_to_train() -> None:
    destinations = {action.dest for action in qc.build_parser()._actions}

    assert qc.QC_SPLIT == "train"
    assert "split" not in destinations


@pytest.mark.parametrize("split", ["test", "stress"])
def test_tuning_cannot_read_a_sealed_split(split, manifest) -> None:
    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        tune.manifest_rows(manifest, split)


@pytest.mark.parametrize("split", ["test", "stress"])
def test_the_clahe_evaluation_refuses_a_sealed_split(split, manifest) -> None:
    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        evaluate.manifest_rows(manifest, split)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_the_clahe_evaluation_accepts_development_splits(split, manifest) -> None:
    rows = evaluate.manifest_rows(manifest, split)

    assert set(rows["split"]) == {split}


def test_the_clahe_evaluation_refuses_an_unknown_split(manifest) -> None:
    with pytest.raises(EvaluationError, match="Unknown split"):
        evaluate.manifest_rows(manifest, "holdout")


def test_the_clahe_evaluation_defaults_to_validation() -> None:
    defaults = {action.dest: action.default for action in evaluate.build_parser()._actions}

    assert defaults["split"] == "validation"
    assert defaults["limit"] == 0


# --------------------------------------------------------------------------
# Guards on canonical outputs
# --------------------------------------------------------------------------


def test_tuning_refuses_limit_to_the_canonical_search_table() -> None:
    with pytest.raises(ValueError, match="debug-only"):
        tune.check_output_policy(30, tune.CANONICAL_SEARCH_CSV)


def test_tuning_allows_a_full_sweep_to_the_canonical_search_table() -> None:
    tune.check_output_policy(0, tune.CANONICAL_SEARCH_CSV)


def test_tuning_allows_limit_to_a_noncanonical_table(tmp_path) -> None:
    tune.check_output_policy(30, tmp_path / "debug_search.csv")


def test_tuning_refuses_to_overwrite_an_existing_frozen_config(tmp_path) -> None:
    """CLAHE must not be quietly retuned after later results exist."""
    existing = tmp_path / "clahe.yaml"
    existing.write_text("clahe: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists and is frozen"):
        tune.check_freeze_policy(existing, overwrite=False)


def test_tuning_allows_a_deliberate_overwrite(tmp_path) -> None:
    existing = tmp_path / "clahe.yaml"
    existing.write_text("clahe: {}\n", encoding="utf-8")

    tune.check_freeze_policy(existing, overwrite=True)


def test_tuning_allows_writing_a_config_that_does_not_exist_yet(tmp_path) -> None:
    tune.check_freeze_policy(tmp_path / "absent.yaml", overwrite=False)


def test_the_clahe_evaluation_refuses_limit_to_canonical_metrics() -> None:
    canonical = evaluate.METRICS_DIR / "clahe_validation_slices.csv"

    with pytest.raises(ValueError, match="debug-only"):
        evaluate.check_metrics_output_policy(30, canonical)


def test_the_clahe_evaluation_allows_limit_elsewhere(tmp_path) -> None:
    evaluate.check_metrics_output_policy(30, tmp_path / "clahe_validation_slices.csv")


# --------------------------------------------------------------------------
# Provenance recorded in the frozen config
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given",
    ["clahe_search.yaml", "configs/clahe_search.yaml", "./configs/clahe_search.yaml"],
)
def test_provenance_records_the_tracked_repo_relative_path(given) -> None:
    """Whatever shorthand the command line uses, the file named is openable."""
    assert tune.search_config_provenance(given) == "configs/clahe_search.yaml"


def test_provenance_resolves_an_absolute_path_inside_the_repo() -> None:
    absolute = (tune.CONFIGS_DIR / "clahe_search.yaml").resolve()

    assert tune.search_config_provenance(absolute) == "configs/clahe_search.yaml"


def test_the_frozen_config_written_records_that_path(tmp_path) -> None:
    written = tune.write_frozen_config(
        ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4)),
        tmp_path / "clahe.yaml",
        "configs/clahe_search.yaml",
    )
    document = yaml.safe_load(written.read_text(encoding="utf-8"))["clahe"]

    assert document["selected_by"]["search_config"] == "configs/clahe_search.yaml"
    # Provenance only: the runtime parameters are untouched by it.
    assert document["clip_limit"] == 0.5
    assert document["tile_grid_size"] == [4, 4]


# --------------------------------------------------------------------------
# Candidate ranking must be patient-weighted
# --------------------------------------------------------------------------


def slice_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """A per-slice table where every metric column carries the same value."""
    records = []
    for index, (subject_id, value) in enumerate(rows):
        record = {
            "subject_id": subject_id,
            "source_archive": "Train_Sets",
            "acquisition_group": "A",
            "relative_dicom_path": f"x/{subject_id}/{index}.dcm",
            "geometric_slice_index": index,
            "body_pixel_fraction": 0.5,
            "body_ssim_interior_fraction": 0.4,
        }
        record.update({name: value for name in METRIC_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


def test_slice_weighting_would_pick_a_different_candidate() -> None:
    """The distinction the whole aggregation policy exists to protect.

    Candidate "uneven" scores 1.0 on a patient with one slice and 0.0 on a
    patient with nine. Candidate "even" scores 0.3 on both. Weighting patients
    equally makes uneven the winner at 0.5 against 0.3; pooling slices makes
    even the winner at 0.3 against 0.1. The tuning code must choose uneven.
    """
    uneven = slice_frame([("4", 1.0)] + [("14", 0.0)] * 9)
    even = slice_frame([("4", 0.3)] + [("14", 0.3)] * 9)

    candidates = [
        ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4)),
        ClaheConfig(clip_limit=1.0, tile_grid_size=(8, 8)),
    ]
    per_candidate = {candidates[0].label: uneven, candidates[1].label: even}
    baseline = dict.fromkeys(METRIC_COLUMNS, 0.0)

    table = tune.candidate_table(candidates, per_candidate, baseline)
    ranked = rank_candidates(table)
    winner = ranked.iloc[0]

    # Patient-weighted values are what landed in the table.
    by_clip = table.set_index("clip_limit")[f"{PATIENT_WEIGHTED_PREFIX}body_ssim"]
    assert by_clip.loc[0.5] == pytest.approx(0.5)
    assert by_clip.loc[1.0] == pytest.approx(0.3)

    # Slice weighting would have reversed the ranking.
    assert slice_weighted_summary(uneven)["body_ssim"]["mean"] == pytest.approx(0.1)
    assert slice_weighted_summary(even)["body_ssim"]["mean"] == pytest.approx(0.3)

    assert winner["clip_limit"] == 0.5
    assert bool(winner["selected"])


def test_the_candidate_table_reports_patient_counts_not_slice_counts() -> None:
    frame = slice_frame([("4", 0.5)] + [("14", 0.5)] * 9)
    config = ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4))

    table = tune.candidate_table(
        [config], {config.label: frame}, dict.fromkeys(METRIC_COLUMNS, 0.0)
    )

    assert int(table.iloc[0]["patients"]) == 2
    assert len(aggregate_slices_to_patients(frame)) == 2


def test_deltas_are_candidate_minus_baseline() -> None:
    frame = slice_frame([("4", 0.8), ("14", 0.8)])
    config = ClaheConfig(clip_limit=0.5, tile_grid_size=(4, 4))
    baseline = dict.fromkeys(METRIC_COLUMNS, 0.5)

    row = tune.candidate_table([config], {config.label: frame}, baseline).iloc[0]

    assert row[f"{PATIENT_WEIGHTED_PREFIX}body_ssim"] == pytest.approx(0.8)
    assert row["delta_vs_degraded_body_ssim"] == pytest.approx(0.3)


def test_the_candidate_table_has_a_row_per_candidate() -> None:
    configs = [ClaheConfig(clip_limit=clip, tile_grid_size=(4, 4)) for clip in (0.5, 1.0, 2.0, 4.0)]
    frames = {config.label: slice_frame([("4", 0.5), ("14", 0.5)]) for config in configs}

    table = tune.candidate_table(configs, frames, dict.fromkeys(METRIC_COLUMNS, 0.0))

    assert len(table) == 4
    assert list(table["clip_limit"]) == [0.5, 1.0, 2.0, 4.0]
