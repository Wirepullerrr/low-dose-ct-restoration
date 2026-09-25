"""Milestone 11A: the frozen held-out protocol, exercised without a test image.

Everything here is synthetic. A miniature repository is built in ``tmp_path``
- DICOM files, a split, a manifest, configs, checkpoints, run records,
validation summaries and a plan - and the protocol is driven through it,
including one complete execution and one failed one. The real repository is
read only for tracked metadata: the real plan, the split, the manifest and
the committed run and metric records. No real image is opened, which the
tests that touch the real repository prove with the same audit-hook monitor
the runner uses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import build_ct_dataset

from ct_restoration import benchmark, holdout
from ct_restoration.benchmark import evaluate_slices, identity_restoration, manifest_rows
from ct_restoration.classical.clahe import ClaheConfig, clahe_restorer
from ct_restoration.config import PROJECT_ROOT
from ct_restoration.data import dataset as dataset_module
from ct_restoration.data.dataset import RestorationDataset, development_rows
from ct_restoration.data.degradation import DegradationConfig
from ct_restoration.evaluation import (
    METRIC_COLUMNS,
    EvaluationConfig,
    HeldOutSplitError,
    HoldoutAccess,
    _issue_holdout_access,
)
from ct_restoration.holdout import (
    DEGRADED_DIGEST_COLUMN,
    FROZEN,
    CheckReport,
    DicomOpenMonitor,
    GitState,
    HoldoutExecutionError,
    HoldoutProtocolError,
    execute_protocol,
    learned_name,
    load_test_plan,
    run_preflight,
    score_shared_inputs,
    validate_plan,
    verify_tracked_inputs,
)
from ct_restoration.models import diagnostics
from ct_restoration.run_layout import checkpoint_path, run_directory

REAL_PLAN = PROJECT_ROOT / "configs/holdout/test_plan.yaml"
SEEDS = (2026, 2027, 2028, 2029, 2030)

#: The miniature cohort: (subject, split, group, archive, slices). Two test
#: patients, one stress patient, and one each of train and validation.
MINI_COHORT = (
    ("2", "train", "B", "Train_Sets", 2),
    ("4", "validation", "B", "Test_Sets", 2),
    ("11", "test", "B", "Test_Sets", 3),
    ("13", "test", "A", "Test_Sets", 2),
    ("1", "stress", "C", "Train_Sets", 2),
)

MINI = replace(
    FROZEN,
    development_frozen_commit="a" * 40,
    subjects=("11", "13"),
    patients=2,
    slices=5,
    stress_subjects=("1",),
)

CLEAN_GIT = GitState(commit="b" * 40, porcelain="", frozen_commit_is_ancestor=True)

FIXED_CLOCK = "2026-01-01T00:00:00+00:00"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stored_pixels(subject_id: str, index: int) -> np.ndarray:
    """A 64x64 slice with a large body disc, distinct per slice."""
    grid = np.arange(64, dtype=np.float64)
    rows, columns = np.meshgrid(grid, grid, indexing="ij")
    body = 1024 + ((int(subject_id) * 29 + index * 13) % 150) + rows * 1.5 + columns
    body[(rows - 32.0) ** 2 + (columns - 32.0) ** 2 > 28.0**2] = 24.0
    return body.astype(np.uint16)


def _write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8", newline="\n")


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")


def build_mini_repository(root: Path) -> Path:
    """A complete miniature repository the protocol can run against."""
    records = []
    for subject, split, group, archive, count in MINI_COHORT:
        for index in range(count):
            key = f"{archive}/CT/{subject}/DICOM_anon/i{index:04d}.dcm"
            path = root / "data/raw/chaos" / key
            path.parent.mkdir(parents=True, exist_ok=True)
            build_ct_dataset(_stored_pixels(subject, index)).save_as(path, enforce_file_format=True)
            records.append(
                {
                    "subject_id": subject,
                    "split": split,
                    "source_archive": archive,
                    "acquisition_group": group,
                    "relative_dicom_path": key,
                    "geometric_slice_position": -50.0 + index,
                    "geometric_slice_index": index,
                }
            )
    splits = root / "data/splits"
    splits.mkdir(parents=True)
    pd.DataFrame(records).to_csv(splits / "chaos_slice_manifest.csv", index=False)
    pd.DataFrame(
        [{"subject_id": s, "split": sp, "acquisition_group": g} for s, sp, g, _, _ in MINI_COHORT]
    ).to_csv(splits / "chaos_patient_split.csv", index=False)

    configs = root / "configs"
    configs.mkdir()
    for name in ("degradation.yaml", "evaluation.yaml", "clahe.yaml"):
        shutil.copyfile(PROJECT_ROOT / "configs" / name, configs / name)
    _write_yaml(
        configs / "baseline.yaml",
        {
            "preprocessing": {
                "window_center": 40,
                "window_width": 400,
                "image_size": [32, 32],
                "interpolation": "area",
            }
        },
    )
    documents = {
        name: yaml.safe_load((configs / f"{file}.yaml").read_text(encoding="utf-8"))
        for name, file in (
            ("preprocessing", "baseline"),
            ("degradation", "degradation"),
            ("evaluation", "evaluation"),
            ("clahe", "clahe"),
        )
    }
    settings = {name: holdout.parse_policy(name, doc) for name, doc in documents.items()}

    learned: dict = {"cnn": {}, "unet": {}}
    runs: dict = {"cnn": {}, "unet": {}}
    for method in ("cnn", "unet"):
        for seed in SEEDS:
            config = (
                f"configs/{method}.yaml"
                if seed == 2026
                else f"configs/multiseed/{method}_seed{seed}.yaml"
            )
            _write_yaml(root / config, {"model": {"name": method}, "training": {"seed": seed}})
            checkpoint = root / checkpoint_path(method, seed)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"synthetic weights {method} {seed}".encode())
            run_dir = root / run_directory(method, seed)
            run_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "epoch": [0, 1, 2, 3],
                    holdout.SELECTION_METRIC: [0.020, 0.015, 0.012, 0.013],
                }
            ).to_csv(run_dir / "training_history.csv", index=False)
            _write_json(
                run_dir / "run_summary.json",
                {
                    "training": {"seed": seed},
                    "config": {"sha256": _sha(root / config)},
                    "checkpoint": {
                        "path": checkpoint_path(method, seed).as_posix(),
                        "sha256": _sha(checkpoint),
                        "tracked_in_git": False,
                    },
                    "checkpoint_selection": {"best_epoch": 2},
                },
            )
            learned[method][seed] = {
                "config": config,
                "config_sha256": _sha(root / config),
                "checkpoint": checkpoint_path(method, seed).as_posix(),
                "checkpoint_sha256": _sha(checkpoint),
                "run_summary": (run_directory(method, seed) / "run_summary.json").as_posix(),
                "training_history": (
                    run_directory(method, seed) / "training_history.csv"
                ).as_posix(),
                "selected_epoch": 2,
            }
            runs[method][seed] = {"config": config, "sha256": _sha(root / config)}
    _write_yaml(root / "configs/multiseed/plan.yaml", {"runs": runs})

    def validation_summary(path: str, offset: float, **extra) -> dict:
        document = {
            "split": "validation",
            "evaluation_config": settings["evaluation"],
            "degradation_config": settings["degradation"],
            "primary_result": {
                metric: {"mean": 0.1 + offset + index / 100}
                for index, metric in enumerate(FROZEN.metrics)
            },
            **extra,
        }
        _write_json(root / path, document)
        return {"summary": path, "sha256_lf": holdout.text_sha256_lf(root / path)}

    reference: dict = {
        "degraded": validation_summary(
            "outputs/metrics/degraded_baseline_validation_summary.json",
            0.0,
            preprocessing=settings["preprocessing"],
        ),
        "clahe": validation_summary(
            "outputs/metrics/clahe_validation_summary.json", 0.01, clahe_config=settings["clahe"]
        ),
        "cnn": {},
        "unet": {},
    }
    for method in ("cnn", "unet"):
        for seed in SEEDS:
            entry = learned[method][seed]
            path = (
                f"outputs/metrics/{method}_validation_summary.json"
                if seed == 2026
                else f"outputs/metrics/multiseed/seed{seed}/{method}_validation_summary.json"
            )
            reference[method][seed] = validation_summary(
                path,
                0.02,
                checkpoint={
                    "seed": seed,
                    "config_sha256": entry["config_sha256"],
                    "provenance": {
                        "recomputed_from_files": {
                            "checkpoint_file_sha256": entry["checkpoint_sha256"]
                        }
                    },
                },
            )

    plan = yaml.safe_load(REAL_PLAN.read_text(encoding="utf-8"))
    plan.update(
        {
            "development_frozen_commit": MINI.development_frozen_commit,
            "expected_subjects": list(MINI.subjects),
            "expected_patients": MINI.patients,
            "expected_slices": MINI.slices,
            "expected_slices_by_subject": {"11": 3, "13": 2},
            "stress_subjects_sealed": list(MINI.stress_subjects),
            "split_file": {
                "path": "data/splits/chaos_patient_split.csv",
                "sha256": _sha(splits / "chaos_patient_split.csv"),
            },
            "manifest_file": {
                "path": "data/splits/chaos_slice_manifest.csv",
                "sha256": _sha(splits / "chaos_slice_manifest.csv"),
            },
            "multiseed_plan": {
                "path": "configs/multiseed/plan.yaml",
                "sha256": _sha(root / "configs/multiseed/plan.yaml"),
            },
            "learned_checkpoints": learned,
            "validation_reference": reference,
        }
    )
    for name, file in (
        ("preprocessing", "baseline"),
        ("degradation", "degradation"),
        ("evaluation", "evaluation"),
        ("clahe", "clahe"),
    ):
        plan["frozen_policies"][name]["sha256_lf"] = holdout.text_sha256_lf(
            configs / f"{file}.yaml"
        )
        plan["frozen_policies"][name]["settings"] = settings[name]
    plan_path = root / "configs/holdout/test_plan.yaml"
    _write_yaml(plan_path, plan)
    return plan_path


def fake_restorer(strength: float):
    """A deterministic stand-in for a learned model: a mild pull towards 0.5."""

    def restore(degraded: np.ndarray) -> np.ndarray:
        return np.clip(degraded + strength * (0.5 - degraded), 0.0, 1.0).astype(np.float32)

    return restore


def fake_loader(plan: dict, root: Path):
    restorers, provenance = {}, {}
    for offset, method in enumerate(("cnn", "unet")):
        for index, seed in enumerate(SEEDS):
            name = learned_name(method, seed)
            restorers[name] = fake_restorer(0.01 * (1 + index) + 0.05 * offset)
            provenance[name] = {"method": method, "seed": seed, "synthetic": True}
    return restorers, provenance


@pytest.fixture
def mini(tmp_path: Path):
    plan_path = build_mini_repository(tmp_path)
    return tmp_path, plan_path


def _preflight(root: Path, plan_path: Path, git: GitState = CLEAN_GIT, loader=fake_loader):
    plan = load_test_plan(plan_path, MINI)
    return run_preflight(plan, root, git, loader, MINI)


def _failed_names(result) -> list[str]:
    return [name for name, _ in result.report.failed]


def _mutate_plan(plan_path: Path, change) -> dict:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    change(plan)
    return plan


def _poison_pixel_readers(monkeypatch) -> list[str]:
    """Make every in-process route to DICOM pixels fail loudly, and record it."""
    calls: list[str] = []

    def refuse(*args, **kwargs):
        calls.append("pixel read")
        raise AssertionError("a pixel reader was called")

    import pydicom

    from ct_restoration.data import dicom

    monkeypatch.setattr(pydicom, "dcmread", refuse)
    monkeypatch.setattr(dicom, "load_ct_hu", refuse)
    monkeypatch.setattr(holdout, "prepare_evaluation_slice", refuse)
    monkeypatch.setattr(benchmark, "prepare_evaluation_slice", refuse)
    monkeypatch.setattr(diagnostics, "prepare_evaluation_slice", refuse)
    return calls


# ==========================================================================
# The real frozen plan, checked against tracked metadata only
# ==========================================================================


def test_the_real_plan_validates_against_the_frozen_protocol():
    plan = load_test_plan(REAL_PLAN)
    assert plan["milestone"] == 11
    assert plan["stage"] == "protocol_frozen_execution_pending"
    assert plan["development_frozen_commit"].startswith("f1fc6dd")


def test_the_real_plan_names_exactly_the_six_test_subjects():
    plan = load_test_plan(REAL_PLAN)
    assert plan["expected_subjects"] == ["11", "13", "19", "25", "29", "32"]
    assert plan["expected_patients"] == 6
    assert plan["expected_slices"] == 941
    assert sum(plan["expected_slices_by_subject"].values()) == 941
    assert plan["stress_subjects_sealed"] == ["1", "6", "39"]
    assert plan["stress_allowed"] is False


def test_the_real_plan_enters_exactly_five_seeds_per_architecture():
    plan = load_test_plan(REAL_PLAN)
    assert plan["methods"] == ["degraded", "clahe", "cnn", "unet"]
    assert plan["statistical_seeds"] == list(SEEDS)
    for method in ("cnn", "unet"):
        assert sorted(plan["learned_checkpoints"][method]) == list(SEEDS)
    digests = [
        plan["learned_checkpoints"][method][seed]["checkpoint_sha256"]
        for method in ("cnn", "unet")
        for seed in SEEDS
    ]
    assert len(set(digests)) == 10
    assert plan["frozen_policies"]["clahe"]["settings"]["clip_limit"] == 0.5
    assert plan["frozen_policies"]["clahe"]["settings"]["tile_grid_size"] == [4, 4]


def test_the_real_plan_agrees_with_every_tracked_record_without_reading_an_image(monkeypatch):
    # Split and manifest hashes, the cohort from metadata, every shared
    # config, the Milestone 10 plan, every run summary, every training
    # history and every validation summary. Tracked files only, so this runs
    # in a fresh clone with no checkpoint and no DICOM - and it proves it
    # reads no image: the pixel readers are poisoned and the imaging root is
    # sealed for the duration.
    calls = _poison_pixel_readers(monkeypatch)
    plan = load_test_plan(REAL_PLAN)
    report = CheckReport()
    with DicomOpenMonitor(PROJECT_ROOT / plan["data_root"]) as sealed:
        verify_tracked_inputs(report, plan, PROJECT_ROOT)
    assert report.failed == []
    assert len(report.checks) == 4 + 1 + 4 + 12 + 10
    assert calls == []
    assert sum(sealed.opened.values()) == sum(sealed.refused.values()) == 0


def test_no_held_out_result_exists_unless_it_is_complete():
    # Valid before and after Milestone 11B: until the protocol runs, nothing
    # exists; after it runs, the directory exists only with its receipt, and
    # an incomplete attempt is never left in the final location.
    final = PROJECT_ROOT / holdout.OUTPUT_ROOT
    staging = PROJECT_ROOT / holdout.STAGING_ROOT
    assert not staging.exists(), "an interrupted attempt needs an integrity review"
    if final.exists():
        receipt = json.loads((final / "execution_receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "complete"
        assert receipt["image_reads"]["stress_images_read"] == 0
    stress = PROJECT_ROOT / "outputs/metrics/holdout/stress"
    assert not stress.exists()


# ==========================================================================
# The plan: every change to the protocol is refused
# ==========================================================================


def test_a_plan_that_permits_stress_is_refused(mini):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p.update(stress_allowed=True))
    with pytest.raises(HoldoutProtocolError, match="stress_allowed"):
        validate_plan(plan, MINI)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_a_prohibition_must_be_the_boolean_false(mini, value):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p.update(best_seed=value))
    with pytest.raises(HoldoutProtocolError, match="best_seed"):
        validate_plan(plan, MINI)


@pytest.mark.parametrize(
    "methods",
    [
        ["degraded", "cnn", "unet"],
        ["degraded", "clahe", "cnn", "unet", "ensemble"],
        ["clahe", "degraded", "cnn", "unet"],
    ],
)
def test_a_plan_that_changes_the_method_set_is_refused(mini, methods):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p.update(methods=methods))
    with pytest.raises(HoldoutProtocolError, match="methods must be exactly"):
        validate_plan(plan, MINI)


@pytest.mark.parametrize(
    "seeds", [[2026, 2027, 2028, 2029], [2026, 2027, 2028, 2029, 2030, 2031], [2027] * 5]
)
def test_a_plan_that_changes_the_seed_set_is_refused(mini, seeds):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p.update(statistical_seeds=seeds))
    with pytest.raises(HoldoutProtocolError, match="statistical_seeds"):
        validate_plan(plan, MINI)


def test_a_missing_learned_seed_is_refused(mini):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p["learned_checkpoints"]["unet"].pop(2030))
    with pytest.raises(HoldoutProtocolError, match="no missing seed, no extra seed"):
        validate_plan(plan, MINI)


def test_an_extra_learned_seed_is_refused(mini):
    _, plan_path = mini

    def add(plan):
        plan["learned_checkpoints"]["cnn"][2031] = dict(plan["learned_checkpoints"]["cnn"][2030])

    plan = _mutate_plan(plan_path, add)
    with pytest.raises(HoldoutProtocolError, match="no missing seed, no extra seed"):
        validate_plan(plan, MINI)


def test_a_plan_with_a_changed_clahe_parameter_is_refused(mini):
    _, plan_path = mini

    def retune(plan):
        plan["frozen_policies"]["clahe"]["settings"]["clip_limit"] = 1.0

    with pytest.raises(HoldoutProtocolError, match="CLAHE is not retuned"):
        validate_plan(_mutate_plan(plan_path, retune), MINI)


def test_a_plan_with_a_visual_output_is_refused(mini):
    _, plan_path = mini

    def add_figure(plan):
        plan["outputs"]["files"] = [*plan["outputs"]["files"], "test_panels.png"]

    with pytest.raises(HoldoutProtocolError, match="output layout"):
        validate_plan(_mutate_plan(plan_path, add_figure), MINI)

    def add_image(plan):
        plan["outputs"]["image_outputs"] = ["test_panels.png"]

    with pytest.raises(HoldoutProtocolError, match="image_outputs"):
        validate_plan(_mutate_plan(plan_path, add_image), MINI)


def test_an_unrecognised_plan_key_is_refused(mini):
    _, plan_path = mini
    plan = _mutate_plan(plan_path, lambda p: p.update(checkpoint_override="x"))
    with pytest.raises(HoldoutProtocolError, match="unrecognised"):
        validate_plan(plan, MINI)


def test_the_receipt_schema_cannot_be_trimmed(mini):
    _, plan_path = mini

    def trim(plan):
        plan["execution_receipt"]["required_fields"].remove("image_reads")

    with pytest.raises(HoldoutProtocolError, match="receipt schema"):
        validate_plan(_mutate_plan(plan_path, trim), MINI)


# ==========================================================================
# Preflight: every frozen input is proved before a test image is read
# ==========================================================================


def test_preflight_passes_on_an_intact_repository_and_only_then_grants_access(mini):
    root, plan_path = mini
    result = _preflight(root, plan_path)
    assert result.report.failed == []
    assert isinstance(result.access, HoldoutAccess)
    assert sorted(result.restorers) == sorted(
        learned_name(method, seed) for method in ("cnn", "unet") for seed in SEEDS
    )


def test_a_missing_checkpoint_is_refused(mini):
    root, plan_path = mini
    (root / checkpoint_path("cnn", 2028)).unlink()
    result = _preflight(root, plan_path)
    assert "checkpoint bytes cnn 2028" in _failed_names(result)
    assert result.access is None and result.restorers is None


def test_a_checkpoint_sha_mismatch_is_refused(mini):
    root, plan_path = mini
    (root / checkpoint_path("unet", 2029)).write_bytes(b"retrained replacement")
    result = _preflight(root, plan_path)
    assert "checkpoint bytes unet 2029" in _failed_names(result)
    assert result.access is None


def test_a_config_sha_mismatch_is_refused(mini):
    root, plan_path = mini
    config = root / "configs/multiseed/cnn_seed2027.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    result = _preflight(root, plan_path)
    assert "tracked evidence cnn 2027" in _failed_names(result)
    assert result.access is None


def test_a_dirty_working_tree_is_refused(mini):
    root, plan_path = mini
    dirty = replace(CLEAN_GIT, porcelain=" M src/ct_restoration/metrics.py\n?? notes.txt\n")
    result = _preflight(root, plan_path, git=dirty)
    assert _failed_names(result) == ["git working tree clean"]
    assert result.access is None


def test_a_head_that_does_not_descend_from_the_frozen_commit_is_refused(mini):
    root, plan_path = mini
    result = _preflight(root, plan_path, git=replace(CLEAN_GIT, frozen_commit_is_ancestor=False))
    assert _failed_names(result) == ["development frozen commit is an ancestor of HEAD"]
    assert result.access is None


def test_a_split_sha_mismatch_is_refused(mini):
    root, plan_path = mini
    split = root / "data/splits/chaos_patient_split.csv"
    split.write_bytes(split.read_bytes().replace(b"\n", b"\r\n"))
    result = _preflight(root, plan_path)
    assert "split file SHA-256" in _failed_names(result)
    assert result.access is None


def test_a_manifest_sha_mismatch_is_refused(mini):
    root, plan_path = mini
    manifest = root / "data/splits/chaos_slice_manifest.csv"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    result = _preflight(root, plan_path)
    assert "manifest file SHA-256" in _failed_names(result)
    assert result.access is None


@pytest.mark.parametrize("location", [holdout.OUTPUT_ROOT, holdout.STAGING_ROOT])
def test_an_existing_output_destination_is_refused(mini, location):
    root, plan_path = mini
    (root / location).mkdir(parents=True)
    result = _preflight(root, plan_path)
    assert len(result.report.failed) == 1
    assert result.report.failed[0][0].startswith("output ")
    assert result.access is None


def test_a_changed_shared_config_is_refused(mini):
    root, plan_path = mini
    config = root / "configs/degradation.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("global_seed: 2026", "global_seed: 2027"),
        encoding="utf-8",
    )
    result = _preflight(root, plan_path)
    assert "frozen degradation config" in _failed_names(result)


def test_a_run_summary_disagreeing_about_the_selected_epoch_is_refused(mini):
    root, plan_path = mini
    summary_path = root / run_directory("unet", 2026) / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"]["best_epoch"] = 3
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = _preflight(root, plan_path)
    assert "tracked evidence unet 2026" in _failed_names(result)


def test_a_model_loader_failure_withholds_access(mini):
    root, plan_path = mini

    def broken(plan, root):
        raise RuntimeError("provenance gate refused")

    result = _preflight(root, plan_path, loader=broken)
    assert _failed_names(result) == ["learned checkpoints loaded and provenance verified"]
    assert result.access is None


def test_preflight_reads_no_image(mini, monkeypatch):
    root, plan_path = mini
    calls = _poison_pixel_readers(monkeypatch)
    with DicomOpenMonitor(root / "data/raw/chaos") as sealed:
        result = _preflight(root, plan_path)
    assert result.report.failed == []
    assert calls == []
    assert sum(sealed.opened.values()) == sum(sealed.refused.values()) == 0


# ==========================================================================
# The access guard: ordinary evaluators refuse test and stress
# ==========================================================================


def _mini_rows(root: Path, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data/splits/chaos_slice_manifest.csv", dtype={"subject_id": str})
    return manifest[manifest["split"] == split].reset_index(drop=True)


def _policies(root: Path):
    configs = root / "configs"
    preprocessing = yaml.safe_load((configs / "baseline.yaml").read_text())["preprocessing"]
    evaluation = EvaluationConfig.from_mapping(
        yaml.safe_load((configs / "evaluation.yaml").read_text())
    )
    degradation = DegradationConfig.from_mapping(
        yaml.safe_load((configs / "degradation.yaml").read_text())
    )
    return preprocessing, evaluation, degradation


@pytest.mark.parametrize("split", ["test", "stress"])
def test_ordinary_evaluators_refuse_sealed_splits(mini, monkeypatch, split):
    root, _ = mini
    manifest = root / "data/splits/chaos_slice_manifest.csv"
    calls = _poison_pixel_readers(monkeypatch)
    preprocessing, evaluation, degradation = _policies(root)
    rows = _mini_rows(root, split)

    with pytest.raises(HeldOutSplitError):
        manifest_rows(manifest, split)
    with pytest.raises(HeldOutSplitError):
        development_rows(manifest, split)
    # The same rows handed straight to a reader, bypassing every name gate.
    with pytest.raises(HeldOutSplitError):
        evaluate_slices(rows, root / "data/raw/chaos", preprocessing, evaluation, degradation)
    with pytest.raises(HeldOutSplitError):
        RestorationDataset(rows, root / "data/raw/chaos", preprocessing, degradation)
    with pytest.raises(HeldOutSplitError):
        diagnostics.raw_output_diagnostics(
            rows, root / "data/raw/chaos", preprocessing, evaluation, degradation, None, "cpu"
        )
    with pytest.raises(HeldOutSplitError):
        score_shared_inputs(
            rows,
            root / "data/raw/chaos",
            preprocessing,
            evaluation,
            degradation,
            {"degraded": identity_restoration},
        )
    assert calls == [], "a pixel reader was reached before the refusal"


def test_rows_without_a_split_label_are_refused_not_assumed_to_be_development(mini, monkeypatch):
    root, _ = mini
    calls = _poison_pixel_readers(monkeypatch)
    preprocessing, evaluation, degradation = _policies(root)
    rows = _mini_rows(root, "train").drop(columns="split")
    with pytest.raises(HeldOutSplitError, match="no 'split' column"):
        evaluate_slices(rows, root / "data/raw/chaos", preprocessing, evaluation, degradation)
    with pytest.raises(HeldOutSplitError, match="no 'split' column"):
        RestorationDataset(rows, root / "data/raw/chaos", preprocessing, degradation)
    assert calls == []


def test_development_readers_still_accept_development_rows(mini):
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    dataset = dataset_module.development_dataset(
        "train",
        root / "data/splits/chaos_slice_manifest.csv",
        root / "data/raw/chaos",
        preprocessing,
    )
    assert len(dataset) == 2
    frame = evaluate_slices(
        _mini_rows(root, "validation"),
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        progress_every=0,
    )
    assert len(frame) == 2


def test_a_holdout_access_cannot_be_constructed_directly():
    with pytest.raises(HeldOutSplitError, match="cannot be constructed directly"):
        HoldoutAccess("0" * 64, "b" * 40)
    with pytest.raises(HeldOutSplitError):
        HoldoutAccess("0" * 64, "b" * 40, issuer=object())


def test_the_holdout_protocol_is_structurally_permitted_to_read_test(mini):
    # Synthetic test rows only. This is the branch Milestone 11B uses on the
    # real split; in Milestone 11A it is exercised here and nowhere else.
    root, _ = mini
    manifest = root / "data/splits/chaos_slice_manifest.csv"
    access = _issue_holdout_access("0" * 64, "b" * 40)
    rows = manifest_rows(manifest, "test", access=access)
    assert sorted(set(rows["subject_id"])) == ["11", "13"]
    assert len(rows) == 5

    preprocessing, evaluation, degradation = _policies(root)
    split_by = holdout.split_by_key(pd.read_csv(manifest, dtype={"subject_id": str}))
    with DicomOpenMonitor(root / "data/raw/chaos", split_by, readable={"test"}) as monitor:
        frames = score_shared_inputs(
            rows,
            root / "data/raw/chaos",
            preprocessing,
            evaluation,
            degradation,
            {"degraded": identity_restoration},
            access,
        )
    assert len(frames["degraded"]) == 5
    assert monitor.distinct("test") == 5
    assert monitor.opened["stress"] == 0


def test_the_holdout_protocol_is_refused_stress_and_development_splits(mini, monkeypatch):
    root, _ = mini
    manifest = root / "data/splits/chaos_slice_manifest.csv"
    access = _issue_holdout_access("0" * 64, "b" * 40)
    calls = _poison_pixel_readers(monkeypatch)
    for split in ("stress", "validation", "train"):
        with pytest.raises(HeldOutSplitError):
            manifest_rows(manifest, split, access=access)
    preprocessing, evaluation, degradation = _policies(root)
    mixed = pd.concat([_mini_rows(root, "test"), _mini_rows(root, "stress")])
    with pytest.raises(HeldOutSplitError, match="stress"):
        score_shared_inputs(
            mixed,
            root / "data/raw/chaos",
            preprocessing,
            evaluation,
            degradation,
            {"degraded": identity_restoration},
            access,
        )
    assert calls == []


def test_the_monitor_refuses_a_stress_file_at_the_file_system(mini):
    # Even a caller that skipped every gate cannot open a stress file while
    # the monitor is active: the open itself raises.
    root, _ = mini
    manifest = pd.read_csv(root / "data/splits/chaos_slice_manifest.csv", dtype={"subject_id": str})
    stress_key = manifest.loc[manifest["split"] == "stress", "relative_dicom_path"].iloc[0]
    test_key = manifest.loc[manifest["split"] == "test", "relative_dicom_path"].iloc[0]
    data_root = root / "data/raw/chaos"
    with DicomOpenMonitor(data_root, holdout.split_by_key(manifest), readable={"test"}) as monitor:
        (data_root / test_key).read_bytes()
        with pytest.raises(HeldOutSplitError, match="stress"):
            (data_root / stress_key).read_bytes()
        with pytest.raises(HeldOutSplitError, match="outside_manifest"):
            (data_root / "Train_Sets/CT/1/unlisted.dcm").open("rb")
    assert monitor.opened["test"] == 1
    assert monitor.refused["stress"] == 1
    # Outside the monitor, nothing is intercepted.
    (data_root / stress_key).stat()


def test_a_sealed_monitor_refuses_every_file_under_the_root(mini):
    root, _ = mini
    data_root = root / "data/raw/chaos"
    any_file = next(data_root.rglob("*.dcm"))
    with DicomOpenMonitor(data_root) as sealed:
        with pytest.raises(HeldOutSplitError):
            any_file.read_bytes()
        (root / "configs/clahe.yaml").read_bytes()  # outside the root: untouched
    assert sealed.refused["outside_manifest"] == 1


# ==========================================================================
# Scoring: the Milestone 5-10 numbers, from one shared input
# ==========================================================================


def test_shared_input_scoring_reproduces_the_development_harness_exactly(mini):
    # One method through the new loop must be the old loop, value for value.
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    rows = manifest_rows(root / "data/splits/chaos_slice_manifest.csv", "validation")
    clahe = clahe_restorer(
        ClaheConfig.from_mapping(yaml.safe_load((root / "configs/clahe.yaml").read_text()))
    )
    frames = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"degraded": identity_restoration, "clahe": clahe},
    )
    for name, restore in (("degraded", identity_restoration), ("clahe", clahe)):
        expected = evaluate_slices(
            rows, root / "data/raw/chaos", preprocessing, evaluation, degradation, restore, 0
        )
        pd.testing.assert_frame_equal(frames[name].drop(columns=DEGRADED_DIGEST_COLUMN), expected)


def test_every_method_sees_one_degraded_realization_per_slice(mini):
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    rows = manifest_rows(root / "data/splits/chaos_slice_manifest.csv", "validation")
    frames = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"a": identity_restoration, "b": fake_restorer(0.1), "c": fake_restorer(0.2)},
    )
    digests = {tuple(frame[DEGRADED_DIGEST_COLUMN]) for frame in frames.values()}
    assert len(digests) == 1
    holdout.require_shared_samples(frames)


def test_a_method_that_modifies_its_input_in_place_cannot_affect_the_others(mini):
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    rows = manifest_rows(root / "data/splits/chaos_slice_manifest.csv", "validation")

    def vandal(degraded: np.ndarray) -> np.ndarray:
        degraded[...] = 0.0  # writes into the array it was given
        return degraded

    alone = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"x": identity_restoration},
    )
    after = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"vandal": vandal, "x": identity_restoration},
    )
    pd.testing.assert_frame_equal(alone["x"], after["x"])


def test_same_sample_with_a_mismatched_degradation_is_refused(mini):
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    rows = manifest_rows(root / "data/splits/chaos_slice_manifest.csv", "validation")
    frames = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"cnn_seed2026": identity_restoration, "unet_seed2026": fake_restorer(0.1)},
    )
    frames["unet_seed2026"].loc[1, DEGRADED_DIGEST_COLUMN] = "0" * 64
    with pytest.raises(HoldoutExecutionError, match="different degraded inputs"):
        holdout.require_shared_samples(frames)


def test_a_unet_cnn_sample_key_mismatch_is_refused(mini):
    root, _ = mini
    preprocessing, evaluation, degradation = _policies(root)
    rows = manifest_rows(root / "data/splits/chaos_slice_manifest.csv", "validation")
    frames = score_shared_inputs(
        rows,
        root / "data/raw/chaos",
        preprocessing,
        evaluation,
        degradation,
        {"cnn_seed2026": identity_restoration, "unet_seed2026": fake_restorer(0.1)},
    )
    frames["unet_seed2026"] = frames["unet_seed2026"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(HoldoutExecutionError, match="same sample keys"):
        holdout.require_shared_samples(frames)


def _slice_table(values_by_subject: dict[str, list[float]]) -> pd.DataFrame:
    """A scored-looking slice table with every metric set to the given values."""
    records = []
    for subject, values in values_by_subject.items():
        for index, value in enumerate(values):
            records.append(
                {
                    "subject_id": subject,
                    "source_archive": "Test_Sets",
                    "acquisition_group": "A",
                    "relative_dicom_path": f"T/{subject}/{index}.dcm",
                    "geometric_slice_index": index,
                    **{metric: value for metric in METRIC_COLUMNS},
                    "body_pixel_fraction": 0.5,
                    "body_ssim_interior_fraction": 0.3,
                    DEGRADED_DIGEST_COLUMN: "0" * 64,
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_non_finite_metric_is_refused(value):
    frame = _slice_table({"11": [0.1, 0.2], "13": [0.3]})
    frame.loc[1, "full_psnr"] = value
    with pytest.raises(HoldoutExecutionError, match="non-finite"):
        holdout.require_finite_metrics(frame, "cnn_seed2027")


def test_a_wrong_patient_count_is_refused():
    plan = {
        "expected_slices_by_subject": {"11": 2, "13": 1},
        "expected_slices": 3,
        "expected_patients": 2,
    }
    frame = _slice_table({"11": [0.1, 0.2]})
    with pytest.raises(HoldoutExecutionError, match="slices by subject"):
        holdout.require_cohort(frame, plan, "degraded")


def test_a_wrong_slice_count_is_refused():
    plan = {
        "expected_slices_by_subject": {"11": 2, "13": 1},
        "expected_slices": 3,
        "expected_patients": 2,
    }
    frame = _slice_table({"11": [0.1], "13": [0.3]})
    with pytest.raises(HoldoutExecutionError, match="slices by subject"):
        holdout.require_cohort(frame, plan, "degraded")
    holdout.require_cohort(_slice_table({"11": [0.1, 0.2], "13": [0.3]}), plan, "degraded")


# ==========================================================================
# The predeclared analysis
# ==========================================================================


def _analysis(cnn: list[float], unet: list[float], degraded: float = 0.5, clahe: float = 0.45):
    """Run the real analysis over scored tables whose every metric is chosen.

    Every metric takes the same per-seed value, so a comparison's direction
    depends on the metric's family: a larger value is better for PSNR and
    SSIM and worse for MAE and MSE.
    """
    scored = {
        "degraded": holdout.summarise_method(_slice_table({"11": [degraded], "13": [degraded]})),
        "clahe": holdout.summarise_method(_slice_table({"11": [clahe], "13": [clahe]})),
    }
    for method, values in (("cnn", cnn), ("unet", unet)):
        for seed, value in zip(SEEDS, values, strict=True):
            scored[learned_name(method, seed)] = holdout.summarise_method(
                _slice_table({"11": [value], "13": [value]})
            )
    validation = {
        "degraded": dict.fromkeys(FROZEN.metrics, 0.5),
        "clahe": dict.fromkeys(FROZEN.metrics, 0.45),
        "cnn": {seed: dict.fromkeys(FROZEN.metrics, 0.6) for seed in SEEDS},
        "unet": {seed: dict.fromkeys(FROZEN.metrics, 0.6) for seed in SEEDS},
    }
    return holdout.analyse(scored, {}, validation, FROZEN)


def test_seed_spread_is_the_sample_standard_deviation_ddof_1():
    cnn = [0.60, 0.61, 0.62, 0.64, 0.66]
    summary = _analysis(cnn, [0.7] * 5)["summary"]
    block = summary["learned_architectures"]["cnn"]["full_psnr"]
    assert block["std_ddof"] == 1
    assert block["std"] == pytest.approx(statistics.stdev(cnn))
    assert block["std"] != pytest.approx(statistics.pstdev(cnn))
    assert block["values"] == cnn
    assert block["min"] == 0.60 and block["max"] == 0.66


def test_lower_is_better_metrics_are_oriented_so_positive_means_better():
    # Learned value 0.6 above degraded 0.5: better for PSNR/SSIM, worse for MAE/MSE.
    summary = _analysis([0.6] * 5, [0.6] * 5)["summary"]
    versus = summary["learned_vs_degraded"]["cnn"]
    assert versus["full_psnr"]["oriented_improvement"]["mean"] == pytest.approx(0.1)
    assert versus["full_psnr"]["seeds_improved"] == 5
    assert versus["full_mae"]["oriented_improvement"]["mean"] == pytest.approx(-0.1)
    assert versus["full_mae"]["seeds_worsened"] == 5
    assert versus["full_mae"]["raw_delta"]["mean"] == pytest.approx(0.1)
    assert holdout.versus_reference("body_mse", 0.002, 0.001) == pytest.approx((-0.001, 0.001))
    assert holdout.versus_reference("body_ssim", 0.8, 0.9) == pytest.approx((0.1, 0.1))


def test_five_of_five_seeds_favouring_the_unet():
    per_metric = _analysis([0.60] * 5, [0.61, 0.62, 0.63, 0.64, 0.65])["summary"]["unet_vs_cnn"]
    block = per_metric["per_metric"]["full_psnr"]
    assert block["seeds_favouring_unet"] == 5
    assert block["permitted_phrase"] == "all 5 seeds favoured the U-Net"
    # Every metric has the same values, so lower-is-better ones favour the CNN.
    assert (
        per_metric["per_metric"]["full_mae"]["permitted_phrase"] == "all 5 seeds favoured the CNN"
    )
    assert per_metric["overall"]["architecture"] is None


def test_four_of_five_seeds_with_the_mean_is_directionally_consistent():
    block = _analysis([0.60] * 5, [0.62, 0.62, 0.62, 0.62, 0.59])["summary"]["unet_vs_cnn"][
        "per_metric"
    ]["full_ssim"]
    assert (block["seeds_favouring_unet"], block["seeds_favouring_cnn"]) == (4, 1)
    assert block["directionally_consistent_for_unet"] is True
    assert (
        block["permitted_phrase"] == "the U-Net was directionally consistent across training seeds"
    )


def test_four_of_five_against_the_mean_is_not_consistent():
    block = _analysis([0.60] * 5, [0.61, 0.61, 0.61, 0.61, 0.40])["summary"]["unet_vs_cnn"][
        "per_metric"
    ]["full_psnr"]
    assert block["seeds_favouring_unet"] == 4
    assert block["oriented_improvement"]["mean"] < 0
    assert block["directionally_consistent_for_unet"] is False
    assert block["permitted_phrase"] is None


def test_three_of_five_is_never_called_consistent():
    block = _analysis([0.60] * 5, [0.65, 0.65, 0.65, 0.59, 0.59])["summary"]["unet_vs_cnn"][
        "per_metric"
    ]["body_psnr"]
    assert (block["seeds_favouring_unet"], block["seeds_favouring_cnn"]) == (3, 2)
    assert block["oriented_improvement"]["mean"] > 0
    assert block["directionally_consistent_for_unet"] is False
    assert block["permitted_phrase"] is None


def test_deterministic_methods_are_not_given_a_seed_spread():
    analysis = _analysis([0.6] * 5, [0.7] * 5)
    deterministic = analysis["summary"]["deterministic_methods"]
    for method in ("degraded", "clahe"):
        assert set(deterministic[method]) == set(FROZEN.metrics)
        assert all(isinstance(value, float) for value in deterministic[method].values())
    assert "not applicable" in deterministic["training_seed_variability"]
    table = analysis["tables"]["deterministic_methods.csv"]
    assert not [column for column in table.columns if "std" in column or "seed" in column]
    assert len(table) == 8


# ==========================================================================
# One complete execution, one failed one
# ==========================================================================


def _execute(root: Path, plan_path: Path, loader=fake_loader, git: GitState = CLEAN_GIT):
    plan = load_test_plan(plan_path, MINI)
    return execute_protocol(
        plan,
        root,
        git,
        loader,
        {"python": "synthetic"},
        MINI,
        clock=lambda: FIXED_CLOCK,
        echo=lambda _: None,
    )


def test_a_complete_execution_writes_exactly_the_frozen_layout_and_a_receipt(mini):
    root, plan_path = mini
    receipt = _execute(root, plan_path)

    final = root / holdout.OUTPUT_ROOT
    assert not (root / holdout.STAGING_ROOT).exists()
    present = {path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()}
    expected = {*holdout.OUTPUT_FILES} | {
        f"seed{seed}/{name}" for seed in SEEDS for name in holdout.PER_SEED_FILES
    }
    assert present == expected
    assert not [path for path in present if Path(path).suffix not in {".csv", ".json", ".txt"}]

    assert receipt["status"] == "complete"
    assert [field for field in holdout.RECEIPT_FIELDS if field not in receipt] == []
    reads = receipt["image_reads"]
    assert reads["test_images_read"] == 5
    assert reads["stress_images_read"] == 0
    assert reads["file_opens_by_split"]["stress"] == reads["refused_opens_by_split"]["stress"] == 0
    assert reads["file_opens_by_split"]["validation"] == reads["file_opens_by_split"]["train"] == 0
    assert reads["preflight_file_opens_under_imaging_root"] == 0
    assert receipt["visual_outputs_created"] is False
    assert receipt["overwrite_occurred"] is False and receipt["retry_occurred"] is False
    assert receipt["actual_slices"] == 5 and receipt["actual_patients"] == 2
    assert len(receipt["learned_checkpoints"]) == 10

    text = (final / "execution_receipt.json").read_text(encoding="utf-8")
    assert not [metric for metric in FROZEN.metrics if metric in text], "receipt holds a metric"

    seed_level = pd.read_csv(final / "seed_level_metrics.csv")
    assert len(seed_level) == 10
    assert len(pd.read_csv(final / "paired_seed_deltas.csv")) == 40
    assert len(pd.read_csv(final / "learned_vs_degraded.csv")) == 80
    assert len(pd.read_csv(final / "learned_vs_clahe.csv")) == 80
    assert len(pd.read_csv(final / "validation_to_test.csv")) == 32

    # Twelve tables, one degraded realization.
    tables = [final / "degraded_test_slices.csv", final / "clahe_test_slices.csv"] + [
        final / f"seed{seed}/{method}_test_slices.csv"
        for seed in SEEDS
        for method in ("cnn", "unet")
    ]
    digests = {tuple(pd.read_csv(path)[DEGRADED_DIGEST_COLUMN]) for path in tables}
    assert len(digests) == 1

    summary = json.loads((final / "holdout_test_summary.json").read_text(encoding="utf-8"))
    assert summary["significance_testing"] is False and summary["best_seed"] is False
    assert summary["inspection_policy"]["stress_files_opened"] == 0


def test_a_second_execution_is_refused_without_reading_a_file(mini, monkeypatch):
    root, plan_path = mini
    _execute(root, plan_path)
    calls = _poison_pixel_readers(monkeypatch)
    with pytest.raises(HoldoutProtocolError, match="output root absent"):
        _execute(root, plan_path)
    assert calls == []


def test_a_failed_execution_is_preserved_and_blocks_any_retry(mini):
    root, plan_path = mini

    def failing_loader(plan, root):
        restorers, provenance = fake_loader(plan, root)
        calls = {"n": 0}

        def breaks_on_the_third_slice(degraded):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated CUDA fault")
            return degraded

        restorers[learned_name("unet", 2029)] = breaks_on_the_third_slice
        return restorers, provenance

    with pytest.raises(HoldoutExecutionError, match="simulated CUDA fault"):
        _execute(root, plan_path, loader=failing_loader)

    staging = root / holdout.STAGING_ROOT
    assert not (root / holdout.OUTPUT_ROOT).exists()
    assert (staging / "opening_record.json").is_file()
    assert (staging / "execution_log.txt").is_file()
    failure = json.loads((staging / "failure_record.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["stage"] == "scoring the test split"
    assert not (staging / "execution_receipt.json").exists()
    assert not [path for path in staging.rglob("*.csv")], "no partial table was written"

    # No automatic retry: the preserved attempt blocks the next one.
    with pytest.raises(HoldoutProtocolError, match="output staging absent"):
        _execute(root, plan_path)


def test_execution_refuses_on_a_dirty_tree_before_creating_anything(mini):
    root, plan_path = mini
    with pytest.raises(HoldoutProtocolError, match="git working tree clean"):
        _execute(root, plan_path, git=replace(CLEAN_GIT, porcelain="?? scratch.py\n"))
    assert not (root / holdout.STAGING_ROOT).exists()
    assert not (root / holdout.OUTPUT_ROOT).exists()


# ==========================================================================
# The runner exposes nothing scientific
# ==========================================================================


def _runner():
    path = PROJECT_ROOT / "scripts/run_holdout_test.py"
    spec = importlib.util.spec_from_file_location("run_holdout_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_holdout_test"] = module
    spec.loader.exec_module(module)
    return module


def test_the_runner_has_no_scientific_option():
    parser = _runner().build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--preflight-only"}


def test_importing_the_runner_reads_no_image(monkeypatch):
    calls = _poison_pixel_readers(monkeypatch)
    with DicomOpenMonitor(PROJECT_ROOT / "data/raw/chaos") as sealed:
        _runner()
    assert calls == []
    assert sum(sealed.opened.values()) == sum(sealed.refused.values()) == 0


def test_the_runner_refuses_to_run_outside_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _runner().main([]) == 5
    assert not (tmp_path / "outputs").exists()
