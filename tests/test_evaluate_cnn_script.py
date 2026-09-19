"""Tests for the CNN evaluation command's integrity gates.

Fully synthetic and dataset-free: no CHAOS file is opened, no network access,
and the "model" is a tiny CNN on small arrays. What is being protected here
is not a number but the conditions under which a number is allowed to be
written.

Three gates, and each exists because of a way the canonical artifacts could
be wrong while looking entirely well-formed on disk:

* the **raw correction diagnostic** must describe the correction the network
  predicted, not the part of it that survived the clamp;
* **sample alignment** must be enforced, and enforced *before* anything is
  written, or a paired comparison can subtract one slice's metric from
  another slice's metric;
* **checkpoint provenance** must be verified against files other than the
  checkpoint, or the command will happily score whatever it is handed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ct_restoration.models.cnn import ResidualCnnConfig, build_model
from ct_restoration.training import SELECTION_METRIC

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_cnn = _load("evaluate_cnn")


# --------------------------------------------------------------------------
# 1. the raw predicted correction is not the post-clamp change
# --------------------------------------------------------------------------


class _ConstantCorrection(torch.nn.Module):
    """A stand-in model that always predicts the same additive correction.

    Not a ResidualCNN: the diagnostic only needs the raw/clamped contract,
    and a fixed correction makes the arithmetic checkable by hand.
    """

    def __init__(self, correction: float) -> None:
        super().__init__()
        self.correction_value = float(correction)
        # The adapter validates every batch against the model's config, so
        # the stub carries a real one and gets the real input checks.
        self.config = ResidualCnnConfig()

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        return degraded + self.correction_value

    def restore(self, degraded: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.forward(degraded), 0.0, 1.0)

    def eval(self):  # noqa: D102 - mirrors nn.Module for the adapter
        return self


def _diagnostics_over(images: list[np.ndarray], correction: float) -> dict:
    """Run the diagnostic over in-memory images, bypassing DICOM entirely."""
    model = _ConstantCorrection(correction)
    rows = pd.DataFrame({"relative_dicom_path": [f"s{i}.dcm" for i in range(len(images))]})
    by_key = dict(zip(rows["relative_dicom_path"], images, strict=True))

    def fake_prepare(path, preprocessing, evaluation):
        return by_key[Path(path).name], None, None

    def fake_degrade(clean, key, config):
        return clean

    original_prepare = evaluate_cnn.prepare_evaluation_slice
    original_degrade = evaluate_cnn.degrade_low_dose_like
    evaluate_cnn.prepare_evaluation_slice = fake_prepare
    evaluate_cnn.degrade_low_dose_like = fake_degrade
    try:
        return evaluate_cnn.raw_output_diagnostics(
            rows, Path("."), {}, None, None, model, torch.device("cpu")
        )
    finally:
        evaluate_cnn.prepare_evaluation_slice = original_prepare
        evaluate_cnn.degrade_low_dose_like = original_degrade


def test_the_predicted_correction_is_measured_from_the_raw_output():
    # Degraded is all zeros and the model subtracts 0.25, so the raw output
    # is -0.25 everywhere and the clamp returns every pixel to 0.
    diagnostics = _diagnostics_over([np.zeros((4, 4), dtype=np.float32)], -0.25)

    # What the network asked for: 0.25 everywhere.
    assert diagnostics["mean_absolute_predicted_correction"] == pytest.approx(0.25)
    # What survived the clamp: nothing at all.
    assert diagnostics["mean_absolute_post_clamp_change_vs_degraded"] == pytest.approx(0.0)


def test_the_two_correction_quantities_are_not_the_same_number():
    diagnostics = _diagnostics_over([np.zeros((4, 4), dtype=np.float32)], -0.25)
    assert (
        diagnostics["mean_absolute_predicted_correction"]
        != diagnostics["mean_absolute_post_clamp_change_vs_degraded"]
    )


def test_the_predicted_correction_is_the_larger_one_when_the_raw_output_is_clipped():
    # Whenever the clamp engages it moves the output back towards the
    # degraded image, so the surviving change can never exceed what was asked
    # for. A diagnostic that reported the smaller number under the larger
    # one's name would understate how hard the network is pushing.
    diagnostics = _diagnostics_over([np.zeros((8, 8), dtype=np.float32)], -0.1)
    assert (
        diagnostics["mean_absolute_predicted_correction"]
        > diagnostics["mean_absolute_post_clamp_change_vs_degraded"]
    )


def test_the_two_quantities_agree_when_nothing_is_clipped():
    # Degraded sits at 0.5 and the correction is small, so the raw output
    # stays inside [0, 1] and the clamp is a no-op.
    diagnostics = _diagnostics_over([np.full((4, 4), 0.5, dtype=np.float32)], 0.2)
    assert diagnostics["mean_absolute_predicted_correction"] == pytest.approx(0.2)
    assert diagnostics["mean_absolute_post_clamp_change_vs_degraded"] == pytest.approx(0.2)
    assert diagnostics["fraction_changed_by_clamp"] == 0.0


def test_the_per_slice_quantiles_also_use_the_raw_output():
    diagnostics = _diagnostics_over([np.zeros((4, 4), dtype=np.float32)] * 3, -0.25)
    predicted = diagnostics["per_slice_mean_absolute_predicted_correction"]
    post_clamp = diagnostics["per_slice_mean_absolute_post_clamp_change"]
    for quantile in ("q05", "q50", "q95"):
        assert predicted[quantile] == pytest.approx(0.25)
        assert post_clamp[quantile] == pytest.approx(0.0)


def test_the_clipping_fractions_describe_the_raw_output():
    diagnostics = _diagnostics_over([np.zeros((4, 4), dtype=np.float32)], -0.25)
    assert diagnostics["fraction_raw_below_zero"] == pytest.approx(1.0)
    assert diagnostics["fraction_raw_above_one"] == pytest.approx(0.0)
    assert diagnostics["fraction_changed_by_clamp"] == pytest.approx(1.0)
    assert diagnostics["raw_minimum"] == pytest.approx(-0.25)
    assert diagnostics["finite_value_failures"] == 0


def test_an_overshoot_above_one_is_counted_too():
    diagnostics = _diagnostics_over([np.ones((4, 4), dtype=np.float32)], 0.3)
    assert diagnostics["fraction_raw_above_one"] == pytest.approx(1.0)
    assert diagnostics["raw_maximum"] == pytest.approx(1.3)
    assert diagnostics["mean_absolute_predicted_correction"] == pytest.approx(0.3)
    assert diagnostics["mean_absolute_post_clamp_change_vs_degraded"] == pytest.approx(0.0)


def test_the_diagnostic_names_its_two_definitions():
    # The names alone caused the original defect; the file should say what
    # each one means rather than leaving a reader to guess.
    diagnostics = _diagnostics_over([np.zeros((2, 2), dtype=np.float32)], -0.1)
    definitions = diagnostics["definitions"]
    assert "raw - degraded" in definitions["predicted_correction"]
    assert "UNCLAMPED" in definitions["predicted_correction"]
    assert "clamp(" in definitions["post_clamp_change"]


def test_the_old_ambiguous_key_is_gone():
    # It named the post-clamp change "the correction". Leaving it in place
    # with a corrected value would silently change the meaning of a key
    # someone may already be reading.
    diagnostics = _diagnostics_over([np.zeros((2, 2), dtype=np.float32)], -0.1)
    assert "mean_absolute_correction_vs_degraded" not in diagnostics
    assert "per_slice_mean_absolute_correction" not in diagnostics


# --------------------------------------------------------------------------
# 2. sample alignment is a hard gate
# --------------------------------------------------------------------------

KEYS = ["a/1.dcm", "a/2.dcm", "b/1.dcm"]


def _reference(tmp_path: Path, keys: list[str], name: str = "ref.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame({"relative_dicom_path": keys}).to_csv(path, index=False)
    return path


def _frame(keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"relative_dicom_path": keys})


def _report(tmp_path: Path, scored: list[str], reference: list[str]) -> dict:
    return evaluate_cnn.check_sample_alignment(
        _frame(scored), {"vs_degraded_baseline": _reference(tmp_path, reference)}
    )


def test_exact_ordered_keys_are_accepted(tmp_path):
    report = _report(tmp_path, KEYS, KEYS)
    evaluate_cnn.require_sample_alignment(report, len(KEYS))  # must not raise
    assert report["vs_degraded_baseline"]["identical_order"] is True
    assert report["duplicates"] == 0


def test_reordered_keys_are_refused(tmp_path):
    # The same set of slices in a different order: every set-based check
    # passes and the comparison is still wrong, row by row.
    report = _report(tmp_path, KEYS, [KEYS[1], KEYS[0], KEYS[2]])
    assert report["vs_degraded_baseline"]["missing"] == 0
    assert report["vs_degraded_baseline"]["extra"] == 0
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="order"):
        evaluate_cnn.require_sample_alignment(report, len(KEYS))


def test_a_missing_key_is_refused(tmp_path):
    report = _report(tmp_path, KEYS[:2], KEYS)
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="missing"):
        evaluate_cnn.require_sample_alignment(report, 2)


def test_an_extra_key_is_refused(tmp_path):
    report = _report(tmp_path, [*KEYS, "c/1.dcm"], KEYS)
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="extra"):
        evaluate_cnn.require_sample_alignment(report, 4)


def test_a_duplicate_key_is_refused(tmp_path):
    report = _report(tmp_path, [*KEYS, KEYS[0]], KEYS)
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="duplicate"):
        evaluate_cnn.require_sample_alignment(report, 4)


def test_a_wrong_row_count_is_refused_even_against_a_matching_reference(tmp_path):
    # The manifest said how many slices there are; scoring a different number
    # is a failure whatever the reference table happens to contain.
    report = _report(tmp_path, KEYS, KEYS)
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="expected 99"):
        evaluate_cnn.require_sample_alignment(report, 99)


def test_a_short_reference_is_counted_as_mismatched_rows(tmp_path):
    # zip() stops at the shorter list, so a truncated reference used to score
    # zero order mismatches and pass.
    report = _report(tmp_path, KEYS, KEYS[:1])
    assert report["vs_degraded_baseline"]["order_mismatches"] == 2
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError):
        evaluate_cnn.require_sample_alignment(report, len(KEYS))


def test_every_declared_reference_is_checked(tmp_path):
    # A second reference that disagrees must fail the gate even when the
    # first one is perfect.
    report = evaluate_cnn.check_sample_alignment(
        _frame(KEYS),
        {
            "vs_degraded_baseline": _reference(tmp_path, KEYS, "base.csv"),
            "vs_clahe": _reference(tmp_path, [KEYS[2], KEYS[1], KEYS[0]], "clahe.csv"),
        },
    )
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="vs_clahe"):
        evaluate_cnn.require_sample_alignment(report, len(KEYS))


def test_the_gate_reports_every_failure_rather_than_the_first(tmp_path):
    report = _report(tmp_path, [*KEYS, KEYS[0]], [*KEYS, "c/9.dcm"])
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError) as error:
        evaluate_cnn.require_sample_alignment(report, 4)
    message = str(error.value)
    assert "duplicate" in message
    assert "missing" in message


# --------------------------------------------------------------------------
# 3. checkpoint provenance is verified against files other than the checkpoint
# --------------------------------------------------------------------------

CONFIG_TEXT = "model:\n  algorithm: residual_cnn_v1\n"
EPOCH = 29
VALUE = 0.0093530865


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def canonical(tmp_path):
    """A miniature but structurally complete canonical run on disk."""
    config_path = tmp_path / "cnn.yaml"
    config_path.write_text(CONFIG_TEXT, encoding="utf-8")
    # Hash the bytes that landed on disk, not the string: text mode rewrites
    # newlines on Windows, and file_sha256 reads bytes.
    config_sha = _sha256(config_path.read_bytes())

    model = build_model(ResidualCnnConfig(hidden_channels=2, depth=2), torch.device("cpu"))
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    checkpoint_sha = _sha256(checkpoint_path.read_bytes())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # A complete, contiguous history: select_best_epoch refuses one with
    # gaps, and rightly so - a hole is a lost epoch, not a tie.
    scores = {epoch: 0.0163 - 0.0002 * epoch for epoch in range(31)}
    scores[EPOCH] = VALUE
    pd.DataFrame({"epoch": list(scores), SELECTION_METRIC: list(scores.values())}).to_csv(
        run_dir / "training_history.csv", index=False
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "config": {"sha256": config_sha},
                "training": {"seed": 2026},
                "checkpoint_selection": {
                    "primary_metric": SELECTION_METRIC,
                    "best_epoch": EPOCH,
                    "best_validation_patient_weighted_full_mae": VALUE,
                },
                "checkpoint": {
                    "sha256": checkpoint_sha,
                    "round_trip": {
                        "state_dict_tensor_mismatches": 0,
                        "probe_prediction_mismatches": 0,
                        "reproduces_saved_model": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = {
        "config_sha256": config_sha,
        "seed": 2026,
        "selection_metric": SELECTION_METRIC,
        "selection_value": VALUE,
        "epoch": EPOCH,
    }
    return {
        "payload": payload,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "run_dir": run_dir,
    }


def _verify(canonical, **overrides):
    payload = dict(canonical["payload"])
    payload.update(overrides)
    return evaluate_cnn.verify_checkpoint_provenance(
        payload, canonical["checkpoint_path"], canonical["config_path"], canonical["run_dir"]
    )


def test_a_consistent_canonical_run_verifies(canonical):
    report = _verify(canonical)
    assert report["verified"] is True
    assert report["checks_failed"] == []
    assert report["recomputed_from_files"]["selected_epoch_from_training_history"] == EPOCH


def test_a_mismatched_config_sha_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="config_sha"):
        _verify(canonical, config_sha256="0" * 64)


def test_editing_the_frozen_config_after_training_is_refused(canonical):
    # The checkpoint's recorded hash is still internally consistent with the
    # run summary; what changed is the file on disk. Hashing the bytes is the
    # only check that notices.
    canonical["config_path"].write_text(CONFIG_TEXT + "# edited\n", encoding="utf-8")
    with pytest.raises(
        evaluate_cnn.EvaluationIntegrityError,
        match="checkpoint_config_sha_matches_frozen_config",
    ):
        _verify(canonical)


def test_a_mismatched_checkpoint_file_sha_is_refused(canonical):
    # A different checkpoint file carrying correct-looking metadata.
    canonical["checkpoint_path"].write_bytes(canonical["checkpoint_path"].read_bytes() + b"\x00")
    with pytest.raises(
        evaluate_cnn.EvaluationIntegrityError, match="checkpoint_file_sha_matches_run_summary"
    ):
        _verify(canonical)


def test_a_non_canonical_seed_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="seed"):
        _verify(canonical, seed=7)


def test_a_mismatched_epoch_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="epoch"):
        _verify(canonical, epoch=30)


def test_an_epoch_the_selection_rule_would_not_have_chosen_is_refused(canonical):
    # The summary and the checkpoint agree with each other on epoch 30, and
    # both are wrong: re-running the predeclared rule over the tracked
    # history picks 29.
    summary_path = canonical["run_dir"] / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"]["best_epoch"] = 30
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(
        evaluate_cnn.EvaluationIntegrityError,
        match="checkpoint_epoch_matches_recomputed_selection",
    ):
        _verify(canonical, epoch=30)


def test_a_wrong_selection_metric_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="selection_metric"):
        _verify(canonical, selection_metric="validation_body_ssim")


def test_a_mismatched_selection_value_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="selection_value"):
        _verify(canonical, selection_value=VALUE + 1e-4)


def test_a_selection_value_within_the_rounding_tolerance_is_accepted(canonical):
    # The tracked summary rounds to 10 decimal places; that much disagreement
    # is representation, not a different checkpoint.
    report = _verify(canonical, selection_value=VALUE + 4e-10)
    assert report["verified"] is True


def test_a_failed_round_trip_is_refused(canonical):
    summary_path = canonical["run_dir"] / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint"]["round_trip"]["probe_prediction_mismatches"] = 3
    summary["checkpoint"]["round_trip"]["reproduces_saved_model"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="round_trip"):
        _verify(canonical)


def test_a_checkpoint_without_provenance_fields_is_refused(canonical):
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="carries no provenance"):
        evaluate_cnn.verify_checkpoint_provenance(
            {"epoch": EPOCH},
            canonical["checkpoint_path"],
            canonical["config_path"],
            canonical["run_dir"],
        )


def test_a_missing_run_record_is_refused(canonical):
    (canonical["run_dir"] / "run_summary.json").unlink()
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="run_summary.json is missing"):
        _verify(canonical)


def test_a_malformed_run_summary_is_refused(canonical):
    (canonical["run_dir"] / "run_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(evaluate_cnn.EvaluationIntegrityError, match="missing the provenance"):
        _verify(canonical)


def test_the_report_records_independently_recomputed_values(canonical):
    # Not a copy of the payload: these come from hashing the files and from
    # re-running the selection rule.
    report = _verify(canonical)
    recomputed = report["recomputed_from_files"]
    assert recomputed["config_sha256"] == _sha256(canonical["config_path"].read_bytes())
    assert recomputed["checkpoint_file_sha256"] == _sha256(
        canonical["checkpoint_path"].read_bytes()
    )
    assert recomputed["selection_value_from_training_history"] == pytest.approx(VALUE)
    assert report["verified_before_any_image_was_scored"] is True


def test_the_frozen_config_is_recorded_relative_to_the_repository():
    # Tracked artifacts carry no absolute filesystem path: it differs between
    # machines, so two runs of an unchanged definition would stop producing
    # byte-identical files, and it would commit a local username.
    from ct_restoration.config import PROJECT_ROOT

    recorded = evaluate_cnn.repository_path(PROJECT_ROOT / "configs" / "cnn.yaml")
    assert recorded == "configs/cnn.yaml"
    assert "\\" not in recorded


def test_a_path_outside_the_repository_falls_back_to_absolute(tmp_path):
    # Documented fallback: a config deliberately kept elsewhere still has to
    # be identified, and there is no relative form for it.
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("{}", encoding="utf-8")
    assert Path(evaluate_cnn.repository_path(outside)).is_absolute()


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"low-dose-ct")
    assert evaluate_cnn.file_sha256(path) == _sha256(b"low-dose-ct")
