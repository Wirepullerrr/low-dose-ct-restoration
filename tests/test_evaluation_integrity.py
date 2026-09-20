"""Tests for the integrity gates every learned method's evaluation must clear.

These cover the shared modules - :mod:`ct_restoration.evaluation_integrity`
and :mod:`ct_restoration.models.diagnostics` - rather than one command,
because the residual CNN and the lightweight U-Net are both scored through
them. One definition of "integrity verified", tested once.

Fully synthetic and dataset-free: no CHAOS file is opened, no network access,
and the "model" is a stub on small arrays. What is being protected here is
not a number but the conditions under which a number is allowed to be
written.

Three gates, and each exists because of a way the canonical artifacts could
be wrong while looking entirely well-formed on disk:

* the **raw correction diagnostic** must describe the correction the network
  predicted, not the part of it that survived the clamp;
* **sample alignment** must be enforced, and enforced *before* anything is
  written, or a paired comparison can subtract one slice's metric from
  another slice's metric;
* **checkpoint provenance** must be verified against files other than the
  checkpoint, or a command will happily score whatever it is handed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ct_restoration import evaluation_integrity
from ct_restoration.models import diagnostics
from ct_restoration.models.cnn import ResidualCnnConfig, build_model
from ct_restoration.training import SELECTION_METRIC

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

    original_prepare = diagnostics.prepare_evaluation_slice
    original_degrade = diagnostics.degrade_low_dose_like
    diagnostics.prepare_evaluation_slice = fake_prepare
    diagnostics.degrade_low_dose_like = fake_degrade
    try:
        return diagnostics.raw_output_diagnostics(
            rows, Path("."), {}, None, None, model, torch.device("cpu")
        )
    finally:
        diagnostics.prepare_evaluation_slice = original_prepare
        diagnostics.degrade_low_dose_like = original_degrade


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
# the QC correction panels cannot show one quantity under the other's name
# --------------------------------------------------------------------------


def test_the_predicted_correction_panel_uses_the_raw_output():
    # Degraded is all zeros and the model subtracts 0.25, so the raw output
    # is -0.25 everywhere and the clamp pulls every pixel back to 0. The
    # predicted correction is therefore -0.25 and the post-clamp change is 0.
    model = _ConstantCorrection(-0.25)
    degraded = np.zeros((4, 4), dtype=np.float32)

    label, image = diagnostics.predicted_correction_panel(model, degraded)
    assert label == diagnostics.PREDICTED_CORRECTION_LABEL
    assert np.allclose(image, -0.25)


def test_the_post_clamp_panel_uses_the_clamped_output():
    model = _ConstantCorrection(-0.25)
    degraded = np.zeros((4, 4), dtype=np.float32)

    label, image = diagnostics.post_clamp_change_panel(model, degraded)
    assert label == diagnostics.POST_CLAMP_CHANGE_LABEL
    assert np.allclose(image, 0.0)


def test_the_two_panels_are_different_images_when_the_clamp_engages():
    # This is the defect the pairing exists to prevent: the two look equally
    # plausible as a figure, and only one of them is the correction.
    model = _ConstantCorrection(-0.25)
    degraded = np.zeros((4, 4), dtype=np.float32)

    _, predicted = diagnostics.predicted_correction_panel(model, degraded)
    _, post_clamp = diagnostics.post_clamp_change_panel(model, degraded)
    assert not np.allclose(predicted, post_clamp)


def test_the_two_panels_agree_when_nothing_is_clipped():
    model = _ConstantCorrection(0.2)
    degraded = np.full((4, 4), 0.5, dtype=np.float32)

    _, predicted = diagnostics.predicted_correction_panel(model, degraded)
    _, post_clamp = diagnostics.post_clamp_change_panel(model, degraded)
    assert np.allclose(predicted, post_clamp)


def test_the_two_labels_are_distinct_and_say_which_output_they_came_from():
    predicted = diagnostics.PREDICTED_CORRECTION_LABEL
    post_clamp = diagnostics.POST_CLAMP_CHANGE_LABEL
    assert predicted != post_clamp
    assert "raw" in predicted
    assert "clamp" in post_clamp
    # "correction" must never be the word used for the post-clamp image.
    assert "correction" in predicted
    assert "correction" not in post_clamp


def test_neither_qc_command_labels_a_clamped_difference_a_correction():
    """A source-level guard, because the figures themselves are git-ignored.

    A regression here would leave no trace in any tracked file, so the check
    has to be on the code. The module docstrings legitimately discuss both
    quantities while explaining the difference, so they are stripped first
    and only executable source is inspected.
    """
    for name in ("qc_cnn.py", "qc_unet.py"):
        path = Path(__file__).resolve().parents[1] / "scripts" / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        prose = ast.get_docstring(tree) or ""
        code = source.replace(prose, "")

        # The correction image must come from the shared helper, which
        # returns it bound to its own label.
        assert "predicted_correction_panel" in code, name
        # The old mislabel, and any hand-built clamped difference called a
        # correction, must both be gone.
        assert "(correction)" not in code, name
        labels = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for label in labels:
            if "correction" in label and label != prose:
                assert "raw" in label, f"{name}: {label!r} names a correction without saying raw"


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
    return evaluation_integrity.check_sample_alignment(
        _frame(scored), {"vs_degraded_baseline": _reference(tmp_path, reference)}
    )


def test_exact_ordered_keys_are_accepted(tmp_path):
    report = _report(tmp_path, KEYS, KEYS)
    evaluation_integrity.require_sample_alignment(report, len(KEYS))  # must not raise
    assert report["vs_degraded_baseline"]["identical_order"] is True
    assert report["duplicates"] == 0


def test_reordered_keys_are_refused(tmp_path):
    # The same set of slices in a different order: every set-based check
    # passes and the comparison is still wrong, row by row.
    report = _report(tmp_path, KEYS, [KEYS[1], KEYS[0], KEYS[2]])
    assert report["vs_degraded_baseline"]["missing"] == 0
    assert report["vs_degraded_baseline"]["extra"] == 0
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="order"):
        evaluation_integrity.require_sample_alignment(report, len(KEYS))


def test_a_missing_key_is_refused(tmp_path):
    report = _report(tmp_path, KEYS[:2], KEYS)
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="missing"):
        evaluation_integrity.require_sample_alignment(report, 2)


def test_an_extra_key_is_refused(tmp_path):
    report = _report(tmp_path, [*KEYS, "c/1.dcm"], KEYS)
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="extra"):
        evaluation_integrity.require_sample_alignment(report, 4)


def test_a_duplicate_key_is_refused(tmp_path):
    report = _report(tmp_path, [*KEYS, KEYS[0]], KEYS)
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="duplicate"):
        evaluation_integrity.require_sample_alignment(report, 4)


def test_a_wrong_row_count_is_refused_even_against_a_matching_reference(tmp_path):
    # The manifest said how many slices there are; scoring a different number
    # is a failure whatever the reference table happens to contain.
    report = _report(tmp_path, KEYS, KEYS)
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="expected 99"):
        evaluation_integrity.require_sample_alignment(report, 99)


def test_a_short_reference_is_counted_as_mismatched_rows(tmp_path):
    # zip() stops at the shorter list, so a truncated reference used to score
    # zero order mismatches and pass.
    report = _report(tmp_path, KEYS, KEYS[:1])
    assert report["vs_degraded_baseline"]["order_mismatches"] == 2
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError):
        evaluation_integrity.require_sample_alignment(report, len(KEYS))


def test_every_declared_reference_is_checked(tmp_path):
    # A second reference that disagrees must fail the gate even when the
    # first one is perfect.
    report = evaluation_integrity.check_sample_alignment(
        _frame(KEYS),
        {
            "vs_degraded_baseline": _reference(tmp_path, KEYS, "base.csv"),
            "vs_clahe": _reference(tmp_path, [KEYS[2], KEYS[1], KEYS[0]], "clahe.csv"),
        },
    )
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="vs_clahe"):
        evaluation_integrity.require_sample_alignment(report, len(KEYS))


def test_the_gate_reports_every_failure_rather_than_the_first(tmp_path):
    report = _report(tmp_path, [*KEYS, KEYS[0]], [*KEYS, "c/9.dcm"])
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError) as error:
        evaluation_integrity.require_sample_alignment(report, 4)
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
    return evaluation_integrity.verify_checkpoint_provenance(
        payload, canonical["checkpoint_path"], canonical["config_path"], canonical["run_dir"]
    )


def test_a_consistent_canonical_run_verifies(canonical):
    report = _verify(canonical)
    assert report["verified"] is True
    assert report["checks_failed"] == []
    assert report["recomputed_from_files"]["selected_epoch_from_training_history"] == EPOCH


def test_a_mismatched_config_sha_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="config_sha"):
        _verify(canonical, config_sha256="0" * 64)


def test_editing_the_frozen_config_after_training_is_refused(canonical):
    # The checkpoint's recorded hash is still internally consistent with the
    # run summary; what changed is the file on disk. Hashing the bytes is the
    # only check that notices.
    canonical["config_path"].write_text(CONFIG_TEXT + "# edited\n", encoding="utf-8")
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError,
        match="checkpoint_config_sha_matches_frozen_config",
    ):
        _verify(canonical)


def test_a_mismatched_checkpoint_file_sha_is_refused(canonical):
    # A different checkpoint file carrying correct-looking metadata.
    canonical["checkpoint_path"].write_bytes(canonical["checkpoint_path"].read_bytes() + b"\x00")
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError,
        match="checkpoint_file_sha_matches_run_summary",
    ):
        _verify(canonical)


def test_a_non_canonical_seed_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="seed"):
        _verify(canonical, seed=7)


def test_a_mismatched_epoch_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="epoch"):
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
        evaluation_integrity.EvaluationIntegrityError,
        match="checkpoint_epoch_matches_recomputed_selection",
    ):
        _verify(canonical, epoch=30)


def test_a_wrong_selection_metric_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="selection_metric"):
        _verify(canonical, selection_metric="validation_body_ssim")


def test_a_mismatched_selection_value_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="selection_value"):
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
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="round_trip"):
        _verify(canonical)


def test_a_checkpoint_without_provenance_fields_is_refused(canonical):
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="carries no provenance"
    ):
        evaluation_integrity.verify_checkpoint_provenance(
            {"epoch": EPOCH},
            canonical["checkpoint_path"],
            canonical["config_path"],
            canonical["run_dir"],
        )


def test_a_missing_run_record_is_refused(canonical):
    (canonical["run_dir"] / "run_summary.json").unlink()
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="run_summary.json is missing"
    ):
        _verify(canonical)


def test_a_malformed_run_summary_is_refused(canonical):
    (canonical["run_dir"] / "run_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="missing the provenance"
    ):
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


# --------------------------------------------------------------------------
# provenance metadata is type-checked, not coerced
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 2026.0),
        ("seed", "2026"),
        ("seed", True),
        ("seed", None),
        ("epoch", 29.0),
        ("epoch", "29"),
        ("epoch", True),
        ("epoch", None),
    ],
)
def test_a_non_integer_checkpoint_seed_or_epoch_is_refused(canonical, field, value):
    # int() would happily turn 29.0, "29" and True into 29. A gate that
    # repairs malformed metadata is checking its own repair, not the file.
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="must be an integer"):
        _verify(canonical, **{field: value})


def test_a_negative_checkpoint_epoch_is_refused(canonical):
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="must be >= 0"):
        _verify(canonical, epoch=-1)


@pytest.mark.parametrize("value", ["0.0093530865", True, None, float("nan"), float("inf")])
def test_a_non_finite_or_non_numeric_selection_value_is_refused(canonical, value):
    # NaN matters on its own: it compares unequal to everything, so without
    # this check it would be reported as a mismatched checkpoint rather than
    # as corrupt metadata.
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="must be (a real|finite)"
    ):
        _verify(canonical, selection_value=value)


def _rewrite_summary(canonical, mutate):
    path = canonical["run_dir"] / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    mutate(summary)
    path.write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize("value", [2026.0, "2026", True, None])
def test_a_non_integer_run_summary_seed_is_refused(canonical, value):
    _rewrite_summary(canonical, lambda s: s["training"].__setitem__("seed", value))
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="run summary training seed"
    ):
        _verify(canonical)


@pytest.mark.parametrize("value", [29.0, "29", True, None])
def test_a_non_integer_run_summary_best_epoch_is_refused(canonical, value):
    _rewrite_summary(
        canonical, lambda s: s["checkpoint_selection"].__setitem__("best_epoch", value)
    )
    with pytest.raises(
        evaluation_integrity.EvaluationIntegrityError, match="run summary best_epoch"
    ):
        _verify(canonical)


def test_a_negative_run_summary_best_epoch_is_refused(canonical):
    _rewrite_summary(canonical, lambda s: s["checkpoint_selection"].__setitem__("best_epoch", -3))
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="must be >= 0"):
        _verify(canonical)


@pytest.mark.parametrize("value", ["0.009", True, None])
def test_a_non_numeric_run_summary_selection_value_is_refused(canonical, value):
    _rewrite_summary(
        canonical,
        lambda s: s["checkpoint_selection"].__setitem__(
            evaluation_integrity.SUMMARY_BEST_VALUE_KEY, value
        ),
    )
    with pytest.raises(evaluation_integrity.EvaluationIntegrityError, match="must be a real"):
        _verify(canonical)


def test_the_strict_helpers_accept_genuine_numpy_scalars():
    # pandas hands back numpy scalars; those are genuine integers and reals
    # and must not be rejected alongside the malformed values above.
    assert evaluation_integrity._require_integer("n", np.int64(29)) == 29
    assert evaluation_integrity._require_finite_real("x", np.float64(0.5)) == 0.5


def test_the_canonical_metadata_still_passes_every_type_check(canonical):
    # The whole point of the hardening is that nothing valid changed.
    assert _verify(canonical)["verified"] is True


def test_the_frozen_config_is_recorded_relative_to_the_repository():
    # Tracked artifacts carry no absolute filesystem path: it differs between
    # machines, so two runs of an unchanged definition would stop producing
    # byte-identical files, and it would commit a local username.
    from ct_restoration.config import PROJECT_ROOT

    recorded = evaluation_integrity.repository_path(PROJECT_ROOT / "configs" / "cnn.yaml")
    assert recorded == "configs/cnn.yaml"
    assert "\\" not in recorded


def test_a_path_outside_the_repository_falls_back_to_absolute(tmp_path):
    # Documented fallback: a config deliberately kept elsewhere still has to
    # be identified, and there is no relative form for it.
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("{}", encoding="utf-8")
    assert Path(evaluation_integrity.repository_path(outside)).is_absolute()


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"low-dose-ct")
    assert evaluation_integrity.file_sha256(path) == _sha256(b"low-dose-ct")
