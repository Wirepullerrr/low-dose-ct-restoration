"""Tests for the U-Net evaluation command's own wiring.

The integrity gates themselves are shared with the CNN and tested in
``test_evaluation_integrity.py``. What is specific to this command, and
therefore tested here, is that it compares against the right set of methods
and reads the eight metrics' directions correctly - the two places a
four-method comparison can quietly go wrong.

Fully synthetic and dataset-free: nothing here opens a CHAOS file or a
checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_unet = _load("evaluate_unet")


def test_the_eight_reported_metrics_are_the_frozen_set():
    assert evaluate_unet.METRICS == (
        "full_mae",
        "full_mse",
        "full_psnr",
        "full_ssim",
        "body_mae",
        "body_mse",
        "body_psnr",
        "body_ssim",
    )


def test_the_degraded_baseline_is_the_bar_and_the_others_are_comparisons():
    # The baseline is separate from the comparison list on purpose: beating
    # no restoration is the requirement, beating CLAHE or the CNN is not.
    assert evaluate_unet.BASELINE_STEM == "degraded_baseline"
    assert evaluate_unet.COMPARISON_STEMS == ("clahe", "cnn")


def test_the_cnn_is_among_the_methods_compared_against():
    # The architecture comparison is the reason this milestone exists; if the
    # CNN dropped out of the list it would vanish silently from the report.
    assert "cnn" in evaluate_unet.COMPARISON_STEMS


@pytest.mark.parametrize("metric", ["full_mae", "full_mse", "body_mae", "body_mse"])
def test_lower_is_better_metrics_improve_when_the_delta_is_negative(metric):
    assert evaluate_unet.improves(metric, -0.001) is True
    assert evaluate_unet.improves(metric, 0.001) is False


@pytest.mark.parametrize("metric", ["full_psnr", "full_ssim", "body_psnr", "body_ssim"])
def test_higher_is_better_metrics_improve_when_the_delta_is_positive(metric):
    assert evaluate_unet.improves(metric, 0.5) is True
    assert evaluate_unet.improves(metric, -0.5) is False


@pytest.mark.parametrize("metric", evaluate_unet.METRICS)
def test_an_exactly_zero_delta_is_never_an_improvement(metric):
    # A tie is not a win. Counting it as one would let a method that changed
    # nothing be reported as better.
    assert evaluate_unet.improves(metric, 0.0) is False


def test_the_run_directory_and_checkpoint_default_to_the_canonical_run():
    arguments = evaluate_unet.build_parser().parse_args([])
    assert arguments.run_dir == "outputs/runs/unet_seed2026"
    assert arguments.checkpoint == "outputs/checkpoints/unet_seed2026_best.pt"
    assert arguments.unet_config == "unet.yaml"
    assert arguments.split == "validation"


def test_the_evaluation_and_degradation_configs_are_the_frozen_shared_ones():
    # A U-Net-specific metric or corruption would make every comparison in
    # this report meaningless.
    arguments = evaluate_unet.build_parser().parse_args([])
    assert arguments.evaluation_config == "evaluation.yaml"
    assert arguments.degradation_config == "degradation.yaml"
    assert arguments.preprocessing_config == "baseline.yaml"


def test_the_qc_command_reads_training_slices_only():
    qc = _load("qc_unet")
    assert qc.QC_SPLIT == "train"
    # No --split option at all: the restriction is structural, not a default.
    actions = {action.dest for action in qc.build_parser()._actions}
    assert "split" not in actions
