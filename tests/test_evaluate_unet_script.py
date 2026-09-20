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


# --------------------------------------------------------------------------
# per-seed metric layout: a new seed must not read or write another's tables
# --------------------------------------------------------------------------

evaluate_cnn = _load("evaluate_cnn")


def test_the_frozen_references_are_read_from_the_reference_directory():
    # The degraded baseline and CLAHE are measured once, not per seed.
    assert evaluate_unet.BASELINE_STEM == "degraded_baseline"
    assert "clahe" in evaluate_unet.COMPARISON_STEMS
    assert "clahe" not in evaluate_unet.SEED_LOCAL_STEMS
    assert evaluate_unet.BASELINE_STEM not in evaluate_unet.SEED_LOCAL_STEMS


def test_the_cnn_comparison_is_read_from_the_same_seed_directory():
    # The CNN is a per-seed learned result. Comparing a seed-2027 U-Net
    # against the seed-2026 CNN would pair two different experiments.
    assert evaluate_unet.SEED_LOCAL_STEMS == ("cnn",)


@pytest.mark.parametrize("module", ["evaluate_cnn", "evaluate_unet"])
def test_both_evaluators_expose_a_separate_reference_directory(module):
    source = (SCRIPTS / f"{module}.py").read_text(encoding="utf-8")
    assert '"--reference-dir"' in source
    assert "reference_dir = Path(arguments.reference_dir)" in source


@pytest.mark.parametrize("module", ["evaluate_cnn", "evaluate_unet"])
def test_the_reference_directory_defaults_to_the_canonical_metrics_root(module):
    loaded = evaluate_cnn if module == "evaluate_cnn" else evaluate_unet
    assert loaded.REFERENCE_DIR == loaded.METRICS_DIR


def test_a_seed_directory_and_the_canonical_directory_are_different_places():
    from ct_restoration.run_layout import METRICS_ROOT, seed_metrics_directory

    assert seed_metrics_directory(2027) != METRICS_ROOT
    assert seed_metrics_directory(2027).as_posix().startswith(METRICS_ROOT.as_posix())
    # Nested under the canonical root, never equal to it: the Milestone 8 and
    # 9 tables sit at the root and no later seed can land on them.
    assert METRICS_ROOT in seed_metrics_directory(2027).parents


def test_the_seed_directory_layout_holds_both_methods_for_that_seed():
    # The documented workflow: CNN seed S and U-Net seed S write into the
    # same per-seed directory, so the U-Net finds its matching CNN there.
    from ct_restoration.run_layout import seed_metrics_directory

    directory = seed_metrics_directory(2027)
    expected = {
        directory / "cnn_validation_patients.csv",
        directory / "unet_validation_patients.csv",
    }
    assert len({path.parent for path in expected}) == 1


# --------------------------------------------------------------------------
# the hold-out seal does not weaken for any seed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "stress"])
@pytest.mark.parametrize("seed", [2026, 2027, 7])
def test_a_sealed_split_is_refused_whatever_seed_is_in_play(split, seed):
    # A seed loop must never imply a split loop. The seal is a property of
    # the split name alone, so no per-seed automation can widen it.
    from ct_restoration.evaluation import HeldOutSplitError, require_development_split
    from ct_restoration.run_layout import run_directory

    run_directory("cnn", seed)  # a seed being in play changes nothing
    with pytest.raises(HeldOutSplitError):
        require_development_split(split)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_the_development_splits_remain_the_only_accepted_ones(split):
    from ct_restoration.evaluation import DEVELOPMENT_SPLITS, require_development_split

    assert require_development_split(split) == split
    assert set(DEVELOPMENT_SPLITS) == {"train", "validation"}


# --------------------------------------------------------------------------
# a per-seed U-Net must be paired with the CNN from its own seed
# --------------------------------------------------------------------------

CNN_SLICES = "cnn_validation_slices.csv"
CNN_PATIENTS = "cnn_validation_patients.csv"


def _seed_dir(tmp_path, *, slices: bool = False, patients: bool = False) -> Path:
    directory = tmp_path / "multiseed" / "seed2027"
    directory.mkdir(parents=True)
    if slices:
        (directory / CNN_SLICES).write_text("x", encoding="utf-8")
    if patients:
        (directory / CNN_PATIENTS).write_text("x", encoding="utf-8")
    return directory


def test_the_canonical_metrics_directory_needs_no_extra_pairing_check():
    # Milestone 9 behaviour, unchanged: the canonical U-Net sits beside the
    # canonical CNN and is scored exactly as before.
    report = evaluate_unet.require_same_seed_cnn(evaluate_unet.METRICS_DIR, "validation")
    assert report["output_dir_is_canonical"] is True
    assert report["same_seed_cnn_required"] is False


def test_a_per_seed_directory_without_any_cnn_is_refused(tmp_path):
    with pytest.raises(evaluate_unet.MissingSeedPairError) as caught:
        evaluate_unet.require_same_seed_cnn(_seed_dir(tmp_path), "validation")
    message = str(caught.value)
    assert CNN_SLICES in message
    assert CNN_PATIENTS in message


def test_a_per_seed_directory_with_only_the_patients_file_is_refused(tmp_path):
    with pytest.raises(evaluate_unet.MissingSeedPairError, match=CNN_SLICES):
        evaluate_unet.require_same_seed_cnn(_seed_dir(tmp_path, patients=True), "validation")


def test_a_per_seed_directory_with_only_the_slices_file_is_refused(tmp_path):
    with pytest.raises(evaluate_unet.MissingSeedPairError, match=CNN_PATIENTS):
        evaluate_unet.require_same_seed_cnn(_seed_dir(tmp_path, slices=True), "validation")


def test_a_per_seed_directory_with_both_cnn_files_is_accepted(tmp_path):
    report = evaluate_unet.require_same_seed_cnn(
        _seed_dir(tmp_path, slices=True, patients=True), "validation"
    )
    assert report["same_seed_cnn_required"] is True
    assert sorted(report["present"]) == sorted([CNN_SLICES, CNN_PATIENTS])


def test_the_canonical_cnn_is_never_used_as_a_fallback_for_a_per_seed_run(tmp_path):
    # The canonical CNN tables exist on disk throughout this test. A per-seed
    # directory without its own CNN must still refuse rather than reach for
    # them: pairing seed 2027's U-Net with seed 2026's CNN would compare two
    # different experiments behind a delta that looks entirely valid.
    if not (evaluate_unet.METRICS_DIR / CNN_SLICES).exists():
        pytest.skip("canonical CNN metrics not present in this checkout")
    with pytest.raises(evaluate_unet.MissingSeedPairError) as caught:
        evaluate_unet.require_same_seed_cnn(_seed_dir(tmp_path), "validation")
    assert "NOT used as a fallback" in str(caught.value)


def test_the_refusal_explains_the_same_seed_requirement(tmp_path):
    with pytest.raises(evaluate_unet.MissingSeedPairError) as caught:
        evaluate_unet.require_same_seed_cnn(_seed_dir(tmp_path), "validation")
    assert "SAME statistical" in str(caught.value)


def test_the_pairing_gate_runs_before_the_checkpoint_is_loaded():
    # Ordering, not just presence: refusing after scoring would still leave
    # the run without a comparison, but only after reading every image.
    import ast

    tree = ast.parse((SCRIPTS / "evaluate_unet.py").read_text(encoding="utf-8"))
    main = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = sorted(
        (node.lineno, node.func.id)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    names = [name for _, name in calls]
    gate = names.index("require_same_seed_cnn")
    for later in ("load_checkpoint", "evaluate_slices", "write_csv", "write_json"):
        if later in names:
            assert gate < names.index(later), f"pairing gate runs after {later}"
