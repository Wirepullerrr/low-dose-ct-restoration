"""Tests for the Milestone 10 analysis rules and the multi-seed summarizer.

Written before any of the eight new runs exists, and entirely synthetic: no
CHAOS image, no checkpoint, no GPU. That is deliberate. These rules decide
what the real results will be allowed to say, so they are fixed and tested
while nobody knows what those results are.

The scenarios cover the ways a multi-seed comparison goes wrong quietly - a
missing seed, a mislabelled one, a sign flipped on a lower-is-better metric,
a NaN averaged into a mean - because none of those look wrong in a table.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

from ct_restoration.multiseed import (
    DIRECTIONAL_PHRASE,
    HEADLINE_METRICS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRICS,
    MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY,
    STATISTICAL_SEEDS,
    STD_DDOF,
    MultiseedError,
    describe_values,
    directional_phrase,
    favours_cnn,
    favours_unet,
    oriented_improvement,
    overall_directional_claim,
    paired_metric_summary,
    raw_delta,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SEEDS = (2026, 2027, 2028, 2029, 2030)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summarize = _load_script("summarize_multiseed")


# ---------------------------------------------------------------------------
# 1. the frozen definitions
# ---------------------------------------------------------------------------


def test_the_seed_set_is_the_five_predeclared_seeds():
    assert STATISTICAL_SEEDS == (2026, 2027, 2028, 2029, 2030)
    assert len(set(STATISTICAL_SEEDS)) == 5


def test_the_eight_metrics_split_into_four_of_each_direction():
    assert set(METRICS) == LOWER_IS_BETTER | HIGHER_IS_BETTER
    assert LOWER_IS_BETTER.isdisjoint(HIGHER_IS_BETTER)
    assert len(LOWER_IS_BETTER) == 4
    assert len(HIGHER_IS_BETTER) == 4


def test_the_headline_metrics_are_the_four_quality_figures_and_do_not_replace_the_rest():
    assert set(HEADLINE_METRICS) == HIGHER_IS_BETTER
    # All eight stay mandatory: the headline four are an ordering, not a filter.
    assert set(HEADLINE_METRICS) < set(METRICS)


def test_the_standard_deviation_is_a_sample_statistic():
    assert STD_DDOF == 1


# ---------------------------------------------------------------------------
# 2. raw delta and oriented improvement (scenario F)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", sorted(HIGHER_IS_BETTER))
def test_for_higher_is_better_metrics_both_quantities_agree(metric):
    # U-Net scored higher, which is better, so both are positive.
    assert raw_delta(metric, cnn_value=31.0, unet_value=31.2) == pytest.approx(0.2)
    assert oriented_improvement(metric, cnn_value=31.0, unet_value=31.2) == pytest.approx(0.2)


@pytest.mark.parametrize("metric", sorted(LOWER_IS_BETTER))
def test_for_lower_is_better_metrics_the_two_quantities_have_opposite_signs(metric):
    # This is the sign trap the oriented quantity exists for: the U-Net is
    # BETTER here, and the raw delta is NEGATIVE.
    assert raw_delta(metric, cnn_value=0.0100, unet_value=0.0090) == pytest.approx(-0.0010)
    assert oriented_improvement(metric, cnn_value=0.0100, unet_value=0.0090) == pytest.approx(
        0.0010
    )


@pytest.mark.parametrize("metric", METRICS)
def test_a_positive_oriented_improvement_always_means_the_unet_did_better(metric):
    better, worse = (31.2, 30.8) if metric in HIGHER_IS_BETTER else (0.009, 0.011)
    assert oriented_improvement(metric, cnn_value=worse, unet_value=better) > 0
    assert favours_unet(metric, cnn_value=worse, unet_value=better) is True
    assert favours_cnn(metric, cnn_value=worse, unet_value=better) is False
    # and the other way round
    assert oriented_improvement(metric, cnn_value=better, unet_value=worse) < 0
    assert favours_cnn(metric, cnn_value=better, unet_value=worse) is True


@pytest.mark.parametrize("metric", METRICS)
def test_an_exact_tie_favours_neither_architecture(metric):
    assert oriented_improvement(metric, 1.0, 1.0) == 0.0
    assert favours_unet(metric, 1.0, 1.0) is False
    assert favours_cnn(metric, 1.0, 1.0) is False


def test_an_unknown_metric_is_refused():
    with pytest.raises(MultiseedError, match="unknown metric"):
        raw_delta("full_rmse", 1.0, 2.0)


# ---------------------------------------------------------------------------
# 3. describe_values: ddof=1 (scenario G for NaN)
# ---------------------------------------------------------------------------


def test_the_standard_deviation_uses_ddof_one():
    # mean 3; squared deviations 4+1+0+1+4 = 10; /(5-1) = 2.5; sqrt = 1.5811...
    described = describe_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert described["mean"] == pytest.approx(3.0)
    assert described["std"] == pytest.approx(math.sqrt(2.5))
    assert described["std_ddof"] == 1
    assert described["n"] == 5
    assert described["min"] == 1.0
    assert described["max"] == 5.0
    # A population std (ddof=0) would be sqrt(2.0); make the difference explicit.
    assert described["std"] != pytest.approx(math.sqrt(2.0))


def test_the_individual_values_are_reported_beside_the_summary():
    described = describe_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert described["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused_rather_than_averaged(bad):
    with pytest.raises(MultiseedError):
        describe_values([1.0, 2.0, bad, 4.0, 5.0])


def test_a_single_value_cannot_have_a_sample_standard_deviation():
    with pytest.raises(MultiseedError, match="at least two values"):
        describe_values([1.0])


# ---------------------------------------------------------------------------
# 4. the paired summary, scenarios A / B / C
# ---------------------------------------------------------------------------


def _pair(metric: str, unet_wins: int, total: int = 5):
    """Build a metric's seed pairs with exactly ``unet_wins`` favouring U-Net."""
    higher = metric in HIGHER_IS_BETTER
    cnn, unet = {}, {}
    for index, seed in enumerate(SEEDS[:total]):
        unet_better = index < unet_wins
        cnn[seed] = 30.0 if higher else 0.010
        if higher:
            unet[seed] = 30.5 if unet_better else 29.5
        else:
            unet[seed] = 0.009 if unet_better else 0.011
    return cnn, unet


@pytest.mark.parametrize("metric", METRICS)
def test_scenario_a_unet_wins_all_five_seeds(metric):
    summary = paired_metric_summary(metric, *_pair(metric, unet_wins=5))
    assert summary["seeds_favouring_unet"] == 5
    assert summary["seeds_favouring_cnn"] == 0
    assert summary["seeds_tied"] == 0
    assert summary["oriented_improvement"]["mean"] > 0
    assert summary["directionally_consistent_for_unet"] is True
    assert directional_phrase(summary) == "all 5 seeds favoured the U-Net"


@pytest.mark.parametrize("metric", METRICS)
def test_scenario_b_cnn_wins_all_five_seeds(metric):
    summary = paired_metric_summary(metric, *_pair(metric, unet_wins=0))
    assert summary["seeds_favouring_cnn"] == 5
    assert summary["directionally_consistent_for_unet"] is False
    assert summary["directionally_consistent_for_cnn"] is True
    assert directional_phrase(summary) == "all 5 seeds favoured the CNN"


@pytest.mark.parametrize("metric", METRICS)
def test_scenario_c_a_three_two_split_is_never_called_consistent(metric):
    summary = paired_metric_summary(metric, *_pair(metric, unet_wins=3))
    assert summary["seeds_favouring_unet"] == 3
    assert summary["seeds_favouring_cnn"] == 2
    assert summary["directionally_consistent_for_unet"] is False
    assert summary["directionally_consistent_for_cnn"] is False
    # No permitted phrase at all: the caller has to describe the split.
    assert directional_phrase(summary) is None


@pytest.mark.parametrize("metric", METRICS)
def test_four_of_five_is_the_threshold_for_directional_consistency(metric):
    assert MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY == 4
    four = paired_metric_summary(metric, *_pair(metric, unet_wins=4))
    assert four["directionally_consistent_for_unet"] is True
    assert directional_phrase(four) == f"the U-Net was {DIRECTIONAL_PHRASE}"


def test_direction_requires_both_the_mean_and_the_count():
    # Four seeds favour the U-Net by a hair; the fifth favours the CNN by a
    # landslide, so the MEAN oriented improvement is negative. The count alone
    # would call this consistent; the rule requires both halves.
    cnn = {2026: 30.0, 2027: 30.0, 2028: 30.0, 2029: 30.0, 2030: 30.0}
    unet = {2026: 30.01, 2027: 30.01, 2028: 30.01, 2029: 30.01, 2030: 20.0}
    summary = paired_metric_summary("full_psnr", cnn, unet)
    assert summary["seeds_favouring_unet"] == 4
    assert summary["oriented_improvement"]["mean"] < 0
    assert summary["directionally_consistent_for_unet"] is False
    assert directional_phrase(summary) is None


def test_the_raw_delta_is_reported_per_seed_and_is_plain_subtraction():
    cnn, unet = _pair("full_mae", unet_wins=5)
    summary = paired_metric_summary("full_mae", cnn, unet)
    assert summary["raw_delta_definition"] == "unet_minus_cnn"
    for seed in SEEDS:
        assert summary["raw_delta_by_seed"][seed] == pytest.approx(unet[seed] - cnn[seed])
        # lower-is-better: raw negative, oriented positive
        assert summary["raw_delta_by_seed"][seed] < 0
        assert summary["oriented_improvement_by_seed"][seed] > 0


def test_scenario_e_mismatched_seed_sets_are_refused():
    cnn, unet = _pair("full_psnr", unet_wins=5)
    unet[9999] = unet.pop(2030)
    with pytest.raises(MultiseedError, match="seed sets differ"):
        paired_metric_summary("full_psnr", cnn, unet)


# ---------------------------------------------------------------------------
# 5. the overall claim
# ---------------------------------------------------------------------------


def test_an_overall_claim_needs_all_eight_metrics_directional():
    summaries = {m: paired_metric_summary(m, *_pair(m, unet_wins=5)) for m in METRICS}
    overall = overall_directional_claim(summaries)
    assert overall["architecture"] == "unet"
    assert overall["requires_all_metrics"] == 8
    assert overall["significance_tested"] is False


def test_seven_of_eight_metrics_is_not_enough_for_an_overall_claim():
    summaries = {m: paired_metric_summary(m, *_pair(m, unet_wins=5)) for m in METRICS}
    summaries["body_ssim"] = paired_metric_summary("body_ssim", *_pair("body_ssim", unet_wins=3))
    overall = overall_directional_claim(summaries)
    assert overall["architecture"] is None
    assert "metric by metric" in overall["statement"]


def test_an_overall_claim_cannot_be_made_from_a_missing_metric_set():
    summaries = {m: paired_metric_summary(m, *_pair(m, unet_wins=5)) for m in METRICS[:4]}
    with pytest.raises(MultiseedError, match="needs all"):
        overall_directional_claim(summaries)


def test_the_overall_claim_carries_no_significance_machinery():
    # Checked structurally rather than by scanning prose: the note deliberately
    # *mentions* p-values in order to disclaim them, so a keyword scan would
    # flag the very sentence that keeps the report honest.
    summaries = {m: paired_metric_summary(m, *_pair(m, unet_wins=5)) for m in METRICS}
    overall = overall_directional_claim(summaries)
    assert overall["significance_tested"] is False
    assert not any(key in overall for key in ("p_value", "p_values", "confidence_interval", "ci"))
    # Nor does any per-metric block acquire one.
    for summary in summaries.values():
        assert not any(key.startswith("p_") for key in summary)


# ---------------------------------------------------------------------------
# 6. the frozen plan on disk
# ---------------------------------------------------------------------------


def test_the_committed_plan_declares_the_five_seeds_and_ten_runs():
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    assert plan["statistical_seeds"] == list(SEEDS)
    assert plan["new_training_seeds"] == [2027, 2028, 2029, 2030]
    assert plan["new_training_runs"] == 8
    assert plan["total_runs"] == 10
    assert plan["retrain_seed_2026"] is False
    assert plan["canonical_existing_seed"] == 2026


def test_the_committed_plan_keeps_the_holdout_sealed():
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    assert plan["development_split"] == "validation"
    assert plan["test_allowed"] is False
    assert plan["stress_allowed"] is False


def test_the_committed_plan_forbids_the_practices_that_would_inflate_a_result():
    analysis = summarize.load_plan(Path("configs/multiseed/plan.yaml"))["seed_level_analysis"]
    assert analysis["significance_testing"] is False
    assert analysis["best_seed_selection"] is False
    assert analysis["seed_exclusion_after_results"] is False
    assert analysis["checkpoint_averaging"] is False
    assert analysis["seed_ensembling"] is False
    assert analysis["outlier_discarding"] is False
    assert analysis["composite_score"] is False
    assert analysis["pool_patients_and_seeds"] is False
    assert analysis["sample_std_ddof"] == STD_DDOF


def test_the_committed_plan_pins_every_config_hash_to_the_file_on_disk():
    import hashlib

    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    for method in ("cnn", "unet"):
        for seed, entry in plan["runs"][method].items():
            path = Path(entry["config"])
            assert path.exists(), f"{method} {seed}: {path} missing"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == entry["sha256"], f"{method} {seed}: {path} hash drifted"


def test_a_plan_that_allows_a_sealed_split_is_refused(tmp_path):
    plan = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    plan["test_allowed"] = True
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    with pytest.raises(MultiseedError, match="sealed split"):
        summarize.load_plan(path)


def test_a_plan_with_a_duplicate_seed_is_refused(tmp_path):
    plan = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    plan["statistical_seeds"] = [2026, 2027, 2027, 2029, 2030]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    with pytest.raises(MultiseedError, match="distinct seeds"):
        summarize.load_plan(path)


def test_a_plan_whose_methods_disagree_with_its_runs_is_refused(tmp_path):
    plan = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    del plan["runs"]["cnn"][2030]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    with pytest.raises(MultiseedError, match="declares seeds"):
        summarize.load_plan(path)


# ---------------------------------------------------------------------------
# 7. the summarizer's per-run validation
# ---------------------------------------------------------------------------


def _summary(seed: int = 2027, **overrides):
    document = {
        "split": "validation",
        "patients": 6,
        "checkpoint": {"config_sha256": "deadbeef", "provenance": {"seed": seed}},
        "primary_result": {metric: {"mean": 1.0} for metric in METRICS},
        "outputs": {},
    }
    document.update(overrides)
    return document


def test_a_complete_run_summary_yields_its_eight_metrics():
    values = summarize.validate_run_summary("cnn", 2027, _summary())
    assert sorted(values) == sorted(METRICS)


def test_scenario_g_a_nan_metric_is_refused(tmp_path):
    document = _summary()
    document["primary_result"]["full_psnr"]["mean"] = float("nan")
    with pytest.raises(MultiseedError, match="full_psnr"):
        summarize.validate_run_summary("unet", 2027, document)


def test_an_infinite_metric_is_refused():
    document = _summary()
    document["primary_result"]["body_mse"]["mean"] = float("inf")
    with pytest.raises(MultiseedError, match="body_mse"):
        summarize.validate_run_summary("unet", 2027, document)


def test_an_incomplete_metric_set_is_refused():
    document = _summary()
    del document["primary_result"]["body_ssim"]
    with pytest.raises(MultiseedError, match="missing metrics"):
        summarize.validate_run_summary("cnn", 2027, document)


def test_a_wrong_patient_count_is_refused():
    with pytest.raises(MultiseedError, match="patients"):
        summarize.validate_run_summary("cnn", 2027, _summary(patients=5))


@pytest.mark.parametrize("split", ["train", "test", "stress"])
def test_a_summary_from_another_split_is_refused(split):
    with pytest.raises(MultiseedError, match="validation"):
        summarize.validate_run_summary("cnn", 2027, _summary(split=split))


@pytest.mark.parametrize("sealed", ["test", "stress"])
def test_an_artifact_referencing_a_sealed_split_is_refused(sealed):
    document = _summary(outputs={"slices_csv": f"outputs/metrics/cnn_{sealed}_slices.csv"})
    with pytest.raises(MultiseedError, match=sealed):
        summarize.validate_run_summary("cnn", 2027, document)


# ---------------------------------------------------------------------------
# 8. seed identity: four witnesses, every one required
# ---------------------------------------------------------------------------


def _witnesses(tmp_path, monkeypatch, *, seed=2027, method="cnn", config_seed=2027):
    """A complete, agreeing witness set on disk, ready to be broken one at a time.

    The run directory is redirected into tmp_path: these tests must never
    create outputs/runs/cnn_seed2027/, which would be a Milestone 10B artifact.
    """
    config = tmp_path / f"{method}_seed{seed}.yaml"
    body = "model:\n  algorithm: x\n"
    if config_seed is not None:
        body += f"training:\n  seed: {config_seed}\n"
    else:
        body += "training:\n  epochs: 30\n"
    config.write_bytes(body.encode("utf-8"))
    sha = hashlib.sha256(config.read_bytes()).hexdigest()

    runs = tmp_path / "runs"
    (runs / f"{method}_seed{seed}").mkdir(parents=True)
    (runs / f"{method}_seed{seed}" / "run_summary.json").write_text(
        json.dumps({"training": {"seed": seed}, "config": {"sha256": sha}}), encoding="utf-8"
    )
    monkeypatch.setattr(summarize, "run_directory", lambda m, s: runs / f"{m}_seed{s}")

    plan = {"runs": {method: {seed: {"config": config.as_posix(), "sha256": sha}}}}
    summary = {
        "checkpoint": {"config_sha256": sha, "seed": seed, "provenance": {"seed": seed}},
    }
    return plan, summary, config, runs / f"{method}_seed{seed}" / "run_summary.json"


def test_every_witness_agreeing_is_accepted(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    report = summarize.verify_seed_identity("cnn", 2027, plan, summary)
    assert report["agree"] is True
    assert report["witnesses_checked"] == 4
    assert report["config_training_seed"] == 2027
    assert report["metric_summary_recorded_seed"] == 2027
    assert report["run_summary_seed"] == 2027


def test_a_run_absent_from_the_plan_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    with pytest.raises(MultiseedError, match="no complete entry"):
        summarize.verify_seed_identity("unet", 2027, plan, summary)


# --- B. the config file ----------------------------------------------------


def test_a_missing_config_file_is_refused(tmp_path, monkeypatch):
    plan, summary, config, _ = _witnesses(tmp_path, monkeypatch)
    config.unlink()
    with pytest.raises(MultiseedError, match="does not exist"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_config_whose_bytes_changed_is_refused(tmp_path, monkeypatch):
    plan, summary, config, _ = _witnesses(tmp_path, monkeypatch)
    config.write_bytes(config.read_bytes() + b"# a comment\n")
    with pytest.raises(MultiseedError, match="hashes to"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_config_without_a_training_seed_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch, config_seed=None)
    with pytest.raises(MultiseedError, match="training.seed is absent"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_config_whose_training_seed_is_wrong_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch, config_seed=2029)
    with pytest.raises(MultiseedError, match="directory name is not evidence"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


@pytest.mark.parametrize("malformed", ['"2027"', "2027.0", "true"])
def test_a_malformed_config_training_seed_is_refused(tmp_path, monkeypatch, malformed):
    plan, summary, config, _ = _witnesses(tmp_path, monkeypatch)
    config.write_bytes(f"training:\n  seed: {malformed}\n".encode())
    plan["runs"]["cnn"][2027]["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    summary["checkpoint"]["config_sha256"] = plan["runs"]["cnn"][2027]["sha256"]
    with pytest.raises(MultiseedError, match="must be an integer"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


# --- C. provenance in the metric summary -----------------------------------


def test_a_metric_summary_without_a_recorded_config_sha_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    del summary["checkpoint"]["config_sha256"]
    with pytest.raises(MultiseedError, match="records no checkpoint config SHA"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_metric_summary_config_sha_that_disagrees_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    summary["checkpoint"]["config_sha256"] = "0" * 64
    with pytest.raises(MultiseedError, match="different experiment definition"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_metric_summary_without_a_recorded_seed_is_refused(tmp_path, monkeypatch):
    # Absence used to read as success. It is evidence of nothing.
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    summary["checkpoint"]["provenance"] = {}
    del summary["checkpoint"]["seed"]
    with pytest.raises(MultiseedError, match="is absent"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_metric_summary_recorded_seed_that_is_wrong_is_refused(tmp_path, monkeypatch):
    plan, summary, _, _ = _witnesses(tmp_path, monkeypatch)
    summary["checkpoint"]["provenance"]["seed"] = 2029
    with pytest.raises(MultiseedError, match="directory name is not evidence"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


# --- D. the training run summary -------------------------------------------


def test_a_missing_run_summary_is_refused(tmp_path, monkeypatch):
    plan, summary, _, run_summary = _witnesses(tmp_path, monkeypatch)
    run_summary.unlink()
    with pytest.raises(MultiseedError, match="is missing"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_run_summary_without_a_training_seed_is_refused(tmp_path, monkeypatch):
    plan, summary, _, run_summary = _witnesses(tmp_path, monkeypatch)
    run_summary.write_text(json.dumps({"training": {}}), encoding="utf-8")
    with pytest.raises(MultiseedError, match="is absent"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_run_summary_seed_that_is_wrong_is_refused(tmp_path, monkeypatch):
    plan, summary, _, run_summary = _witnesses(tmp_path, monkeypatch)
    run_summary.write_text(json.dumps({"training": {"seed": 2029}}), encoding="utf-8")
    with pytest.raises(MultiseedError, match="directory name is not evidence"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


@pytest.mark.parametrize("malformed", ["2027", 2027.0, True])
def test_a_malformed_run_summary_seed_is_refused(tmp_path, monkeypatch, malformed):
    plan, summary, _, run_summary = _witnesses(tmp_path, monkeypatch)
    run_summary.write_text(json.dumps({"training": {"seed": malformed}}), encoding="utf-8")
    with pytest.raises(MultiseedError, match="must be an integer"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


def test_a_run_summary_config_sha_that_disagrees_is_refused(tmp_path, monkeypatch):
    plan, summary, _, run_summary = _witnesses(tmp_path, monkeypatch)
    run_summary.write_text(
        json.dumps({"training": {"seed": 2027}, "config": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(MultiseedError, match="disagree about"):
        summarize.verify_seed_identity("cnn", 2027, plan, summary)


# --- the canonical seed-2026 runs must keep passing ------------------------


@pytest.mark.parametrize("method", ["cnn", "unet"])
def test_the_canonical_seed_2026_run_satisfies_every_witness(method):
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    path = Path(f"outputs/metrics/{method}_validation_summary.json")
    if not path.exists():
        pytest.skip("canonical metrics not present in this checkout")
    report = summarize.verify_seed_identity(
        method, 2026, plan, json.loads(path.read_text(encoding="utf-8"))
    )
    assert report["agree"] is True
    assert report["config_training_seed"] == 2026
    assert report["metric_summary_recorded_seed"] == 2026
    assert report["run_summary_seed"] == 2026


# ---------------------------------------------------------------------------
# 9. the experiment must be complete, and must contain nothing extra
# ---------------------------------------------------------------------------


def _experiment(tmp_path, monkeypatch, seeds=SEEDS, methods=("cnn", "unet")):
    """Write synthetic per-seed metric summaries and point the script at them."""
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    root = tmp_path / "metrics"
    root.mkdir()
    multiseed_root = root / "multiseed"
    multiseed_root.mkdir()

    def seed_dir(seed: int) -> Path:
        directory = multiseed_root / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    monkeypatch.setattr(summarize, "CANONICAL_METRICS", root)
    monkeypatch.setattr(summarize, "MULTISEED_METRICS_ROOT", multiseed_root)
    monkeypatch.setattr(summarize, "seed_metrics_directory", seed_dir)
    # Identity is covered by its own tests above; here the subject is
    # completeness, so the witness check is stubbed out deliberately.
    monkeypatch.setattr(summarize, "verify_seed_identity", lambda *a, **k: {"agree": True})

    for method in methods:
        for index, seed in enumerate(seeds):
            directory = root if seed == 2026 else seed_dir(seed)
            values = {}
            for metric in METRICS:
                base = 30.0 if metric in HIGHER_IS_BETTER else 0.010
                bonus = (0.1 if metric in HIGHER_IS_BETTER else -0.001) if method == "unet" else 0.0
                values[metric] = {"mean": base + bonus + index * 0.001}
            (directory / f"{method}_validation_summary.json").write_text(
                json.dumps(
                    {
                        "split": "validation",
                        "patients": 6,
                        "checkpoint": {"config_sha256": "x", "provenance": {"seed": seed}},
                        "primary_result": values,
                        "outputs": {},
                    }
                ),
                encoding="utf-8",
            )
    return plan, multiseed_root


def test_a_complete_experiment_is_summarised(tmp_path, monkeypatch):
    plan, _ = _experiment(tmp_path, monkeypatch)
    collected = summarize.collect(plan)
    assert sorted(collected["cnn"]) == list(SEEDS)
    assert sorted(collected["unet"]) == list(SEEDS)
    summary = summarize.build_summary(plan, collected)
    assert summary["seeds"] == list(SEEDS)
    assert summary["per_metric"]["full_psnr"]["seeds_favouring_unet"] == 5
    assert summary["overall"]["architecture"] == "unet"


def test_exactly_the_five_planned_seeds_are_accepted(tmp_path, monkeypatch):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    assert summarize.discover_unplanned_seeds(plan, multiseed_root) == []
    assert sorted(summarize.collect(plan)["cnn"]) == list(SEEDS)


def test_scenario_d_a_missing_seed_refuses_the_whole_summary(tmp_path, monkeypatch):
    plan, _ = _experiment(tmp_path, monkeypatch, seeds=(2026, 2027, 2028, 2029))
    with pytest.raises(MultiseedError) as caught:
        summarize.collect(plan)
    message = str(caught.value)
    assert "incomplete" in message
    assert "seed 2030" in message
    assert "which runs finished" in message


def test_a_missing_seed_for_only_one_architecture_is_still_refused(tmp_path, monkeypatch):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    (multiseed_root / "seed2029" / "unet_validation_summary.json").unlink()
    with pytest.raises(MultiseedError, match="incomplete"):
        summarize.collect(plan)


def _plant_unplanned(root: Path, seed: int, names: tuple[str, ...]) -> Path:
    directory = root / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("{}", encoding="utf-8")
    return directory


def test_scenario_h_an_unplanned_seed_result_is_refused_not_ignored(tmp_path, monkeypatch):
    # The correction: an extra seed must stop the experiment. Quietly leaving
    # it out would make the report a summary of the seeds that suited it.
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    _plant_unplanned(
        multiseed_root, 2031, ("cnn_validation_summary.json", "unet_validation_summary.json")
    )
    assert summarize.discover_unplanned_seeds(plan, multiseed_root) == [2031]
    with pytest.raises(MultiseedError) as caught:
        summarize.collect(plan)
    message = str(caught.value)
    assert "2031" in message
    assert "pre-registration" in message


def test_an_unplanned_seed_with_only_a_cnn_result_is_refused(tmp_path, monkeypatch):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    _plant_unplanned(multiseed_root, 2031, ("cnn_validation_summary.json",))
    with pytest.raises(MultiseedError, match="2031"):
        summarize.collect(plan)


def test_an_unplanned_seed_with_only_a_unet_result_is_refused(tmp_path, monkeypatch):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    _plant_unplanned(multiseed_root, 2031, ("unet_validation_summary.json",))
    with pytest.raises(MultiseedError, match="2031"):
        summarize.collect(plan)


@pytest.mark.parametrize(
    "name",
    [
        "cnn_validation_patients.csv",
        "cnn_validation_slices.csv",
        "unet_validation_patients.csv",
        "unet_validation_slices.csv",
    ],
)
def test_any_learned_artifact_marks_an_unplanned_seed(tmp_path, monkeypatch, name):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    _plant_unplanned(multiseed_root, 2031, (name,))
    assert summarize.discover_unplanned_seeds(plan, multiseed_root) == [2031]


def test_several_unplanned_seeds_are_all_named(tmp_path, monkeypatch):
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    _plant_unplanned(multiseed_root, 2031, ("cnn_validation_summary.json",))
    _plant_unplanned(multiseed_root, 1999, ("unet_validation_summary.json",))
    assert summarize.discover_unplanned_seeds(plan, multiseed_root) == [1999, 2031]


def test_documentation_under_the_multiseed_root_is_not_a_result(tmp_path, monkeypatch):
    # A README beside the seed directories is documentation, not a run.
    plan, multiseed_root = _experiment(tmp_path, monkeypatch)
    (multiseed_root / "README.md").write_text("notes", encoding="utf-8")
    (multiseed_root / "seed2031").mkdir()
    (multiseed_root / "seed2031" / "README.md").write_text("notes", encoding="utf-8")
    (multiseed_root / "notes").mkdir()
    (multiseed_root / "notes" / "cnn_validation_summary.json").write_text("{}", encoding="utf-8")
    assert summarize.discover_unplanned_seeds(plan, multiseed_root) == []
    assert sorted(summarize.collect(plan)["cnn"]) == list(SEEDS)


def test_the_real_experiment_on_disk_is_now_complete():
    # Milestone 10A asserted the opposite: that only seed 2026 existed and the
    # summarizer had to refuse. Milestone 10B executed the pre-registered runs,
    # so the assertion is inverted rather than deleted - the real on-disk state
    # is still checked against the plan, just against the phase the project is
    # actually in. The refusal behaviour itself is not lost: it is covered on
    # synthetic trees by test_scenario_d_a_missing_seed_refuses_the_whole_summary
    # and test_a_missing_seed_for_only_one_architecture_is_still_refused, which
    # can construct an incomplete experiment without deleting a real run.
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    collected = summarize.collect(plan)
    for method in ("cnn", "unet"):
        assert sorted(collected[method]) == list(SEEDS)
    assert summarize.discover_unplanned_seeds(plan) == []


# ---------------------------------------------------------------------------
# 9b. the plan hash recorded is the plan that was actually loaded
# ---------------------------------------------------------------------------


def test_the_summary_records_the_hash_of_the_plan_actually_loaded(tmp_path, monkeypatch):
    plan, _ = _experiment(tmp_path, monkeypatch)
    collected = summarize.collect(plan)

    # A byte-different copy of the plan, loaded from another path.
    copy = tmp_path / "other_plan.yaml"
    original = Path("configs/multiseed/plan.yaml").read_bytes()
    copy.write_bytes(original + b"\n# an extra trailing comment\n")
    copied_plan = summarize.load_plan(copy)

    expected = hashlib.sha256(copy.read_bytes()).hexdigest()
    canonical = hashlib.sha256(original).hexdigest()
    assert expected != canonical

    summary = summarize.build_summary(copied_plan, collected)
    assert summary["plan_sha256"] == expected
    assert summary["plan_sha256"] != canonical
    assert summary["plan_path"] == copy.as_posix()


def test_the_default_plan_hash_is_the_canonical_one(tmp_path, monkeypatch):
    plan, _ = _experiment(tmp_path, monkeypatch)
    summary = summarize.build_summary(plan, summarize.collect(plan))
    canonical = hashlib.sha256(Path("configs/multiseed/plan.yaml").read_bytes()).hexdigest()
    assert summary["plan_sha256"] == canonical
    assert summary["plan_path"] == "configs/multiseed/plan.yaml"


def test_a_plan_file_may_not_declare_the_reserved_source_key(tmp_path):
    document = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    document[summarize.PLAN_SOURCE_KEY] = {"path": "elsewhere", "sha256": "0" * 64}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(MultiseedError, match="reserved key"):
        summarize.load_plan(path)


# ---------------------------------------------------------------------------
# 9c. the plan must still be the five-seed plan at runtime
# ---------------------------------------------------------------------------


def _plan_with(tmp_path, seeds):
    document = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    document["statistical_seeds"] = list(seeds)
    for method in ("cnn", "unet"):
        template = document["runs"][method][2026]
        document["runs"][method] = {seed: dict(template) for seed in seeds}
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_a_plan_that_grew_a_sixth_seed_is_refused(tmp_path):
    path = _plan_with(tmp_path, [2026, 2027, 2028, 2029, 2030, 2031])
    with pytest.raises(MultiseedError, match="pre-registered Milestone 10"):
        summarize.load_plan(path)


def test_a_plan_that_lost_a_seed_is_refused(tmp_path):
    path = _plan_with(tmp_path, [2026, 2027, 2028, 2029])
    with pytest.raises(MultiseedError, match="pre-registered Milestone 10"):
        summarize.load_plan(path)


def test_a_plan_with_a_replaced_seed_is_refused(tmp_path):
    path = _plan_with(tmp_path, [2026, 2027, 2028, 2029, 2099])
    with pytest.raises(MultiseedError, match="pre-registered Milestone 10"):
        summarize.load_plan(path)


def test_a_plan_with_the_wrong_methods_is_refused(tmp_path):
    document = yaml.safe_load(Path("configs/multiseed/plan.yaml").read_text(encoding="utf-8"))
    document["methods"] = ["cnn"]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(MultiseedError, match="plan methods must be"):
        summarize.load_plan(path)


def test_the_committed_plan_matches_the_predeclared_constants():
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    assert sorted(plan["statistical_seeds"]) == sorted(STATISTICAL_SEEDS)
    assert sorted(plan["methods"]) == sorted(("cnn", "unet"))


# ---------------------------------------------------------------------------
# 10. the ten planned destinations, and degradation independence
# ---------------------------------------------------------------------------


def test_every_planned_run_gets_a_distinct_destination():
    from ct_restoration.run_layout import checkpoint_path, run_directory

    runs = [run_directory(m, s) for m in ("cnn", "unet") for s in SEEDS]
    checkpoints = [checkpoint_path(m, s) for m in ("cnn", "unet") for s in SEEDS]
    assert len(set(runs)) == 10
    assert len(set(checkpoints)) == 10
    assert len(set(runs) | set(checkpoints)) == 20


@pytest.mark.parametrize("seed", [2027, 2028, 2029, 2030])
def test_the_new_seed_destinations_are_the_documented_shape(seed):
    from ct_restoration.run_layout import checkpoint_path, run_directory, seed_metrics_directory

    assert run_directory("cnn", seed) == Path(f"outputs/runs/cnn_seed{seed}")
    assert run_directory("unet", seed) == Path(f"outputs/runs/unet_seed{seed}")
    assert checkpoint_path("cnn", seed) == Path(f"outputs/checkpoints/cnn_seed{seed}_best.pt")
    assert checkpoint_path("unet", seed) == Path(f"outputs/checkpoints/unet_seed{seed}_best.pt")
    assert seed_metrics_directory(seed) == Path(f"outputs/metrics/multiseed/seed{seed}")


def test_the_seed_2026_destinations_are_untouched_by_the_plan():
    from ct_restoration.run_layout import checkpoint_path, run_directory

    assert run_directory("cnn", 2026) == Path("outputs/runs/cnn_seed2026")
    assert run_directory("unet", 2026) == Path("outputs/runs/unet_seed2026")
    assert checkpoint_path("cnn", 2026) == Path("outputs/checkpoints/cnn_seed2026_best.pt")
    assert checkpoint_path("unet", 2026) == Path("outputs/checkpoints/unet_seed2026_best.pt")


def test_every_planned_run_is_recorded_exactly_once_in_tracked_artifacts():
    # Milestone 10A asserted that none of these existed yet. Milestone 10B
    # produced them, so what is worth protecting is no longer their absence but
    # their exactness: one run, one recorded checkpoint and one metric set per
    # planned run, and nothing in the tracked output trees beyond the plan.
    #
    # Tracked evidence only. The checkpoint binaries are git-ignored, so a
    # fresh clone has none and a unit test must not need them. What a clone
    # does have is each run summary's record of the single checkpoint that run
    # selected - its path and SHA-256 - which is also what the evaluation
    # provenance gate checks a regenerated checkpoint against.
    from ct_restoration.run_layout import (
        MULTISEED_METRICS_ROOT,
        RUNS_ROOT,
        checkpoint_path,
        run_directory,
        seed_metrics_directory,
    )

    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    canonical_seed = plan["canonical_existing_seed"]
    hex_digits = set("0123456789abcdef")
    recorded = []

    for method in ("cnn", "unet"):
        for seed in SEEDS:
            run_dir = run_directory(method, seed)
            assert (run_dir / "training_history.csv").is_file(), f"{method} {seed} history"
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            assert summary["training"]["seed"] == seed

            checkpoint = summary["checkpoint"]
            assert checkpoint["path"] == checkpoint_path(method, seed).as_posix()
            assert checkpoint["tracked_in_git"] is False
            digest = checkpoint["sha256"]
            assert len(digest) == 64 and set(digest) <= hex_digits, f"{method} {seed} sha"
            recorded.append(digest)

            if seed != canonical_seed:
                metrics = seed_metrics_directory(seed) / f"{method}_validation_summary.json"
                assert metrics.is_file(), f"{method} {seed} metric summary missing"

    # Ten runs, ten distinct selected checkpoints. The same digest recorded
    # twice would mean two runs claim one set of weights.
    assert len(recorded) == 10
    assert len(set(recorded)) == 10

    # Nothing beyond the plan in the tracked trees: no extra run directory and
    # no extra seed directory, whatever it is named.
    expected_runs = sorted(f"{method}_seed{seed}" for method in ("cnn", "unet") for seed in SEEDS)
    assert sorted(entry.name for entry in RUNS_ROOT.iterdir() if entry.is_dir()) == expected_runs
    expected_seed_dirs = sorted(f"seed{seed}" for seed in SEEDS if seed != canonical_seed)
    assert (
        sorted(entry.name for entry in MULTISEED_METRICS_ROOT.iterdir() if entry.is_dir())
        == expected_seed_dirs
    )
    assert summarize.discover_unplanned_seeds(plan) == []


def test_the_training_seed_cannot_reach_the_degradation():
    # Structural, not by inspection: the Dataset constructor has no seed
    # parameter, so a per-training-seed corruption is not expressible.
    import inspect

    from ct_restoration.data.dataset import development_dataset
    from ct_restoration.data.degradation import derive_sample_seed

    assert "seed" not in inspect.signature(development_dataset).parameters
    assert "global_seed" in inspect.signature(derive_sample_seed).parameters
    # The per-slice seed is a function of (global_seed, key, algorithm) only.
    key = "subject=01/series=CT/instance=0042"
    assert derive_sample_seed(2026, key) == derive_sample_seed(2026, key)
    assert derive_sample_seed(2026, key) != derive_sample_seed(2027, key)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("method", ["cnn", "unet"])
def test_each_planned_run_declares_its_seed_and_shares_one_degradation(method, seed):
    """The parameter is the run under test, and it is genuinely used.

    For each of the ten planned runs: its own config declares that training
    seed, and the degradation it will use is the single frozen one - not a
    per-run corruption. The training seed is deliberately NOT fed into
    derive_sample_seed; the whole point is that it cannot get there.
    """
    from ct_restoration.config import load_config
    from ct_restoration.data.degradation import DegradationConfig, derive_sample_seed

    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    entry = plan["runs"][method][seed]

    # 1. this run's own config declares this training seed
    config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    assert config["training"]["seed"] == seed

    # 2. and nothing in it mentions a degradation seed of its own
    assert "degradation" not in config
    assert "global_seed" not in yaml.dump(config)

    # 3. the degradation every run uses is the one frozen config
    policy = plan["degradation_policy"]
    assert policy["config"] == "configs/degradation.yaml"
    assert policy["frozen_across_all_training_seeds"] is True
    assert policy["per_training_seed_degradation_seed"] is False

    # 4. so for a fixed key the corruption is identical for this run and for
    #    the canonical seed-2026 run - the degradation seed comes from the
    #    frozen file, never from `seed` above.
    degradation = DegradationConfig.from_mapping(load_config("degradation.yaml"))
    key = "subject=01/series=CT/instance=0042"
    assert degradation.global_seed == 2026
    assert derive_sample_seed(degradation.global_seed, key) == derive_sample_seed(2026, key)


def test_the_ten_planned_configs_declare_ten_run_seeds_and_two_distinct_values():
    # Read across the whole plan at once: five distinct seeds, each appearing
    # once per architecture, and every config agreeing with its plan slot.
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    declared = {}
    for method in ("cnn", "unet"):
        for seed, entry in plan["runs"][method].items():
            config = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
            declared[(method, seed)] = config["training"]["seed"]
    assert len(declared) == 10
    assert all(value == seed for (_, seed), value in declared.items())
    assert sorted({seed for _, seed in declared}) == sorted(SEEDS)


def test_the_plan_declares_degradation_frozen_across_seeds():
    plan = summarize.load_plan(Path("configs/multiseed/plan.yaml"))
    policy = plan["degradation_policy"]
    assert policy["frozen_across_all_training_seeds"] is True
    assert policy["config"] == "configs/degradation.yaml"
    assert policy["per_training_seed_degradation_seed"] is False
    assert policy["stochastic_by_epoch"] is False
