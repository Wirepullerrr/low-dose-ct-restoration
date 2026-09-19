"""Tests for the paired patient-level comparison between two methods.

Fully synthetic. The tables are small enough that the right answer can be read
off by eye, which is the point: a paired comparison that silently drops a
patient or flips a sign would still produce plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ct_restoration.evaluation import (
    METRIC_COLUMNS,
    EvaluationError,
    improved,
    metric_direction,
    paired_patient_deltas,
    summarise_paired_deltas,
)


def patient_table(rows: list[tuple[str, float]], slice_count: int = 10) -> pd.DataFrame:
    """A patient table where every metric column carries the same value."""
    records = []
    for subject_id, value in rows:
        record = {
            "subject_id": subject_id,
            "source_archive": "Train_Sets",
            "acquisition_group": "A",
            "slice_count": slice_count,
        }
        record.update({f"mean_{name}": value for name in METRIC_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Metric direction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("full_mae", "lower"),
        ("body_mae", "lower"),
        ("full_mse", "lower"),
        ("body_mse", "lower"),
        ("full_psnr", "higher"),
        ("body_psnr", "higher"),
        ("full_ssim", "higher"),
        ("body_ssim", "higher"),
    ],
)
def test_every_metric_has_a_declared_direction(metric, expected) -> None:
    assert metric_direction(metric) == expected


def test_an_unknown_metric_family_is_refused() -> None:
    with pytest.raises(EvaluationError, match="which direction improves"):
        metric_direction("body_entropy")


@pytest.mark.parametrize(
    ("metric", "delta", "is_better"),
    [
        ("body_psnr", 0.5, True),
        ("body_psnr", -0.5, False),
        ("body_ssim", 0.01, True),
        ("body_ssim", -0.01, False),
        ("body_mae", -0.01, True),
        ("body_mae", 0.01, False),
        ("full_mse", -1e-5, True),
        ("full_mse", 1e-5, False),
    ],
)
def test_improvement_respects_the_metric_direction(metric, delta, is_better) -> None:
    assert improved(metric, delta) is is_better


@pytest.mark.parametrize("metric", ["body_psnr", "body_mae"])
def test_a_zero_delta_is_not_an_improvement(metric) -> None:
    assert improved(metric, 0.0) is False


# --------------------------------------------------------------------------
# Delta computation and sign convention
# --------------------------------------------------------------------------


def test_delta_is_method_minus_baseline() -> None:
    method = patient_table([("4", 0.9)])
    baseline = patient_table([("4", 0.7)])

    deltas = paired_patient_deltas(method, baseline)

    row = deltas.iloc[0]
    assert row["baseline_body_ssim"] == pytest.approx(0.7)
    assert row["method_body_ssim"] == pytest.approx(0.9)
    assert row["delta_body_ssim"] == pytest.approx(0.2)


def test_a_worse_method_gives_a_negative_higher_is_better_delta() -> None:
    deltas = paired_patient_deltas(patient_table([("4", 0.6)]), patient_table([("4", 0.8)]))

    assert deltas.iloc[0]["delta_body_psnr"] == pytest.approx(-0.2)
    assert not improved("body_psnr", float(deltas.iloc[0]["delta_body_psnr"]))


def test_a_better_method_gives_a_negative_lower_is_better_delta() -> None:
    """For MAE the improvement is negative; the sign convention is uniform."""
    deltas = paired_patient_deltas(patient_table([("4", 0.01)]), patient_table([("4", 0.03)]))

    assert deltas.iloc[0]["delta_body_mae"] == pytest.approx(-0.02)
    assert improved("body_mae", float(deltas.iloc[0]["delta_body_mae"]))


def test_patients_come_back_in_canonical_order() -> None:
    method = patient_table([("34", 0.5), ("4", 0.5), ("17", 0.5)])
    baseline = patient_table([("4", 0.4), ("17", 0.4), ("34", 0.4)])

    deltas = paired_patient_deltas(method, baseline)

    assert list(deltas["subject_id"]) == ["4", "17", "34"]


def test_every_metric_gets_a_delta_column() -> None:
    deltas = paired_patient_deltas(patient_table([("4", 0.5)]), patient_table([("4", 0.4)]))

    for name in METRIC_COLUMNS:
        assert f"delta_{name}" in deltas
        assert f"method_{name}" in deltas
        assert f"baseline_{name}" in deltas


# --------------------------------------------------------------------------
# Strict alignment
# --------------------------------------------------------------------------


def test_different_patient_sets_are_refused() -> None:
    """A silent inner join would quietly shrink the comparison."""
    method = patient_table([("4", 0.5), ("14", 0.5)])
    baseline = patient_table([("4", 0.4), ("17", 0.4)])

    with pytest.raises(EvaluationError, match="same patients on both sides"):
        paired_patient_deltas(method, baseline)


def test_a_missing_patient_is_refused() -> None:
    method = patient_table([("4", 0.5), ("14", 0.5), ("17", 0.5)])
    baseline = patient_table([("4", 0.4), ("14", 0.4)])

    with pytest.raises(EvaluationError, match="only in method"):
        paired_patient_deltas(method, baseline)


def test_a_duplicated_patient_is_refused() -> None:
    method = patient_table([("4", 0.5), ("4", 0.6)])
    baseline = patient_table([("4", 0.4)])

    with pytest.raises(EvaluationError, match="repeats subject"):
        paired_patient_deltas(method, baseline)


def test_mismatched_slice_counts_are_refused() -> None:
    """Different slice counts mean the two methods did not score the same data."""
    method = patient_table([("4", 0.5)], slice_count=10)
    baseline = patient_table([("4", 0.4)], slice_count=11)

    with pytest.raises(EvaluationError, match="slice_count disagrees"):
        paired_patient_deltas(method, baseline)


def test_mismatched_acquisition_group_is_refused() -> None:
    method = patient_table([("4", 0.5)])
    baseline = patient_table([("4", 0.4)])
    baseline.loc[0, "acquisition_group"] = "B"

    with pytest.raises(EvaluationError, match="acquisition_group disagrees"):
        paired_patient_deltas(method, baseline)


def test_a_missing_metric_column_is_refused() -> None:
    method = patient_table([("4", 0.5)]).drop(columns=["mean_body_ssim"])
    baseline = patient_table([("4", 0.4)])

    with pytest.raises(EvaluationError, match="missing column"):
        paired_patient_deltas(method, baseline)


def test_an_empty_table_is_refused() -> None:
    with pytest.raises(EvaluationError, match="is empty"):
        paired_patient_deltas(pd.DataFrame(), patient_table([("4", 0.4)]))


# --------------------------------------------------------------------------
# Summarising the deltas
# --------------------------------------------------------------------------


def test_the_summary_counts_improvements_by_direction() -> None:
    """Four patients better, two worse, on a higher-is-better metric."""
    method = patient_table([(str(i), value) for i, value in enumerate([1, 1, 1, 1, 0, 0], 1)])
    baseline = patient_table([(str(i), 0.5) for i in range(1, 7)])

    summary = summarise_paired_deltas(paired_patient_deltas(method, baseline))

    assert summary["patients"] == 6
    assert summary["body_ssim"]["count_improved"] == 4
    assert summary["body_ssim"]["count_worsened"] == 2
    assert summary["body_ssim"]["count_tied"] == 0
    # The same raw deltas are improvements the other way round for MAE.
    assert summary["body_mae"]["count_improved"] == 2
    assert summary["body_mae"]["count_worsened"] == 4


def test_the_summary_records_which_sign_is_an_improvement() -> None:
    summary = summarise_paired_deltas(
        paired_patient_deltas(patient_table([("4", 0.5)]), patient_table([("4", 0.4)]))
    )

    assert summary["body_psnr"]["improvement_sign"] == "positive"
    assert summary["body_mae"]["improvement_sign"] == "negative"


def test_ties_are_counted_separately() -> None:
    method = patient_table([("4", 0.5), ("14", 0.5)])
    baseline = patient_table([("4", 0.5), ("14", 0.4)])

    summary = summarise_paired_deltas(paired_patient_deltas(method, baseline))

    assert summary["body_ssim"]["count_tied"] == 1
    assert summary["body_ssim"]["count_improved"] == 1
    assert summary["body_ssim"]["count_worsened"] == 0


def test_the_summary_reports_the_spread_of_the_deltas() -> None:
    values = [0.1, 0.2, 0.3]
    method = patient_table([(str(i), 0.5 + v) for i, v in enumerate(values, 1)])
    baseline = patient_table([(str(i), 0.5) for i in range(1, 4)])

    summary = summarise_paired_deltas(paired_patient_deltas(method, baseline))["body_ssim"]

    assert summary["mean"] == pytest.approx(0.2)
    assert summary["median"] == pytest.approx(0.2)
    assert summary["min"] == pytest.approx(0.1)
    assert summary["max"] == pytest.approx(0.3)
    assert summary["std"] == pytest.approx(np.std(values, ddof=1))


def test_a_single_patient_reports_undefined_spread_as_null() -> None:
    summary = summarise_paired_deltas(
        paired_patient_deltas(patient_table([("4", 0.6)]), patient_table([("4", 0.5)]))
    )

    assert summary["body_ssim"]["std"] is None


def test_a_mean_delta_can_hide_opposite_patient_outcomes() -> None:
    """Why the counts are reported and not just the mean.

    Three patients helped a lot and three harmed a lot averages to nothing,
    and looks identical to a method that did nothing at all.
    """
    swinging = paired_patient_deltas(
        patient_table([(str(i), v) for i, v in enumerate([0.9, 0.9, 0.9, 0.1, 0.1, 0.1], 1)]),
        patient_table([(str(i), 0.5) for i in range(1, 7)]),
    )
    inert = paired_patient_deltas(
        patient_table([(str(i), 0.5) for i in range(1, 7)]),
        patient_table([(str(i), 0.5) for i in range(1, 7)]),
    )

    swinging_summary = summarise_paired_deltas(swinging)["body_ssim"]
    inert_summary = summarise_paired_deltas(inert)["body_ssim"]

    assert swinging_summary["mean"] == pytest.approx(inert_summary["mean"], abs=1e-12)
    assert (swinging_summary["count_improved"], swinging_summary["count_worsened"]) == (3, 3)
    assert (inert_summary["count_improved"], inert_summary["count_worsened"]) == (0, 0)


def test_summarising_an_empty_delta_table_is_refused() -> None:
    with pytest.raises(EvaluationError, match="empty delta table"):
        summarise_paired_deltas(pd.DataFrame())
