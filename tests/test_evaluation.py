"""Tests for the evaluation policy: body mask, aggregation, hold-out gate.

Fully synthetic. The body mask is exercised on HU phantoms rather than on CHAOS
slices, and the aggregation on small hand-built tables where the right answer
can be read off by eye.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ct_restoration.config import load_config
from ct_restoration.evaluation import (
    DEVELOPMENT_SPLITS,
    METRIC_COLUMNS,
    SEALED_SPLITS,
    AggregationSettings,
    BodyMaskSettings,
    EvaluationConfig,
    EvaluationError,
    HeldOutSplitError,
    aggregate_slices_to_patients,
    body_mask_from_hu,
    group_breakdown,
    require_development_split,
    slice_weighted_summary,
    ssim_interior_mask,
    summarise_patients,
)


def hu_phantom(
    size: int = 512,
    radius: int = 150,
    air_hole_radius: int = 0,
    speck: bool = False,
) -> np.ndarray:
    """An HU slice: soft tissue disc on air, optionally with extras."""
    image = np.full((size, size), -1000.0)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    distance = (grid_y - size // 2) ** 2 + (grid_x - size // 2) ** 2
    image[distance < radius**2] = 50.0
    if air_hole_radius:
        image[distance < air_hole_radius**2] = -900.0
    if speck:
        image[10:40, 10:40] = 60.0
    return image


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_committed_evaluation_config_matches_the_defaults() -> None:
    """The tracked YAML is the canonical policy; code defaults must agree."""
    loaded = EvaluationConfig.from_mapping(load_config("evaluation.yaml"))

    assert loaded == EvaluationConfig()
    assert loaded.data_range == 1.0
    assert loaded.ssim.win_size == 11
    assert loaded.ssim.sigma == 1.5
    assert loaded.ssim.gaussian_weights is True
    assert loaded.ssim.use_sample_covariance is False
    assert loaded.body_mask.threshold_hu == -500.0
    assert loaded.body_mask.connectivity == 8
    assert loaded.aggregation.primary_unit == "patient"


def test_evaluation_config_refuses_unknown_keys() -> None:
    document = load_config("evaluation.yaml")["evaluation"]

    with pytest.raises(EvaluationError, match="unrecognised key"):
        EvaluationConfig.from_mapping({**document, "extra": 1})


def test_evaluation_config_refuses_missing_keys() -> None:
    with pytest.raises(EvaluationError, match="missing key"):
        EvaluationConfig.from_mapping({"data_range": 1.0})


def test_evaluation_config_refuses_an_unknown_ssim_key() -> None:
    document = load_config("evaluation.yaml")["evaluation"]
    broken = {**document, "ssim": {**document["ssim"], "truncate": 3.5}}

    with pytest.raises(EvaluationError, match="'ssim' has unrecognised key"):
        EvaluationConfig.from_mapping(broken)


def test_a_disagreeing_data_range_is_refused() -> None:
    from ct_restoration.metrics import SsimSettings

    with pytest.raises(EvaluationError, match="two dynamic ranges"):
        EvaluationConfig(data_range=2.0, ssim=SsimSettings(data_range=1.0))


@pytest.mark.parametrize("connectivity", [4, 6, 0])
def test_a_non_eight_connectivity_is_refused(connectivity) -> None:
    with pytest.raises(EvaluationError, match="connectivity must be 8"):
        BodyMaskSettings(connectivity=connectivity)


def test_an_interpolating_mask_resize_is_refused() -> None:
    """Any interpolation would produce fractional values in a binary mask."""
    with pytest.raises(EvaluationError, match="must be 'nearest'"):
        BodyMaskSettings(resize_interpolation="area")


def test_a_slice_primary_unit_is_refused() -> None:
    with pytest.raises(EvaluationError, match="not an independent experimental unit"):
        AggregationSettings(primary_unit="slice")


# --------------------------------------------------------------------------
# Body mask
# --------------------------------------------------------------------------


def test_the_mask_finds_the_body_disc() -> None:
    mask = body_mask_from_hu(hu_phantom(), (256, 256))

    assert mask.dtype == bool
    assert mask.shape == (256, 256)
    assert mask[128, 128]
    assert not mask[0, 0]
    # A radius-150 disc in a 512 frame is about pi*150^2/512^2 of the area.
    assert mask.mean() == pytest.approx(np.pi * 150**2 / 512**2, abs=0.01)


def test_a_smaller_disconnected_object_is_removed() -> None:
    """The table rail and stray specks must not join the evaluation region."""
    mask = body_mask_from_hu(hu_phantom(speck=True), (256, 256))

    assert mask[128, 128]
    assert not mask[10, 10]


def test_an_enclosed_air_hole_is_filled() -> None:
    """Bowel gas and lung base are inside the body and must be evaluated."""
    with_hole = body_mask_from_hu(hu_phantom(air_hole_radius=40), (256, 256))
    without_hole = body_mask_from_hu(hu_phantom(), (256, 256))

    assert with_hole[128, 128]
    assert int(with_hole.sum()) == int(without_hole.sum())


def test_background_is_excluded() -> None:
    mask = body_mask_from_hu(hu_phantom(), (256, 256))

    assert not mask[:8, :].any()
    assert not mask[-8:, :].any()


def test_the_threshold_separates_air_from_tissue() -> None:
    """Everything below -500 HU that is not enclosed stays outside the mask."""
    image = np.full((64, 64), -600.0)
    image[20:40, 20:40] = -400.0

    mask = body_mask_from_hu(image, (64, 64))

    assert mask[30, 30]
    assert not mask[5, 5]


def test_resizing_uses_nearest_neighbour_semantics() -> None:
    """A downsized binary mask stays binary, with no interpolated edge."""
    mask = body_mask_from_hu(hu_phantom(size=512, radius=150), (256, 256))

    assert mask.dtype == bool
    assert set(np.unique(mask).tolist()) <= {False, True}


def test_the_mask_can_be_built_at_another_size() -> None:
    assert body_mask_from_hu(hu_phantom(), (128, 128)).shape == (128, 128)
    assert body_mask_from_hu(hu_phantom(), (256, 192)).shape == (256, 192)


def test_a_slice_with_no_body_is_refused() -> None:
    with pytest.raises(EvaluationError, match="contains no body"):
        body_mask_from_hu(np.full((64, 64), -1000.0), (64, 64))


def test_the_mask_does_not_modify_its_input() -> None:
    image = hu_phantom()
    before = image.copy()

    body_mask_from_hu(image, (256, 256))

    assert np.array_equal(image, before)


@pytest.mark.parametrize("shape", [(64,), (4, 4, 3), ()])
def test_a_non_2d_hu_slice_is_refused(shape) -> None:
    with pytest.raises(EvaluationError, match="2D HU slice"):
        body_mask_from_hu(np.full(shape, 50.0), (64, 64))


def test_a_non_finite_hu_slice_is_refused() -> None:
    image = hu_phantom(size=64, radius=20)
    image[0, 0] = np.nan

    with pytest.raises(EvaluationError, match="finite HU"):
        body_mask_from_hu(image, (64, 64))


# --------------------------------------------------------------------------
# SSIM interior mask
# --------------------------------------------------------------------------


def test_the_interior_is_a_subset_of_the_body() -> None:
    body = body_mask_from_hu(hu_phantom(), (256, 256))

    interior = ssim_interior_mask(body, 11)

    assert interior.dtype == bool
    assert interior.any()
    assert not (interior & ~body).any()
    assert interior.sum() < body.sum()


def test_the_interior_excludes_pixels_whose_window_leaves_the_body() -> None:
    body = np.zeros((40, 40), dtype=bool)
    body[10:30, 10:30] = True  # a 20x20 square

    interior = ssim_interior_mask(body, 11)

    # An 11x11 window needs 5 pixels of margin on every side.
    assert interior[15:25, 15:25].all()
    assert not interior[10, 10]
    assert int(interior.sum()) == 10 * 10


def test_the_interior_excludes_the_image_border() -> None:
    """scikit-image cannot compute a meaningful SSIM there either."""
    body = np.ones((40, 40), dtype=bool)

    interior = ssim_interior_mask(body, 11)

    assert not interior[:5, :].any()
    assert not interior[:, :5].any()
    assert interior[5:35, 5:35].all()


def test_a_body_thinner_than_the_window_gives_an_empty_interior() -> None:
    """Allowed to be empty here; the caller must handle it explicitly."""
    body = np.zeros((40, 40), dtype=bool)
    body[20:23, 20:23] = True

    assert not ssim_interior_mask(body, 11).any()


def test_the_interior_does_not_modify_its_input() -> None:
    body = body_mask_from_hu(hu_phantom(), (256, 256))
    before = body.copy()

    ssim_interior_mask(body, 11)

    assert np.array_equal(body, before)


@pytest.mark.parametrize("win_size", [2, 10, 1, 0])
def test_an_invalid_interior_window_is_refused(win_size) -> None:
    with pytest.raises(EvaluationError, match="odd integer"):
        ssim_interior_mask(np.ones((20, 20), dtype=bool), win_size)


# --------------------------------------------------------------------------
# Aggregation: the patient is the experimental unit
# --------------------------------------------------------------------------


def metric_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """A slice table where every metric column carries the same value."""
    records = []
    for subject_id, value in rows:
        record = {
            "subject_id": subject_id,
            "source_archive": "Train_Sets",
            "acquisition_group": "A",
        }
        record.update({name: value for name in METRIC_COLUMNS})
        records.append(record)
    return pd.DataFrame(records)


def test_slice_counts_do_not_buy_extra_primary_weight() -> None:
    """The distinction this whole benchmark's aggregation exists to protect.

    Patient A contributes one slice scoring 0. Patient B contributes nine
    slices each scoring 1. Pooling slices gives 0.9, which says the result is
    almost entirely determined by whichever patient happened to be scanned
    with a finer slice spacing. Weighting patients equally gives 0.5, which is
    what the primary evaluator must return.
    """
    frame = metric_frame([("A", 0.0)] + [("B", 1.0)] * 9)

    patients = aggregate_slices_to_patients(frame)
    primary = summarise_patients(patients)
    secondary = slice_weighted_summary(frame)

    assert list(patients["slice_count"]) == [1, 9]
    assert primary["full_mae"]["mean"] == pytest.approx(0.5)
    assert secondary["full_mae"]["mean"] == pytest.approx(0.9)
    assert primary["full_mae"]["mean"] != secondary["full_mae"]["mean"]


def test_patient_means_are_correct() -> None:
    frame = metric_frame([("A", 0.2), ("A", 0.4), ("B", 1.0)])

    patients = aggregate_slices_to_patients(frame)

    by_subject = patients.set_index("subject_id")
    assert by_subject.loc["A", "mean_full_mae"] == pytest.approx(0.3)
    assert by_subject.loc["B", "mean_full_mae"] == pytest.approx(1.0)
    assert by_subject.loc["A", "slice_count"] == 2


def test_the_split_mean_weights_every_patient_equally() -> None:
    frame = metric_frame([("A", 0.0)] * 100 + [("B", 1.0)] * 2 + [("C", 0.5)])

    primary = summarise_patients(aggregate_slices_to_patients(frame))

    assert primary["patients"] == 3
    assert primary["full_mae"]["mean"] == pytest.approx((0.0 + 1.0 + 0.5) / 3)


def test_the_summary_reports_spread_across_patients() -> None:
    frame = metric_frame([("A", 0.0), ("B", 1.0), ("C", 0.5)])

    primary = summarise_patients(aggregate_slices_to_patients(frame))

    assert primary["full_mae"]["min"] == pytest.approx(0.0)
    assert primary["full_mae"]["max"] == pytest.approx(1.0)
    assert primary["full_mae"]["median"] == pytest.approx(0.5)
    assert primary["full_mae"]["std"] == pytest.approx(np.std([0.0, 1.0, 0.5], ddof=1))


def test_a_single_patient_reports_undefined_spread_as_null() -> None:
    """None, not NaN, so the tracked summary stays strictly valid JSON."""
    primary = summarise_patients(aggregate_slices_to_patients(metric_frame([("A", 0.4)])))

    assert primary["full_mae"]["std"] is None
    assert primary["full_mae"]["mean"] == pytest.approx(0.4)


def test_inconsistent_patient_attributes_are_refused() -> None:
    frame = metric_frame([("A", 0.1), ("A", 0.2)])
    frame.loc[1, "acquisition_group"] = "B"

    with pytest.raises(EvaluationError, match="inconsistent acquisition_group"):
        aggregate_slices_to_patients(frame)


def test_aggregating_an_empty_table_is_refused() -> None:
    with pytest.raises(EvaluationError, match="empty slice table"):
        aggregate_slices_to_patients(pd.DataFrame())


def test_a_missing_metric_column_is_refused() -> None:
    frame = metric_frame([("A", 0.1)]).drop(columns=["body_ssim"])

    with pytest.raises(EvaluationError, match="missing metric column"):
        aggregate_slices_to_patients(frame)


def test_the_group_breakdown_is_patient_weighted() -> None:
    frame = metric_frame([("A", 0.0), ("B", 1.0)])
    frame.loc[frame["subject_id"] == "B", "acquisition_group"] = "B"

    breakdown = group_breakdown(aggregate_slices_to_patients(frame))

    assert breakdown["A"]["patients"] == 1
    assert breakdown["A"]["full_mae"]["mean"] == pytest.approx(0.0)
    assert breakdown["B"]["full_mae"]["mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Hold-out discipline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "validation"])
def test_development_splits_are_accepted(split) -> None:
    assert require_development_split(split) == split


@pytest.mark.parametrize("split", ["test", "stress"])
def test_sealed_splits_are_refused(split) -> None:
    """There must be no easy accidental path to spending held-out data."""
    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        require_development_split(split)


@pytest.mark.parametrize("split", ["", "Test", "TRAIN", "holdout", "val"])
def test_unknown_splits_are_refused(split) -> None:
    with pytest.raises(EvaluationError, match="Unknown split"):
        require_development_split(split)


def test_the_sealed_and_development_sets_are_disjoint_and_complete() -> None:
    assert set(DEVELOPMENT_SPLITS) & set(SEALED_SPLITS) == set()
    assert set(DEVELOPMENT_SPLITS) | set(SEALED_SPLITS) == {
        "train",
        "validation",
        "test",
        "stress",
    }


def test_a_sealed_split_error_is_an_evaluation_error() -> None:
    """Callers catching the general error still cannot proceed past the gate."""
    assert issubclass(HeldOutSplitError, EvaluationError)
