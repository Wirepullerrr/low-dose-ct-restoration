"""Tests for the supervised restoration Dataset.

Fully synthetic: real DICOM files written to a temporary directory, never the
CHAOS download.

The risk in a Dataset is not that it crashes. It is that it quietly returns
something slightly different from what the benchmark measures - a renormalized
tensor, a fresh noise realization each epoch, a mask derived from the target -
and training proceeds looking perfectly healthy on numbers that no longer mean
what the report says they mean. Almost everything here checks an equality that
would still "work" if it were broken.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from ct_restoration.data.dataset import (
    FORBIDDEN_SAMPLE_FIELDS,
    PAIR_GENERATION_VERSION,
    SAMPLE_FIELDS,
    DatasetError,
    RestorationDataset,
    canonical_order,
    development_dataset,
    development_rows,
    sample_key_is_portable,
)
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like
from ct_restoration.data.dicom import load_ct_hu
from ct_restoration.data.preprocessing import preprocess_ct_slice
from ct_restoration.evaluation import EvaluationError, HeldOutSplitError


@pytest.fixture
def train_dataset(synthetic_cohort) -> RestorationDataset:
    root, manifest, preprocessing = synthetic_cohort
    return development_dataset("train", manifest, root, preprocessing)


@pytest.fixture
def validation_dataset(synthetic_cohort) -> RestorationDataset:
    root, manifest, preprocessing = synthetic_cohort
    return development_dataset("validation", manifest, root, preprocessing)


def direct_pair(root, preprocessing, key, degradation=None):
    """The clean/degraded pair built by calling the frozen functions directly."""
    image_hu = load_ct_hu(root / key)
    clean = preprocess_ct_slice(
        image_hu,
        window_center=preprocessing["window_center"],
        window_width=preprocessing["window_width"],
        size=tuple(preprocessing["image_size"]),
        interpolation=preprocessing["interpolation"],
    )
    return clean, degrade_low_dose_like(clean, key, degradation or DegradationConfig())


# --------------------------------------------------------------------------
# Size and canonical ordering
# --------------------------------------------------------------------------


def test_the_dataset_holds_every_slice_of_its_split(train_dataset, validation_dataset) -> None:
    assert len(train_dataset) == 3 + 5 + 2 + 6
    assert len(validation_dataset) == 4 + 3
    assert train_dataset.patients == ("2", "3", "21", "31")
    assert validation_dataset.patients == ("4", "23")


def test_samples_are_in_natural_subject_then_slice_order(train_dataset) -> None:
    ordered = list(zip(train_dataset.subject_ids, _slice_indices(train_dataset), strict=True))

    # Natural subject order puts 3 before 21, which a plain string sort would not.
    assert [subject for subject, _ in ordered] == ["2"] * 3 + ["3"] * 5 + ["21"] * 2 + ["31"] * 6
    for subject in train_dataset.patients:
        indices = [index for name, index in ordered if name == subject]
        assert indices == sorted(indices)


def _slice_indices(dataset: RestorationDataset) -> list[int]:
    return [dataset[position]["geometric_slice_index"] for position in range(len(dataset))]


def test_shuffling_the_input_rows_cannot_change_the_order(synthetic_cohort) -> None:
    """An accidentally reordered CSV must not rename which slice index 0 is."""
    root, manifest, preprocessing = synthetic_cohort
    shuffled = manifest.sample(frac=1.0, random_state=7).reset_index(drop=True)

    ordered = development_dataset("train", manifest, root, preprocessing)
    reordered = development_dataset("train", shuffled, root, preprocessing)

    assert ordered.sample_keys == reordered.sample_keys
    assert ordered.subject_ids == reordered.subject_ids


def test_reversed_input_rows_cannot_change_the_order(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort
    reversed_rows = manifest.iloc[::-1].reset_index(drop=True)

    assert (
        development_dataset("train", reversed_rows, root, preprocessing).sample_keys
        == development_dataset("train", manifest, root, preprocessing).sample_keys
    )


def test_canonical_order_refuses_a_repeated_sample_key(synthetic_cohort) -> None:
    _, manifest, _ = synthetic_cohort
    doubled = pd.concat([manifest, manifest.head(1)], ignore_index=True)

    with pytest.raises(DatasetError, match="repeat sample key"):
        canonical_order(doubled)


@pytest.mark.parametrize("column", ["subject_id", "relative_dicom_path", "geometric_slice_index"])
def test_canonical_order_refuses_missing_columns(synthetic_cohort, column) -> None:
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="missing column"):
        canonical_order(manifest.drop(columns=[column]))


def test_an_empty_row_set_is_refused(synthetic_cohort) -> None:
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="empty"):
        canonical_order(manifest.iloc[0:0])


# --------------------------------------------------------------------------
# The sample contract
# --------------------------------------------------------------------------


def test_a_sample_carries_exactly_the_declared_fields(train_dataset) -> None:
    sample = train_dataset[0]

    assert tuple(sample) == SAMPLE_FIELDS


def test_a_sample_never_carries_evaluation_only_information(train_dataset) -> None:
    """The body mask is derived from the target; a model must not see it."""
    for position in range(len(train_dataset)):
        sample = train_dataset[position]
        assert set(sample).isdisjoint(FORBIDDEN_SAMPLE_FIELDS)
        assert "body_mask" not in sample
        assert "clean_hu" not in sample


def test_metadata_is_enough_to_audit_the_pairing(train_dataset) -> None:
    sample = train_dataset[0]

    assert sample["subject_id"] == "2"
    assert sample["source_archive"] == "Train_Sets"
    assert sample["acquisition_group"] == "B"
    assert isinstance(sample["geometric_slice_index"], int)
    assert sample["sample_key"].endswith(".dcm")


def test_the_sample_key_is_the_portable_relative_posix_path(
    train_dataset, synthetic_cohort
) -> None:
    """One input to the derivation of the slice's degradation seed.

    Not the numeric seed itself, but the key it is derived from, so a
    machine-local spelling would derive a different seed for the same image.
    """
    root, _, _ = synthetic_cohort
    for key in train_dataset.sample_keys:
        assert sample_key_is_portable(key)
        assert "\\" not in key
        assert str(root) not in key


@pytest.mark.parametrize("key", ["/abs/path.dcm", "C:/data/x.dcm", "Train_Sets\\CT\\2\\a.dcm", ""])
def test_non_portable_keys_are_recognised(key) -> None:
    assert not sample_key_is_portable(key)


# --------------------------------------------------------------------------
# The tensor contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("half", ["degraded", "clean"])
def test_both_halves_are_single_channel_float32_in_range(train_dataset, half) -> None:
    for position in range(len(train_dataset)):
        tensor = train_dataset[position][half]

        assert tensor.shape == (1, 16, 16)
        assert tensor.dtype is torch.float32
        assert bool(torch.isfinite(tensor).all())
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0


def test_the_tensor_owns_its_storage(train_dataset) -> None:
    """Writing to a returned tensor must not reach any shared buffer."""
    sample = train_dataset[0]
    before = sample["clean"].clone()
    sample["clean"][0, 0, 0] = 0.5

    assert train_dataset[0]["clean"].equal(before)


def test_the_dataset_does_not_renormalize(train_dataset, synthetic_cohort) -> None:
    """[0,1] stays [0,1]: no [-1,1] shift, no z-scoring, no per-image min/max."""
    root, _, preprocessing = synthetic_cohort
    key = train_dataset.sample_keys[0]
    clean, _ = direct_pair(root, preprocessing, key)

    assert train_dataset[0]["clean"].squeeze(0).numpy() == pytest.approx(clean)


# --------------------------------------------------------------------------
# The pair is a pure function of the slice
# --------------------------------------------------------------------------


def test_the_clean_half_equals_the_frozen_preprocessing(train_dataset, synthetic_cohort) -> None:
    root, _, preprocessing = synthetic_cohort
    for position in range(len(train_dataset)):
        sample = train_dataset[position]
        clean, _ = direct_pair(root, preprocessing, sample["sample_key"])

        assert sample["clean"].squeeze(0).numpy().tobytes() == clean.tobytes()


def test_the_degraded_half_equals_the_frozen_degradation(train_dataset, synthetic_cohort) -> None:
    """No new noise realization: the Dataset reproduces the frozen corruption."""
    root, _, preprocessing = synthetic_cohort
    for position in range(len(train_dataset)):
        sample = train_dataset[position]
        _, degraded = direct_pair(root, preprocessing, sample["sample_key"])

        assert sample["degraded"].squeeze(0).numpy().tobytes() == degraded.tobytes()


def test_repeated_access_returns_byte_identical_pairs(train_dataset) -> None:
    for position in range(len(train_dataset)):
        first = train_dataset[position]
        second = train_dataset[position]

        assert first["clean"].numpy().tobytes() == second["clean"].numpy().tobytes()
        assert first["degraded"].numpy().tobytes() == second["degraded"].numpy().tobytes()


def test_indexing_another_slice_first_changes_nothing(train_dataset) -> None:
    alone = train_dataset[2]
    train_dataset[7]
    train_dataset[0]
    after = train_dataset[2]

    assert alone["degraded"].equal(after["degraded"])
    assert alone["clean"].equal(after["clean"])


def test_the_global_numpy_rng_cannot_alter_a_pair(train_dataset) -> None:
    np.random.seed(1)
    first = train_dataset[3]

    np.random.seed(999)
    np.random.random(10_000)
    second = train_dataset[3]

    assert first["degraded"].equal(second["degraded"])


def test_the_global_torch_rng_cannot_alter_a_pair(train_dataset) -> None:
    torch.manual_seed(1)
    first = train_dataset[3]

    torch.manual_seed(999)
    torch.rand(1000)
    second = train_dataset[3]

    assert first["degraded"].equal(second["degraded"])


def test_two_slices_of_one_patient_get_different_noise(train_dataset) -> None:
    """Seeded per sample key, so a patient is not one repeated realization."""
    first = train_dataset[0]
    second = train_dataset[1]

    assert first["subject_id"] == second["subject_id"]
    assert not first["degraded"].equal(second["degraded"])


def test_the_degradation_differs_from_the_clean_target(train_dataset) -> None:
    sample = train_dataset[0]

    assert not sample["degraded"].equal(sample["clean"])


# --------------------------------------------------------------------------
# The Dataset does not write through to its caller
# --------------------------------------------------------------------------


def test_indexing_does_not_mutate_the_caller_manifest_or_config(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort
    manifest_before = manifest.copy(deep=True)
    preprocessing_before = copy.deepcopy(preprocessing)

    dataset = development_dataset("train", manifest, root, preprocessing)
    for position in range(len(dataset)):
        dataset[position]

    pd.testing.assert_frame_equal(manifest, manifest_before)
    assert preprocessing == preprocessing_before


def test_mutating_the_caller_config_afterwards_cannot_change_the_dataset(
    synthetic_cohort,
) -> None:
    root, manifest, preprocessing = synthetic_cohort
    dataset = development_dataset("train", manifest, root, preprocessing)
    before = dataset[0]["clean"].clone()

    preprocessing["window_width"] = 4000

    assert dataset[0]["clean"].equal(before)


# --------------------------------------------------------------------------
# Hold-out discipline
# --------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "validation"])
def test_development_splits_are_accepted(synthetic_cohort, split) -> None:
    root, manifest, preprocessing = synthetic_cohort

    assert len(development_dataset(split, manifest, root, preprocessing)) > 0


@pytest.mark.parametrize("split", ["test", "stress"])
def test_sealed_splits_are_refused(synthetic_cohort, split) -> None:
    """PyTorch existing is not a reason to loosen the hold-out gate."""
    root, manifest, preprocessing = synthetic_cohort

    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        development_dataset(split, manifest, root, preprocessing)


@pytest.mark.parametrize("split", ["test", "stress"])
def test_sealed_splits_are_refused_before_any_file_is_opened(synthetic_cohort, split) -> None:
    _, manifest, _ = synthetic_cohort

    with pytest.raises(HeldOutSplitError):
        development_rows(manifest, split)


def test_an_unknown_split_is_refused(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort

    with pytest.raises(EvaluationError, match="Unknown split"):
        development_dataset("holdout", manifest, root, preprocessing)


def test_a_split_with_no_rows_is_refused(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort
    without_validation = manifest[manifest["split"] != "validation"]

    with pytest.raises(DatasetError, match="no rows for split"):
        development_dataset("validation", without_validation, root, preprocessing)


# --------------------------------------------------------------------------
# Indexing and configuration guards
# --------------------------------------------------------------------------


def test_an_out_of_range_index_is_refused(train_dataset) -> None:
    with pytest.raises(IndexError):
        train_dataset[len(train_dataset)]


def test_a_negative_index_addresses_from_the_end(train_dataset) -> None:
    assert train_dataset[-1]["sample_key"] == train_dataset[len(train_dataset) - 1]["sample_key"]


@pytest.mark.parametrize("index", [1.0, "0", True, None])
def test_a_non_integer_index_is_refused(train_dataset, index) -> None:
    with pytest.raises(DatasetError, match="index must be an integer"):
        train_dataset[index]


@pytest.mark.parametrize("field", ["window_center", "window_width", "image_size", "interpolation"])
def test_an_incomplete_preprocessing_config_is_refused(synthetic_cohort, field) -> None:
    root, manifest, preprocessing = synthetic_cohort
    del preprocessing[field]

    with pytest.raises(DatasetError, match=field):
        development_dataset("train", manifest, root, preprocessing)


# --------------------------------------------------------------------------
# Worker safety
# --------------------------------------------------------------------------


def test_the_dataset_survives_a_pickle_round_trip(train_dataset) -> None:
    """A DataLoader worker process reconstructs the Dataset by unpickling it."""
    revived = pickle.loads(pickle.dumps(train_dataset))

    assert revived.sample_keys == train_dataset.sample_keys
    assert revived[0]["degraded"].equal(train_dataset[0]["degraded"])
    assert revived[0]["clean"].equal(train_dataset[0]["clean"])


def test_the_pair_generation_contract_is_versioned() -> None:
    assert PAIR_GENERATION_VERSION == "frozen_clean_plus_degradation_v1"


def test_a_nondefault_degradation_config_reaches_the_pair(synthetic_cohort) -> None:
    """The Dataset uses the config it was given, not a hard-coded default."""
    root, manifest, preprocessing = synthetic_cohort
    canonical = development_dataset("train", manifest, root, preprocessing)
    other = development_dataset(
        "train", manifest, root, preprocessing, DegradationConfig(global_seed=7)
    )

    assert canonical[0]["clean"].equal(other[0]["clean"])
    assert not canonical[0]["degraded"].equal(other[0]["degraded"])


# --------------------------------------------------------------------------
# The audit command's own policy logic
# --------------------------------------------------------------------------


def _audit_module():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
    spec = importlib.util.spec_from_file_location("audit_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_dataset"] = module
    spec.loader.exec_module(module)
    return module


def test_the_audit_has_no_split_option() -> None:
    """No way to point the audit at test or stress."""
    audit = _audit_module()

    assert "split" not in {action.dest for action in audit.build_parser()._actions}


def test_the_audit_refuses_limit_to_the_canonical_summary() -> None:
    audit = _audit_module()

    with pytest.raises(ValueError, match="debug-only"):
        audit.check_output_policy(30, audit.CANONICAL_SUMMARY)


def test_the_audit_allows_a_full_run_to_the_canonical_summary() -> None:
    audit = _audit_module()

    audit.check_output_policy(0, audit.CANONICAL_SUMMARY)


def test_the_audit_allows_limit_elsewhere(tmp_path) -> None:
    audit = _audit_module()

    audit.check_output_policy(30, tmp_path / "debug.json")


def test_the_audit_sequence_digest_is_order_sensitive() -> None:
    """Two epochs drawing the same slices in different orders must differ."""
    audit = _audit_module()

    assert audit.sequence_digest(["a", "b"]) != audit.sequence_digest(["b", "a"])
    assert audit.sequence_digest(["a", "b"]) == audit.sequence_digest(["a", "b"])


def test_the_audit_probe_positions_are_deterministic_and_spread() -> None:
    audit = _audit_module()

    assert audit.probe_positions(885, 8) == audit.probe_positions(885, 8)
    assert audit.probe_positions(885, 8)[0] == 0
    assert audit.probe_positions(885, 8)[-1] == 884
    assert audit.probe_positions(5, 8) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# The canonical slice-identity contract
# --------------------------------------------------------------------------
#
# The canonical order sorts on (subject_id, geometric_slice_index). Those two
# fields therefore have to identify a slice on their own: if a pair were
# duplicated, the tie would fall back to input row order and "canonical" would
# become a property of the CSV instead of the manifest.


def test_two_paths_claiming_one_subject_and_slice_index_are_refused(
    synthetic_cohort,
) -> None:
    root, manifest, _ = synthetic_cohort
    clashing = manifest.copy()
    first = clashing.index[0]
    clashing.loc[first, "relative_dicom_path"] = str(
        clashing.loc[first, "relative_dicom_path"]
    ).replace(".dcm", "_copy.dcm")
    clashing = pd.concat([manifest, clashing.loc[[first]]], ignore_index=True)

    # Distinct paths, so the sample-key check passes; the position clashes.
    assert clashing["relative_dicom_path"].is_unique
    with pytest.raises(DatasetError, match="to more than one"):
        canonical_order(clashing)


def test_a_repeated_sample_key_is_still_refused(synthetic_cohort) -> None:
    """The original duplicate-path rejection is retained."""
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="repeat sample key"):
        canonical_order(pd.concat([manifest, manifest.head(1)], ignore_index=True))


def test_the_frozen_manifest_satisfies_both_uniqueness_rules() -> None:
    """The real tracked manifest must keep parsing unchanged."""
    frame = pd.read_csv("data/splits/chaos_slice_manifest.csv", dtype={"subject_id": str})

    assert frame["relative_dicom_path"].is_unique
    assert not frame.duplicated(subset=["subject_id", "geometric_slice_index"]).any()


# --------------------------------------------------------------------------
# Slice indices are not silently coerced
# --------------------------------------------------------------------------


def _with_slice_index(manifest, value):
    changed = manifest.copy()
    changed["geometric_slice_index"] = changed["geometric_slice_index"].astype(object)
    changed.loc[changed.index[0], "geometric_slice_index"] = value
    return changed


@pytest.mark.parametrize("value", [3.8, -0.5, "0", "abc", None, [0]])
def test_a_malformed_slice_index_is_refused(synthetic_cohort, value) -> None:
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="geometric_slice_index"):
        canonical_order(_with_slice_index(manifest, value))


@pytest.mark.parametrize("value", [True, False])
def test_a_bool_slice_index_is_refused(synthetic_cohort, value) -> None:
    """``True`` would otherwise read as slice 1 and reorder a patient."""
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="must be an integer, got bool"):
        canonical_order(_with_slice_index(manifest, value))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_missing_or_infinite_slice_index_is_refused(synthetic_cohort, value) -> None:
    """A missing manifest cell reaches pandas as NaN in a float column."""
    _, manifest, _ = synthetic_cohort

    with pytest.raises(DatasetError, match="finite integer"):
        canonical_order(_with_slice_index(manifest, value))


def test_a_whole_float_slice_index_is_accepted(synthetic_cohort) -> None:
    """pandas promotes an int column to float if one cell is missing.

    Converting an exactly integral float back is lossless, so it is allowed;
    3.8 in the test above is not, and is refused.
    """
    _, manifest, _ = synthetic_cohort
    promoted = manifest.copy()
    promoted["geometric_slice_index"] = promoted["geometric_slice_index"].astype(float)

    assert list(canonical_order(promoted)["geometric_slice_index"]) == list(
        canonical_order(manifest)["geometric_slice_index"]
    )


def test_the_frozen_manifest_still_parses_as_integers() -> None:
    frame = pd.read_csv("data/splits/chaos_slice_manifest.csv", dtype={"subject_id": str})

    assert frame["geometric_slice_index"].dtype.kind == "i"
    assert (
        canonical_order(frame[frame["split"] == "validation"])["geometric_slice_index"].dtype.kind
        in "iu"
    )


# --------------------------------------------------------------------------
# Sample keys are checked where the Dataset is built, not only in the audit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "/Train_Sets/CT/2/a.dcm",
        "C:/Train_Sets/CT/2/a.dcm",
        r"Train_Sets\CT\2\a.dcm",
        "Train_Sets/../CT/2/a.dcm",
        "../a.dcm",
    ],
)
def test_a_non_portable_sample_key_is_refused_at_construction(synthetic_cohort, key) -> None:
    _, manifest, _ = synthetic_cohort
    broken = manifest.copy()
    broken.loc[broken.index[0], "relative_dicom_path"] = key

    with pytest.raises(DatasetError, match="non-portable sample key"):
        canonical_order(broken)


def test_a_parent_traversal_segment_is_not_portable() -> None:
    assert not sample_key_is_portable("Train_Sets/../CT/2/a.dcm")
    assert sample_key_is_portable("Train_Sets/CT/2/..hidden.dcm")


def test_every_frozen_manifest_key_is_portable() -> None:
    frame = pd.read_csv("data/splits/chaos_slice_manifest.csv", dtype={"subject_id": str})

    assert all(sample_key_is_portable(key) for key in frame["relative_dicom_path"])


# --------------------------------------------------------------------------
# image_size is not silently coerced
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size", [[256.7, 256], [256, 256.0], ["256", 256], [True, 256], [0, 256], [-8, 8]]
)
def test_a_malformed_image_size_is_refused(synthetic_cohort, size) -> None:
    root, manifest, preprocessing = synthetic_cohort
    preprocessing["image_size"] = size

    with pytest.raises(DatasetError, match="image_size"):
        development_dataset("train", manifest, root, preprocessing)


@pytest.mark.parametrize("size", [[16], [16, 16, 16], "1616", 16])
def test_an_image_size_that_is_not_a_pair_is_refused(synthetic_cohort, size) -> None:
    root, manifest, preprocessing = synthetic_cohort
    preprocessing["image_size"] = size

    with pytest.raises(DatasetError, match="image_size"):
        development_dataset("train", manifest, root, preprocessing)


def test_the_committed_preprocessing_image_size_is_accepted(synthetic_cohort) -> None:
    """[256, 256] from configs/baseline.yaml must behave exactly as before."""
    from ct_restoration.config import load_config

    root, manifest, preprocessing = synthetic_cohort
    committed = load_config("baseline.yaml")["preprocessing"]

    assert committed["image_size"] == [256, 256]
    dataset = development_dataset(
        "train", manifest, root, {**preprocessing, "image_size": committed["image_size"]}
    )
    assert dataset.image_size == (256, 256)


@pytest.mark.parametrize("size", [(16, 16), [16, 16]])
def test_a_numpy_integer_image_size_is_accepted(synthetic_cohort, size) -> None:
    root, manifest, preprocessing = synthetic_cohort
    preprocessing["image_size"] = [np.int64(size[0]), np.int32(size[1])]

    assert development_dataset("train", manifest, root, preprocessing).image_size == (16, 16)
