"""Tests for the training and validation DataLoader contracts.

Fully synthetic, on the miniature on-disk cohort.

The distinction these protect is that **batching changes grouping, not
sampling**. A loader that quietly dropped a ragged final batch, or reshuffled
under a different batch size, would still train and still report numbers - it
would just no longer be traversing what the audit says it traverses.
"""

from __future__ import annotations

import pytest
import torch

from ct_restoration.data.dataset import development_dataset, flatten_sample_keys
from ct_restoration.data.loaders import make_training_loader, make_validation_loader
from ct_restoration.data.sampling import PatientBalancedSampler

SEED = 2026


@pytest.fixture
def train_dataset(synthetic_cohort):
    root, manifest, preprocessing = synthetic_cohort
    return development_dataset("train", manifest, root, preprocessing)


@pytest.fixture
def validation_dataset(synthetic_cohort):
    root, manifest, preprocessing = synthetic_cohort
    return development_dataset("validation", manifest, root, preprocessing)


def keys_of(loader) -> list[str]:
    return flatten_sample_keys(list(loader))


# --------------------------------------------------------------------------
# The training loader
# --------------------------------------------------------------------------


def test_the_training_loader_is_driven_by_the_patient_balanced_sampler(train_dataset) -> None:
    loader = make_training_loader(train_dataset, batch_size=4, seed=SEED)

    assert isinstance(loader.sampler, PatientBalancedSampler)
    assert loader.sampler.epoch_size == len(train_dataset)


def test_the_training_loader_does_not_shuffle_on_its_own(train_dataset) -> None:
    """Ordering is the sampler's job; a second source would be unauditable."""
    loader = make_training_loader(train_dataset, batch_size=4, seed=SEED)

    assert keys_of(loader) == [
        train_dataset.sample_keys[index] for index in loader.sampler.epoch_indices(0)
    ]


def test_the_training_loader_keeps_the_ragged_final_batch(train_dataset) -> None:
    """drop_last would silently discard whichever patients landed at the end."""
    loader = make_training_loader(train_dataset, batch_size=7, seed=SEED)
    batches = list(loader)

    assert loader.drop_last is False
    assert sum(len(batch["sample_key"]) for batch in batches) == len(train_dataset)
    assert len(batches[-1]["sample_key"]) == len(train_dataset) % 7


def test_the_training_loader_yields_the_full_epoch_however_it_is_batched(
    train_dataset,
) -> None:
    for batch_size in (1, 3, 7, len(train_dataset), len(train_dataset) + 5):
        loader = make_training_loader(train_dataset, batch_size=batch_size, seed=SEED)

        assert len(keys_of(loader)) == len(train_dataset)


def test_training_batch_size_changes_grouping_not_sampling(train_dataset) -> None:
    """Same seed, same epoch, different batch size: identical ordered keys."""
    small = make_training_loader(train_dataset, batch_size=1, seed=SEED, epoch=2)
    large = make_training_loader(train_dataset, batch_size=7, seed=SEED, epoch=2)

    assert keys_of(small) == keys_of(large)


def test_advancing_the_epoch_changes_the_training_sequence(train_dataset) -> None:
    loader = make_training_loader(train_dataset, batch_size=4, seed=SEED, epoch=0)
    first = keys_of(loader)

    loader.sampler.set_epoch(1)
    second = keys_of(loader)

    loader.sampler.set_epoch(0)
    again = keys_of(loader)

    assert first != second
    assert first == again


def test_advancing_the_epoch_does_not_change_any_pair(train_dataset) -> None:
    """Sampling order may change between epochs; the corruption may not.

    The reference comes from the Dataset rather than from epoch 0, because a
    single epoch does not necessarily draw every slice: a long scan is
    sampled as a subset, so epoch 5 can legitimately show a slice epoch 0 did
    not.
    """
    reference = {
        train_dataset.sample_keys[position]: train_dataset[position]["degraded"]
        for position in range(len(train_dataset))
    }

    loader = make_training_loader(train_dataset, batch_size=1, seed=SEED, epoch=0)
    for epoch in (0, 5):
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            key = batch["sample_key"][0]
            assert batch["degraded"][0].equal(reference[key])


def test_the_training_loader_balances_patients_across_one_epoch(train_dataset) -> None:
    loader = make_training_loader(train_dataset, batch_size=4, seed=SEED)
    by_key = dict(zip(train_dataset.sample_keys, train_dataset.subject_ids, strict=True))

    drawn: dict[str, int] = {}
    for key in keys_of(loader):
        drawn[by_key[key]] = drawn.get(by_key[key], 0) + 1

    assert len(drawn) == len(train_dataset.patients)
    assert max(drawn.values()) - min(drawn.values()) <= 1


def test_a_training_batch_carries_both_halves_as_stacked_tensors(train_dataset) -> None:
    loader = make_training_loader(train_dataset, batch_size=4, seed=SEED)
    batch = next(iter(loader))

    assert batch["degraded"].shape == (4, 1, 16, 16)
    assert batch["clean"].shape == (4, 1, 16, 16)
    assert batch["degraded"].dtype is torch.float32
    assert len(batch["sample_key"]) == 4


def test_a_batch_carries_no_evaluation_mask(train_dataset) -> None:
    batch = next(iter(make_training_loader(train_dataset, batch_size=4, seed=SEED)))

    assert "body_mask" not in batch
    assert "clean_hu" not in batch
    assert set(batch) == {
        "degraded",
        "clean",
        "subject_id",
        "sample_key",
        "source_archive",
        "acquisition_group",
        "geometric_slice_index",
    }


# --------------------------------------------------------------------------
# The validation loader
# --------------------------------------------------------------------------


def test_validation_visits_every_sample_exactly_once(validation_dataset) -> None:
    keys = keys_of(make_validation_loader(validation_dataset, batch_size=2))

    assert len(keys) == len(validation_dataset)
    assert len(set(keys)) == len(validation_dataset)
    assert sorted(keys) == sorted(validation_dataset.sample_keys)


def test_validation_iterates_in_canonical_manifest_order(validation_dataset) -> None:
    keys = keys_of(make_validation_loader(validation_dataset, batch_size=3))

    assert keys == list(validation_dataset.sample_keys)


def test_validation_is_never_resampled(validation_dataset) -> None:
    """No patient balancing here: the figure must stay comparable."""
    loader = make_validation_loader(validation_dataset, batch_size=2)

    assert not isinstance(loader.sampler, PatientBalancedSampler)
    assert loader.drop_last is False
    assert keys_of(loader) == keys_of(loader)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 100])
def test_validation_order_is_independent_of_batch_size(validation_dataset, batch_size) -> None:
    keys = keys_of(make_validation_loader(validation_dataset, batch_size=batch_size))

    assert keys == list(validation_dataset.sample_keys)


def test_validation_keeps_the_ragged_final_batch(validation_dataset) -> None:
    batches = list(make_validation_loader(validation_dataset, batch_size=4))

    assert sum(len(batch["sample_key"]) for batch in batches) == len(validation_dataset)
    assert len(batches[-1]["sample_key"]) == len(validation_dataset) % 4


def test_validation_batch_size_does_not_change_the_tensors(validation_dataset) -> None:
    one = {
        batch["sample_key"][0]: batch["degraded"][0].clone()
        for batch in make_validation_loader(validation_dataset, batch_size=1)
    }
    for batch in make_validation_loader(validation_dataset, batch_size=3):
        for position, key in enumerate(batch["sample_key"]):
            assert batch["degraded"][position].equal(one[key])
