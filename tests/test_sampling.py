"""Tests for the patient-balanced training sampler.

Fully synthetic and dataset-free: the sampler only ever sees a list of subject
ids, so the whole policy can be exercised on hand-built cohorts where the
correct answer is known by construction.

The sampler is the one place the experiment's unit of analysis is expressed in
the training loop. If it silently degenerated into slice-uniform traversal,
nothing would crash and no metric would look wrong - the long scans would just
quietly dominate the optimizer. Most of what follows exists to make that
degeneration impossible to introduce unnoticed.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby

import numpy as np
import pytest
import torch

from ct_restoration.data.sampling import (
    ALGORITHM_VERSION,
    PatientBalancedSampler,
    SamplingError,
    derive_epoch_seed,
)


def cohort(**slices_per_patient: int) -> list[str]:
    """A subject-id-per-index list, the only thing the sampler consumes."""
    ids: list[str] = []
    for patient, count in slices_per_patient.items():
        ids.extend([patient] * count)
    return ids


def counts(sampler: PatientBalancedSampler, subject_ids: list[str], epoch: int) -> Counter:
    return Counter(subject_ids[index] for index in sampler.epoch_indices(epoch))


# --------------------------------------------------------------------------
# The reason the sampler exists
# --------------------------------------------------------------------------


def test_patient_balancing_beats_slice_uniform_exposure() -> None:
    """The whole training policy, in one eight-line example.

    Patient A has 2 slices, patient B has 8. Visiting every slice once - the
    ordinary DataLoader ``shuffle=True`` behaviour - exposes B four times as
    often as A, purely because B's scan is longer. Patient balancing gives
    each of them 5 of the 10 draws, even though A must then repeat slices.
    """
    subject_ids = cohort(A=2, B=8)

    slice_uniform = Counter(subject_ids)
    assert (slice_uniform["A"], slice_uniform["B"]) == (2, 8)

    sampler = PatientBalancedSampler(subject_ids, seed=2026)
    balanced = counts(sampler, subject_ids, epoch=0)

    assert sampler.epoch_size == 10
    assert (balanced["A"], balanced["B"]) == (5, 5)


def test_acquisition_groups_are_deliberately_not_balanced() -> None:
    """Patients are balanced; A/B group mass is left at the cohort's own 10/15.

    Ten group-A patients and fifteen group-B patients, each weighted equally
    as a patient, leaves group B with 15/25 of the draws. Forcing 50/50 would
    be a second intervention the frozen cohort does not justify.
    """
    group_a = {f"a{index:02d}": 4 for index in range(10)}
    group_b = {f"b{index:02d}": 40 for index in range(15)}
    subject_ids = cohort(**group_a, **group_b)

    sampler = PatientBalancedSampler(subject_ids, seed=2026, epoch_size=250)
    drawn = counts(sampler, subject_ids, epoch=0)

    assert {drawn[name] for name in group_a} == {10}
    assert {drawn[name] for name in group_b} == {10}
    assert sum(drawn[name] for name in group_b) == 150  # 15/25 of 250, not half


# --------------------------------------------------------------------------
# Epoch size and exact quotas
# --------------------------------------------------------------------------


def test_an_epoch_is_exactly_the_dataset_size_by_default() -> None:
    sampler = PatientBalancedSampler(cohort(p1=3, p2=7, p3=5), seed=1)

    assert sampler.epoch_size == 15
    assert len(sampler) == 15
    assert len(sampler.epoch_indices(0)) == 15


def test_the_real_training_shape_gives_166_and_167() -> None:
    """The frozen train split: 4160 slices over 25 patients."""
    subject_ids = cohort(**{f"p{index:02d}": 4160 // 25 for index in range(25)})
    sampler = PatientBalancedSampler(subject_ids, seed=2026, epoch_size=4160)

    quotas = sampler.patient_quotas(0)

    assert (4160 // 25, 4160 % 25) == (166, 10)
    assert sum(quotas.values()) == 4160
    assert sorted(Counter(quotas.values()).items()) == [(166, 15), (167, 10)]


@pytest.mark.parametrize("epoch", range(6))
def test_every_patient_is_within_one_sample_of_every_other(epoch) -> None:
    subject_ids = cohort(p1=3, p2=20, p3=7, p4=11, p5=2, p6=9, p7=30)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    drawn = counts(sampler, subject_ids, epoch)

    assert sum(drawn.values()) == sampler.epoch_size
    assert len(drawn) == len(sampler.patients)
    assert max(drawn.values()) - min(drawn.values()) <= 1


def test_realized_counts_match_the_declared_quotas() -> None:
    """The quotas are a promise; this checks the drawn sequence keeps it."""
    subject_ids = cohort(p1=3, p2=20, p3=7, p4=11)
    sampler = PatientBalancedSampler(subject_ids, seed=5)

    for epoch in range(4):
        assert dict(counts(sampler, subject_ids, epoch)) == sampler.patient_quotas(epoch)


def test_every_patient_appears_in_every_epoch() -> None:
    subject_ids = cohort(p1=1, p2=50, p3=2)
    sampler = PatientBalancedSampler(subject_ids, seed=3)

    for epoch in range(5):
        assert set(counts(sampler, subject_ids, epoch)) == set(sampler.patients)


def test_an_epoch_smaller_than_the_patient_count_is_refused() -> None:
    with pytest.raises(SamplingError, match="no samples at all"):
        PatientBalancedSampler(cohort(p1=2, p2=2, p3=2), seed=1, epoch_size=2)


def test_an_empty_cohort_is_refused() -> None:
    with pytest.raises(SamplingError, match="empty dataset"):
        PatientBalancedSampler([], seed=1)


# --------------------------------------------------------------------------
# The rotating remainder
# --------------------------------------------------------------------------


def test_the_extra_quota_rotates_rather_than_favouring_the_same_patients() -> None:
    subject_ids = cohort(**{f"p{index}": 3 for index in range(5)})
    sampler = PatientBalancedSampler(subject_ids, seed=1, epoch_size=17)  # 3 each, 2 extra

    per_epoch = [sampler.extra_quota_patients(epoch) for epoch in range(5)]

    assert all(len(extra) == 2 for extra in per_epoch)
    assert len(set(per_epoch)) > 1
    # Over five epochs every patient takes the extra draw the same number of
    # times: 5 epochs x 2 extras / 5 patients = 2 each.
    tally = Counter(patient for extra in per_epoch for patient in extra)
    assert set(tally.values()) == {2}


def test_the_rotation_on_the_real_training_shape_is_even_over_five_epochs() -> None:
    """25 patients, remainder 10: starts cycle 0, 10, 20, 5, 15 then repeat."""
    subject_ids = cohort(**{f"p{index:02d}": 166 for index in range(25)})
    sampler = PatientBalancedSampler(subject_ids, seed=2026, epoch_size=4160)

    tally = Counter(
        patient for epoch in range(5) for patient in sampler.extra_quota_patients(epoch)
    )

    assert len(tally) == 25
    assert set(tally.values()) == {2}


def test_no_patient_is_starved_of_the_extra_quota_over_many_epochs() -> None:
    subject_ids = cohort(**{f"p{index}": 4 for index in range(7)})
    sampler = PatientBalancedSampler(subject_ids, seed=9, epoch_size=30)  # 4 each, 2 extra

    tally = Counter(
        patient for epoch in range(14) for patient in sampler.extra_quota_patients(epoch)
    )

    assert set(tally) == set(sampler.patients)


def test_quotas_do_not_depend_on_input_order() -> None:
    """The canonical patient list comes from the ids, not from row order."""
    forward = PatientBalancedSampler(cohort(p1=3, p2=5, p3=2), seed=4, epoch_size=11)
    shuffled_ids = cohort(p3=2, p1=3, p2=5)
    scrambled = PatientBalancedSampler(shuffled_ids, seed=4, epoch_size=11)

    assert forward.patients == scrambled.patients
    assert forward.patient_quotas(0) == scrambled.patient_quotas(0)
    assert forward.extra_quota_patients(2) == scrambled.extra_quota_patients(2)


def test_patients_are_in_natural_not_lexicographic_order() -> None:
    sampler = PatientBalancedSampler(cohort(**{"2": 2, "3": 2, "21": 2, "31": 2}), seed=1)

    assert sampler.patients == ("2", "3", "21", "31")


# --------------------------------------------------------------------------
# Within-patient cycles
# --------------------------------------------------------------------------


def test_a_short_scan_covers_every_slice_before_repeating_any() -> None:
    """Equal patient weight forces repeats; it must not force early repeats."""
    subject_ids = cohort(short=3, long=30)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    order = sampler.patient_draw_order("short", epoch=0)

    assert len(order) in (16, 17)
    assert sorted(order[:3]) == [0, 1, 2]  # a full permutation first
    assert sorted(order[3:6]) == [0, 1, 2]  # then another


def test_a_long_scan_uses_a_subset_without_repeating_inside_one_epoch() -> None:
    subject_ids = cohort(short=3, long=30)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    order = sampler.patient_draw_order("long", epoch=0)

    assert len(set(order)) == len(order)
    assert len(order) < 30  # a subset of the scan, this epoch


def test_a_long_scan_covers_different_slices_in_different_epochs() -> None:
    subject_ids = cohort(short=3, long=30)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    first = set(sampler.patient_draw_order("long", epoch=0))
    second = set(sampler.patient_draw_order("long", epoch=1))

    assert first != second


def test_draw_order_is_refused_for_an_unknown_patient() -> None:
    sampler = PatientBalancedSampler(cohort(p1=3), seed=1)

    with pytest.raises(SamplingError, match="unknown subject_id"):
        sampler.patient_draw_order("nobody")


# --------------------------------------------------------------------------
# The final shuffle
# --------------------------------------------------------------------------


def test_the_epoch_is_shuffled_rather_than_arriving_in_patient_blocks() -> None:
    subject_ids = cohort(p1=20, p2=20, p3=20, p4=20)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    drawn = [subject_ids[index] for index in sampler.epoch_indices(0)]
    blocked = sorted(drawn)

    assert drawn != blocked
    # Consecutive-same-patient runs should be far below the unshuffled 20.
    longest = max(len(list(members)) for _, members in groupby(drawn))
    assert longest < 10


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_same_seed_and_epoch_reproduce_the_sequence_exactly() -> None:
    subject_ids = cohort(p1=3, p2=9, p3=5)

    first = PatientBalancedSampler(subject_ids, seed=2026).epoch_indices(0)
    second = PatientBalancedSampler(subject_ids, seed=2026).epoch_indices(0)

    assert first == second


def test_returning_to_an_earlier_epoch_reproduces_it() -> None:
    sampler = PatientBalancedSampler(cohort(p1=3, p2=9, p3=5), seed=2026)

    sampler.set_epoch(0)
    original = list(sampler)
    sampler.set_epoch(3)
    list(sampler)
    sampler.set_epoch(0)

    assert list(sampler) == original


def test_a_different_epoch_gives_a_different_sequence() -> None:
    sampler = PatientBalancedSampler(cohort(p1=8, p2=9, p3=7), seed=2026)

    assert sampler.epoch_indices(0) != sampler.epoch_indices(1)
    assert sampler.epoch_indices(1) != sampler.epoch_indices(2)


def test_a_different_seed_gives_a_different_sequence() -> None:
    subject_ids = cohort(p1=8, p2=9, p3=7)

    assert PatientBalancedSampler(subject_ids, seed=2026).epoch_indices(
        0
    ) != PatientBalancedSampler(subject_ids, seed=2027).epoch_indices(0)


def test_asking_for_an_epoch_does_not_advance_the_sampler() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=4), seed=1)

    sampler.epoch_indices(5)

    assert sampler.epoch == 0
    assert list(sampler) == list(sampler)


def test_iterating_twice_yields_the_same_sequence() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    assert list(sampler) == list(sampler)


# --------------------------------------------------------------------------
# set_epoch is strict
# --------------------------------------------------------------------------


def test_set_epoch_selects_the_epoch() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    sampler.set_epoch(4)

    assert sampler.epoch == 4
    assert list(sampler) == list(sampler.epoch_indices(4))


@pytest.mark.parametrize("epoch", [1.0, "1", None, [1]])
def test_set_epoch_refuses_a_non_integer(epoch) -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be an integer"):
        sampler.set_epoch(epoch)


@pytest.mark.parametrize("epoch", [True, False])
def test_set_epoch_refuses_a_bool(epoch) -> None:
    """``True`` is an int in Python and would silently mean epoch 1."""
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be an integer"):
        sampler.set_epoch(epoch)


def test_set_epoch_refuses_a_negative_epoch() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="non-negative"):
        sampler.set_epoch(-1)


@pytest.mark.parametrize("seed", [1.0, "1", True, None])
def test_a_non_integer_seed_is_refused(seed) -> None:
    with pytest.raises(SamplingError, match="seed must be an integer"):
        PatientBalancedSampler(cohort(p1=4, p2=6), seed=seed)


@pytest.mark.parametrize("size", [3.0, "3", True])
def test_a_non_integer_epoch_size_is_refused(size) -> None:
    with pytest.raises(SamplingError, match="epoch_size must be an integer"):
        PatientBalancedSampler(cohort(p1=4, p2=6), seed=1, epoch_size=size)


# --------------------------------------------------------------------------
# Global state is never touched
# --------------------------------------------------------------------------


def test_the_global_numpy_rng_is_neither_read_nor_written() -> None:
    """Sampling must not consume draws another part of a run is relying on."""
    np.random.seed(1234)
    before = np.random.get_state()

    sampler = PatientBalancedSampler(cohort(p1=9, p2=11, p3=4), seed=2026)
    sampler.epoch_indices(0)
    sampler.set_epoch(2)
    list(sampler)

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_the_global_numpy_rng_cannot_change_the_sequence() -> None:
    subject_ids = cohort(p1=9, p2=11, p3=4)

    np.random.seed(1)
    first = PatientBalancedSampler(subject_ids, seed=2026).epoch_indices(0)

    np.random.seed(999)
    np.random.random(10_000)
    second = PatientBalancedSampler(subject_ids, seed=2026).epoch_indices(0)

    assert first == second


def test_the_global_torch_rng_is_untouched() -> None:
    """This module uses no PyTorch randomness at all."""
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()

    list(PatientBalancedSampler(cohort(p1=9, p2=11), seed=2026))

    assert torch.equal(before, torch.random.get_rng_state())


# --------------------------------------------------------------------------
# Seed derivation
# --------------------------------------------------------------------------


def test_the_algorithm_is_versioned() -> None:
    assert ALGORITHM_VERSION == "patient_balanced_v1"
    assert PatientBalancedSampler(cohort(p1=2), seed=1).algorithm == ALGORITHM_VERSION


def test_seed_derivation_is_deterministic_across_calls() -> None:
    assert derive_epoch_seed(2026, 3, "patient:14") == derive_epoch_seed(2026, 3, "patient:14")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((2026, 0, "a"), (2026, 1, "a")),
        ((2026, 0, "a"), (2027, 0, "a")),
        ((2026, 0, "a"), (2026, 0, "b")),
    ],
)
def test_seed_derivation_separates_its_inputs(left, right) -> None:
    assert derive_epoch_seed(*left) != derive_epoch_seed(*right)


def test_the_algorithm_version_is_part_of_the_seed() -> None:
    assert derive_epoch_seed(2026, 0, "a") != derive_epoch_seed(
        2026, 0, "a", algorithm_version="patient_balanced_v2"
    )


def test_the_payload_separator_is_refused_in_a_stream_label() -> None:
    with pytest.raises(SamplingError, match="U\\+001F"):
        derive_epoch_seed(2026, 0, "patient\x1f14")


def test_the_payload_separator_is_refused_in_a_subject_id() -> None:
    with pytest.raises(SamplingError, match="U\\+001F"):
        PatientBalancedSampler(["a\x1fb", "a\x1fb"], seed=1)


# --------------------------------------------------------------------------
# Every epoch entry point is strict in the same way
# --------------------------------------------------------------------------
#
# set_epoch was already strict. The explicit-epoch query methods and the
# public seed helper took the same argument through a silent int() instead, so
# derive_epoch_seed(2026.5, ...) quietly hashed a different experiment than the
# caller asked for and epoch_indices(-1) returned a usable but meaningless
# sequence.


@pytest.mark.parametrize("base_seed", [2026.5, 2026.0, "2026", True, None])
def test_derive_epoch_seed_refuses_a_non_integer_base_seed(base_seed) -> None:
    with pytest.raises(SamplingError, match="base_seed must be an integer"):
        derive_epoch_seed(base_seed, 0, "epoch_shuffle")


@pytest.mark.parametrize("epoch", [1.0, 1.5, "1", True, None])
def test_derive_epoch_seed_refuses_a_non_integer_epoch(epoch) -> None:
    with pytest.raises(SamplingError, match="epoch must be an integer"):
        derive_epoch_seed(2026, epoch, "epoch_shuffle")


def test_derive_epoch_seed_refuses_a_negative_epoch() -> None:
    with pytest.raises(SamplingError, match="epoch must be non-negative"):
        derive_epoch_seed(2026, -1, "epoch_shuffle")


@pytest.mark.parametrize("epoch", [-1, -100])
def test_patient_quotas_refuses_a_negative_epoch(epoch) -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be non-negative"):
        sampler.patient_quotas(epoch)


def test_patient_draw_order_refuses_a_negative_epoch() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be non-negative"):
        sampler.patient_draw_order("p1", -1)


def test_epoch_indices_refuses_a_negative_epoch() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be non-negative"):
        sampler.epoch_indices(-1)


@pytest.mark.parametrize("method", ["patient_quotas", "epoch_indices"])
@pytest.mark.parametrize("epoch", [1.0, "1", True])
def test_explicit_epoch_methods_refuse_a_non_integer(method, epoch) -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be an integer"):
        getattr(sampler, method)(epoch)


def test_extra_quota_patients_refuses_a_negative_epoch() -> None:
    sampler = PatientBalancedSampler(cohort(p1=4, p2=6), seed=1)

    with pytest.raises(SamplingError, match="epoch must be non-negative"):
        sampler.extra_quota_patients(-1)


def test_stricter_validation_did_not_change_any_derived_seed() -> None:
    """Every valid integer argument must still hash to what it hashed before.

    These are the two stream shapes the sampler actually uses, pinned so the
    audited epoch sequences cannot drift under a refactor of the guards.
    """
    assert derive_epoch_seed(2026, 0, "epoch_shuffle") == 6066294244247046968
    assert derive_epoch_seed(2026, 3, "patient:14") == 10776177018209623446


def test_numpy_integers_are_still_accepted() -> None:
    assert derive_epoch_seed(np.int64(2026), np.int32(0), "epoch_shuffle") == derive_epoch_seed(
        2026, 0, "epoch_shuffle"
    )


# --------------------------------------------------------------------------
# What cross-epoch coverage does and does not promise
# --------------------------------------------------------------------------


def test_a_long_scan_is_sampled_without_replacement_within_one_epoch() -> None:
    """The guarantee the v1 sampler does make."""
    subject_ids = cohort(short=3, long=60)
    sampler = PatientBalancedSampler(subject_ids, seed=2026)

    order = sampler.patient_draw_order("long", epoch=0)

    assert len(order) < 60
    assert len(set(order)) == len(order)


def test_the_sampler_keeps_no_cross_epoch_cursor() -> None:
    """Each epoch derives a fresh permutation; nothing tracks what was missed.

    Coverage broadens across epochs, but the v1 algorithm makes no claim that
    every slice has appeared by any particular finite epoch - and this shows
    why: epoch n is computed from n alone, with no memory of epoch n-1.
    """
    subject_ids = cohort(short=3, long=60)
    first = PatientBalancedSampler(subject_ids, seed=2026)
    second = PatientBalancedSampler(subject_ids, seed=2026)

    second.epoch_indices(1)
    second.epoch_indices(7)

    # Having asked for other epochs changes nothing about epoch 2.
    assert first.epoch_indices(2) == second.epoch_indices(2)
