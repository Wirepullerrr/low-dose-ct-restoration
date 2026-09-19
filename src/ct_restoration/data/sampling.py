"""The patient-balanced training sampler.

Why not ordinary slice shuffling
--------------------------------
The 25 training patients hold between 78 and 294 slices. Traversing every
slice once per epoch would give the longest scan 294/78 times the optimizer
weight of the shortest, purely because of how long that patient's scan was.
Scan length is an acquisition-protocol fact, not a statement about how much
that patient should matter.

Patient is already the primary experimental unit everywhere else in this
benchmark: the split is patient-level, and the reported metrics average slices
within a patient before averaging patients. A training policy weighted by
slice count would be the one place the experiment silently switched units.

So the canonical training policy weights **patients** equally.

What this does not claim
------------------------
Balancing patients does **not** make slices statistically independent.
Adjacent slices of one CT scan remain highly correlated, and nothing here
changes that. It only stops scan length from directly determining one
patient's total training weight.

What is deliberately not balanced
---------------------------------
Acquisition group, source archive and slice position are left alone. Training
holds 10 group-A and 15 group-B patients; equal per-patient weight therefore
leaves group B contributing 15/25 of the patient mass, which is the cohort's
own composition. Forcing A and B to 50/50 would be a second sampling
intervention that the frozen cohort definition does not justify.

Why not WeightedRandomSampler
-----------------------------
``torch.utils.data.WeightedRandomSampler`` balances patients only in
expectation: any single epoch can over- or under-draw a patient by a wide
margin, and an audit can only report the realized counts after the fact. The
counts here are enforced exactly, so "every patient received 166 or 167
samples this epoch" is a property of the algorithm rather than an observation
about one run.

The algorithm
-------------
For an epoch of ``epoch_size`` draws over ``n`` patients::

    base, remainder = divmod(epoch_size, n)

``remainder`` patients receive ``base + 1`` draws and the rest receive
``base``, so patient exposure differs by at most one sample inside an epoch.
Which patients receive the extra draw rotates with the epoch: the canonical
patient list is cycled by ``(epoch * remainder) % n`` positions and the first
``remainder`` patients of that rotation take the extra. On the real training
split that is ``divmod(4160, 25) == (166, 10)``, so ten patients get 167 and
fifteen get 166, and the ten rotate through ``start`` values 0, 10, 20, 5, 15
before repeating - five epochs in which every patient takes the extra draw
exactly twice.

Within a patient, indices come from shuffled full passes: permute all of that
patient's slices, consume the permutation, and reshuffle for another full pass
if more draws are needed. A short scan therefore covers every one of its
slices before repeating any. A long scan is sampled without replacement up to
its quota, so it contributes a subset of itself; each epoch derives a new
deterministic permutation, so those subsets generally change and coverage
broadens across epochs. Note what that is **not**: the v1 sampler keeps no
cross-epoch cursor, so it does not guarantee that every slice of a long scan
has appeared by any particular finite epoch. Finally the whole epoch sequence
is shuffled so batches intermix patients instead of arriving in contiguous
per-patient blocks.

Randomness
----------
Every draw comes from a private ``Generator`` seeded by SHA-256 over the
algorithm version, the base seed, the epoch and a stream label. The global
NumPy RNG is never read or written, and this module uses no PyTorch
randomness at all. The sampler's RNG is entirely separate from the per-slice
degradation RNG: changing which slice is drawn is allowed, changing what
corruption that slice carries is not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from ct_restoration.data.splits import subject_sort_key

#: The only sampling algorithm this module implements.
ALGORITHM_VERSION = "patient_balanced_v1"

#: Separator for the hashed seed payload, mirroring the degradation module's
#: contract: the byte is forbidden in every field, so no two distinct
#: (version, seed, epoch, stream) tuples can serialize to the same payload.
_FIELD_SEPARATOR = "\x1f"

#: Bytes of the SHA-256 digest used as a seed.
_SEED_BYTES = 8

#: Stream label for the final whole-epoch shuffle.
_EPOCH_SHUFFLE_STREAM = "epoch_shuffle"

#: Stream label prefix for one patient's within-patient cycle.
_PATIENT_STREAM_PREFIX = "patient"


class SamplingError(ValueError):
    """The sampler configuration or an epoch argument is not usable."""


def _require_integer(name: str, value: object) -> int:
    """Accept only a genuine integer. ``bool`` is an ``int`` subclass; refuse it."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise SamplingError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}. "
            "It is not rounded, truncated or parsed from a string: a silently "
            "altered value would change the whole sampled sequence."
        )
    return int(value)


def _require_epoch(value: object) -> int:
    """Accept only a non-negative integer epoch.

    Every entry point that takes an epoch runs through here, so ``set_epoch``
    and the explicit-epoch query methods cannot disagree about what an epoch
    is. ``True`` would otherwise read as epoch 1 and ``-1`` would silently
    derive a perfectly usable but meaningless RNG stream.
    """
    epoch = _require_integer("epoch", value)
    if epoch < 0:
        raise SamplingError(f"epoch must be non-negative, got {epoch}")
    return epoch


def derive_epoch_seed(
    base_seed: int,
    epoch: int,
    stream: str,
    algorithm_version: str = ALGORITHM_VERSION,
) -> int:
    """Derive one RNG stream's seed from the sampling definition.

    SHA-256 over the algorithm version, the base seed, the epoch and a stream
    label. Python's :func:`hash` is unusable here: it is randomized per process
    for strings, so the same epoch would sample differently in different runs.
    Nothing here reads the clock, the process id or any global state.

    The arguments are validated rather than coerced, to the same standard the
    sampler itself applies: ``int(2026.5)`` would otherwise serialize a
    different experiment than the caller asked for, and silently.

    Returns:
        A 64-bit integer seed, identical on every platform, process and run.

    Raises:
        SamplingError: ``base_seed`` is not an integer, ``epoch`` is not a
            non-negative integer, or ``stream`` contains the payload
            separator.
    """
    seed = _require_integer("base_seed", base_seed)
    index = _require_epoch(epoch)
    if _FIELD_SEPARATOR in str(stream):
        raise SamplingError(
            f"stream label must not contain U+001F, the reserved payload separator; got {stream!r}"
        )
    payload = _FIELD_SEPARATOR.join((str(algorithm_version), str(seed), str(index), str(stream)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


class PatientBalancedSampler(Sampler[int]):
    """Draw a fixed-size epoch in which every patient contributes equally.

    Args:
        subject_ids: the subject of each dataset index, in dataset order.
        seed: base seed for this sampler. Supplied by the caller rather than
            frozen here: a training run owns its own seed, and the sampler
            must not quietly pin the eventual network training seed.
        epoch: the starting epoch. Change it with :meth:`set_epoch`.
        epoch_size: draws per epoch. Defaults to the dataset size, so one
            epoch is the same amount of optimizer work as a plain traversal.
    """

    def __init__(
        self,
        subject_ids: Sequence[str],
        seed: int,
        epoch: int = 0,
        epoch_size: int | None = None,
    ) -> None:
        if len(subject_ids) == 0:
            raise SamplingError("cannot sample from an empty dataset")

        self._seed = _require_integer("seed", seed)
        self._subject_ids = tuple(str(value) for value in subject_ids)
        for subject_id in set(self._subject_ids):
            if _FIELD_SEPARATOR in subject_id:
                raise SamplingError(
                    f"subject_id must not contain U+001F, the reserved payload "
                    f"separator; got {subject_id!r}"
                )

        # Canonical patient order, derived from the ids themselves, so the
        # sequence cannot depend on the row order of the input manifest.
        self._patients: tuple[str, ...] = tuple(
            sorted(set(self._subject_ids), key=subject_sort_key)
        )
        grouped: dict[str, list[int]] = {patient: [] for patient in self._patients}
        for index, subject_id in enumerate(self._subject_ids):
            grouped[subject_id].append(index)
        self._indices_by_patient: dict[str, np.ndarray] = {
            patient: np.asarray(indices, dtype=np.int64) for patient, indices in grouped.items()
        }

        size = (
            len(self._subject_ids)
            if epoch_size is None
            else _require_integer("epoch_size", epoch_size)
        )
        if size < len(self._patients):
            raise SamplingError(
                f"epoch_size {size} is smaller than the {len(self._patients)} patients; "
                "at least one patient would receive no samples at all."
            )
        self._epoch_size = size
        self._epoch = 0
        self.set_epoch(epoch)

    # -- structure --------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return ALGORITHM_VERSION

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def epoch_size(self) -> int:
        return self._epoch_size

    @property
    def patients(self) -> tuple[str, ...]:
        """Distinct patients, in canonical natural order."""
        return self._patients

    def patient_slice_counts(self) -> dict[str, int]:
        """How many dataset indices each patient actually has."""
        return {patient: int(len(indices)) for patient, indices in self._indices_by_patient.items()}

    # -- epoch control ----------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose sequence :meth:`__iter__` will yield.

        Same seed and same epoch always give exactly the same index sequence;
        a different epoch generally gives a different one. Re-setting an
        earlier epoch reproduces it exactly, because the sequence is derived
        from the epoch number rather than accumulated in sampler state.

        Raises:
            SamplingError: ``epoch`` is not a non-negative integer.
        """
        self._epoch = _require_epoch(epoch)

    # -- the algorithm ----------------------------------------------------

    def patient_quotas(self, epoch: int | None = None) -> dict[str, int]:
        """How many draws each patient receives in one epoch.

        ``base = epoch_size // n_patients`` for everyone, plus one extra for
        ``remainder`` patients taken from the canonical list cycled by
        ``(epoch * remainder) % n_patients`` positions.

        Raises:
            SamplingError: an explicit ``epoch`` is not a non-negative integer.
        """
        target = self._epoch if epoch is None else _require_epoch(epoch)
        count = len(self._patients)
        base, remainder = divmod(self._epoch_size, count)
        quotas = dict.fromkeys(self._patients, base)
        if remainder:
            start = (target * remainder) % count
            for offset in range(remainder):
                quotas[self._patients[(start + offset) % count]] += 1
        return quotas

    def extra_quota_patients(self, epoch: int | None = None) -> tuple[str, ...]:
        """The patients receiving ``base + 1`` draws in one epoch, in canonical order."""
        quotas = self.patient_quotas(epoch)
        base = self._epoch_size // len(self._patients)
        return tuple(patient for patient in self._patients if quotas[patient] > base)

    def patient_draw_order(self, subject_id: str, epoch: int | None = None) -> tuple[int, ...]:
        """One patient's drawn indices for an epoch, before the global shuffle.

        Exposed so the within-patient cycle can be audited directly: the first
        ``len(pool)`` entries are a permutation of every one of that patient's
        slices, so a short scan covers all of them before repeating any.

        Raises:
            SamplingError: an explicit ``epoch`` is not a non-negative
                integer, or ``subject_id`` is not in this cohort.
        """
        target = self._epoch if epoch is None else _require_epoch(epoch)
        if subject_id not in self._indices_by_patient:
            raise SamplingError(f"unknown subject_id {subject_id!r}")
        quota = self.patient_quotas(target)[subject_id]
        return tuple(self._draw_for_patient(subject_id, quota, target))

    def _draw_for_patient(self, subject_id: str, quota: int, epoch: int) -> list[int]:
        pool = self._indices_by_patient[subject_id]
        generator = np.random.default_rng(
            derive_epoch_seed(self._seed, epoch, f"{_PATIENT_STREAM_PREFIX}:{subject_id}")
        )
        drawn: list[int] = []
        while len(drawn) < quota:
            permutation = generator.permutation(pool)
            drawn.extend(int(value) for value in permutation[: quota - len(drawn)])
        return drawn

    def epoch_indices(self, epoch: int | None = None) -> tuple[int, ...]:
        """The complete shuffled index sequence for one epoch.

        A pure function of ``(algorithm, seed, epoch)`` and the dataset's
        patient structure: it reads and writes no global state, and asking for
        an epoch does not advance anything.

        Raises:
            SamplingError: an explicit ``epoch`` is not a non-negative integer.
        """
        target = self._epoch if epoch is None else _require_epoch(epoch)
        quotas = self.patient_quotas(target)

        drawn: list[int] = []
        for patient in self._patients:
            drawn.extend(self._draw_for_patient(patient, quotas[patient], target))

        # Without this the epoch would arrive as contiguous per-patient blocks
        # and every batch would hold one or two patients.
        sequence = np.asarray(drawn, dtype=np.int64)
        generator = np.random.default_rng(
            derive_epoch_seed(self._seed, target, _EPOCH_SHUFFLE_STREAM)
        )
        generator.shuffle(sequence)
        return tuple(int(value) for value in sequence)

    def __iter__(self) -> Iterator[int]:
        return iter(self.epoch_indices(self._epoch))

    def __len__(self) -> int:
        return self._epoch_size

    def __repr__(self) -> str:
        return (
            f"PatientBalancedSampler(algorithm={ALGORITHM_VERSION!r}, seed={self._seed}, "
            f"epoch={self._epoch}, epoch_size={self._epoch_size}, "
            f"patients={len(self._patients)})"
        )
