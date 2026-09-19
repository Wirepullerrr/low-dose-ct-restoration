"""DataLoader construction for the learned methods.

Two loaders, two different jobs.

Training
--------
Ordering is the sampler's job, so the loader passes ``shuffle=False`` and hands
ordering to :class:`PatientBalancedSampler`. ``drop_last=False`` matters more
than it looks: dropping a partial final batch would silently discard whichever
patients happened to land at the end of the shuffled epoch, so the realized
per-patient counts would no longer match the counts the sampler audit reports.

Batch size is a caller argument, not a frozen benchmark parameter. The eventual
CNN and U-Net batch sizes are model decisions and belong to those milestones;
nothing here declares them.

Validation
----------
Never resampled. Validation exists to produce a comparable number, so it
visits all 885 slices exactly once, in the frozen canonical manifest order,
with no shuffling and no dropped batch. Patient balancing would change which
slices contribute and make the figure incomparable with the degraded-baseline
and CLAHE numbers already measured.

Batching changes grouping, not sampling. For both loaders, changing the batch
size regroups the same samples in the same order; it cannot change which
samples appear, their order, or their values.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from ct_restoration.data.dataset import RestorationDataset
from ct_restoration.data.sampling import PatientBalancedSampler


def make_training_loader(
    dataset: RestorationDataset,
    batch_size: int,
    seed: int,
    epoch: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    epoch_size: int | None = None,
) -> DataLoader:
    """A training loader ordered by the patient-balanced sampler.

    ``shuffle`` and ``drop_last`` are not parameters: the sampler owns
    ordering, and dropping the tail would break the audited per-patient counts.

    Advance epochs with ``loader.sampler.set_epoch(n)``, which reseeds the
    sequence deterministically without touching the degradation of any slice.
    """
    sampler = PatientBalancedSampler(
        dataset.subject_ids, seed=seed, epoch=epoch, epoch_size=epoch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_validation_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """A sequential loader visiting every sample exactly once, in canonical order.

    Typed against the base ``Dataset`` so a ``Subset`` of a
    :class:`RestorationDataset` can be wrapped too, which is what the worker
    comparison in the audit needs.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
