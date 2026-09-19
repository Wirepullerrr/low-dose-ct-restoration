"""Audit the Dataset / DataLoader layer on real training and validation data.

    uv run python scripts/audit_dataset.py --root data/raw/chaos

Development splits only, by construction
----------------------------------------
There is no ``--split`` option. The audit reads train and validation, and the
helper it builds datasets with refuses test and stress. Nothing here can be
pointed at a sealed split.

What it establishes
-------------------
* the Dataset produces the declared tensor contract on every real slice;
* repeated access returns byte-identical pairs;
* for a small set of deterministic probes per split, Dataset output is
  byte-identical to calling the frozen preprocessing and degradation functions
  independently, so the training pipeline and the evaluation pipeline are fed
  the same images. The probes are a spot check on the wiring; what covers
  every item is that the Dataset calls those same frozen functions for all of
  them;
* the sampler's realized per-patient counts match the algorithm's promise;
* the extra-quota patients rotate across epochs;
* validation is visited exactly once, in canonical order, whatever the batch
  size.

Output
------
``outputs/audit/dataset_dataloader_summary.json`` - counts, hashes and
checks. No image pixels, no timestamps; regenerating from an unchanged
definition rewrites it byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import write_json  # noqa: E402
from ct_restoration.config import load_config  # noqa: E402
from ct_restoration.data.dataset import (  # noqa: E402
    FORBIDDEN_SAMPLE_FIELDS,
    PAIR_GENERATION_VERSION,
    SAMPLE_FIELDS,
    RestorationDataset,
    development_dataset,
    flatten_sample_keys,
    sample_key_is_portable,
)
from ct_restoration.data.degradation import (  # noqa: E402
    DegradationConfig,
    degrade_low_dose_like,
)
from ct_restoration.data.dicom import load_ct_hu  # noqa: E402
from ct_restoration.data.loaders import (  # noqa: E402
    make_training_loader,
    make_validation_loader,
)
from ct_restoration.data.preprocessing import preprocess_ct_slice  # noqa: E402
from ct_restoration.data.sampling import (  # noqa: E402
    ALGORITHM_VERSION,
    PatientBalancedSampler,
)

#: The tracked summary this command writes.
CANONICAL_SUMMARY = Path("outputs/audit/dataset_dataloader_summary.json")

#: Seed used for the Milestone 7 sampler audit. Not a frozen training seed:
#: future runs supply their own, see configs/data_loader.yaml.
AUDIT_SEED = 2026

#: Epochs audited.
AUDIT_EPOCHS = (0, 1, 2)

#: Epochs whose extra-quota patient set is recorded, to show the rotation.
ROTATION_EPOCHS = tuple(range(8))

#: Batch sizes compared for invariance. 17 does not divide 885 or 4160, so a
#: ragged final batch is exercised rather than avoided.
BATCH_SIZES = (1, 17)

#: How many deterministic samples per split are compared against the frozen
#: pipeline called directly.
PAIR_PROBES = 8


def check_output_policy(limit: int, summary_path: Path) -> None:
    """Refuse to write a partial audit over the canonical tracked summary.

    Raises:
        ValueError: ``limit`` is set and the output is the canonical path.
    """
    if limit and Path(summary_path).resolve() == CANONICAL_SUMMARY.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite the canonical tracked summary "
            f"{CANONICAL_SUMMARY.as_posix()} with a partial audit. "
            "Re-run without --limit, or pass a noncanonical --output."
        )


def sequence_digest(values: list[str]) -> str:
    """SHA-256 of an ordered sample-key sequence.

    Order-sensitive on purpose: two epochs drawing the same multiset of slices
    in different orders must hash differently, or the reproducibility check
    would be checking nothing.
    """
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def probe_positions(size: int, count: int) -> list[int]:
    """Deterministic, spread-out sample positions. No randomness."""
    if size <= count:
        return list(range(size))
    return [round(position * (size - 1) / (count - 1)) for position in range(count)]


def scan_split(
    dataset: RestorationDataset, limit: int = 0, progress_every: int = 500
) -> dict[str, Any]:
    """Walk every sample once, checking the tensor contract on real data.

    ``limit`` caps only how many slices are opened; the reported patient and
    slice counts always describe the whole split, and ``scanned_samples``
    records how much of it the tensor checks actually covered.
    """
    expected_shape = (1, *dataset.image_size)
    scanned = len(dataset) if limit <= 0 else min(int(limit), len(dataset))
    finite_failures = 0
    range_failures = 0
    shape_failures = 0
    dtype_failures = 0
    forbidden_field_hits: set[str] = set()
    field_mismatches = 0
    nonportable_keys = 0

    minimum = float("inf")
    maximum = float("-inf")
    keys: list[str] = []

    for position in range(scanned):
        sample = dataset[position]
        keys.append(sample["sample_key"])

        if tuple(sample.keys()) != SAMPLE_FIELDS:
            field_mismatches += 1
        forbidden_field_hits |= set(sample) & set(FORBIDDEN_SAMPLE_FIELDS)
        if not sample_key_is_portable(sample["sample_key"]):
            nonportable_keys += 1

        for half in ("degraded", "clean"):
            tensor = sample[half]
            if tuple(tensor.shape) != expected_shape:
                shape_failures += 1
            if tensor.dtype is not torch.float32:
                dtype_failures += 1
            if not bool(torch.isfinite(tensor).all()):
                finite_failures += 1
            low, high = float(tensor.min()), float(tensor.max())
            if low < 0.0 or high > 1.0:
                range_failures += 1
            minimum = min(minimum, low)
            maximum = max(maximum, high)

        if progress_every and (position + 1) % progress_every == 0:
            print(f"  scanned {position + 1:>5} / {scanned}")

    return {
        "patients": len(dataset.patients),
        "samples": len(dataset),
        "scanned_samples": scanned,
        "unique_sample_keys": len(set(keys)),
        "tensor_shape": list(expected_shape),
        "tensor_dtype": "torch.float32",
        "shape_failures": shape_failures,
        "dtype_failures": dtype_failures,
        "finite_failures": finite_failures,
        "range_failures": range_failures,
        "observed_min": minimum,
        "observed_max": maximum,
        "sample_field_order_mismatches": field_mismatches,
        "forbidden_fields_present": sorted(forbidden_field_hits),
        "nonportable_sample_keys": nonportable_keys,
        "canonical_sequence_sha256": sequence_digest(list(dataset.sample_keys)),
        "scanned_sequence_sha256": sequence_digest(keys),
        "slices_per_patient": {
            patient: len(indices) for patient, indices in dataset.indices_by_patient().items()
        },
    }


def repeated_access_check(dataset: RestorationDataset, positions: list[int]) -> dict[str, Any]:
    """Index each probe twice, with an unrelated index in between."""
    mismatches = 0
    other = (positions[0] + len(dataset) // 2) % len(dataset)
    for position in positions:
        first = dataset[position]
        dataset[other]  # deliberately interleave a different slice
        second = dataset[position]
        for half in ("degraded", "clean"):
            if first[half].numpy().tobytes() != second[half].numpy().tobytes():
                mismatches += 1
    return {"probes": len(positions), "byte_mismatches": mismatches}


def global_rng_independence_check(
    dataset: RestorationDataset, positions: list[int]
) -> dict[str, Any]:
    """Reseed the global NumPy and PyTorch RNGs between two identical reads."""
    mismatches = 0
    for position in positions:
        np.random.seed(1)
        torch.manual_seed(1)
        first = dataset[position]

        np.random.seed(999)
        np.random.random(10_000)
        torch.manual_seed(999)
        torch.rand(1000)
        second = dataset[position]

        for half in ("degraded", "clean"):
            if first[half].numpy().tobytes() != second[half].numpy().tobytes():
                mismatches += 1
    return {"probes": len(positions), "byte_mismatches": mismatches}


def direct_pipeline_equivalence(
    dataset: RestorationDataset,
    root: Path,
    preprocessing: dict[str, Any],
    degradation: DegradationConfig,
    positions: list[int],
) -> dict[str, Any]:
    """Compare Dataset output against the frozen functions called directly.

    A spot check on the wiring, over ``PAIR_PROBES`` deterministic positions
    per split rather than the whole split: if these arrays are byte-identical,
    the Dataset is not transforming anything on the way out, and a model
    trains on the images the benchmark scores. What covers every item is that
    :meth:`RestorationDataset.__getitem__` calls these same frozen functions
    for all of them, which the per-slice scan then range-checks.
    """
    clean_mismatches = 0
    degraded_mismatches = 0
    checked_keys: list[str] = []

    for position in positions:
        sample = dataset[position]
        key = sample["sample_key"]
        checked_keys.append(key)

        image_hu = load_ct_hu(root / key)
        clean = preprocess_ct_slice(
            image_hu,
            window_center=preprocessing["window_center"],
            window_width=preprocessing["window_width"],
            size=tuple(preprocessing["image_size"]),
            interpolation=preprocessing["interpolation"],
        )
        degraded = degrade_low_dose_like(clean, key, degradation)

        if sample["clean"].squeeze(0).numpy().tobytes() != clean.tobytes():
            clean_mismatches += 1
        if sample["degraded"].squeeze(0).numpy().tobytes() != degraded.tobytes():
            degraded_mismatches += 1

    return {
        "probes": len(positions),
        "clean_byte_mismatches": clean_mismatches,
        "degraded_byte_mismatches": degraded_mismatches,
        "probe_sample_keys": checked_keys,
    }


def sampler_epoch_report(
    sampler: PatientBalancedSampler, dataset: RestorationDataset, epoch: int
) -> dict[str, Any]:
    """Counts and hashes for one epoch of the patient-balanced sampler."""
    indices = sampler.epoch_indices(epoch)
    subject_ids = dataset.subject_ids
    keys = [dataset.sample_keys[index] for index in indices]

    per_patient: dict[str, int] = dict.fromkeys(sampler.patients, 0)
    for index in indices:
        per_patient[subject_ids[index]] += 1

    distribution: dict[str, int] = {}
    for count in sorted(set(per_patient.values())):
        distribution[str(count)] = sum(1 for value in per_patient.values() if value == count)

    unique_indices = set(indices)
    return {
        "total_draws": len(indices),
        "patients": len(per_patient),
        "min_samples_per_patient": min(per_patient.values()),
        "max_samples_per_patient": max(per_patient.values()),
        "patient_count_distribution": distribution,
        "unique_slices_sampled": len(unique_indices),
        "repeated_slice_selections": len(indices) - len(unique_indices),
        "train_slices_not_sampled": len(dataset) - len(unique_indices),
        "extra_quota_patients": list(sampler.extra_quota_patients(epoch)),
        "samples_per_patient": dict(per_patient),
        "sequence_sha256": sequence_digest(keys),
    }


def within_patient_cycle_report(sampler: PatientBalancedSampler, epoch: int) -> dict[str, Any]:
    """Show that a short scan covers every slice before repeating any."""
    counts = sampler.patient_slice_counts()
    shortest = min(counts, key=lambda patient: (counts[patient], patient))
    pool_size = counts[shortest]
    order = sampler.patient_draw_order(shortest, epoch)
    first_pass = order[:pool_size]
    return {
        "shortest_patient": shortest,
        "slices_available": pool_size,
        "quota": len(order),
        "first_pass_is_a_full_permutation": len(set(first_pass)) == pool_size,
        "repeats_before_full_coverage": pool_size - len(set(first_pass)),
    }


def loader_key_sequence(loader) -> list[str]:
    """Flatten the sample keys a DataLoader yields, in order."""
    return flatten_sample_keys(list(loader))


def main() -> int:
    arguments = build_parser().parse_args()
    output = Path(arguments.output)

    try:
        check_output_policy(arguments.limit, output)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))

    print(f"pair generation : {PAIR_GENERATION_VERSION}")
    print(f"sampler         : {ALGORITHM_VERSION}, audit seed {AUDIT_SEED}")
    print("splits read     : train, validation")
    print("test and stress image content is NOT read by this command")
    print()

    datasets = {
        split: development_dataset(split, arguments.manifest, root, preprocessing, degradation)
        for split in ("train", "validation")
    }

    summary: dict[str, Any] = {
        "dataset_contract": {
            "pair_generation": PAIR_GENERATION_VERSION,
            "sample_fields": list(SAMPLE_FIELDS),
            "forbidden_fields_checked": list(FORBIDDEN_SAMPLE_FIELDS),
            "model_input": "degraded",
            "supervised_target": "clean",
            "tensor_shape": [1, 256, 256],
            "tensor_dtype": "torch.float32",
            "value_range": [0.0, 1.0],
            "model_normalization": "none",
            "augmentation": "none",
            "cache": "none",
            "preprocessing": dict(preprocessing),
            "degradation_algorithm": degradation.algorithm,
            "degradation_global_seed": degradation.global_seed,
        },
        "splits_read": ["train", "validation"],
        "splits_sealed": ["test", "stress"],
        "test_images_read": 0,
        "stress_images_read": 0,
    }

    for split, dataset in datasets.items():
        print(f"scanning {split}: {len(dataset)} slices, {len(dataset.patients)} patients")
        positions = probe_positions(len(dataset), PAIR_PROBES)
        summary[split] = {
            **scan_split(dataset, arguments.limit),
            "repeated_access": repeated_access_check(dataset, positions),
            "global_rng_independence": global_rng_independence_check(dataset, positions),
            "direct_pipeline_equivalence": direct_pipeline_equivalence(
                dataset, root, preprocessing, degradation, positions
            ),
        }
        print()

    train = datasets["train"]
    sampler = PatientBalancedSampler(train.subject_ids, seed=AUDIT_SEED)
    base, remainder = divmod(sampler.epoch_size, len(sampler.patients))

    print(f"sampler epochs  : {list(AUDIT_EPOCHS)}")
    epochs = {str(epoch): sampler_epoch_report(sampler, train, epoch) for epoch in AUDIT_EPOCHS}
    repeat_of_epoch_zero = sampler_epoch_report(sampler, train, 0)["sequence_sha256"]

    rotation = {str(epoch): list(sampler.extra_quota_patients(epoch)) for epoch in ROTATION_EPOCHS}
    extra_counts: dict[str, int] = dict.fromkeys(sampler.patients, 0)
    for patients in rotation.values():
        for patient in patients:
            extra_counts[patient] += 1

    summary["training_sampler"] = {
        "algorithm": ALGORITHM_VERSION,
        "audit_seed": AUDIT_SEED,
        "epoch_size": sampler.epoch_size,
        "patients": len(sampler.patients),
        "base_quota": base,
        "remainder": remainder,
        "quota_rule": "(epoch * remainder) % n_patients selects the rotation start",
        "within_patient": "shuffled_cycles",
        "final_epoch_shuffle": True,
        "epochs": epochs,
        "epoch_0_recomputed_sha256": repeat_of_epoch_zero,
        "epoch_0_reproducible": repeat_of_epoch_zero == epochs["0"]["sequence_sha256"],
        "epoch_0_differs_from_epoch_1": (
            epochs["0"]["sequence_sha256"] != epochs["1"]["sequence_sha256"]
        ),
        "different_seed_differs": (
            PatientBalancedSampler(train.subject_ids, seed=AUDIT_SEED + 1).epoch_indices(0)
            != sampler.epoch_indices(0)
        ),
        "extra_quota_rotation": rotation,
        "extra_quota_times_in_first_8_epochs": extra_counts,
        "within_patient_cycle": within_patient_cycle_report(sampler, 0),
    }

    print("validation loader: batch-size invariance")
    validation = datasets["validation"]
    validation_keys = {
        str(size): loader_key_sequence(make_validation_loader(validation, batch_size=size))
        for size in BATCH_SIZES
    }
    canonical_validation = list(validation.sample_keys)
    first, second = (validation_keys[str(size)] for size in BATCH_SIZES)
    summary["validation_loader"] = {
        "batch_sizes_compared": list(BATCH_SIZES),
        "identical_key_sequence": first == second,
        "matches_canonical_manifest_order": first == canonical_validation,
        "samples_visited": len(first),
        "unique_samples_visited": len(set(first)),
        "every_sample_exactly_once": sorted(first) == sorted(canonical_validation),
        "shuffle": False,
        "drop_last": False,
        "sequence_sha256": sequence_digest(first),
    }
    print()

    print("training loader : batch-size invariance")
    training_keys = {
        str(size): loader_key_sequence(
            make_training_loader(train, batch_size=size, seed=AUDIT_SEED, epoch=0)
        )
        for size in BATCH_SIZES
    }
    train_first, train_second = (training_keys[str(size)] for size in BATCH_SIZES)
    sampler_keys = [train.sample_keys[index] for index in sampler.epoch_indices(0)]
    summary["training_loader"] = {
        "batch_sizes_compared": list(BATCH_SIZES),
        "identical_key_sequence": train_first == train_second,
        "matches_sampler_sequence": train_first == sampler_keys,
        "draws": len(train_first),
        "shuffle": False,
        "drop_last": False,
        "sequence_sha256": sequence_digest(train_first),
    }
    print()

    summary["worker_check"] = worker_check(validation, arguments.workers)

    write_json(summary, output)
    print(f"summary         : {output}")
    print(
        f"train           : {summary['train']['patients']} patients, "
        f"{summary['train']['samples']} slices"
    )
    print(
        f"validation      : {summary['validation']['patients']} patients, "
        f"{summary['validation']['samples']} slices"
    )
    print(
        f"sampler epoch 0 : {epochs['0']['total_draws']} draws, "
        f"{epochs['0']['patients']} patients, "
        f"{epochs['0']['min_samples_per_patient']}-"
        f"{epochs['0']['max_samples_per_patient']} per patient"
    )
    return 0


def worker_check(dataset: RestorationDataset, workers: int) -> dict[str, Any]:
    """Compare a small loader run with 0 workers against ``workers`` workers.

    Reported rather than asserted: DataLoader worker processes use spawn on
    Windows, and a failure here is an environment fact worth recording, not a
    reason to add fragile hacks around it.
    """
    from torch.utils.data import Subset

    probe = Subset(dataset, list(range(min(8, len(dataset)))))
    reference = list(make_validation_loader(probe, batch_size=4, num_workers=0))

    if workers <= 0:
        return {"compared_num_workers": [0], "status": "not attempted"}

    try:
        other = list(make_validation_loader(probe, batch_size=4, num_workers=workers))
    except Exception as error:  # noqa: BLE001 - an environment fact, recorded
        return {
            "compared_num_workers": [0, workers],
            "status": "unavailable in this environment",
            "detail": f"{type(error).__name__}",
        }

    keys_match = flatten_sample_keys(reference) == flatten_sample_keys(other)
    tensor_mismatches = 0
    for left, right in zip(reference, other, strict=True):
        for half in ("degraded", "clean"):
            if left[half].numpy().tobytes() != right[half].numpy().tobytes():
                tensor_mismatches += 1
    return {
        "compared_num_workers": [0, workers],
        "status": "compared",
        "samples_compared": len(probe),
        "identical_sample_keys": keys_match,
        "tensor_byte_mismatches": tensor_mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--output", default=CANONICAL_SUMMARY.as_posix())
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="worker count to compare against 0; pass 0 to skip the comparison",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug only: skip the full per-slice scan; refused for the canonical summary",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
