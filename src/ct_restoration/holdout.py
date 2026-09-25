"""The Milestone 11 held-out test protocol: one frozen, one-shot final measurement.

Everything in this module was written, committed and tested before a single
held-out test image was read. That is the point of it. By the time the test
split is opened there is nothing left to decide: which methods, which ten
checkpoints, which metrics, which aggregation, which comparisons, which words
a result may use, what happens on a failure. The test result is a measurement
of frozen decisions, never an input to a new one.

The frozen plan is :file:`configs/holdout/test_plan.yaml`. This module refuses
it unless it agrees with the constants below, with the committed split and
manifest, with every tracked Milestone 8-10 artifact it cites and with the
bytes of every checkpoint, so neither the plan nor the code can drift alone.

What is new here, and what is not
---------------------------------
Nothing that computes a number is new. Every slice is read by
:func:`~ct_restoration.evaluation.prepare_evaluation_slice`, degraded by
:func:`~ct_restoration.data.degradation.degrade_low_dose_like`, scored by
:func:`~ct_restoration.metrics.slice_metrics`, aggregated by
:func:`~ct_restoration.evaluation.aggregate_slices_to_patients`, and compared
across seeds by :mod:`ct_restoration.multiseed` - the Milestone 5-10 code.

What is new is the loop's shape. The development commands score one method
per pass. Here each test slice is read **once**, degraded **once**, and that
one degraded array is handed to all twelve scored methods: the degraded
baseline, CLAHE, five CNN checkpoints and five U-Net checkpoints. So every
method is measured on one degraded realization by construction rather than by
regenerating it twelve times, each test file is opened once, and the SHA-256
of the degraded input is written beside every metric row, so any reader can
check that all twelve tables scored identical inputs.

Three layers stand between a caller and a test pixel
----------------------------------------------------
1. the split-name gate, which opens test only with a
   :class:`~ct_restoration.evaluation.HoldoutAccess`, issued by
   :func:`run_preflight` after every check passes;
2. the row-label gate at every function that reads image content;
3. :class:`DicomOpenMonitor`, an audit hook that sees every file the process
   opens under the imaging root and refuses any that is not a test slice -
   and, during preflight, refuses them all. Its counts are the receipt's
   image-read record, measured at the file system rather than asserted.

Stress is never opened by any of them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ct_restoration.benchmark import (
    RestoreFn,
    finalise_slice_frame,
    identity_restoration,
    manifest_rows,
    write_csv,
    write_json,
)
from ct_restoration.classical.clahe import ClaheConfig, clahe_restorer
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like
from ct_restoration.data.splits import subject_sort_key
from ct_restoration.evaluation import (
    METRIC_COLUMNS,
    EvaluationConfig,
    HeldOutSplitError,
    HoldoutAccess,
    _issue_holdout_access,
    aggregate_slices_to_patients,
    paired_patient_deltas,
    prepare_evaluation_slice,
    require_rows_split_access,
    slice_weighted_summary,
    summarise_paired_deltas,
    summarise_patients,
)
from ct_restoration.evaluation_integrity import file_sha256
from ct_restoration.metrics import slice_metrics
from ct_restoration.multiseed import (
    HIGHER_IS_BETTER,
    METRICS,
    MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY,
    STATISTICAL_SEEDS,
    describe_values,
    directional_phrase,
    oriented_improvement,
    overall_directional_claim,
    paired_metric_summary,
)
from ct_restoration.run_layout import checkpoint_path, run_directory
from ct_restoration.training import SELECTION_METRIC, select_best_epoch

# --------------------------------------------------------------------------
# The frozen protocol
# --------------------------------------------------------------------------

#: The frozen plan, relative to the repository root.
PLAN_PATH = Path("configs/holdout/test_plan.yaml")

#: Where a complete execution lands, and where it is assembled first.
OUTPUT_ROOT = "outputs/metrics/holdout/test"
STAGING_ROOT = "outputs/metrics/holdout/test.incomplete"

#: Every file a complete execution writes at the output root, and no other.
OUTPUT_FILES: tuple[str, ...] = (
    "degraded_test_slices.csv",
    "degraded_test_patients.csv",
    "degraded_test_summary.json",
    "clahe_test_slices.csv",
    "clahe_test_patients.csv",
    "clahe_test_summary.json",
    "clahe_vs_degraded_test_patient_deltas.csv",
    "deterministic_methods.csv",
    "seed_level_metrics.csv",
    "paired_seed_deltas.csv",
    "learned_vs_degraded.csv",
    "learned_vs_clahe.csv",
    "validation_to_test.csv",
    "holdout_test_summary.json",
    "opening_record.json",
    "execution_log.txt",
    "execution_receipt.json",
)

#: Every file a complete execution writes in each ``seed<N>/`` directory.
PER_SEED_FILES: tuple[str, ...] = (
    "cnn_test_slices.csv",
    "cnn_test_patients.csv",
    "cnn_test_summary.json",
    "unet_test_slices.csv",
    "unet_test_patients.csv",
    "unet_test_summary.json",
    "cnn_vs_degraded_test_patient_deltas.csv",
    "cnn_vs_clahe_test_patient_deltas.csv",
    "unet_vs_degraded_test_patient_deltas.csv",
    "unet_vs_clahe_test_patient_deltas.csv",
    "unet_vs_cnn_test_patient_deltas.csv",
)

#: The only file types a held-out execution may write. No image, no array,
#: no model: nothing a viewer could open as a picture of a test patient.
PERMITTED_OUTPUT_SUFFIXES: frozenset[str] = frozenset({".csv", ".json", ".txt"})

#: Per-slice column recording the SHA-256 of the one degraded input every
#: method on that slice was scored against.
DEGRADED_DIGEST_COLUMN = "degraded_input_sha256"

#: Tolerance when checking that a mean of per-patient paired deltas equals
#: the difference of the two patient-weighted means. Both are sums of the
#: same float64 numbers in a different order, so only rounding separates them.
LINEARITY_TOLERANCE = 1e-12

#: Every check the plan's boolean prohibitions must pass: each must be the
#: YAML boolean ``false`` itself, not a string, not a zero, not absent.
PROHIBITIONS: tuple[str, ...] = (
    "stress_allowed",
    "visual_inspection_allowed",
    "image_outputs_allowed",
    "retraining_allowed",
    "fine_tuning_allowed",
    "hyperparameter_change_allowed",
    "checkpoint_selection_after_test",
    "best_seed",
    "representative_seed",
    "checkpoint_averaging",
    "model_ensembling",
    "calibration_on_test",
    "per_patient_method_selection",
    "overwrite_allowed",
    "automatic_retry_allowed",
    "significance_testing",
    "confidence_intervals",
    "p_values",
    "composite_score",
    "seed_ranking",
    "seed_exclusion",
)

#: The aggregation, exactly as the plan must state it.
AGGREGATION: dict[str, Any] = {
    "slice_to_patient": "arithmetic_mean",
    "patient_to_split": "equal_patient_arithmetic_mean",
    "primary_unit": "patient",
    "slice_weighted_summary": "secondary_descriptive_only",
    "pool_patients_and_seeds": False,
    "seed_summary_statistics": ["values", "mean", "sample_std", "min", "max"],
    "sample_std_ddof": 1,
}

#: The comparisons, exactly as the plan must state them.
COMPARISONS: dict[str, Any] = {
    "unet_vs_cnn": {
        "pairing": "same_statistical_seed",
        "raw_delta": "unet_minus_cnn",
        "oriented_improvement": "positive_means_unet_better",
    },
    "learned_vs_degraded": {
        "reference": "one_deterministic_degraded_benchmark_shared_by_every_seed",
        "raw_delta": "learned_minus_degraded",
        "oriented_improvement": "positive_means_learned_better",
    },
    "learned_vs_clahe": {
        "reference": "one_deterministic_validation_selected_clahe_shared_by_every_seed",
        "raw_delta": "learned_minus_clahe",
        "oriented_improvement": "positive_means_learned_better",
    },
    "clahe_vs_degraded": {
        "raw_delta": "clahe_minus_degraded",
        "oriented_improvement": "positive_means_clahe_better",
        "training_seed_variability": "not_applicable_deterministic_methods",
    },
    "validation_to_test": {
        "secondary_descriptive_only": True,
        "raw_change": "test_minus_validation",
        "oriented_change": "positive_means_test_numerically_better",
        "learned_basis": "five_seed_architecture_mean_on_both_splits",
        "forbidden_terms": [
            "generalization error estimate",
            "significant degradation",
            "domain shift",
        ],
    },
}

#: Every top-level key the plan must carry, and no other. A key nobody reads
#: is a key that looks like it changed something and did not.
PLAN_KEYS: frozenset[str] = frozenset(
    {
        "milestone",
        "stage",
        "protocol_version",
        "development_frozen_commit",
        "frozen_before_any_test_image_is_read",
        "split",
        "expected_subjects",
        "expected_patients",
        "expected_slices",
        "expected_slices_by_subject",
        "stress_subjects_sealed",
        "data_root",
        "split_file",
        "manifest_file",
        "multiseed_plan",
        *PROHIBITIONS,
        "methods",
        "device",
        "frozen_policies",
        "statistical_seeds",
        "learned_checkpoints",
        "metrics",
        "aggregation",
        "comparisons",
        "validation_reference",
        "interpretation_rules",
        "claim_rules",
        "failure_policy",
        "outputs",
        "execution_receipt",
    }
)

#: The fields every learned checkpoint entry carries.
CHECKPOINT_FIELDS: frozenset[str] = frozenset(
    {
        "config",
        "config_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "run_summary",
        "training_history",
        "selected_epoch",
    }
)

#: The fields the execution receipt must contain. Frozen here and in the
#: plan, so the schema cannot be adjusted after the fact to omit one.
RECEIPT_FIELDS: tuple[str, ...] = (
    "receipt_schema",
    "status",
    "git",
    "test_plan",
    "split_sha256",
    "manifest_sha256",
    "environment",
    "learned_checkpoints",
    "shared_configs",
    "test_subjects",
    "expected_patients",
    "expected_slices",
    "actual_patients",
    "actual_slices",
    "test_opening_started_utc",
    "completed_utc",
    "image_reads",
    "visual_outputs_created",
    "overwrite_occurred",
    "retry_occurred",
    "prior_attempt_detected",
    "outputs",
    "metric_values_in_receipt",
)

#: The shared frozen configs and how each is parsed for comparison.
SHARED_POLICIES: tuple[str, ...] = ("preprocessing", "degradation", "evaluation", "clahe")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class FrozenProtocol:
    """The protocol constants a plan must agree with.

    The real protocol is :data:`FROZEN`. The class exists so the synthetic
    tests can run the whole machinery on a miniature cohort; the runner takes
    no argument that could substitute another one.
    """

    milestone: int = 11
    stage: str = "protocol_frozen_execution_pending"
    protocol_version: str = "m11_holdout_test_v1"
    development_frozen_commit: str = "f1fc6dd6714e8e818d27b9c4e52dc3aadbd4318e"
    split: str = "test"
    subjects: tuple[str, ...] = ("11", "13", "19", "25", "29", "32")
    patients: int = 6
    slices: int = 941
    stress_subjects: tuple[str, ...] = ("1", "6", "39")
    data_root: str = "data/raw/chaos"
    methods: tuple[str, ...] = ("degraded", "clahe", "cnn", "unet")
    learned_methods: tuple[str, ...] = ("cnn", "unet")
    seeds: tuple[int, ...] = STATISTICAL_SEEDS
    metrics: tuple[str, ...] = METRICS
    device: str = "cuda"
    clahe_clip_limit: float = 0.5
    clahe_tile_grid_size: tuple[int, int] = (4, 4)
    degradation_global_seed: int = 2026
    output_files: tuple[str, ...] = OUTPUT_FILES
    per_seed_files: tuple[str, ...] = PER_SEED_FILES


FROZEN = FrozenProtocol()


class HoldoutProtocolError(RuntimeError):
    """The frozen protocol cannot be executed as it stands. Nothing was scored."""


class HoldoutExecutionError(RuntimeError):
    """Execution stopped after the test split was opened.

    Never caught and retried. The partial record stays where it is, the
    reason is written beside it, and whether anything may be re-run is a
    decision for an integrity review, not for this code.
    """


# --------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------


def text_sha256_lf(path: Path) -> str:
    """SHA-256 of a text file with CRLF read as LF.

    For tracked text files whose line endings Git does not pin. Under
    ``core.autocrlf=true`` their bytes differ between checkouts while their
    content does not, so a raw byte hash would refuse a scientifically
    identical file on one machine and accept it on another. Files the
    checkpoints hash byte for byte are pinned LF in ``.gitattributes`` and
    are checked with a raw digest instead.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and bytes."""
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{values.dtype.str}|{values.shape}|".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutProtocolError(message)


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def _clahe_settings(frozen: FrozenProtocol) -> dict[str, Any]:
    return {
        "algorithm": "opencv_clahe_v1",
        "input_quantization_bits": 8,
        "clip_limit": frozen.clahe_clip_limit,
        "tile_grid_size": list(frozen.clahe_tile_grid_size),
    }


def validate_plan(plan: Any, frozen: FrozenProtocol = FROZEN) -> dict[str, Any]:
    """Check the plan against the frozen constants. Reads nothing from disk.

    Refuses rather than repairs: a plan that permits stress, adds a seed,
    drops a method or names an image output is a different protocol, however
    well-formed it is otherwise.

    Raises:
        HoldoutProtocolError: naming the first disagreement.
    """
    _require(isinstance(plan, dict), f"the plan must be a mapping, got {type(plan).__name__}")
    missing = sorted(PLAN_KEYS - set(plan))
    unknown = sorted(set(plan) - PLAN_KEYS)
    _require(not missing, f"the plan is missing key(s) {missing}")
    _require(not unknown, f"the plan has unrecognised key(s) {unknown}; nothing would read them")

    _require(plan["milestone"] == frozen.milestone, f"milestone must be {frozen.milestone}")
    _require(plan["stage"] == frozen.stage, f"stage must be {frozen.stage!r}")
    _require(
        plan["protocol_version"] == frozen.protocol_version,
        f"protocol_version must be {frozen.protocol_version!r}",
    )
    _require(
        plan["development_frozen_commit"] == frozen.development_frozen_commit,
        f"development_frozen_commit must be {frozen.development_frozen_commit}",
    )
    _require(
        plan["frozen_before_any_test_image_is_read"] is True,
        "frozen_before_any_test_image_is_read must be true",
    )

    for name in PROHIBITIONS:
        _require(
            plan[name] is False,
            f"{name} must be the boolean false, got {plan[name]!r}. The protocol forbids it; "
            "a plan that does not is a different protocol.",
        )

    _require(
        plan["split"] == frozen.split, f"split must be {frozen.split!r}, got {plan['split']!r}"
    )
    _require(
        plan["expected_subjects"] == list(frozen.subjects),
        f"expected_subjects must be {list(frozen.subjects)}, got {plan['expected_subjects']!r}",
    )
    _require(
        plan["expected_patients"] == frozen.patients == len(frozen.subjects),
        f"expected_patients must be {frozen.patients}",
    )
    _require(plan["expected_slices"] == frozen.slices, f"expected_slices must be {frozen.slices}")
    by_subject = plan["expected_slices_by_subject"]
    _require(
        isinstance(by_subject, dict) and sorted(by_subject) == sorted(frozen.subjects),
        "expected_slices_by_subject must name exactly the expected subjects",
    )
    counts = list(by_subject.values())
    _require(
        all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts),
        "expected_slices_by_subject values must be positive integers",
    )
    _require(
        sum(counts) == frozen.slices,
        f"expected_slices_by_subject sums to {sum(counts)}, not {frozen.slices}",
    )
    _require(
        plan["stress_subjects_sealed"] == list(frozen.stress_subjects),
        f"stress_subjects_sealed must be {list(frozen.stress_subjects)}",
    )
    _require(
        not set(frozen.stress_subjects) & set(frozen.subjects),
        "a stress subject may not also be a test subject",
    )
    _require(plan["data_root"] == frozen.data_root, f"data_root must be {frozen.data_root!r}")

    for name in ("split_file", "manifest_file", "multiseed_plan"):
        entry = plan[name]
        _require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"{name} must have exactly a path and a sha256",
        )
        _require(_is_digest(entry["sha256"]), f"{name}.sha256 is not a SHA-256 digest")

    _require(
        plan["methods"] == list(frozen.methods),
        f"methods must be exactly {list(frozen.methods)}, got {plan['methods']!r}. The method "
        "set is part of the protocol; no method is added, dropped or substituted.",
    )
    _require(plan["device"] == frozen.device, f"device must be {frozen.device!r}")

    policies = plan["frozen_policies"]
    _require(
        isinstance(policies, dict) and sorted(policies) == sorted(SHARED_POLICIES),
        f"frozen_policies must be exactly {sorted(SHARED_POLICIES)}",
    )
    for name in SHARED_POLICIES:
        entry = policies[name]
        _require(isinstance(entry, dict), f"frozen_policies.{name} must be a mapping")
        for key in ("config", "sha256_lf", "settings"):
            _require(key in entry, f"frozen_policies.{name} is missing {key!r}")
        _require(_is_digest(entry["sha256_lf"]), f"frozen_policies.{name}.sha256_lf is malformed")
    _require(
        policies["clahe"]["settings"] == _clahe_settings(frozen),
        f"the CLAHE settings must be {_clahe_settings(frozen)}; CLAHE is not retuned on test",
    )
    _require(
        policies["clahe"].get("retuning_allowed") is False, "CLAHE retuning_allowed must be false"
    )
    degradation = policies["degradation"]
    _require(
        degradation["settings"].get("global_seed") == frozen.degradation_global_seed,
        f"the degradation global_seed must be {frozen.degradation_global_seed}",
    )
    _require(
        degradation.get("one_realization_per_slice_shared_by_every_method") is True
        and degradation.get("depends_on_training_seed") is False,
        "the degradation must be one realization per slice, shared, independent of training seed",
    )

    seeds = plan["statistical_seeds"]
    _require(
        seeds == list(frozen.seeds),
        f"statistical_seeds must be exactly {list(frozen.seeds)}, got {seeds!r}. No seed is "
        "added, dropped or chosen.",
    )
    learned = plan["learned_checkpoints"]
    _require(
        isinstance(learned, dict) and sorted(learned) == sorted(frozen.learned_methods),
        f"learned_checkpoints must be exactly {sorted(frozen.learned_methods)}",
    )
    digests: list[str] = []
    for method in frozen.learned_methods:
        entries = learned[method]
        _require(isinstance(entries, dict), f"learned_checkpoints.{method} must be a mapping")
        declared = sorted(entries)
        _require(
            declared == sorted(frozen.seeds),
            f"learned_checkpoints.{method} declares seeds {declared}, but the protocol's are "
            f"{sorted(frozen.seeds)}. Every frozen checkpoint enters exactly once; no missing "
            "seed, no extra seed.",
        )
        for seed in frozen.seeds:
            entry = entries[seed]
            label = f"learned_checkpoints.{method}.{seed}"
            _require(
                isinstance(entry, dict) and set(entry) == CHECKPOINT_FIELDS,
                f"{label} must have exactly {sorted(CHECKPOINT_FIELDS)}",
            )
            _require(_is_digest(entry["config_sha256"]), f"{label}.config_sha256 is malformed")
            _require(
                _is_digest(entry["checkpoint_sha256"]), f"{label}.checkpoint_sha256 is malformed"
            )
            _require(
                entry["checkpoint"] == checkpoint_path(method, seed).as_posix(),
                f"{label}.checkpoint must be {checkpoint_path(method, seed).as_posix()}",
            )
            run_dir = run_directory(method, seed)
            _require(
                entry["run_summary"] == (run_dir / "run_summary.json").as_posix()
                and entry["training_history"] == (run_dir / "training_history.csv").as_posix(),
                f"{label} must point at the tracked records in {run_dir.as_posix()}",
            )
            epoch = entry["selected_epoch"]
            _require(
                isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 1,
                f"{label}.selected_epoch must be an integer >= 1 (epoch 0 is never selectable)",
            )
            digests.append(entry["checkpoint_sha256"])
    _require(len(set(digests)) == len(digests), "two planned checkpoints share one SHA-256")

    _require(plan["metrics"] == list(frozen.metrics), f"metrics must be {list(frozen.metrics)}")
    _require(plan["aggregation"] == AGGREGATION, f"aggregation must be exactly {AGGREGATION}")
    _require(plan["comparisons"] == COMPARISONS, "comparisons must be exactly the frozen set")

    reference = plan["validation_reference"]
    _require(
        isinstance(reference, dict) and sorted(reference) == sorted(frozen.methods),
        f"validation_reference must cover exactly {sorted(frozen.methods)}",
    )
    for method in ("degraded", "clahe"):
        entry = reference[method]
        _require(
            isinstance(entry, dict) and set(entry) == {"summary", "sha256_lf"},
            f"validation_reference.{method} must have exactly a summary and a sha256_lf",
        )
    for method in frozen.learned_methods:
        _require(
            sorted(reference[method]) == sorted(frozen.seeds),
            f"validation_reference.{method} must cover exactly the frozen seeds",
        )

    rules = plan["interpretation_rules"]
    consistent = rules.get("directionally_consistent", {})
    _require(
        consistent.get("requires_minimum_seeds_favouring") == MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY
        and consistent.get("of_seeds") == len(frozen.seeds)
        and consistent.get("requires_mean_oriented_improvement_same_direction") is True,
        "the directional rule must be the Milestone 10 rule: the mean and at least "
        f"{MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY} of {len(frozen.seeds)} seeds",
    )
    _require(
        rules.get("three_of_five_may_be_called_consistent") is False,
        "three of five seeds may never be called consistent",
    )
    _require(rules.get("causal_claims") is False, "causal_claims must be false")

    claims = plan["claim_rules"]
    _require(
        claims.get("requires_positive_architecture_mean_improvement") is True
        and claims.get("requires_computed_from_frozen_test_outputs") is True,
        "a claim must require a positive five-seed mean computed from the frozen outputs",
    )
    failure = plan["failure_policy"]
    _require(
        failure.get("test_result_may_trigger") == []
        and failure.get("silent_patch_and_rerun") is False
        and failure.get("restart_requires_integrity_review") is True,
        "the failure policy must forbid every result-driven change and any silent rerun",
    )

    outputs = plan["outputs"]
    _require(
        outputs.get("root") == OUTPUT_ROOT and outputs.get("staging") == STAGING_ROOT,
        f"outputs must go to {OUTPUT_ROOT} via {STAGING_ROOT}",
    )
    _require(
        outputs.get("files") == list(frozen.output_files)
        and outputs.get("per_seed_directory") == "seed{seed}"
        and outputs.get("per_seed_files") == list(frozen.per_seed_files),
        "the output layout must be exactly the frozen layout",
    )
    _require(outputs.get("image_outputs") == [], "image_outputs must be an empty list")
    for name in [*frozen.output_files, *frozen.per_seed_files]:
        _require(
            Path(name).suffix in PERMITTED_OUTPUT_SUFFIXES,
            f"{name} is not a permitted output type; no visual output may be planned",
        )
    receipt = plan["execution_receipt"]
    _require(
        receipt.get("path") == f"{OUTPUT_ROOT}/execution_receipt.json"
        and receipt.get("required_fields") == list(RECEIPT_FIELDS)
        and receipt.get("metric_values") is False,
        "the execution receipt schema must be exactly the frozen one, with no metric values",
    )
    return plan


def load_test_plan(path: Path, frozen: FrozenProtocol = FROZEN) -> dict[str, Any]:
    """Read and validate the plan. Returns it with its own SHA-256 attached."""
    _require(Path(path).is_file(), f"{Path(path).as_posix()} is missing; there is no protocol")
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_plan(plan, frozen)
    plan = dict(plan)
    plan["_source"] = {"path": Path(path).as_posix(), "sha256": file_sha256(Path(path))}
    return plan


# --------------------------------------------------------------------------
# Preflight: every check, before any test image is read
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GitState:
    """What the repository looks like at the moment of preflight."""

    commit: str
    porcelain: str
    frozen_commit_is_ancestor: bool
    branch: str | None = None
    head_in_origin_main: bool | None = None


def read_git_state(root: Path, frozen: FrozenProtocol = FROZEN) -> GitState:
    """Ask git, read-only. ``git status`` counts untracked files too."""

    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )

    commit = git("rev-parse", "HEAD").stdout.strip()
    porcelain = git("status", "--porcelain", "--untracked-files=all").stdout
    ancestor = (
        git("merge-base", "--is-ancestor", frozen.development_frozen_commit, "HEAD").returncode == 0
    )
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None
    origin = git("rev-parse", "--verify", "--quiet", "refs/remotes/origin/main")
    published = (
        git("merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main").returncode == 0
        if origin.returncode == 0
        else None
    )
    return GitState(commit, porcelain, ancestor, branch, published)


@dataclass
class CheckReport:
    """Every preflight check, in order, with its outcome. Refuses as a whole."""

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(passed), detail))
        return bool(passed)

    def attempt(self, name: str, check: Callable[[], str | None]) -> bool:
        """Run one check; any exception is that check's failure, not a crash."""
        try:
            detail = check() or ""
        except Exception as error:  # noqa: BLE001 - recorded as the check's failure
            return self.add(name, False, f"{type(error).__name__}: {error}")
        return self.add(name, True, detail)

    @property
    def failed(self) -> list[tuple[str, str]]:
        return [(name, detail) for name, passed, detail in self.checks if not passed]

    @property
    def passed_all(self) -> bool:
        return bool(self.checks) and not self.failed

    def require_all(self) -> None:
        if not self.passed_all:
            lines = "\n  ".join(f"{name}: {detail}" for name, detail in self.failed)
            raise HoldoutProtocolError(
                f"preflight refused - {len(self.failed)} of {len(self.checks)} checks failed; "
                f"no test image was read:\n  {lines}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in self.checks
            ],
            "passed": sum(1 for _, passed, _ in self.checks if passed),
            "failed": len(self.failed),
        }


def verify_repository(report: CheckReport, git: GitState, frozen: FrozenProtocol = FROZEN) -> None:
    """The evaluating code is exactly one commit, with nothing uncommitted."""
    report.add(
        "git commit recorded",
        bool(_HEX40.match(git.commit or "")),
        git.commit or "no commit",
    )
    dirty = [line for line in git.porcelain.splitlines() if line.strip()]
    report.add(
        "git working tree clean",
        not dirty,
        "clean" if not dirty else f"{len(dirty)} modified or untracked: {dirty[:5]}",
    )
    report.add(
        "development frozen commit is an ancestor of HEAD",
        git.frozen_commit_is_ancestor,
        frozen.development_frozen_commit,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _equal(label: str, actual: Any, expected: Any) -> str:
    if actual != expected:
        raise HoldoutProtocolError(f"{label} is {actual!r}, expected {expected!r}")
    return f"{label} verified"


def parse_policy(name: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """A shared config, parsed the way its command parses it."""
    if name == "preprocessing":
        settings = document["preprocessing"]
        return {
            "window_center": settings["window_center"],
            "window_width": settings["window_width"],
            "image_size": list(settings["image_size"]),
            "interpolation": settings["interpolation"],
        }
    if name == "degradation":
        return DegradationConfig.from_mapping(dict(document)).as_dict()
    if name == "evaluation":
        return EvaluationConfig.from_mapping(dict(document)).as_dict()
    if name == "clahe":
        return ClaheConfig.from_mapping(dict(document)).as_dict()
    raise HoldoutProtocolError(f"unknown policy {name!r}")


def verify_tracked_inputs(
    report: CheckReport, plan: dict[str, Any], root: Path, frozen: FrozenProtocol = FROZEN
) -> None:
    """Every input the plan cites that is tracked in git. Metadata only.

    Reads CSV, YAML and JSON. Never a DICOM, never a checkpoint's bytes -
    so it runs in a fresh clone, which has neither.
    """
    split_path = root / plan["split_file"]["path"]
    manifest_path = root / plan["manifest_file"]["path"]
    report.attempt(
        "split file SHA-256",
        lambda: _equal("split SHA-256", file_sha256(split_path), plan["split_file"]["sha256"]),
    )
    report.attempt(
        "manifest file SHA-256",
        lambda: _equal(
            "manifest SHA-256", file_sha256(manifest_path), plan["manifest_file"]["sha256"]
        ),
    )

    def cohort_from_split() -> str:
        table = pd.read_csv(split_path, dtype={"subject_id": str})
        test = sorted(table.loc[table["split"] == frozen.split, "subject_id"], key=subject_sort_key)
        stress = sorted(table.loc[table["split"] == "stress", "subject_id"], key=subject_sort_key)
        _equal("split-file test subjects", test, list(frozen.subjects))
        _equal("split-file stress subjects", stress, list(frozen.stress_subjects))
        return f"test {test}; stress {stress}"

    def cohort_from_manifest() -> str:
        manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
        test = manifest[manifest["split"] == frozen.split]
        counts = {str(k): int(v) for k, v in test.groupby("subject_id").size().items()}
        _equal("manifest test slices by subject", counts, dict(plan["expected_slices_by_subject"]))
        _equal("manifest test slices", len(test), frozen.slices)
        _equal("manifest test patients", test["subject_id"].nunique(), frozen.patients)
        elsewhere = manifest[
            manifest["subject_id"].isin(frozen.subjects) & (manifest["split"] != frozen.split)
        ]
        _equal("test-subject rows labelled with another split", len(elsewhere), 0)
        stress = manifest[manifest["split"] == "stress"]
        _equal(
            "manifest stress subjects",
            sorted(set(stress["subject_id"]), key=subject_sort_key),
            list(frozen.stress_subjects),
        )
        _equal("duplicate test sample keys", int(test["relative_dicom_path"].duplicated().sum()), 0)
        return f"{len(test)} test slices over {test['subject_id'].nunique()} patients"

    report.attempt("test cohort from the split file", cohort_from_split)
    report.attempt("test cohort from the manifest", cohort_from_manifest)

    multiseed_path = root / plan["multiseed_plan"]["path"]
    report.attempt(
        "Milestone 10 plan SHA-256",
        lambda: _equal(
            "multiseed plan SHA-256", file_sha256(multiseed_path), plan["multiseed_plan"]["sha256"]
        ),
    )

    policies = plan["frozen_policies"]
    for name in SHARED_POLICIES:
        entry = policies[name]

        def policy_check(entry: dict[str, Any] = entry, name: str = name) -> str:
            path = root / entry["config"]
            _equal(f"{entry['config']} SHA-256 (LF)", text_sha256_lf(path), entry["sha256_lf"])
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            return _equal(f"{name} policy", parse_policy(name, document), entry["settings"])

        report.attempt(f"frozen {name} config", policy_check)

    reference = plan["validation_reference"]

    def reference_check(method: str, entry: dict[str, Any], seed: int | None) -> str:
        path = root / entry["summary"]
        _equal(f"{entry['summary']} SHA-256 (LF)", text_sha256_lf(path), entry["sha256_lf"])
        summary = _read_json(path)
        _equal(f"{entry['summary']} split", summary.get("split"), "validation")
        _equal(
            f"{entry['summary']} evaluation policy",
            summary.get("evaluation_config"),
            policies["evaluation"]["settings"],
        )
        _equal(
            f"{entry['summary']} degradation policy",
            summary.get("degradation_config"),
            policies["degradation"]["settings"],
        )
        if method == "degraded":
            _equal(
                f"{entry['summary']} preprocessing",
                summary.get("preprocessing"),
                policies["preprocessing"]["settings"],
            )
        if method == "clahe":
            _equal(
                f"{entry['summary']} CLAHE config",
                summary.get("clahe_config"),
                policies["clahe"]["settings"],
            )
        if seed is not None:
            planned = plan["learned_checkpoints"][method][seed]
            recorded = summary["checkpoint"]
            _equal(f"{entry['summary']} seed", recorded.get("seed"), seed)
            _equal(
                f"{entry['summary']} config SHA-256",
                recorded.get("config_sha256"),
                planned["config_sha256"],
            )
            _equal(
                f"{entry['summary']} checkpoint SHA-256",
                recorded["provenance"]["recomputed_from_files"]["checkpoint_file_sha256"],
                planned["checkpoint_sha256"],
            )
        for metric in frozen.metrics:
            value = summary["primary_result"][metric]["mean"]
            if not (isinstance(value, int | float) and math.isfinite(float(value))):
                raise HoldoutProtocolError(f"{entry['summary']} {metric} is not finite")
        return f"{entry['summary']} verified"

    for method in ("degraded", "clahe"):
        report.attempt(
            f"validation reference {method}",
            lambda method=method: reference_check(method, reference[method], None),
        )
    for method in frozen.learned_methods:
        for seed in frozen.seeds:
            report.attempt(
                f"validation reference {method} {seed}",
                lambda method=method, seed=seed: reference_check(
                    method, reference[method][seed], seed
                ),
            )

    multiseed = None
    if multiseed_path.is_file():
        multiseed = yaml.safe_load(multiseed_path.read_text(encoding="utf-8"))

    for method in frozen.learned_methods:
        for seed in frozen.seeds:
            entry = plan["learned_checkpoints"][method][seed]

            def learned_check(method: str = method, seed: int = seed, entry=entry) -> str:
                config = root / entry["config"]
                _equal(f"{entry['config']} SHA-256", file_sha256(config), entry["config_sha256"])
                if multiseed is None:
                    raise HoldoutProtocolError("the Milestone 10 plan is missing")
                pinned = multiseed["runs"][method][seed]
                _equal(f"M10 plan config for {method} {seed}", pinned["config"], entry["config"])
                _equal(
                    f"M10 plan SHA for {method} {seed}", pinned["sha256"], entry["config_sha256"]
                )
                document = yaml.safe_load(config.read_text(encoding="utf-8"))
                _equal(f"{entry['config']} training.seed", document["training"]["seed"], seed)

                summary = _read_json(root / entry["run_summary"])
                _equal(f"{entry['run_summary']} seed", summary["training"]["seed"], seed)
                _equal(
                    f"{entry['run_summary']} config SHA-256",
                    summary["config"]["sha256"],
                    entry["config_sha256"],
                )
                _equal(
                    f"{entry['run_summary']} checkpoint path",
                    summary["checkpoint"]["path"],
                    entry["checkpoint"],
                )
                _equal(
                    f"{entry['run_summary']} checkpoint SHA-256",
                    summary["checkpoint"]["sha256"],
                    entry["checkpoint_sha256"],
                )
                _equal(
                    f"{entry['run_summary']} checkpoint tracked_in_git",
                    summary["checkpoint"]["tracked_in_git"],
                    False,
                )
                _equal(
                    f"{entry['run_summary']} selected epoch",
                    summary["checkpoint_selection"]["best_epoch"],
                    entry["selected_epoch"],
                )
                history = pd.read_csv(root / entry["training_history"])
                recomputed = int(
                    select_best_epoch(history, SELECTION_METRIC, epoch_zero_eligible=False)
                )
                _equal(
                    f"{entry['training_history']} recomputed selection",
                    recomputed,
                    entry["selected_epoch"],
                )
                return f"{method} {seed}: config, run summary and history agree"

            report.attempt(f"tracked evidence {method} {seed}", learned_check)


def verify_checkpoint_binaries(
    report: CheckReport, plan: dict[str, Any], root: Path, frozen: FrozenProtocol = FROZEN
) -> None:
    """Every checkpoint exists and is, byte for byte, the one the plan froze."""
    for method in frozen.learned_methods:
        for seed in frozen.seeds:
            entry = plan["learned_checkpoints"][method][seed]
            path = root / entry["checkpoint"]

            def binary_check(path: Path = path, entry=entry) -> str:
                if not path.is_file():
                    raise HoldoutProtocolError(
                        f"{entry['checkpoint']} is missing. It is git-ignored, so it must be "
                        "the original file, whose SHA-256 is frozen; a retrained replacement "
                        "is a different checkpoint and would fail this check anyway."
                    )
                return _equal(
                    f"{entry['checkpoint']} SHA-256", file_sha256(path), entry["checkpoint_sha256"]
                )

            report.attempt(f"checkpoint bytes {method} {seed}", binary_check)


def verify_destinations_absent(report: CheckReport, plan: dict[str, Any], root: Path) -> None:
    """Nothing is ever overwritten, and a previous attempt is never resumed."""
    for key in ("root", "staging"):
        location = plan["outputs"][key]
        report.add(
            f"output {key} absent",
            not (root / location).exists(),
            f"{location} {'exists' if (root / location).exists() else 'absent'}",
        )


@dataclass
class PreflightResult:
    """The outcome of preflight. ``access`` exists only if every check passed."""

    report: CheckReport
    access: HoldoutAccess | None = None
    restorers: dict[str, RestoreFn] | None = None
    provenance: dict[str, Any] | None = None


#: ``(plan, root) -> ({name: restorer}, {name: provenance record})`` for the
#: ten learned checkpoints. Injected so the synthetic tests need no torch
#: checkpoint; the runner passes :func:`load_learned_restorers`.
ModelLoader = Callable[[dict[str, Any], Path], tuple[dict[str, RestoreFn], dict[str, Any]]]


def learned_name(method: str, seed: int) -> str:
    return f"{method}_seed{seed}"


def run_preflight(
    plan: dict[str, Any],
    root: Path,
    git: GitState,
    load_models: ModelLoader,
    frozen: FrozenProtocol = FROZEN,
) -> PreflightResult:
    """Every check before the test split is opened, then - only then - access.

    Nothing here reads a test image: the checks read metadata, tracked JSON,
    CSV and YAML, and checkpoint bytes, and the caller runs this inside a
    sealed :class:`DicomOpenMonitor` so that stays true mechanically.
    """
    report = CheckReport()
    report.add("plan validated against the frozen protocol", True, plan["_source"]["path"])
    verify_repository(report, git, frozen)
    failures_before_inputs = len(report.failed)
    verify_tracked_inputs(report, plan, root, frozen)
    verify_checkpoint_binaries(report, plan, root, frozen)
    inputs_verified = len(report.failed) == failures_before_inputs
    verify_destinations_absent(report, plan, root)

    # The checkpoints are loaded and their provenance proved whenever every
    # file they depend on checked out, even if the tree is dirty, so a
    # preflight-only run reports on them too. Access still needs everything.
    restorers: dict[str, RestoreFn] | None = None
    provenance: dict[str, Any] | None = None
    if inputs_verified:
        try:
            restorers, provenance = load_models(plan, root)
            expected = {
                learned_name(method, seed)
                for method in frozen.learned_methods
                for seed in frozen.seeds
            }
            _equal("loaded learned checkpoints", sorted(restorers), sorted(expected))
            report.add("learned checkpoints loaded and provenance verified", True, "10 of 10")
        except Exception as error:  # noqa: BLE001 - recorded as the check's failure
            report.add(
                "learned checkpoints loaded and provenance verified",
                False,
                f"{type(error).__name__}: {error}",
            )
            restorers = provenance = None

    access = None
    if report.passed_all:
        access = _issue_holdout_access(plan["_source"]["sha256"], git.commit)
    return PreflightResult(report, access, restorers, provenance)


def load_learned_restorers(
    plan: dict[str, Any], root: Path, device: str = "cuda"
) -> tuple[dict[str, RestoreFn], dict[str, Any]]:
    """The ten frozen checkpoints, loaded and proved, as benchmark restorers.

    Each goes through the same provenance gate the validation evaluations
    cleared - config bytes, checkpoint bytes, and the tracked history re-run
    through the predeclared selection rule - before it is accepted.
    """
    import torch

    from ct_restoration.evaluation_integrity import (
        config_training_seed,
        verify_checkpoint_provenance,
    )
    from ct_restoration.models import cnn, unet
    from ct_restoration.models.adapter import torch_restorer

    builders = {
        "cnn": (cnn.ResidualCnnConfig, cnn.build_model, cnn.CANONICAL_PARAMETER_COUNT),
        "unet": (
            unet.LightweightResidualUnetConfig,
            unet.build_model,
            unet.CANONICAL_PARAMETER_COUNT,
        ),
    }
    trainers = {"cnn": "scripts/train_cnn.py", "unet": "scripts/train_unet.py"}
    target = torch.device(device)
    restorers: dict[str, RestoreFn] = {}
    provenance: dict[str, Any] = {}
    for method in FROZEN.learned_methods:
        config_class, build, parameters = builders[method]
        for seed in FROZEN.seeds:
            entry = plan["learned_checkpoints"][method][seed]
            config_path = root / entry["config"]
            checkpoint = root / entry["checkpoint"]
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            record = verify_checkpoint_provenance(
                payload,
                checkpoint,
                config_path,
                root / Path(entry["run_summary"]).parent,
                method=f"{method} seed {seed} (Milestone 11 held-out test)",
                trainer=trainers[method],
                expected_seed=config_training_seed(document),
            )
            _equal(
                f"{method} {seed} checkpoint epoch", int(payload["epoch"]), entry["selected_epoch"]
            )
            model = build(config_class.from_mapping(document), target)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            _equal(f"{method} {seed} parameter count", model.parameter_count(), parameters)
            name = learned_name(method, seed)
            restorers[name] = torch_restorer(model, target)
            provenance[name] = {
                "method": method,
                "seed": seed,
                "epoch": int(payload["epoch"]),
                "config": entry["config"],
                "config_sha256": entry["config_sha256"],
                "checkpoint": entry["checkpoint"],
                "checkpoint_sha256": record["recomputed_from_files"]["checkpoint_file_sha256"],
                "trainable_parameters": int(model.parameter_count()),
                "provenance": record,
            }
    return restorers, provenance


# --------------------------------------------------------------------------
# The file-system record of what was read
# --------------------------------------------------------------------------

_ACTIVE_MONITORS: list[DicomOpenMonitor] = []
_HOOK_INSTALLED = False


def _audit(event: str, arguments: tuple) -> None:
    if event != "open" or not _ACTIVE_MONITORS:
        return
    for monitor in tuple(_ACTIVE_MONITORS):
        monitor._observe(arguments)


def _normalise(path: Any) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    try:
        return os.path.normcase(os.path.abspath(os.fsdecode(path)))
    except TypeError:
        return None


class DicomOpenMonitor:
    """Counts, and polices, every file the process opens under the imaging root.

    Built on :func:`sys.addaudithook`. Python raises an ``open`` audit event
    for every file opened through ``builtins.open``, ``io.open`` or
    ``os.open`` - underneath pydicom, NumPy, OpenCV's Python layer or anything
    else that could read a DICOM file - so the count is taken where files are
    actually opened, not where a loop believes it read one.

    A file whose manifest split is not in ``readable`` is refused: the open
    raises before a byte is read. A file under the root that the manifest does
    not name at all is refused too. With ``readable`` empty the whole root is
    sealed, which is how preflight runs.

    The hook is installed once per process and does nothing while no monitor
    is active, so leaving one behind in a test session costs a function call.
    """

    def __init__(
        self,
        data_root: Path,
        split_by_key: Mapping[str, str] | None = None,
        readable: Iterable[str] = (),
    ) -> None:
        self._root = os.path.normcase(os.path.abspath(data_root))
        self._prefix = self._root.rstrip("\\/") + os.sep
        self._split_by_path = {
            _normalise(Path(data_root) / key): split for key, split in (split_by_key or {}).items()
        }
        self.readable = frozenset(readable)
        self.opened: Counter[str] = Counter()
        self.refused: Counter[str] = Counter()
        self._distinct: dict[str, set[str]] = {}

    def _observe(self, arguments: tuple) -> None:
        path = _normalise(arguments[0]) if arguments else None
        if path is None or not (path == self._root or path.startswith(self._prefix)):
            return
        split = self._split_by_path.get(path, "outside_manifest")
        if split not in self.readable:
            self.refused[split] += 1
            relative = path[len(self._prefix) :] if path.startswith(self._prefix) else "."
            raise HeldOutSplitError(
                f"refused to open a {split!r} file under the imaging root ({relative}); this "
                f"process may read {sorted(self.readable) or 'nothing'} there."
            )
        self.opened[split] += 1
        self._distinct.setdefault(split, set()).add(path)

    def distinct(self, split: str) -> int:
        return len(self._distinct.get(split, ()))

    def __enter__(self) -> DicomOpenMonitor:
        global _HOOK_INSTALLED
        if not _HOOK_INSTALLED:
            sys.addaudithook(_audit)
            _HOOK_INSTALLED = True
        _ACTIVE_MONITORS.append(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        _ACTIVE_MONITORS.remove(self)

    def record(self) -> dict[str, Any]:
        splits = ("train", "validation", "test", "stress", "outside_manifest")
        return {
            "measured_by": "sys.addaudithook 'open' events under the imaging root",
            "readable_splits": sorted(self.readable),
            "file_opens_by_split": {split: int(self.opened.get(split, 0)) for split in splits},
            "distinct_files_opened_by_split": {split: self.distinct(split) for split in splits},
            "refused_opens_by_split": {split: int(self.refused.get(split, 0)) for split in splits},
        }


def split_by_key(manifest: pd.DataFrame) -> dict[str, str]:
    """Every manifest sample key and its split, for the monitor."""
    return dict(zip(manifest["relative_dicom_path"], manifest["split"].astype(str), strict=True))


# --------------------------------------------------------------------------
# Scoring: one read, one degradation, every method
# --------------------------------------------------------------------------


def score_shared_inputs(
    rows: pd.DataFrame,
    data_root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    restorers: Mapping[str, RestoreFn],
    access: HoldoutAccess | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> dict[str, pd.DataFrame]:
    """Score every method on every row, from one shared degraded input per slice.

    Each row is read once and degraded once. Every method receives its own
    copy of that one array, so no method can alter what the next one sees -
    and in case one tried in place, the shared array is re-hashed after all
    methods have run and a change stops the evaluation.

    The per-slice records are the Milestone 5-10 records field for field, so
    a single-method call reproduces :func:`~ct_restoration.benchmark.evaluate_slices`
    exactly; the tests hold it to that. One column is added: the SHA-256 of
    the degraded input.

    Raises:
        HeldOutSplitError: a row is not open to this caller.
        HoldoutExecutionError: a method modified the shared degraded input.
    """
    require_rows_split_access(rows, access)
    if not restorers:
        raise HoldoutExecutionError("no method to score")
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in restorers}
    digests: list[str] = []
    total = len(rows)
    for position, row in enumerate(rows.itertuples(), start=1):
        key = row.relative_dicom_path
        clean, body, interior = prepare_evaluation_slice(data_root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation)
        digest = array_sha256(degraded)
        for name, restore in restorers.items():
            restored = restore(degraded.copy())
            measured = slice_metrics(clean, restored, body, interior, evaluation.ssim)
            records[name].append(
                {
                    "subject_id": row.subject_id,
                    "source_archive": row.source_archive,
                    "acquisition_group": row.acquisition_group,
                    "relative_dicom_path": key,
                    "geometric_slice_index": int(row.geometric_slice_index),
                    **measured,
                    "body_pixel_fraction": float(body.mean()),
                    "body_ssim_interior_fraction": float(interior.mean()),
                }
            )
        if array_sha256(degraded) != digest:
            raise HoldoutExecutionError(
                f"the shared degraded input for slice {position} changed while the methods "
                "ran; one method modified its input in place, so the methods did not all "
                "see the same image"
            )
        digests.append(digest)
        if progress and progress_every and (position % progress_every == 0 or position == total):
            progress(f"  scored slice {position:>5} / {total} with {len(restorers)} methods")

    frames: dict[str, pd.DataFrame] = {}
    for name, rows_for_method in records.items():
        frame = finalise_slice_frame(rows_for_method)
        frame[DEGRADED_DIGEST_COLUMN] = digests
        frames[name] = frame
    return frames


# --------------------------------------------------------------------------
# Checks on the scored tables, before anything is written
# --------------------------------------------------------------------------


def require_finite_metrics(frame: pd.DataFrame, name: str) -> None:
    """A NaN or infinite metric stops the run; it is never averaged in."""
    values = frame[list(METRIC_COLUMNS)].to_numpy(dtype=np.float64)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise HoldoutExecutionError(
            f"{name}: {bad} non-finite metric value(s). A non-finite metric means the "
            "measurement is broken; it is reported as a failure, never summarised."
        )


def require_cohort(frame: pd.DataFrame, plan: dict[str, Any], name: str) -> None:
    """Exactly the frozen patients and exactly their frozen slice counts."""
    counts = {str(k): int(v) for k, v in frame.groupby("subject_id").size().items()}
    if counts != dict(plan["expected_slices_by_subject"]):
        raise HoldoutExecutionError(
            f"{name}: scored slices by subject {counts}, expected "
            f"{dict(plan['expected_slices_by_subject'])}"
        )
    if len(frame) != plan["expected_slices"] or len(counts) != plan["expected_patients"]:
        raise HoldoutExecutionError(
            f"{name}: {len(frame)} slices over {len(counts)} patients, expected "
            f"{plan['expected_slices']} over {plan['expected_patients']}"
        )


def require_shared_samples(frames: Mapping[str, pd.DataFrame]) -> None:
    """Every table scored the same slices, in the same order, from the same input.

    Two tables that agree on sample keys but disagree on the degraded-input
    digest scored different corruptions of the same slice - a comparison
    between them would be between corruptions, not methods.
    """
    names = list(frames)
    if not names:
        raise HoldoutExecutionError("no scored table")
    first = frames[names[0]]
    keys = list(first["relative_dicom_path"])
    if len(set(keys)) != len(keys):
        raise HoldoutExecutionError(f"{names[0]}: duplicate sample keys")
    digests = list(first[DEGRADED_DIGEST_COLUMN])
    for name in names[1:]:
        frame = frames[name]
        if list(frame["relative_dicom_path"]) != keys:
            raise HoldoutExecutionError(
                f"{name} and {names[0]} did not score the same sample keys in the same order"
            )
        if list(frame[DEGRADED_DIGEST_COLUMN]) != digests:
            mismatched = sum(
                1 for a, b in zip(frame[DEGRADED_DIGEST_COLUMN], digests, strict=True) if a != b
            )
            raise HoldoutExecutionError(
                f"{name} and {names[0]} scored different degraded inputs on {mismatched} "
                "slice(s); every method must see one degraded realization per slice"
            )


# --------------------------------------------------------------------------
# The predeclared analysis
# --------------------------------------------------------------------------


def _means(primary: Mapping[str, Any], metrics: Iterable[str]) -> dict[str, float]:
    return {metric: float(primary[metric]["mean"]) for metric in metrics}


def versus_reference(
    metric: str, reference_value: float, candidate_value: float
) -> tuple[float, float]:
    """``(raw, oriented)``: ``candidate - reference``, and positive-is-candidate-better.

    The orientation is :func:`ct_restoration.multiseed.oriented_improvement`
    itself, with the reference in the first position, so every comparison in
    this milestone uses the sign convention Milestone 10 froze.
    """
    raw = float(candidate_value) - float(reference_value)
    return raw, oriented_improvement(metric, reference_value, candidate_value)


def seed_reference_summary(
    metric: str, reference_value: float, candidate_by_seed: Mapping[int, float]
) -> dict[str, Any]:
    """One learned architecture against one deterministic reference, across seeds."""
    seeds = sorted(candidate_by_seed)
    raw = {
        seed: versus_reference(metric, reference_value, candidate_by_seed[seed])[0]
        for seed in seeds
    }
    oriented = {
        seed: versus_reference(metric, reference_value, candidate_by_seed[seed])[1]
        for seed in seeds
    }
    stats = describe_values([oriented[seed] for seed in seeds])
    improved = sum(1 for value in oriented.values() if value > 0.0)
    worsened = sum(1 for value in oriented.values() if value < 0.0)
    return {
        "metric": metric,
        "direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
        "reference_value": float(reference_value),
        "raw_delta_by_seed": raw,
        "raw_delta": describe_values([raw[seed] for seed in seeds]),
        "oriented_improvement_by_seed": oriented,
        "oriented_improvement": stats,
        "seeds_improved": improved,
        "seeds_worsened": worsened,
        "seeds_tied": len(seeds) - improved - worsened,
        "mean_improvement_positive": stats["mean"] > 0.0,
        "all_seeds_improved": improved == len(seeds),
    }


@dataclass
class ScoredMethod:
    """One method's slice table, patient table and split-level summaries."""

    slices: pd.DataFrame
    patients: pd.DataFrame
    primary: dict[str, Any]
    secondary: dict[str, Any]


def summarise_method(slices: pd.DataFrame) -> ScoredMethod:
    patients = aggregate_slices_to_patients(slices)
    return ScoredMethod(
        slices, patients, summarise_patients(patients), slice_weighted_summary(slices)
    )


def load_validation_values(
    plan: dict[str, Any], root: Path, frozen: FrozenProtocol = FROZEN
) -> dict[str, Any]:
    """The Milestone 5-10 validation figures the secondary comparison reads."""
    reference = plan["validation_reference"]
    values: dict[str, Any] = {}
    for method in ("degraded", "clahe"):
        summary = _read_json(root / reference[method]["summary"])
        values[method] = _means(summary["primary_result"], frozen.metrics)
    for method in frozen.learned_methods:
        values[method] = {
            seed: _means(
                _read_json(root / reference[method][seed]["summary"])["primary_result"],
                frozen.metrics,
            )
            for seed in frozen.seeds
        }
    return values


def analyse(
    scored: Mapping[str, ScoredMethod],
    plan: dict[str, Any],
    validation: Mapping[str, Any],
    frozen: FrozenProtocol = FROZEN,
) -> dict[str, Any]:
    """Every predeclared comparison. Returns tables and the summary block.

    No seed is ranked, dropped or singled out, no p-value or interval is
    computed, and nothing here chooses between methods: every number is a
    description of frozen checkpoints on six held-out patients.
    """
    metrics = frozen.metrics
    seeds = list(frozen.seeds)
    degraded = _means(scored["degraded"].primary, metrics)
    clahe = _means(scored["clahe"].primary, metrics)
    learned = {
        method: {
            seed: _means(scored[learned_name(method, seed)].primary, metrics) for seed in seeds
        }
        for method in frozen.learned_methods
    }

    # Paired per-patient tables. Built through paired_patient_deltas, which
    # refuses mismatched patients or slice counts, and checked against the
    # seed-level difference so the two views cannot disagree.
    patient_deltas: dict[str, pd.DataFrame] = {
        "clahe_vs_degraded": paired_patient_deltas(
            scored["clahe"].patients, scored["degraded"].patients
        )
    }
    for seed in seeds:
        cnn = scored[learned_name("cnn", seed)].patients
        unet = scored[learned_name("unet", seed)].patients
        for method, frame in (("cnn", cnn), ("unet", unet)):
            patient_deltas[f"seed{seed}/{method}_vs_degraded"] = paired_patient_deltas(
                frame, scored["degraded"].patients
            )
            patient_deltas[f"seed{seed}/{method}_vs_clahe"] = paired_patient_deltas(
                frame, scored["clahe"].patients
            )
        patient_deltas[f"seed{seed}/unet_vs_cnn"] = paired_patient_deltas(unet, cnn)

    def check_linear(name: str, deltas: pd.DataFrame, candidate: dict, reference: dict) -> None:
        for metric in metrics:
            paired = float(deltas[f"delta_{metric}"].mean())
            difference = candidate[metric] - reference[metric]
            if abs(paired - difference) > LINEARITY_TOLERANCE:
                raise HoldoutExecutionError(
                    f"{name} {metric}: the mean per-patient delta and the difference of "
                    "patient-weighted means disagree beyond rounding"
                )

    check_linear("clahe_vs_degraded", patient_deltas["clahe_vs_degraded"], clahe, degraded)
    for seed in seeds:
        for method in frozen.learned_methods:
            check_linear(
                f"{method} {seed} vs degraded",
                patient_deltas[f"seed{seed}/{method}_vs_degraded"],
                learned[method][seed],
                degraded,
            )
            check_linear(
                f"{method} {seed} vs clahe",
                patient_deltas[f"seed{seed}/{method}_vs_clahe"],
                learned[method][seed],
                clahe,
            )
        check_linear(
            f"unet vs cnn {seed}",
            patient_deltas[f"seed{seed}/unet_vs_cnn"],
            learned["unet"][seed],
            learned["cnn"][seed],
        )

    # --- deterministic methods: measured once, no seed variability ---------
    clahe_counts = summarise_paired_deltas(patient_deltas["clahe_vs_degraded"])
    deterministic_rows = []
    clahe_vs_degraded: dict[str, Any] = {}
    for metric in metrics:
        raw, oriented = versus_reference(metric, degraded[metric], clahe[metric])
        block = clahe_counts[metric]
        clahe_vs_degraded[metric] = {
            "raw_delta_clahe_minus_degraded": raw,
            "oriented_improvement_positive_is_clahe": oriented,
            "patients_improved": block["count_improved"],
            "patients_worsened": block["count_worsened"],
            "patients_tied": block["count_tied"],
        }
        deterministic_rows.append(
            {
                "metric": metric,
                "degraded": degraded[metric],
                "clahe": clahe[metric],
                "raw_delta_clahe_minus_degraded": raw,
                "oriented_improvement_positive_is_clahe": oriented,
                "patients_improved": block["count_improved"],
                "patients_worsened": block["count_worsened"],
                "patients_tied": block["count_tied"],
            }
        )

    # --- learned architectures: five seeds, described, never ranked --------
    architectures = {
        method: {
            metric: {
                "by_seed": {seed: learned[method][seed][metric] for seed in seeds},
                **describe_values([learned[method][seed][metric] for seed in seeds]),
            }
            for metric in metrics
        }
        for method in frozen.learned_methods
    }

    per_metric = {
        metric: paired_metric_summary(
            metric,
            {seed: learned["cnn"][seed][metric] for seed in seeds},
            {seed: learned["unet"][seed][metric] for seed in seeds},
        )
        for metric in metrics
    }
    for block in per_metric.values():
        block["permitted_phrase"] = directional_phrase(block)

    references = {"degraded": degraded, "clahe": clahe}
    versus: dict[str, dict[str, Any]] = {}
    long_tables: dict[str, list[dict[str, Any]]] = {"degraded": [], "clahe": []}
    for reference_name, reference_values in references.items():
        versus[reference_name] = {}
        for method in frozen.learned_methods:
            versus[reference_name][method] = {
                metric: seed_reference_summary(
                    metric,
                    reference_values[metric],
                    {seed: learned[method][seed][metric] for seed in seeds},
                )
                for metric in metrics
            }
            for seed in seeds:
                counts = summarise_paired_deltas(
                    patient_deltas[f"seed{seed}/{method}_vs_{reference_name}"]
                )
                for metric in metrics:
                    raw, oriented = versus_reference(
                        metric, reference_values[metric], learned[method][seed][metric]
                    )
                    long_tables[reference_name].append(
                        {
                            "method": method,
                            "seed": seed,
                            "metric": metric,
                            reference_name: reference_values[metric],
                            "learned": learned[method][seed][metric],
                            f"raw_delta_learned_minus_{reference_name}": raw,
                            "oriented_improvement_positive_is_learned": oriented,
                            "patients_improved": counts[metric]["count_improved"],
                            "patients_worsened": counts[metric]["count_worsened"],
                            "patients_tied": counts[metric]["count_tied"],
                        }
                    )

    # --- validation to test: secondary, descriptive only ------------------
    validation_rows = []
    validation_block: dict[str, Any] = {}
    for method in frozen.methods:
        if method in ("degraded", "clahe"):
            before = validation[method]
            after = references[method]
            basis = "deterministic method, measured once on each split"
        else:
            before = {
                metric: float(np.mean([validation[method][seed][metric] for seed in seeds]))
                for metric in metrics
            }
            after = {
                metric: float(np.mean([learned[method][seed][metric] for seed in seeds]))
                for metric in metrics
            }
            basis = "five-seed architecture mean on both splits"
        validation_block[method] = {}
        for metric in metrics:
            raw, oriented = versus_reference(metric, before[metric], after[metric])
            validation_block[method][metric] = {
                "validation": before[metric],
                "test": after[metric],
                "raw_change_test_minus_validation": raw,
                "oriented_change_positive_is_test_numerically_better": oriented,
            }
            validation_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "validation": before[metric],
                    "test": after[metric],
                    "raw_change_test_minus_validation": raw,
                    "oriented_change_positive_is_test_numerically_better": oriented,
                    "basis": basis,
                }
            )

    seed_rows = [
        {"method": method, "seed": seed, **learned[method][seed]}
        for method in frozen.learned_methods
        for seed in seeds
    ]
    paired_rows = [
        {
            "seed": seed,
            "metric": metric,
            "cnn": learned["cnn"][seed][metric],
            "unet": learned["unet"][seed][metric],
            "raw_delta_unet_minus_cnn": per_metric[metric]["raw_delta_by_seed"][seed],
            "oriented_improvement_positive_is_unet": per_metric[metric][
                "oriented_improvement_by_seed"
            ][seed],
        }
        for metric in metrics
        for seed in seeds
    ]

    summary = {
        "deterministic_methods": {
            "degraded": degraded,
            "clahe": clahe,
            "training_seed_variability": (
                "not applicable: both are deterministic methods, measured once. No seed "
                "standard deviation exists for them and none is reported."
            ),
            "clahe_vs_degraded": clahe_vs_degraded,
        },
        "learned_architectures": architectures,
        "unet_vs_cnn": {
            "pairing": "same statistical seed",
            "per_metric": per_metric,
            "overall": overall_directional_claim(per_metric),
        },
        "learned_vs_degraded": versus["degraded"],
        "learned_vs_clahe": versus["clahe"],
        "validation_to_test": {
            "secondary_descriptive_only": True,
            "note": (
                "Describes whether held-out figures were numerically higher or lower than "
                "the development figures. It is not a generalization-error estimate, and "
                "no difference here is called significant or attributed to a shift."
            ),
            "per_method": validation_block,
        },
    }
    tables = {
        "deterministic_methods.csv": pd.DataFrame(deterministic_rows),
        "seed_level_metrics.csv": pd.DataFrame(seed_rows),
        "paired_seed_deltas.csv": pd.DataFrame(paired_rows),
        "learned_vs_degraded.csv": pd.DataFrame(long_tables["degraded"]),
        "learned_vs_clahe.csv": pd.DataFrame(long_tables["clahe"]),
        "validation_to_test.csv": pd.DataFrame(validation_rows),
    }
    return {"summary": summary, "tables": tables, "patient_deltas": patient_deltas}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment_facts(device: str | None = None) -> dict[str, Any]:
    """Versions the receipt records. No hostname, no user name."""
    import cv2
    import pydicom
    import skimage
    import torch

    from ct_restoration.reproducibility import describe_environment

    try:
        uv = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        uv = None
    return {
        "python": platform.python_version(),
        "uv": uv or None,
        "platform": platform.platform(terse=True),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "scikit_image": skimage.__version__,
        "pydicom": pydicom.__version__,
        "torch": describe_environment(device or ("cuda" if torch.cuda.is_available() else None)),
    }


def _method_summary(
    name: str,
    scored: ScoredMethod,
    plan: dict[str, Any],
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    policies = plan["frozen_policies"]
    block: dict[str, Any] = {
        "milestone": 11,
        "result_class": (
            "HELD-OUT TEST RESULT, measured once under the frozen Milestone 11 protocol. "
            "Six held-out CHAOS patients, synthetic low-dose-like degradation."
        ),
        "method": name,
        "split": plan["split"],
        "patients": int(len(scored.patients)),
        "slices": int(len(scored.slices)),
        "subject_ids": sorted(scored.patients["subject_id"], key=subject_sort_key),
        "primary_result": {
            "unit": "patient",
            "description": (
                "PRIMARY figure. Per-slice metrics averaged within each patient, then the six "
                "patients averaged with equal weight."
            ),
            **scored.primary,
        },
        "secondary_slice_weighted_summary": {
            "unit": "slice",
            "description": "SECONDARY slice-weighted descriptive summary, NOT the result.",
            **scored.secondary,
        },
        "preprocessing": policies["preprocessing"]["settings"],
        "degradation_config": policies["degradation"]["settings"],
        "evaluation_config": policies["evaluation"]["settings"],
    }
    if name == "clahe":
        block["clahe_config"] = policies["clahe"]["settings"]
    if provenance is not None:
        block["checkpoint"] = dict(provenance)
    return block


def execute_protocol(
    plan: dict[str, Any],
    root: Path,
    git: GitState,
    load_models: ModelLoader,
    environment: Mapping[str, Any],
    frozen: FrozenProtocol = FROZEN,
    clock: Callable[[], str] = utc_now,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The whole frozen protocol, once. Returns the execution receipt.

    Order is the safeguard. Everything that can refuse without reading a test
    image refuses first, inside a sealed monitor. Only then is the staging
    directory created - exclusively, so a previous attempt is never resumed -
    and the opening recorded. Only then is a test file opened. Nothing is
    written to the final location until every check on the scored tables has
    passed, and the receipt is the last file written.

    Raises:
        HoldoutProtocolError: a preflight check failed. No test image was read
            and nothing was written.
        HoldoutExecutionError: execution stopped after opening. The staging
            directory, its log and a failure record are left exactly where
            they are; nothing is retried.
    """
    data_root = root / plan["data_root"]
    with DicomOpenMonitor(data_root) as sealed:
        preflight = run_preflight(plan, root, git, load_models, frozen)
    preflight.report.require_all()
    if sum(sealed.opened.values()) or sum(sealed.refused.values()):
        raise HoldoutProtocolError("preflight touched the imaging root; refusing to open test")
    access, restorers = preflight.access, preflight.restorers
    assert access is not None and restorers is not None  # guaranteed by require_all

    staging = root / plan["outputs"]["staging"]
    final = root / plan["outputs"]["root"]
    staging.mkdir(parents=True, exist_ok=False)
    log_path = staging / "execution_log.txt"

    def log(line: str) -> None:
        echo(line)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    started = clock()
    opening = {
        "milestone": 11,
        "event": "held-out test split opened",
        "test_opening_started_utc": started,
        "commit": git.commit,
        "working_tree_clean": True,
        "test_plan": plan["_source"],
        "preflight": preflight.report.as_dict(),
        "note": (
            "Written before the first test image was read. If no execution_receipt.json "
            "accompanies this file, the execution did not complete."
        ),
    }
    with (staging / "opening_record.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(opening, handle, indent=2)
        handle.write("\n")
    log(f"test split opened at {started} under commit {git.commit}")

    stage = "reading manifest rows"
    try:
        manifest_path = root / plan["manifest_file"]["path"]
        manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
        rows = manifest_rows(manifest_path, plan["split"], access=access)
        require_cohort(rows, plan, "test manifest rows")

        # Parsed from the frozen files, exactly as the development commands
        # parse them; preflight has already proved each equals the plan.
        documents = {
            name: yaml.safe_load(
                (root / plan["frozen_policies"][name]["config"]).read_text(encoding="utf-8")
            )
            for name in SHARED_POLICIES
        }
        preprocessing = dict(documents["preprocessing"]["preprocessing"])
        degradation = DegradationConfig.from_mapping(documents["degradation"])
        evaluation = EvaluationConfig.from_mapping(documents["evaluation"])
        clahe = ClaheConfig.from_mapping(documents["clahe"])
        methods: dict[str, RestoreFn] = {
            "degraded": identity_restoration,
            "clahe": clahe_restorer(clahe),
        }
        for method in frozen.learned_methods:
            for seed in frozen.seeds:
                methods[learned_name(method, seed)] = restorers[learned_name(method, seed)]

        stage = "scoring the test split"
        log(f"scoring {len(rows)} test slices with {len(methods)} methods; no metric is printed")
        with DicomOpenMonitor(
            data_root, split_by_key(manifest), readable={plan["split"]}
        ) as monitor:
            frames = score_shared_inputs(
                rows, data_root, preprocessing, evaluation, degradation, methods, access, log
            )
        reads = monitor.record()
        if reads["distinct_files_opened_by_split"]["test"] != plan["expected_slices"]:
            raise HoldoutExecutionError(
                f"{reads['distinct_files_opened_by_split']['test']} distinct test files were "
                f"opened, expected {plan['expected_slices']}"
            )
        for split in ("train", "validation", "stress", "outside_manifest"):
            if reads["file_opens_by_split"][split] or reads["refused_opens_by_split"][split]:
                raise HoldoutExecutionError(f"a {split!r} file was touched during scoring")

        stage = "checking the scored tables"
        for name, frame in frames.items():
            require_finite_metrics(frame, name)
            require_cohort(frame, plan, name)
        require_shared_samples(frames)
        scored = {name: summarise_method(frame) for name, frame in frames.items()}

        stage = "analysing"
        validation = load_validation_values(plan, root, frozen)
        analysis = analyse(scored, plan, validation, frozen)

        stage = "writing"
        written: list[str] = []

        def put(relative: str, content: Any) -> None:
            path = staging / relative
            if path.exists():
                raise HoldoutExecutionError(f"{relative} already exists in staging")
            if isinstance(content, pd.DataFrame):
                write_csv(content, path)
            else:
                write_json(content, path)
            written.append(relative)

        provenance = preflight.provenance or {}
        for name in ("degraded", "clahe"):
            put(f"{name}_test_slices.csv", scored[name].slices)
            put(f"{name}_test_patients.csv", scored[name].patients)
            put(f"{name}_test_summary.json", _method_summary(name, scored[name], plan, None))
        put(
            "clahe_vs_degraded_test_patient_deltas.csv",
            analysis["patient_deltas"]["clahe_vs_degraded"],
        )
        for seed in frozen.seeds:
            for method in frozen.learned_methods:
                name = learned_name(method, seed)
                put(f"seed{seed}/{method}_test_slices.csv", scored[name].slices)
                put(f"seed{seed}/{method}_test_patients.csv", scored[name].patients)
                put(
                    f"seed{seed}/{method}_test_summary.json",
                    _method_summary(name, scored[name], plan, provenance.get(name)),
                )
                for reference in ("degraded", "clahe"):
                    put(
                        f"seed{seed}/{method}_vs_{reference}_test_patient_deltas.csv",
                        analysis["patient_deltas"][f"seed{seed}/{method}_vs_{reference}"],
                    )
            put(
                f"seed{seed}/unet_vs_cnn_test_patient_deltas.csv",
                analysis["patient_deltas"][f"seed{seed}/unet_vs_cnn"],
            )
        for relative, table in analysis["tables"].items():
            put(relative, table)

        put(
            "holdout_test_summary.json",
            {
                "milestone": 11,
                "result_class": (
                    "HELD-OUT TEST RESULT, measured once. Six held-out CHAOS patients under "
                    "synthetic low-dose-like degradation; five predeclared training seeds per "
                    "learned architecture, each seed's frozen checkpoint scored exactly once. "
                    "Descriptive: no significance test, no confidence interval, no best seed."
                ),
                "test_plan": plan["_source"],
                "commit": git.commit,
                "split": plan["split"],
                "patients": plan["expected_patients"],
                "slices": plan["expected_slices"],
                "subject_ids": list(plan["expected_subjects"]),
                "unit_of_analysis": (
                    "patient within each method-seed; then statistical training seed for the "
                    "learned architectures. The six patients and five seeds are never pooled."
                ),
                "pooled_patients_and_seeds": False,
                "significance_testing": False,
                "confidence_intervals": False,
                "best_seed": False,
                "composite_score": False,
                **analysis["summary"],
                "claim_rules": plan["claim_rules"],
                "interpretation_rules": plan["interpretation_rules"],
                "inspection_policy": {
                    "test_files_opened": reads["distinct_files_opened_by_split"]["test"],
                    "stress_files_opened": reads["file_opens_by_split"]["stress"],
                    "visual_outputs_created": False,
                },
            },
        )

        expected = {*frozen.output_files} - {
            "execution_log.txt",
            "opening_record.json",
            "execution_receipt.json",
        }
        expected |= {
            f"seed{seed}/{name}" for seed in frozen.seeds for name in frozen.per_seed_files
        }
        if set(written) != expected:
            raise HoldoutExecutionError(
                f"written files differ from the frozen layout: extra "
                f"{sorted(set(written) - expected)}, missing {sorted(expected - set(written))}"
            )
        visual = [
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.suffix not in PERMITTED_OUTPUT_SUFFIXES
        ]
        if visual:
            raise HoldoutExecutionError(f"non-permitted output file(s) present: {visual}")

        stage = "writing the receipt"
        completed = clock()
        log(f"all tables written; completed at {completed}")
        receipt = {
            "receipt_schema": "m11_execution_receipt_v1",
            "status": "complete",
            "git": {
                "commit": git.commit,
                "branch": git.branch,
                "working_tree_clean_before_opening": True,
                "porcelain_before_opening": git.porcelain,
                "development_frozen_commit": frozen.development_frozen_commit,
                "development_frozen_commit_is_ancestor": git.frozen_commit_is_ancestor,
                "head_in_origin_main": git.head_in_origin_main,
            },
            "test_plan": plan["_source"],
            "split_sha256": plan["split_file"]["sha256"],
            "manifest_sha256": plan["manifest_file"]["sha256"],
            "environment": dict(environment),
            "learned_checkpoints": [
                {
                    "method": method,
                    "seed": seed,
                    **{
                        key: plan["learned_checkpoints"][method][seed][key]
                        for key in (
                            "config",
                            "config_sha256",
                            "checkpoint",
                            "checkpoint_sha256",
                            "selected_epoch",
                            "run_summary",
                        )
                    },
                    "verified_by_preflight": True,
                }
                for method in frozen.learned_methods
                for seed in frozen.seeds
            ],
            "shared_configs": [
                {
                    "name": name,
                    "path": plan["frozen_policies"][name]["config"],
                    "sha256_lf": plan["frozen_policies"][name]["sha256_lf"],
                }
                for name in SHARED_POLICIES
            ],
            "test_subjects": list(plan["expected_subjects"]),
            "expected_patients": plan["expected_patients"],
            "expected_slices": plan["expected_slices"],
            "actual_patients": int(rows["subject_id"].nunique()),
            "actual_slices": int(len(rows)),
            "test_opening_started_utc": started,
            "completed_utc": completed,
            "image_reads": {
                **reads,
                "preflight_file_opens_under_imaging_root": int(sum(sealed.opened.values())),
                "test_images_read": reads["distinct_files_opened_by_split"]["test"],
                "stress_images_read": reads["file_opens_by_split"]["stress"],
            },
            "visual_outputs_created": False,
            "overwrite_occurred": False,
            "retry_occurred": False,
            "prior_attempt_detected": False,
            "outputs": sorted([*written, "opening_record.json", "execution_log.txt"]),
            "metric_values_in_receipt": False,
        }
        missing = [name for name in RECEIPT_FIELDS if name not in receipt]
        if missing:
            raise HoldoutExecutionError(f"receipt is missing {missing}")
        write_json(receipt, staging / "execution_receipt.json")

        present = {
            path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
        }
        layout = {*frozen.output_files} | {
            f"seed{seed}/{name}" for seed in frozen.seeds for name in frozen.per_seed_files
        }
        if present != layout:
            raise HoldoutExecutionError(
                f"the staged files differ from the frozen layout: extra "
                f"{sorted(present - layout)}, missing {sorted(layout - present)}"
            )
    except BaseException as error:
        failure = {
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_utc": clock(),
            "policy": (
                "Nothing is retried and nothing here is deleted. Partial tables in this "
                "directory are not to be read for results. Whether any re-run is permitted "
                "is a decision for an integrity review."
            ),
        }
        try:
            write_json(failure, staging / "failure_record.json")
        except Exception:  # noqa: BLE001 - the original error is what matters
            pass
        if isinstance(error, HoldoutExecutionError | KeyboardInterrupt):
            raise
        raise HoldoutExecutionError(f"execution stopped during {stage}: {error}") from error

    # Outside the failure handler on purpose: once the complete record has
    # moved, nothing may write a failure record back into a new staging copy.
    os.rename(staging, final)
    echo(f"complete: {plan['outputs']['root']}")
    return receipt
