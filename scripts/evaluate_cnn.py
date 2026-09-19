"""Evaluate the selected CNN checkpoint through the frozen benchmark.

    uv run python scripts/evaluate_cnn.py --split validation

Runs the checkpoint chosen by the predeclared rule over a development split,
through the same harness, the same clean targets, the same degraded inputs,
the same evaluation masks and the same metric code as the no-restoration
baseline and CLAHE. Then it compares the three, per patient.

The model reaches the benchmark through the harness's one-argument
restoration contract: degraded image in, restored image out. It never sees
the clean reference, the body mask or the patient identity. That is the same
contract CLAHE is held to, which is what makes the comparison a comparison.

Only after selection
--------------------
This runs once, on the already-selected checkpoint. The checkpoint was chosen
on patient-weighted validation full-frame MAE alone, predeclared in
:file:`configs/cnn.yaml`. The eight full and body metrics below are reported,
not optimized against: nothing in Milestone 8 changes after seeing them.

Development splits only
-----------------------
``--split`` accepts ``train`` and ``validation``. ``test`` and ``stress`` are
refused; their image content stays sealed until the final benchmark.

Outputs
-------
``outputs/metrics/cnn_<split>_slices.csv``
``outputs/metrics/cnn_<split>_patients.csv``
``outputs/metrics/cnn_<split>_summary.json``
``outputs/metrics/cnn_vs_degraded_baseline_<split>_patient_deltas.csv``
``outputs/metrics/cnn_vs_clahe_<split>_patient_deltas.csv``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import (  # noqa: E402
    METRICS_DIR,
    check_metrics_output_policy,
    evaluate_slices,
    manifest_rows,
    write_csv,
    write_json,
)
from ct_restoration.config import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    resolve_config_path,
)
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like  # noqa: E402
from ct_restoration.evaluation import (  # noqa: E402
    EvaluationConfig,
    aggregate_slices_to_patients,
    group_breakdown,
    paired_patient_deltas,
    prepare_evaluation_slice,
    require_development_split,
    slice_weighted_summary,
    summarise_paired_deltas,
    summarise_patients,
)
from ct_restoration.models.adapter import (  # noqa: E402
    raw_restore_array,
    restore_array,
    torch_restorer,
)
from ct_restoration.models.cnn import ResidualCnnConfig, build_model  # noqa: E402
from ct_restoration.training import SELECTION_METRIC, select_best_epoch  # noqa: E402

#: Identity of the method being measured.
METHOD_NAME = "cnn"

#: Stem of every output file. The split is appended.
OUTPUT_STEM = "cnn"

#: The mandatory reference every method is judged against.
BASELINE_STEM = "degraded_baseline"

#: The classical method, reported beside it for context.
CLASSICAL_STEM = "clahe"

#: Quantiles reported for the per-slice diagnostics.
QUANTILES = (0.05, 0.5, 0.95)

#: The canonical Milestone 8 training seed. A checkpoint carrying any other
#: seed is not the run this command is allowed to report.
CANONICAL_SEED = 2026

#: Directory holding the canonical run's tracked history and summary.
CANONICAL_RUN_DIR = Path("outputs/runs/cnn_seed2026")

#: Tolerance when comparing one selection value against another. The tracked
#: run summary rounds every float to 10 decimal places, so half a unit in the
#: tenth place is the largest disagreement rounding alone can produce. It is a
#: representation tolerance, not a tolerance for disagreeing about which
#: epoch was selected.
SELECTION_VALUE_TOLERANCE = 5e-10


class EvaluationIntegrityError(RuntimeError):
    """A precondition for canonical evaluation was not met.

    Raised instead of returning a report nobody reads. Every condition these
    gates check is one that would leave the written artifacts wrong in a way
    no later reader could detect from the files themselves.
    """


def repository_path(path: Path) -> str:
    """A path as written into a tracked artifact: relative to the repository.

    Tracked outputs in this project carry no absolute filesystem path. An
    absolute one would differ between machines, so two runs of an unchanged
    definition would stop producing byte-identical files, and it would put a
    local username into a committed artifact.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device, config: ResidualCnnConfig):
    """Rebuild the selected model from its checkpoint.

    Loaded with ``weights_only=True``: restoring a checkpoint must never
    execute code that happens to be inside the file.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def raw_output_diagnostics(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    model,
    device: torch.device,
) -> dict[str, Any]:
    """How much work the final clamp is doing, and how large the correction is.

    Two different quantities, deliberately not merged. The model is defined as
    ``raw = degraded + predicted_correction`` and ``restored = clamp(raw)``,
    so:

    * the **predicted correction** is ``raw - degraded`` - what the network
      actually asked for, which is what says how hard it is pushing;
    * the **post-clamp change** is ``restored - degraded`` - what survived
      into the reported image.

    Wherever the raw output leaves [0, 1] these differ, and on this model that
    is around half the frame, so reporting one under the other's name would
    understate the correction by a wide margin.

    Descriptive only. None of this becomes an objective: the point is to know
    whether clamping is a rare safety boundary or a structural part of the
    method, which changes how the reported metrics should be read.
    """
    pixels = 0
    below = above = clamp_changed = 0
    predicted_sum = 0.0
    post_clamp_sum = 0.0
    per_slice_predicted: list[float] = []
    per_slice_post_clamp: list[float] = []
    raw_minimum = float("inf")
    raw_maximum = float("-inf")
    finite_failures = 0

    for row in rows.itertuples():
        key = row.relative_dicom_path
        clean, _, _ = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation).astype(np.float64)

        raw = raw_restore_array(model, degraded, device).astype(np.float64)
        restored = restore_array(model, degraded, device).astype(np.float64)
        if not np.all(np.isfinite(raw)):
            finite_failures += 1

        pixels += raw.size
        below += int((raw < 0.0).sum())
        above += int((raw > 1.0).sum())
        clamp_changed += int((raw != restored).sum())
        raw_minimum = min(raw_minimum, float(raw.min()))
        raw_maximum = max(raw_maximum, float(raw.max()))

        # The predicted correction comes from the RAW output. Taking it from
        # the clamped one would silently report the post-clamp change instead.
        predicted = np.abs(raw - degraded)
        post_clamp = np.abs(restored - degraded)
        predicted_sum += float(predicted.sum())
        post_clamp_sum += float(post_clamp.sum())
        per_slice_predicted.append(float(predicted.mean()))
        per_slice_post_clamp.append(float(post_clamp.mean()))

    def quantiles(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}

    return {
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
        "fraction_raw_below_zero": below / pixels,
        "fraction_raw_above_one": above / pixels,
        "fraction_changed_by_clamp": clamp_changed / pixels,
        "mean_absolute_predicted_correction": predicted_sum / pixels,
        "per_slice_mean_absolute_predicted_correction": quantiles(per_slice_predicted),
        "mean_absolute_post_clamp_change_vs_degraded": post_clamp_sum / pixels,
        "per_slice_mean_absolute_post_clamp_change": quantiles(per_slice_post_clamp),
        "finite_value_failures": finite_failures,
        "definitions": {
            "predicted_correction": "abs(raw - degraded), from the UNCLAMPED model output",
            "post_clamp_change": "abs(clamp(raw, 0, 1) - degraded), what reached the metrics",
            "why_both": (
                "They differ wherever the raw output leaves [0, 1]. Reporting either "
                "one under the other's name misstates how hard the network is pushing."
            ),
        },
        "note": (
            "Descriptive diagnostics of the clamp and the predicted correction. Reported "
            "metrics use the clamped image; nothing here is an optimization objective."
        ),
    }


def check_sample_alignment(slice_frame: pd.DataFrame, others: dict[str, Path]) -> dict[str, Any]:
    """The CNN must have scored exactly the slices the other methods scored.

    Not a formality: a paired per-patient comparison is only paired if both
    sides ran on the same images in the same order.
    """
    keys = list(slice_frame["relative_dicom_path"])
    report: dict[str, Any] = {
        "rows": len(keys),
        "unique_rows": len(set(keys)),
        "duplicates": len(keys) - len(set(keys)),
        "references": sorted(others),
    }
    for name, path in others.items():
        other = list(pd.read_csv(path)["relative_dicom_path"])
        paired = sum(1 for a, b in zip(keys, other, strict=False) if a != b)
        report[name] = {
            "reference_rows": len(other),
            "missing": len(set(other) - set(keys)),
            "extra": len(set(keys) - set(other)),
            # zip() stops at the shorter list, so a length difference would
            # otherwise hide every unpaired row. Count those as mismatches.
            "order_mismatches": paired + abs(len(keys) - len(other)),
            "identical_order": keys == other,
        }
    return report


def require_sample_alignment(report: dict[str, Any], expected_rows: int) -> None:
    """Refuse the evaluation unless every paired comparison is genuinely paired.

    A report nobody acts on is not a check. Each condition below, left
    unenforced, produces per-patient deltas that subtract one slice's metric
    from a different slice's metric while looking perfectly well-formed on
    disk.

    Raises:
        EvaluationIntegrityError: any alignment condition failed.
    """
    failures: list[str] = []
    if report["rows"] != expected_rows:
        failures.append(f"scored {report['rows']} rows, expected {expected_rows}")
    if report["unique_rows"] != expected_rows:
        failures.append(f"{report['unique_rows']} unique sample keys, expected {expected_rows}")
    if report["duplicates"]:
        failures.append(f"{report['duplicates']} duplicate sample keys")

    for name in report["references"]:
        entry = report[name]
        for field in ("missing", "extra", "order_mismatches"):
            if entry[field]:
                failures.append(f"{name}: {entry[field]} {field.replace('_', ' ')}")
        if not entry["identical_order"]:
            failures.append(f"{name}: sample keys are not in identical order")

    if failures:
        raise EvaluationIntegrityError(
            "canonical evaluation refused - the CNN did not score the same slices, in the "
            "same order, as the methods it would be compared against: " + "; ".join(failures)
        )


def verify_checkpoint_provenance(
    payload: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Refuse to score a checkpoint that is not the frozen canonical run's.

    Reading a checkpoint's own metadata and copying it into the report proves
    nothing: a file can say anything about itself. Every field here is checked
    against something computed independently - the config file's bytes, the
    checkpoint file's bytes, the tracked training history re-run through the
    predeclared selection rule, and the tracked run summary.

    The gate runs before a single validation image is opened, so a wrong
    checkpoint costs an error message rather than a plausible-looking table.

    Raises:
        EvaluationIntegrityError: the checkpoint is not the canonical one, or
            the run it claims to come from is not internally consistent.
    """
    required = ("config_sha256", "seed", "selection_metric", "selection_value", "epoch")
    absent = [field for field in required if field not in payload]
    if absent:
        raise EvaluationIntegrityError(
            "checkpoint carries no provenance for " + ", ".join(absent) + ". It was not "
            "written by scripts/train_cnn.py and cannot be identified as the canonical run."
        )

    summary_path = run_dir / "run_summary.json"
    history_path = run_dir / "training_history.csv"
    for path in (summary_path, history_path):
        if not path.exists():
            raise EvaluationIntegrityError(
                f"{path.as_posix()} is missing; checkpoint provenance cannot be verified "
                "without the tracked record of the run it came from."
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = pd.read_csv(history_path)
    try:
        summary_config_sha = str(summary["config"]["sha256"])
        summary_seed = int(summary["training"]["seed"])
        selection = summary["checkpoint_selection"]
        summary_metric = str(selection["primary_metric"])
        summary_epoch = int(selection["best_epoch"])
        summary_value = float(selection["best_validation_patient_weighted_full_mae"])
        summary_checkpoint_sha = str(summary["checkpoint"]["sha256"])
        round_trip = summary["checkpoint"]["round_trip"]
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationIntegrityError(
            f"{summary_path.as_posix()} is missing the provenance fields this gate needs "
            f"({error}); it was not written by the canonical training command."
        ) from error

    config_sha = file_sha256(config_path)
    checkpoint_sha = file_sha256(checkpoint_path)

    # Re-derive the selected epoch from the tracked history with the same
    # predeclared rule the training run used, rather than trusting the epoch
    # either the checkpoint or the summary claims.
    recomputed_epoch = select_best_epoch(history, SELECTION_METRIC, epoch_zero_eligible=False)
    recomputed_value = float(
        history.loc[history["epoch"] == recomputed_epoch, SELECTION_METRIC].iloc[0]
    )

    def close(left: Any, right: Any) -> bool:
        return abs(float(left) - float(right)) <= SELECTION_VALUE_TOLERANCE

    checks = {
        "checkpoint_config_sha_matches_frozen_config": (
            str(payload["config_sha256"]) == config_sha
        ),
        "checkpoint_config_sha_matches_run_summary": (
            str(payload["config_sha256"]) == summary_config_sha
        ),
        "checkpoint_seed_is_canonical": int(payload["seed"]) == CANONICAL_SEED,
        "checkpoint_seed_matches_run_summary": int(payload["seed"]) == summary_seed,
        "run_summary_seed_is_canonical": summary_seed == CANONICAL_SEED,
        "checkpoint_selection_metric_is_predeclared": (
            str(payload["selection_metric"]) == SELECTION_METRIC
        ),
        "run_summary_selection_metric_is_predeclared": summary_metric == SELECTION_METRIC,
        "checkpoint_epoch_matches_run_summary": int(payload["epoch"]) == summary_epoch,
        "checkpoint_epoch_matches_recomputed_selection": (
            int(payload["epoch"]) == int(recomputed_epoch)
        ),
        "checkpoint_selection_value_matches_history": close(
            payload["selection_value"], recomputed_value
        ),
        "checkpoint_selection_value_matches_run_summary": close(
            payload["selection_value"], summary_value
        ),
        "checkpoint_file_sha_matches_run_summary": checkpoint_sha == summary_checkpoint_sha,
        "round_trip_state_dict_clean": (round_trip.get("state_dict_tensor_mismatches") == 0),
        "round_trip_predictions_clean": round_trip.get("probe_prediction_mismatches") == 0,
        "round_trip_reproduces_saved_model": round_trip.get("reproduces_saved_model") is True,
    }

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise EvaluationIntegrityError(
            "canonical evaluation refused - this checkpoint is not the frozen Milestone 8 "
            "run, or that run is not internally consistent. Failed: " + ", ".join(failed)
        )

    return {
        "verified": True,
        "verified_before_any_image_was_scored": True,
        "checks_passed": sorted(checks),
        "checks_failed": failed,
        "recomputed_from_files": {
            "config_sha256": config_sha,
            "checkpoint_file_sha256": checkpoint_sha,
            "selected_epoch_from_training_history": int(recomputed_epoch),
            "selection_value_from_training_history": recomputed_value,
        },
        "selection_rule_reapplied": (
            f"lowest {SELECTION_METRIC}, ties to the earlier epoch, epoch 0 ineligible"
        ),
        "selection_value_tolerance": SELECTION_VALUE_TOLERANCE,
        "sources": {
            "frozen_config": repository_path(config_path),
            "run_summary": summary_path.as_posix(),
            "training_history": history_path.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
        },
        "note": (
            "Each field is checked against something computed independently of the "
            "checkpoint - the config bytes, the checkpoint bytes, and the tracked history "
            "re-run through the predeclared selection rule - not copied from the payload."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        help="development split to evaluate: train or validation. test and stress are refused.",
    )
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/cnn_seed2026_best.pt")
    parser.add_argument("--cnn-config", default="cnn.yaml")
    parser.add_argument(
        "--run-dir",
        default=CANONICAL_RUN_DIR.as_posix(),
        help="tracked run directory the checkpoint must be provably from",
    )
    parser.add_argument("--evaluation-config", default="evaluation.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--output-dir", default=METRICS_DIR.as_posix())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug only: cap the slice count; refused when writing canonical metrics",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output_dir = Path(arguments.output_dir)

    try:
        split = require_development_split(arguments.split)
        slices_path = output_dir / f"{OUTPUT_STEM}_{split}_slices.csv"
        check_metrics_output_policy(arguments.limit, slices_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    device = torch.device(arguments.device)
    model_config = ResidualCnnConfig.from_mapping(load_config(arguments.cnn_config))
    evaluation = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    checkpoint_path = Path(arguments.checkpoint)
    model, payload = load_checkpoint(checkpoint_path, device, model_config)

    # Before any validation image is opened: prove this checkpoint is the
    # frozen canonical run's. A wrong checkpoint should cost an error, not a
    # plausible-looking table.
    try:
        provenance = verify_checkpoint_provenance(
            payload,
            checkpoint_path,
            resolve_config_path(arguments.cnn_config),
            Path(arguments.run_dir),
        )
    except EvaluationIntegrityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 5
    print(f"provenance      : verified against {arguments.run_dir} before scoring anything")

    rows = manifest_rows(Path(arguments.manifest), split)
    if arguments.limit:
        rows = rows.head(arguments.limit)
    subjects = sorted(rows["subject_id"].unique(), key=int)

    print(f"method          : {METHOD_NAME} ({model_config.algorithm})")
    print(f"checkpoint      : {checkpoint_path} (epoch {payload['epoch']}, seed {payload['seed']})")
    print(f"selected on     : {payload['selection_metric']} = {payload['selection_value']:.8f}")
    print(f"parameters      : {model.parameter_count():,}")
    print(f"device          : {device}")
    print(f"split           : {split} ({len(subjects)} patients, {len(rows)} slices)")
    print("test and stress image content is NOT read by this command")
    print()

    slice_frame = evaluate_slices(
        rows, root, preprocessing, evaluation, degradation, torch_restorer(model, device)
    )
    patient_frame = aggregate_slices_to_patients(slice_frame)
    primary = summarise_patients(patient_frame)
    secondary = slice_weighted_summary(slice_frame)

    baseline_patients = pd.read_csv(
        output_dir / f"{BASELINE_STEM}_{split}_patients.csv", dtype={"subject_id": str}
    )
    baseline_deltas = paired_patient_deltas(patient_frame, baseline_patients)
    baseline_paired = summarise_paired_deltas(baseline_deltas)

    classical_path = output_dir / f"{CLASSICAL_STEM}_{split}_patients.csv"
    classical_deltas = None
    if classical_path.exists():
        classical_patients = pd.read_csv(classical_path, dtype={"subject_id": str})
        classical_deltas = paired_patient_deltas(patient_frame, classical_patients)

    # The alignment gate runs BEFORE anything is written. A paired comparison
    # against the wrong slices, already on disk, is indistinguishable from a
    # correct one to every later reader.
    references = {"vs_degraded_baseline": output_dir / f"{BASELINE_STEM}_{split}_slices.csv"}
    classical_slices = output_dir / f"{CLASSICAL_STEM}_{split}_slices.csv"
    if classical_slices.exists():
        references["vs_clahe"] = classical_slices
    alignment = check_sample_alignment(slice_frame, references)
    alignment["enforced"] = not arguments.limit
    if alignment["enforced"]:
        try:
            require_sample_alignment(alignment, len(rows))
        except EvaluationIntegrityError as error:
            print(f"error: {error}", file=sys.stderr)
            print("no canonical artifact was written.", file=sys.stderr)
            return 6
        print(f"alignment       : verified against {', '.join(sorted(references))} before writing")

    diagnostics = raw_output_diagnostics(
        rows, root, preprocessing, evaluation, degradation, model, device
    )

    slices_csv = write_csv(slice_frame, slices_path)
    patients_csv = write_csv(patient_frame, output_dir / f"{OUTPUT_STEM}_{split}_patients.csv")
    baseline_deltas_csv = write_csv(
        baseline_deltas,
        output_dir / f"{OUTPUT_STEM}_vs_{BASELINE_STEM}_{split}_patient_deltas.csv",
    )

    comparisons: dict[str, Any] = {}
    if classical_deltas is not None:
        classical_csv = write_csv(
            classical_deltas,
            output_dir / f"{OUTPUT_STEM}_vs_{CLASSICAL_STEM}_{split}_patient_deltas.csv",
        )
        comparisons["paired_comparison_vs_clahe"] = {
            "description": (
                "delta = CNN - selected CLAHE, per patient. Descriptive context only: the "
                "bar a restoration method has to clear is no restoration, not CLAHE."
            ),
            "baseline": classical_path.name,
            "patient_deltas_csv": classical_csv.as_posix(),
            **summarise_paired_deltas(classical_deltas),
        }

    beats = {
        metric: bool(baseline_paired[metric]["mean"] < 0)
        if metric.endswith(("mae", "mse"))
        else bool(baseline_paired[metric]["mean"] > 0)
        for metric in (
            "full_mae",
            "full_mse",
            "full_psnr",
            "full_ssim",
            "body_mae",
            "body_mse",
            "body_psnr",
            "body_ssim",
        )
    }

    summary = {
        "milestone": "8 - small residual CNN, single-seed development run",
        "result_class": (
            "SINGLE-SEED VALIDATION DEVELOPMENT RESULT, not a final benchmark result. One "
            "training seed shows what this run did; it does not establish that the "
            "architecture is stable. The test split remains sealed."
        ),
        "method": METHOD_NAME,
        "method_description": (
            "A 28,353-parameter residual CNN predicting an additive correction to the frozen "
            "degraded image, clamped to [0, 1]. It receives the degraded image only: no "
            "clean reference, no body mask, no patient identity."
        ),
        "split": split,
        "patients": int(len(patient_frame)),
        "slices": int(len(slice_frame)),
        "subject_ids": subjects,
        "model_config": model_config.as_dict(),
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "epoch": int(payload["epoch"]),
            "seed": int(payload["seed"]),
            "config_sha256": str(payload["config_sha256"]),
            "selection_metric": str(payload["selection_metric"]),
            "selection_value": float(payload["selection_value"]),
            "selection_note": (
                "Chosen on patient-weighted validation full-frame MAE alone, predeclared "
                "before training. The metrics below are reported, not optimized against."
            ),
            "provenance": provenance,
        },
        "evaluation_config": evaluation.as_dict(),
        "degradation_config": degradation.as_dict(),
        "primary_result": {
            "unit": "patient",
            "description": (
                "PRIMARY figure. Per-slice metrics averaged within each patient, then "
                "patients averaged with equal weight."
            ),
            **primary,
        },
        "secondary_slice_weighted_summary": {
            "unit": "slice",
            "description": (
                "SECONDARY slice-weighted descriptive summary, NOT the benchmark result."
            ),
            **secondary,
        },
        "paired_comparison_vs_degraded_baseline": {
            "description": (
                "delta = CNN - degraded baseline, per patient, same patients and same "
                "slices. Positive is an improvement for PSNR and SSIM; negative is an "
                "improvement for MAE and MSE. No significance test is run: six patients is "
                "a small descriptive sample."
            ),
            "baseline": f"{BASELINE_STEM}_{split}_patients.csv",
            "patient_deltas_csv": baseline_deltas_csv.as_posix(),
            **baseline_paired,
        },
        **comparisons,
        "verdict": {
            "question": "Did the selected CNN beat no restoration on each metric?",
            "beats_degraded_baseline": beats,
            "metrics_improved": sum(1 for value in beats.values() if value),
            "metrics_total": len(beats),
        },
        "raw_output_diagnostics": diagnostics,
        "sample_alignment": alignment,
        "acquisition_group_breakdown": {
            "description": (
                "Descriptive only, patient-weighted within each group, three patients each. "
                "No significance test and no general claim about acquisition settings."
            ),
            "groups": group_breakdown(patient_frame),
        },
        "inspection_policy": {
            "train_images_read": split == "train",
            "validation_images_read": split == "validation",
            "test_images_read": 0,
            "stress_images_read": 0,
            "validation_images_visually_inspected": False,
            "note": (
                "This command cannot evaluate test or stress: require_development_split "
                "refuses them. The checkpoint was selected on validation MAE only."
            ),
        },
        "interpretation": {
            "mae": "lower is better",
            "mse": "lower is better",
            "psnr": "higher is better, in decibels",
            "ssim": "higher is better, 1.0 for identical images",
            "caution": (
                "A single-seed validation development result. No value here indicates "
                "clinical adequacy, and a PSNR difference is in decibels, never a "
                "percentage."
            ),
        },
        "outputs": {
            "slices_csv": slices_csv.as_posix(),
            "patients_csv": patients_csv.as_posix(),
            "patient_deltas_csv": baseline_deltas_csv.as_posix(),
        },
    }
    summary_path = write_json(summary, output_dir / f"{OUTPUT_STEM}_{split}_summary.json")

    print()
    header = f"{'region':<8}{'metric':<7}{'degraded':>13}{'CLAHE':>13}{'CNN':>13}{'delta':>13}"
    print(f"SELECTED CNN vs DEGRADED BASELINE ({split}, patient-weighted)")
    print(header)
    print("-" * len(header))
    classical = None
    if classical_path.exists():
        classical = summarise_patients(pd.read_csv(classical_path, dtype={"subject_id": str}))
    for region in ("full", "body"):
        for metric in ("mae", "mse", "psnr", "ssim"):
            name = f"{region}_{metric}"
            base = float(baseline_paired[name]["mean"])
            cnn_value = float(primary[name]["mean"])
            degraded_value = cnn_value - base
            clahe_value = float(classical[name]["mean"]) if classical else float("nan")
            print(
                f"{region:<8}{metric:<7}{degraded_value:>13.6f}{clahe_value:>13.6f}"
                f"{cnn_value:>13.6f}{base:>+13.6f}"
            )
    print("-" * len(header))
    for metric in ("body_ssim", "body_psnr", "full_ssim", "full_psnr"):
        block = baseline_paired[metric]
        print(
            f"  {metric:<10} improved {block['count_improved']}/{summary['patients']} patients, "
            f"mean delta {block['mean']:+.6f}"
        )
    print(f"\nsummary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
