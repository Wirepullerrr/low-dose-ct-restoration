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
import sys
from pathlib import Path
from typing import Any

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
from ct_restoration.config import load_config, resolve_config_path  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig  # noqa: E402
from ct_restoration.evaluation import (  # noqa: E402
    EvaluationConfig,
    aggregate_slices_to_patients,
    group_breakdown,
    paired_patient_deltas,
    require_development_split,
    slice_weighted_summary,
    summarise_paired_deltas,
    summarise_patients,
)
from ct_restoration.evaluation_integrity import (  # noqa: E402
    EvaluationIntegrityError,
    check_sample_alignment,
    config_training_seed,
    require_sample_alignment,
    verify_checkpoint_provenance,
)
from ct_restoration.models.adapter import (  # noqa: E402
    torch_restorer,
)
from ct_restoration.models.cnn import ResidualCnnConfig, build_model  # noqa: E402
from ct_restoration.models.diagnostics import raw_output_diagnostics  # noqa: E402

#: Identity of the method being measured.
METHOD_NAME = "cnn"

#: Stem of every output file. The split is appended.
OUTPUT_STEM = "cnn"

#: The mandatory reference every method is judged against.
#: Where the frozen non-learned references live: the degraded baseline and
#: CLAHE. Those are measured once and never per seed, so an additional
#: statistical seed reads them from here rather than needing its own copy.
#: Separate from ``--output-dir`` so a seed can write into
#: ``outputs/metrics/multiseed/seed<N>/`` without the references following it.
REFERENCE_DIR = METRICS_DIR

BASELINE_STEM = "degraded_baseline"

#: The classical method, reported beside it for context.
CLASSICAL_STEM = "clahe"

#: Quantiles reported for the per-slice diagnostics.
QUANTILES = (0.05, 0.5, 0.95)

#: Directory holding the canonical run's tracked history and summary.
CANONICAL_RUN_DIR = Path("outputs/runs/cnn_seed2026")


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
    parser.add_argument(
        "--reference-dir",
        default=REFERENCE_DIR.as_posix(),
        help="directory holding the frozen degraded-baseline and CLAHE artifacts. "
        "Stays canonical when --output-dir points at a per-seed directory.",
    )
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
    reference_dir = Path(arguments.reference_dir)

    try:
        split = require_development_split(arguments.split)
        slices_path = output_dir / f"{OUTPUT_STEM}_{split}_slices.csv"
        check_metrics_output_policy(arguments.limit, slices_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    device = torch.device(arguments.device)
    model_document = load_config(arguments.cnn_config)
    model_config = ResidualCnnConfig.from_mapping(model_document)
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
            expected_seed=config_training_seed(model_document),
            method="residual CNN (Milestone 8)",
            trainer="scripts/train_cnn.py",
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
        reference_dir / f"{BASELINE_STEM}_{split}_patients.csv", dtype={"subject_id": str}
    )
    baseline_deltas = paired_patient_deltas(patient_frame, baseline_patients)
    baseline_paired = summarise_paired_deltas(baseline_deltas)

    classical_path = reference_dir / f"{CLASSICAL_STEM}_{split}_patients.csv"
    classical_deltas = None
    if classical_path.exists():
        classical_patients = pd.read_csv(classical_path, dtype={"subject_id": str})
        classical_deltas = paired_patient_deltas(patient_frame, classical_patients)

    # The alignment gate runs BEFORE anything is written. A paired comparison
    # against the wrong slices, already on disk, is indistinguishable from a
    # correct one to every later reader.
    references = {"vs_degraded_baseline": reference_dir / f"{BASELINE_STEM}_{split}_slices.csv"}
    classical_slices = reference_dir / f"{CLASSICAL_STEM}_{split}_slices.csv"
    if classical_slices.exists():
        references["vs_clahe"] = classical_slices
    alignment = check_sample_alignment(slice_frame, references)
    alignment["enforced"] = not arguments.limit
    if alignment["enforced"]:
        try:
            require_sample_alignment(alignment, len(rows), method="the CNN")
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
