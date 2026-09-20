"""Evaluate the selected U-Net checkpoint through the frozen benchmark.

    uv run python scripts/evaluate_unet.py --split validation

Runs the checkpoint chosen by the predeclared rule over a development split,
through the same harness, the same clean targets, the same degraded inputs,
the same evaluation masks and the same metric code as the no-restoration
baseline, CLAHE and the residual CNN. Then it compares all four.

The comparison this milestone is really about
---------------------------------------------
CNN versus U-Net. Both are learned residual restoration models trained under
a deliberately identical policy - same data, same corruption, same sampler,
same loss, same optimizer, same seed, same epochs, same batch size, same
checkpoint criterion - so the intended difference between them is the
architecture. The paired per-patient deltas against the CNN are reported for
all eight metrics, with the count of patients improved for each.

The bar a restoration method has to clear is still *no restoration*. Beating
the CNN is interesting; beating the degraded baseline is the requirement.

Only after selection
--------------------
This runs once, on the already-selected checkpoint. The checkpoint was chosen
on patient-weighted validation full-frame MAE alone, predeclared in
:file:`configs/unet.yaml`. The eight full and body metrics below are
reported, not optimized against: nothing in Milestone 9 changes after seeing
them.

Two hard gates
--------------
Checkpoint provenance is verified before a single image is opened, and
ordered sample alignment against all three reference tables is enforced
before anything is written. Both refuse rather than warn; both live in
:mod:`ct_restoration.evaluation_integrity` and are the same code the CNN
evaluation clears.

Development splits only
-----------------------
``--split`` accepts ``train`` and ``validation``. ``test`` and ``stress`` are
refused; their image content stays sealed until the final benchmark.

Outputs
-------
``outputs/metrics/unet_<split>_slices.csv``
``outputs/metrics/unet_<split>_patients.csv``
``outputs/metrics/unet_<split>_summary.json``
``outputs/metrics/unet_vs_degraded_baseline_<split>_patient_deltas.csv``
``outputs/metrics/unet_vs_clahe_<split>_patient_deltas.csv``
``outputs/metrics/unet_vs_cnn_<split>_patient_deltas.csv``
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
    require_sample_alignment,
    verify_checkpoint_provenance,
)
from ct_restoration.models.adapter import torch_restorer  # noqa: E402
from ct_restoration.models.diagnostics import raw_output_diagnostics  # noqa: E402
from ct_restoration.models.unet import (  # noqa: E402
    CANONICAL_PARAMETER_COUNT,
    LightweightResidualUnetConfig,
    build_model,
)

#: Identity of the method being measured.
METHOD_NAME = "unet"

#: Stem of every output file. The split is appended.
OUTPUT_STEM = "unet"

#: The mandatory reference every method is judged against.
BASELINE_STEM = "degraded_baseline"

#: Every other method this one is compared against, in report order. The
#: degraded baseline is the bar; these are context, and the CNN is the
#: architecture comparison this milestone exists for.
COMPARISON_STEMS = ("clahe", "cnn")

#: Directory holding the canonical run's tracked history and summary.
CANONICAL_RUN_DIR = Path("outputs/runs/unet_seed2026")

#: The eight reported metrics, in report order.
METRICS = (
    "full_mae",
    "full_mse",
    "full_psnr",
    "full_ssim",
    "body_mae",
    "body_mse",
    "body_psnr",
    "body_ssim",
)


def improves(metric: str, mean_delta: float) -> bool:
    """Does a mean paired delta point the right way for this metric?

    MAE and MSE are lower-is-better, so an improvement is negative; PSNR and
    SSIM are higher-is-better, so an improvement is positive.
    """
    return mean_delta < 0 if metric.endswith(("mae", "mse")) else mean_delta > 0


def load_checkpoint(path: Path, device: torch.device, config: LightweightResidualUnetConfig):
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
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unet_seed2026_best.pt")
    parser.add_argument("--unet-config", default="unet.yaml")
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
    model_config = LightweightResidualUnetConfig.from_mapping(load_config(arguments.unet_config))
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
            resolve_config_path(arguments.unet_config),
            Path(arguments.run_dir),
            method="lightweight U-Net (Milestone 9)",
            trainer="scripts/train_unet.py",
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

    # Deltas against the other measured methods, computed before any write so
    # a failure costs nothing on disk.
    other_deltas: dict[str, pd.DataFrame] = {}
    other_patients: dict[str, pd.DataFrame] = {}
    for stem in COMPARISON_STEMS:
        path = output_dir / f"{stem}_{split}_patients.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"subject_id": str})
            other_patients[stem] = frame
            other_deltas[stem] = paired_patient_deltas(patient_frame, frame)

    # The alignment gate runs BEFORE anything is written. A paired comparison
    # against the wrong slices, already on disk, is indistinguishable from a
    # correct one to every later reader.
    references = {"vs_degraded_baseline": output_dir / f"{BASELINE_STEM}_{split}_slices.csv"}
    for stem in COMPARISON_STEMS:
        candidate = output_dir / f"{stem}_{split}_slices.csv"
        if candidate.exists():
            references[f"vs_{stem}"] = candidate
    alignment = check_sample_alignment(slice_frame, references)
    alignment["enforced"] = not arguments.limit
    if alignment["enforced"]:
        try:
            require_sample_alignment(alignment, len(rows), method="the U-Net")
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

    descriptions = {
        "clahe": (
            "delta = U-Net - selected CLAHE, per patient. Descriptive context only: the "
            "bar a restoration method has to clear is no restoration, not CLAHE."
        ),
        "cnn": (
            "delta = U-Net - residual CNN, per patient. THE ARCHITECTURE COMPARISON this "
            "milestone exists for: both are learned residual models trained under an "
            "identical policy, so the intended difference between them is the model. Still "
            "a single-seed validation comparison, and no significance test is run."
        ),
    }
    comparisons: dict[str, Any] = {}
    for stem, deltas in other_deltas.items():
        csv_path = write_csv(
            deltas, output_dir / f"{OUTPUT_STEM}_vs_{stem}_{split}_patient_deltas.csv"
        )
        comparisons[f"paired_comparison_vs_{stem}"] = {
            "description": descriptions[stem],
            "baseline": f"{stem}_{split}_patients.csv",
            "patient_deltas_csv": csv_path.as_posix(),
            **summarise_paired_deltas(deltas),
        }

    beats_baseline = {
        metric: bool(improves(metric, baseline_paired[metric]["mean"])) for metric in METRICS
    }

    architecture_verdict: dict[str, Any] = {}
    if "cnn" in other_deltas:
        cnn_paired = summarise_paired_deltas(other_deltas["cnn"])
        architecture_verdict = {
            "question": (
                "Does the lightweight U-Net improve over the residual CNN on each metric, "
                "under the same frozen corruption and the same training policy?"
            ),
            "per_metric": {
                metric: {
                    "mean_delta": cnn_paired[metric]["mean"],
                    "unet_better": bool(improves(metric, cnn_paired[metric]["mean"])),
                    "patients_improved": cnn_paired[metric]["count_improved"],
                    "patients_worsened": cnn_paired[metric]["count_worsened"],
                    "patients_tied": cnn_paired[metric]["count_tied"],
                }
                for metric in METRICS
            },
            "metrics_improved": sum(
                1 for metric in METRICS if improves(metric, cnn_paired[metric]["mean"])
            ),
            "metrics_total": len(METRICS),
            "caution": (
                "The two architectures differ in more than receptive field: pooling, a "
                "decoder, concatenative skips, parameter count and the whole computational "
                "graph all change together. A difference here cannot be attributed to any "
                "one of them, and one seed each is not stability evidence."
            ),
        }

    summary = {
        "milestone": "9 - lightweight residual U-Net, single-seed development run",
        "result_class": (
            "SINGLE-SEED VALIDATION DEVELOPMENT RESULT, not a final benchmark result. One "
            "training seed shows what this run did; it does not establish that the "
            "architecture is stable. The test split remains sealed."
        ),
        "method": METHOD_NAME,
        "method_description": (
            "A 116,753-parameter lightweight residual U-Net - two downsampling levels, 16 "
            "base channels, concatenative skip connections, a 1x1 correction head - "
            "predicting an additive correction to the frozen degraded image, clamped to "
            "[0, 1]. It receives the degraded image only: no clean reference, no body "
            "mask, no patient identity."
        ),
        "comparability_with_cnn": (
            "Seed, epochs, batch size, loss, optimizer and its hyperparameters, sampler, "
            "augmentation policy and checkpoint-selection rule are identical to the "
            "Milestone 8 residual CNN's and were not adjusted for this architecture. "
            "Neither recipe was ever hyperparameter-tuned: the CNN's values were one "
            "predeclared development configuration. The U-Net therefore inherits the CNN "
            "benchmark's predeclared training recipe rather than receiving "
            "architecture-specific tuning, and is measured under that recipe rather than "
            "at its best."
        ),
        "split": split,
        "patients": int(len(patient_frame)),
        "slices": int(len(slice_frame)),
        "subject_ids": subjects,
        "model_config": model_config.as_dict(),
        "model_size": {
            "trainable_parameters": int(model.parameter_count()),
            "canonical_parameters": CANONICAL_PARAMETER_COUNT,
            "cnn_parameters": 28353,
            "parameter_ratio_vs_cnn": round(model.parameter_count() / 28353, 4),
            "maximum_deepest_path_receptive_field_pixels": int(model.receptive_field),
            "receptive_field_note": (
                "Maximum over paths. Concatenative skips provide shallower routes carrying "
                "smaller-scale local information, so not every contribution to an output "
                "pixel arrives through a 44x44 window. A statement about pixels only, not "
                "anatomical or clinical context. Parameter count is not latency; latency "
                "has not been measured for any method."
            ),
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "epoch": int(payload["epoch"]),
            "seed": int(payload["seed"]),
            "config_sha256": str(payload["config_sha256"]),
            "selection_metric": str(payload["selection_metric"]),
            "selection_value": float(payload["selection_value"]),
            "selection_note": (
                "Chosen on patient-weighted validation full-frame MAE alone, predeclared "
                "before training and identical to the CNN's criterion. The metrics below "
                "are reported, not optimized against."
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
                "delta = U-Net - degraded baseline, per patient, same patients and same "
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
            "question": "Did the selected U-Net beat no restoration on each metric?",
            "beats_degraded_baseline": beats_baseline,
            "metrics_improved": sum(1 for value in beats_baseline.values() if value),
            "metrics_total": len(beats_baseline),
        },
        "architecture_comparison_vs_cnn": architecture_verdict,
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

    others = {stem: summarise_patients(frame) for stem, frame in other_patients.items()}
    print()
    header = (
        f"{'region':<8}{'metric':<7}{'degraded':>13}{'CLAHE':>13}"
        f"{'CNN':>13}{'U-Net':>13}{'vs degraded':>14}{'vs CNN':>13}"
    )
    print(f"FOUR METHODS ON {split.upper()} (patient-weighted, equal patient weight)")
    print(header)
    print("-" * len(header))
    cnn_paired = summarise_paired_deltas(other_deltas["cnn"]) if "cnn" in other_deltas else None
    for region in ("full", "body"):
        for metric in ("mae", "mse", "psnr", "ssim"):
            name = f"{region}_{metric}"
            versus_baseline = float(baseline_paired[name]["mean"])
            unet_value = float(primary[name]["mean"])
            degraded_value = unet_value - versus_baseline
            clahe_value = (
                float(others["clahe"][name]["mean"]) if "clahe" in others else float("nan")
            )
            cnn_value = float(others["cnn"][name]["mean"]) if "cnn" in others else float("nan")
            versus_cnn = float(cnn_paired[name]["mean"]) if cnn_paired else float("nan")
            print(
                f"{region:<8}{metric:<7}{degraded_value:>13.6f}{clahe_value:>13.6f}"
                f"{cnn_value:>13.6f}{unet_value:>13.6f}"
                f"{versus_baseline:>+14.6f}{versus_cnn:>+13.6f}"
            )
    print("-" * len(header))
    print(f"vs no restoration: beat it on {summary['verdict']['metrics_improved']}/8 metrics")
    if cnn_paired:
        print(f"vs residual CNN  : better on {architecture_verdict['metrics_improved']}/8 metrics")
        for metric in METRICS:
            block = architecture_verdict["per_metric"][metric]
            verdict = "U-Net better" if block["unet_better"] else "CNN better  "
            print(
                f"  {metric:<10} {verdict}  mean delta {block['mean_delta']:+.6f}  "
                f"patients improved {block['patients_improved']}/{summary['patients']}"
            )
    print(f"\nsummary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
