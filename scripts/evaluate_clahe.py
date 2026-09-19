"""Evaluate the frozen CLAHE configuration against the degraded baseline.

    uv run python scripts/evaluate_clahe.py --split validation

Runs the single configuration frozen in :file:`configs/clahe.yaml` over a
development split, through the same harness, the same clean targets, the same
degraded inputs, the same evaluation masks and the same metric code as the
no-restoration baseline. Then it compares the two **per patient**.

Paired, not just averaged
-------------------------
Two split-level means tell you which number is larger. Six paired per-patient
deltas tell you whether the method helped everyone a little, helped some and
harmed others, or was carried by one patient. Those are different findings and
they matter for whether a result would hold up on new patients, so the paired
deltas are what this command reports.

No significance test is run. Six patients is a small descriptive sample, and a
p-value here would dress up model selection as inference.

Development splits only
-----------------------
``--split`` accepts ``train`` and ``validation``. ``test`` and ``stress`` are
refused: their image content stays sealed until the final benchmark.

Outputs
-------
``outputs/metrics/clahe_<split>_slices.csv``
``outputs/metrics/clahe_<split>_patients.csv``
``outputs/metrics/clahe_<split>_summary.json``

Same schema as the degraded baseline, same canonical row order, so the two
per-slice tables line up row for row and a paired comparison can be audited by
reading them side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import (  # noqa: E402
    METRICS_DIR,
    check_metrics_output_policy,
    evaluate_slices,
    manifest_rows,
    write_csv,
    write_json,
)
from ct_restoration.classical.clahe import (  # noqa: E402
    QUANTIZATION_ERROR_BOUND,
    ClaheConfig,
    apply_clahe,
    clahe_restorer,
)
from ct_restoration.config import load_config  # noqa: E402
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

#: Identity of the method being measured.
METHOD_NAME = "clahe"

#: Stem of every output file. The split is appended.
OUTPUT_STEM = "clahe"

#: The method this one is compared against.
BASELINE_STEM = "degraded_baseline"

#: Quantiles reported for the per-slice change diagnostics.
QUANTILES = (0.05, 0.5, 0.95)


def clahe_diagnostics(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    config: ClaheConfig,
) -> dict[str, Any]:
    """Descriptive technical facts about what CLAHE did to the degraded image.

    Deliberately plain quantities: how much the output moved, how much of it
    sits at the clip bounds, whether anything came back non-finite. No derived
    "contrast improvement" figure is invented; there is no agreed definition of
    one and it would read as a result rather than a diagnostic.
    """
    at_zero = at_one = pixels = 0
    change_sum = 0.0
    per_slice_change: list[float] = []
    finite_failures = range_failures = 0

    for row in rows.itertuples():
        key = row.relative_dicom_path
        clean, _, _ = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation)
        restored = apply_clahe(degraded, config)

        if not np.all(np.isfinite(restored)):
            finite_failures += 1
        if float(restored.min()) < 0.0 or float(restored.max()) > 1.0:
            range_failures += 1

        change = np.abs(restored.astype(np.float64) - degraded.astype(np.float64))
        pixels += restored.size
        at_zero += int((restored <= 0.0).sum())
        at_one += int((restored >= 1.0).sum())
        change_sum += float(change.sum())
        per_slice_change.append(float(change.mean()))

    changes = np.asarray(per_slice_change, dtype=np.float64)
    return {
        "fraction_output_at_zero": at_zero / pixels,
        "fraction_output_at_one": at_one / pixels,
        "mean_absolute_change_vs_degraded": change_sum / pixels,
        "per_slice_mean_absolute_change": {
            f"q{int(q * 100):02d}": float(np.quantile(changes, q)) for q in QUANTILES
        },
        "finite_value_failures": finite_failures,
        "range_failures": range_failures,
        "quantization_policy": (
            f"fixed round(x * 255) over the whole [0,1] benchmark range, never per-image "
            f"min/max; round-trip error at most {QUANTIZATION_ERROR_BOUND:.8f} in normalized "
            "units. The 8-bit conversion is part of the method definition."
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
    parser.add_argument("--clahe-config", default="clahe.yaml")
    parser.add_argument("--evaluation-config", default="evaluation.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--output-dir", default=METRICS_DIR.as_posix())
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
    config = ClaheConfig.from_mapping(load_config(arguments.clahe_config))
    evaluation = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    rows = manifest_rows(Path(arguments.manifest), split)
    if arguments.limit:
        rows = rows.head(arguments.limit)
    subjects = sorted(rows["subject_id"].unique(), key=int)

    print(f"method          : {METHOD_NAME} ({config.algorithm})")
    print(
        f"config          : clip_limit {config.clip_limit:g}, "
        f"tile_grid_size [{config.tile_rows}, {config.tile_columns}], "
        f"{config.input_quantization_bits}-bit input"
    )
    print(f"split           : {split} ({len(subjects)} patients, {len(rows)} slices)")
    print("test and stress image content is NOT read by this command")
    print()

    slice_frame = evaluate_slices(
        rows, root, preprocessing, evaluation, degradation, clahe_restorer(config)
    )
    patient_frame = aggregate_slices_to_patients(slice_frame)
    primary = summarise_patients(patient_frame)
    secondary = slice_weighted_summary(slice_frame)

    baseline_patients = pd.read_csv(
        output_dir / f"{BASELINE_STEM}_{split}_patients.csv", dtype={"subject_id": str}
    )
    deltas = paired_patient_deltas(patient_frame, baseline_patients)
    paired = summarise_paired_deltas(deltas)

    slices_csv = write_csv(slice_frame, slices_path)
    patients_csv = write_csv(patient_frame, output_dir / f"{OUTPUT_STEM}_{split}_patients.csv")
    deltas_csv = write_csv(
        deltas, output_dir / f"{OUTPUT_STEM}_vs_{BASELINE_STEM}_{split}_patient_deltas.csv"
    )

    diagnostics = clahe_diagnostics(rows, root, preprocessing, evaluation, degradation, config)
    primary_metric = "body_ssim"
    beats = bool(paired[primary_metric]["mean"] > 0)

    summary = {
        "milestone": "6 - CLAHE classical baseline",
        "method": METHOD_NAME,
        "method_description": (
            "Contrast Limited Adaptive Histogram Equalization applied to the frozen degraded "
            "image. Not a denoiser and not trained. It receives the degraded image only: no "
            "clean reference, no body mask, no patient identity."
        ),
        "split": split,
        "patients": int(len(patient_frame)),
        "slices": int(len(slice_frame)),
        "subject_ids": subjects,
        "clahe_config": config.as_dict(),
        "evaluation_config": evaluation.as_dict(),
        "degradation_config": degradation.as_dict(),
        "primary_result": {
            "unit": "patient",
            "description": (
                "PRIMARY figure. Per-slice metrics averaged within each patient, then patients "
                "averaged with equal weight."
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
                "delta = CLAHE - degraded baseline, per patient, same patients and same "
                "slices. Positive is an improvement for PSNR and SSIM; negative is an "
                "improvement for MAE and MSE. No significance test is run: six patients is a "
                "small descriptive sample."
            ),
            "baseline": f"{BASELINE_STEM}_{split}_patients.csv",
            "patient_deltas_csv": deltas_csv.as_posix(),
            **paired,
        },
        "verdict": {
            "predeclared_primary_metric": primary_metric,
            "beats_degraded_baseline_on_primary_metric": beats,
            "statement": (
                f"The frozen CLAHE configuration "
                f"{'improves' if beats else 'does NOT improve'} patient-weighted "
                f"{primary_metric} relative to no restoration on the {split} split "
                f"(mean paired delta {paired[primary_metric]['mean']:+.6f})."
            ),
        },
        "clahe_diagnostics": diagnostics,
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
            "test_images_read": False,
            "stress_images_read": False,
            "note": (
                "This command cannot evaluate test or stress: require_development_split "
                "refuses them. CLAHE parameters were selected on validation only."
            ),
        },
        "interpretation": {
            "mae": "lower is better",
            "mse": "lower is better",
            "psnr": "higher is better, in decibels",
            "ssim": "higher is better, 1.0 for identical images",
            "caution": (
                "A validation development result, not a final benchmark result. No value here "
                "indicates clinical adequacy, and a PSNR difference is in decibels, never a "
                "percentage."
            ),
        },
        "outputs": {
            "slices_csv": slices_csv.as_posix(),
            "patients_csv": patients_csv.as_posix(),
            "patient_deltas_csv": deltas_csv.as_posix(),
        },
    }

    summary_path = output_dir / f"{OUTPUT_STEM}_{split}_summary.json"
    write_json(summary, summary_path)

    print()
    header = f"{'region':<12}{'metric':<8}{'degraded':>13}{'CLAHE':>13}{'delta':>13}"
    print(f"SELECTED CLAHE vs DEGRADED BASELINE ({split}, patient-weighted)")
    print(header)
    print("-" * len(header))
    for region in ("full", "body"):
        for metric in ("mae", "mse", "psnr", "ssim"):
            name = f"{region}_{metric}"
            base = float(baseline_patients[f"mean_{name}"].mean())
            value = float(primary[name]["mean"])
            print(f"{region:<12}{metric:<8}{base:>13.6f}{value:>13.6f}{value - base:>+13.6f}")
    print("-" * len(header))

    print()
    print(f"PAIRED PATIENT DELTAS (CLAHE - degraded baseline), {len(deltas)} patients")
    columns = ("body_psnr", "body_ssim", "full_psnr", "full_ssim")
    header = f"{'subject':<9}{'group':<7}" + "".join(f"{f'd_{c}':>16}" for c in columns)
    print(header)
    print("-" * len(header))
    for row in deltas.itertuples():
        line = f"{row.subject_id:<9}{row.acquisition_group:<7}"
        line += "".join(f"{getattr(row, f'delta_{c}'):>+16.6f}" for c in columns)
        print(line)
    print("-" * len(header))
    line = f"{'mean':<9}{'':<7}"
    line += "".join(f"{paired[c]['mean']:>+16.6f}" for c in columns)
    print(line)
    line = f"{'improved':<9}{'':<7}"
    for column in columns:
        line += f"{paired[column]['count_improved']}/{len(deltas)}".rjust(16)
    print(line)
    print()
    print(f"VERDICT: {summary['verdict']['statement']}")
    print()
    print(f"slices csv      : {slices_csv}")
    print(f"patients csv    : {patients_csv}")
    print(f"deltas csv      : {deltas_csv}")
    print(f"summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
