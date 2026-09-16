"""Measure the degraded baseline: how much damage the frozen degradation does.

    uv run python scripts/evaluate_degraded_baseline.py --split validation

The degraded baseline is the **no-restoration** method. Its "restored" output
is the degraded image itself, so the numbers below are simply

    clean reference  vs.  frozen degraded input

with no enhancement and no model involved. Every later method - CLAHE, the
residual CNN, the U-Net - is compared against exactly this, computed by exactly
the same metric code on exactly the same slices and masks.

Development splits only
-----------------------
``--split`` accepts ``train`` and ``validation``. ``test`` and ``stress`` are
refused outright: their image content stays sealed until the final benchmark,
after every method decision is frozen, and this command offers no path to
spend it early. The final evaluation will be a separate, explicitly final
command.

What one slice costs
--------------------
Load DICOM, convert to HU, build the evaluation body mask from that clean HU,
build the clean 256x256 reference with the Milestone 1 preprocessing, generate
the degraded image with the frozen Milestone 4 function keyed by the manifest's
``relative_dicom_path``, then score the pair in both regions.

Outputs, all tracked and all deterministic
------------------------------------------
``outputs/metrics/degraded_baseline_<split>_slices.csv``
    one row per slice.
``outputs/metrics/degraded_baseline_<split>_patients.csv``
    one row per patient, the mean of that patient's slices.
``outputs/metrics/degraded_baseline_<split>_summary.json``
    the PRIMARY patient-weighted result, plus a secondary slice-weighted
    figure kept only for transparency.

No timestamps, no pixel data, no DICOM UIDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.config import ensure_dir, load_config  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like  # noqa: E402
from ct_restoration.evaluation import (  # noqa: E402
    METRIC_COLUMNS,
    EvaluationConfig,
    aggregate_slices_to_patients,
    group_breakdown,
    prepare_evaluation_slice,
    require_development_split,
    slice_weighted_summary,
    summarise_patients,
)
from ct_restoration.metrics import slice_metrics  # noqa: E402

#: Directory holding the tracked metric tables.
METRICS_DIR = Path("outputs/metrics")

#: Stem of every output file. The split is appended, so a debug run on train
#: cannot overwrite the validation result.
OUTPUT_STEM = "degraded_baseline"

#: Identity of the method being measured. Later methods reuse this pipeline
#: with a different name and a real restoration step.
METHOD_NAME = "degraded_baseline"

SLICE_COLUMNS = (
    "subject_id",
    "source_archive",
    "acquisition_group",
    "relative_dicom_path",
    "geometric_slice_index",
    *METRIC_COLUMNS,
    "body_pixel_fraction",
    "body_ssim_interior_fraction",
)

#: Decimal places for floats in tracked outputs. Enough for an MSE around
#: 1e-4 to keep every significant digit.
_ROUND = 10


def _round(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    if isinstance(value, float | np.floating):
        return round(float(value), _ROUND)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV with a fixed line ending so its checksum is portable."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return path


def check_output_policy(limit: int, slices_path: Path) -> None:
    """Refuse to write a partial evaluation over a canonical tracked table.

    Raises:
        ValueError: ``limit`` is set and the output is a canonical path.
    """
    if limit and Path(slices_path).resolve().parent == METRICS_DIR.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite canonical tracked metrics in "
            f"{METRICS_DIR.as_posix()} with a partial evaluation. "
            "Re-run without --limit, or pass a noncanonical --output-dir."
        )


def manifest_rows(manifest_path: Path, split: str) -> pd.DataFrame:
    """Rows of one development split, sorted deterministically.

    Raises:
        HeldOutSplitError: the split is sealed.
        ValueError: the manifest holds no rows for that split.
    """
    require_development_split(split)
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    rows = manifest[manifest["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"No split == {split} rows in {manifest_path}")
    rows["subject_order"] = rows["subject_id"].astype(int)
    rows = rows.sort_values(["subject_order", "geometric_slice_index"])
    return rows.drop(columns="subject_order").reset_index(drop=True)


def evaluate_split(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Score the degraded baseline on every row. One row in, one row out."""
    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows.itertuples(), start=1):
        key = row.relative_dicom_path
        clean, body, interior = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        # The degraded image IS the baseline's output: no restoration happens.
        degraded = degrade_low_dose_like(clean, key, degradation)

        measured = slice_metrics(clean, degraded, body, interior, evaluation.ssim)
        records.append(
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
        if position % progress_every == 0 or position == len(rows):
            print(f"  processed {position:>5} / {len(rows)}")

    frame = pd.DataFrame(records)[list(SLICE_COLUMNS)]
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(_ROUND)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        help="development split to evaluate: train or validation. test and stress are refused.",
    )
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
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
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    slices_path = output_dir / f"{OUTPUT_STEM}_{split}_slices.csv"
    try:
        check_output_policy(arguments.limit, slices_path)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    evaluation = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    rows = manifest_rows(Path(arguments.manifest), split)
    if arguments.limit:
        rows = rows.head(arguments.limit)
    subjects = sorted(rows["subject_id"].unique(), key=int)

    print(f"method          : {METHOD_NAME} (no restoration; output is the degraded image)")
    print(f"split           : {split}")
    print(f"patients        : {len(subjects)}")
    print(f"slices          : {len(rows)}")
    print(f"degradation     : {degradation.algorithm}, seed {degradation.global_seed}")
    mask_settings = evaluation.body_mask
    print(f"body mask       : HU > {mask_settings.threshold_hu}, largest component, holes filled")
    print(
        f"ssim            : win {evaluation.ssim.win_size}, "
        f"sigma {evaluation.ssim.sigma}, data_range {evaluation.data_range}"
    )
    print("test and stress image content is NOT read by this command")
    print()

    slice_frame = evaluate_split(rows, root, preprocessing, evaluation, degradation)
    patient_frame = aggregate_slices_to_patients(slice_frame)
    primary = summarise_patients(patient_frame)
    secondary = slice_weighted_summary(slice_frame)

    slices_csv = write_csv(slice_frame, slices_path)
    patients_csv = write_csv(patient_frame, output_dir / f"{OUTPUT_STEM}_{split}_patients.csv")

    coverage = slice_frame["body_pixel_fraction"]
    interior_coverage = slice_frame["body_ssim_interior_fraction"]
    summary = {
        "milestone": "5 - evaluation framework and degraded baseline",
        "method": METHOD_NAME,
        "method_description": (
            "No restoration. The degraded image is returned unchanged, so these numbers are "
            "the image-quality cost of the frozen synthetic degradation and the floor every "
            "later method is compared against."
        ),
        "split": split,
        "patients": int(len(patient_frame)),
        "slices": int(len(slice_frame)),
        "subject_ids": subjects,
        "evaluation_config": evaluation.as_dict(),
        "degradation_config": degradation.as_dict(),
        "preprocessing": {
            "window_center": preprocessing["window_center"],
            "window_width": preprocessing["window_width"],
            "image_size": list(preprocessing["image_size"]),
            "interpolation": preprocessing["interpolation"],
        },
        "primary_result": {
            "unit": "patient",
            "description": (
                "PRIMARY benchmark figure. Per-slice metrics are averaged within each patient, "
                "then patients are averaged with equal weight. The 'mean' entry of each metric "
                "is the headline estimate."
            ),
            **primary,
        },
        "secondary_slice_weighted_summary": {
            "unit": "slice",
            "description": (
                "SECONDARY slice-weighted descriptive summary, NOT the benchmark result. It "
                "pools every slice equally, which treats correlated slices from one patient as "
                "independent observations. Reported only to show how much unequal slice counts "
                "would move the number."
            ),
            **secondary,
        },
        "acquisition_group_breakdown": {
            "description": (
                "Descriptive only, patient-weighted within each group. With three patients per "
                "group this can reveal a glaring imbalance and nothing more. No significance "
                "test, no claim about acquisition settings in general, and no conclusion drawn "
                "from a small difference."
            ),
            "groups": group_breakdown(patient_frame),
        },
        "body_mask_coverage": {
            "body_pixel_fraction": {
                "min": float(coverage.min()),
                "median": float(coverage.median()),
                "max": float(coverage.max()),
            },
            "ssim_interior_fraction": {
                "min": float(interior_coverage.min()),
                "median": float(interior_coverage.median()),
                "max": float(interior_coverage.max()),
            },
        },
        "inspection_policy": {
            "train_images_read": split == "train",
            "validation_images_read": split == "validation",
            "test_images_read": False,
            "stress_images_read": False,
            "note": (
                "This command cannot evaluate test or stress: require_development_split refuses "
                "them. Those splits stay sealed until the final benchmark, which will be a "
                "separate explicitly final command run after every method decision is frozen."
            ),
        },
        "interpretation": {
            "mae": "lower is better, in normalized image units",
            "mse": "lower is better, in squared normalized image units",
            "psnr": "higher is better, in decibels; a logarithmic restatement of MSE",
            "ssim": "higher is better, 1.0 for identical images",
            "caution": (
                "These values describe this synthetic benchmark only. No PSNR or SSIM value "
                "here is 'good' or 'clinically acceptable', and a PSNR difference is a "
                "difference in decibels, never a percentage."
            ),
        },
        "outputs": {
            "slices_csv": slices_csv.as_posix(),
            "patients_csv": patients_csv.as_posix(),
        },
    }

    summary_path = output_dir / f"{OUTPUT_STEM}_{split}_summary.json"
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_round(summary), handle, indent=2, allow_nan=False)
        handle.write("\n")

    print()
    print(f"PRIMARY patient-weighted result, {len(patient_frame)} patients, equal weight")
    header = f"{'region':<12}{'MAE':>12}{'MSE':>14}{'PSNR (dB)':>12}{'SSIM':>10}"
    print(header)
    print("-" * len(header))
    for region, label in (("full", "full frame"), ("body", "body region")):
        line = f"{label:<12}"
        line += f"{primary[f'{region}_mae']['mean']:>12.6f}"
        line += f"{primary[f'{region}_mse']['mean']:>14.8f}"
        line += f"{primary[f'{region}_psnr']['mean']:>12.4f}"
        line += f"{primary[f'{region}_ssim']['mean']:>10.6f}"
        print(line)
    print("-" * len(header))
    print()
    print("SECONDARY slice-weighted descriptive values (not the benchmark result)")
    print(header)
    print("-" * len(header))
    for region, label in (("full", "full frame"), ("body", "body region")):
        line = f"{label:<12}"
        line += f"{secondary[f'{region}_mae']['mean']:>12.6f}"
        line += f"{secondary[f'{region}_mse']['mean']:>14.8f}"
        line += f"{secondary[f'{region}_psnr']['mean']:>12.4f}"
        line += f"{secondary[f'{region}_ssim']['mean']:>10.6f}"
        print(line)
    print("-" * len(header))
    print()
    print(f"patients        : {len(patient_frame)}")
    print(f"slices          : {len(slice_frame)}")
    print(
        f"body coverage   : min {coverage.min():.4f}  median {coverage.median():.4f}  "
        f"max {coverage.max():.4f}"
    )
    print(
        f"ssim interior   : min {interior_coverage.min():.4f}  "
        f"median {interior_coverage.median():.4f}  max {interior_coverage.max():.4f}"
    )
    print(f"zero-mask failures            : {int((coverage <= 0).sum())}")
    print(f"zero-ssim-interior failures   : {int((interior_coverage <= 0).sum())}")
    print()
    print(f"slices csv      : {slices_csv}")
    print(f"patients csv    : {patients_csv}")
    print(f"summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
