"""Sweep the predeclared CLAHE grid on validation and freeze the winner.

    uv run python scripts/tune_clahe.py --root data/raw/chaos

Validation only, by construction
--------------------------------
There is no ``--split`` option. Tuning reads the validation split and nothing
else: that is what validation is for, and offering a flag would be offering a
way to tune against test. Test and stress stay sealed until the final
benchmark.

What it does
------------
For each of the 885 validation slices, rebuild the clean reference, regenerate
the exact frozen degraded image from the same sample key, rebuild the exact
frozen evaluation masks, then apply all 12 candidates to that one degraded
image and score each with the shared Milestone 5 metric code. Slice metrics are
averaged within a patient and patients are averaged with equal weight, so a
candidate is ranked on six patients rather than 885 correlated slices.

The predeclared rule in :file:`configs/clahe_search.yaml` then picks one
winner, and :file:`configs/clahe.yaml` is written with it. That file is frozen
once written: this command refuses to overwrite it without ``--overwrite``, so
CLAHE cannot be quietly retuned after CNN or U-Net results exist.

Outputs
-------
``outputs/metrics/clahe_validation_search.csv``
    one row per candidate, with metrics, deltas against the degraded baseline,
    the rank and the winner. No timestamp; regenerating is byte-identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import (  # noqa: E402
    METRICS_DIR,
    ROUND,
    finalise_slice_frame,
    manifest_rows,
    write_csv,
)
from ct_restoration.classical.clahe import ClaheConfig, apply_clahe  # noqa: E402
from ct_restoration.classical.search import (  # noqa: E402
    PATIENT_WEIGHTED_PREFIX,
    PRIMARY_METRIC,
    ClaheSearchSpace,
    rank_candidates,
    selected_config,
)
from ct_restoration.config import CONFIGS_DIR, PROJECT_ROOT, load_config  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like  # noqa: E402
from ct_restoration.evaluation import (  # noqa: E402
    METRIC_COLUMNS,
    EvaluationConfig,
    aggregate_slices_to_patients,
    prepare_evaluation_slice,
    summarise_patients,
)
from ct_restoration.metrics import slice_metrics  # noqa: E402

#: The split this command reads. Not configurable: see the module docstring.
TUNING_SPLIT = "validation"

#: The tracked search table.
CANONICAL_SEARCH_CSV = METRICS_DIR / "clahe_validation_search.csv"

#: The frozen runtime definition this command writes.
CANONICAL_CLAHE_CONFIG = Path("configs/clahe.yaml")

#: The degraded baseline every delta is measured against.
BASELINE_PATIENTS_CSV = METRICS_DIR / "degraded_baseline_validation_patients.csv"


def check_output_policy(limit: int, search_csv: Path) -> None:
    """Refuse to write a partial sweep over the canonical tracked table.

    Raises:
        ValueError: ``limit`` is set and the output is the canonical path.
    """
    if limit and Path(search_csv).resolve() == CANONICAL_SEARCH_CSV.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite the canonical tracked search "
            f"table {CANONICAL_SEARCH_CSV.as_posix()} with a partial sweep. "
            "Re-run without --limit, or pass a noncanonical --output-dir."
        )


def search_config_provenance(search_config: str | Path) -> str:
    """The repo-relative path of the search config, for the provenance field.

    Mirrors how :func:`load_config` resolves a bare filename, so what gets
    recorded in ``configs/clahe.yaml`` is the tracked file that was actually
    read - ``configs/clahe_search.yaml`` - rather than whatever shorthand the
    command line happened to use. Provenance only: it never touches the
    selected runtime parameters.
    """
    candidate = Path(search_config)
    if (
        not candidate.exists()
        and not candidate.is_absolute()
        and (CONFIGS_DIR / candidate).exists()
    ):
        candidate = CONFIGS_DIR / candidate
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def check_freeze_policy(config_path: Path, overwrite: bool) -> None:
    """Refuse to silently retune a CLAHE configuration that is already frozen.

    Raises:
        ValueError: the config exists and ``--overwrite`` was not given.
    """
    if config_path.exists() and not overwrite:
        raise ValueError(
            f"{config_path.as_posix()} already exists and is frozen. Re-running the sweep "
            "would overwrite a selection that later results may already be measured against. "
            "Pass --overwrite only if retuning is a deliberate, documented decision."
        )


def sweep(
    rows: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    candidates: list[ClaheConfig],
    progress_every: int = 100,
) -> dict[str, pd.DataFrame]:
    """Score every candidate on every slice, reusing one degraded image.

    Each slice is loaded, preprocessed, masked and degraded exactly once, and
    all candidates then see that identical degraded array. Regenerating the
    degradation per candidate would be wasteful and, worse, would invite a
    future edit that gave different candidates different noise.

    Returns:
        ``{candidate label: per-slice metric frame}``.
    """
    records: dict[str, list[dict[str, Any]]] = {config.label: [] for config in candidates}

    for position, row in enumerate(rows.itertuples(), start=1):
        key = row.relative_dicom_path
        clean, body, interior = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation)

        shared = {
            "subject_id": row.subject_id,
            "source_archive": row.source_archive,
            "acquisition_group": row.acquisition_group,
            "relative_dicom_path": key,
            "geometric_slice_index": int(row.geometric_slice_index),
            "body_pixel_fraction": float(body.mean()),
            "body_ssim_interior_fraction": float(interior.mean()),
        }
        for config in candidates:
            restored = apply_clahe(degraded, config)
            measured = slice_metrics(clean, restored, body, interior, evaluation.ssim)
            records[config.label].append({**shared, **measured})

        if position % progress_every == 0 or position == len(rows):
            print(f"  processed {position:>5} / {len(rows)} slices x {len(candidates)} candidates")

    return {label: finalise_slice_frame(rows_) for label, rows_ in records.items()}


def candidate_table(
    candidates: list[ClaheConfig],
    per_candidate: dict[str, pd.DataFrame],
    baseline: dict[str, float],
) -> pd.DataFrame:
    """One patient-weighted row per candidate, with deltas against baseline.

    Sign convention, applied uniformly: ``delta = CLAHE - degraded baseline``.
    So a **positive** delta is an improvement for PSNR and SSIM and a
    **negative** delta is an improvement for MAE and MSE.
    """
    rows: list[dict[str, Any]] = []
    for config in candidates:
        patients = aggregate_slices_to_patients(per_candidate[config.label])
        summary = summarise_patients(patients)
        row: dict[str, Any] = {
            "clip_limit": float(config.clip_limit),
            "tile_grid_rows": config.tile_rows,
            "tile_grid_cols": config.tile_columns,
            "patients": int(summary["patients"]),
        }
        for name in METRIC_COLUMNS:
            value = float(summary[name]["mean"])
            row[f"{PATIENT_WEIGHTED_PREFIX}{name}"] = round(value, ROUND)
            row[f"delta_vs_degraded_{name}"] = round(value - baseline[name], ROUND)
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_patient_weighted(patients_csv: Path) -> dict[str, float]:
    """The degraded baseline's patient-weighted means, from its tracked table.

    Read from the committed Milestone 5 output rather than recomputed, so a
    delta is provably against the same numbers the baseline published.
    """
    if not patients_csv.exists():
        raise FileNotFoundError(
            f"Degraded baseline patient table not found at {patients_csv}. "
            "Run scripts/evaluate_degraded_baseline.py first."
        )
    frame = pd.read_csv(patients_csv, dtype={"subject_id": str})
    return {name: float(frame[f"mean_{name}"].mean()) for name in METRIC_COLUMNS}


def write_frozen_config(config: ClaheConfig, path: Path, search_config: str) -> Path:
    """Write the canonical runtime definition of the selected CLAHE."""
    text = f"""# Canonical CLAHE configuration for this benchmark.
#
# FROZEN. Selected by the predeclared sweep in {search_config}, on the
# validation split only, using patient-weighted mean body SSIM as the primary
# metric with the predeclared tie-breakers. Written by scripts/tune_clahe.py;
# not hand-edited.
#
# Do not retune CLAHE after CNN or U-Net results exist. A changed
# configuration here is a different method, and earlier comparisons against it
# would no longer describe the same benchmark.
#
# input_quantization_bits is part of the method definition, not an
# implementation detail: OpenCV CLAHE needs an integer image, and the fixed
# mapping round(x * 255) over the whole [0,1] benchmark range is what this
# method is defined to use.

clahe:
  algorithm: {config.algorithm}
  input_quantization_bits: {config.input_quantization_bits}
  clip_limit: {config.clip_limit:g}
  tile_grid_size: [{config.tile_rows}, {config.tile_columns}]

  selected_by:
    split: {TUNING_SPLIT}
    primary_metric: {PRIMARY_METRIC}
    aggregation: patient_weighted
    search_config: {search_config}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--search-config", default="clahe_search.yaml")
    parser.add_argument("--evaluation-config", default="evaluation.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--output-dir", default=METRICS_DIR.as_posix())
    parser.add_argument("--clahe-config", default=CANONICAL_CLAHE_CONFIG.as_posix())
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow rewriting an already frozen configs/clahe.yaml (deliberate retune only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug only: cap the slice count; refused when writing the canonical table",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output_dir = Path(arguments.output_dir)
    search_csv = output_dir / CANONICAL_SEARCH_CSV.name
    config_path = Path(arguments.clahe_config)

    try:
        check_output_policy(arguments.limit, search_csv)
        check_freeze_policy(config_path, arguments.overwrite)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    space = ClaheSearchSpace.from_mapping(load_config(arguments.search_config))
    evaluation = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]
    candidates = space.candidates()

    rows = manifest_rows(Path(arguments.manifest), TUNING_SPLIT)
    if arguments.limit:
        rows = rows.head(arguments.limit)
    subjects = sorted(rows["subject_id"].unique(), key=int)

    print(f"search space    : {len(candidates)} candidates from {arguments.search_config}")
    print(f"clip limits     : {[float(v) for v in space.clip_limit]}")
    print(f"tile grids      : {[list(g) for g in space.tile_grid_size]}")
    print(f"primary metric  : patient-weighted {PRIMARY_METRIC} (predeclared)")
    print(f"tuning split    : {TUNING_SPLIT} ({len(subjects)} patients, {len(rows)} slices)")
    print("test and stress image content is NOT read by this command")
    print()

    per_candidate = sweep(rows, root, preprocessing, evaluation, degradation, candidates)
    baseline = baseline_patient_weighted(BASELINE_PATIENTS_CSV)
    ranked = rank_candidates(candidate_table(candidates, per_candidate, baseline))
    write_csv(ranked, search_csv)
    winner = selected_config(ranked)

    print()
    header = (
        f"{'clip':>6}{'grid':>9}{'body_ssim':>12}{'body_psnr':>12}"
        f"{'full_ssim':>12}{'full_psnr':>12}{'rank':>6}  selected"
    )
    print(header)
    print("-" * (len(header) + 2))
    for row in ranked.itertuples():
        grid = f"{row.tile_grid_rows}x{row.tile_grid_cols}"
        print(
            f"{row.clip_limit:>6g}{grid:>9}"
            f"{row.patient_weighted_body_ssim:>12.6f}{row.patient_weighted_body_psnr:>12.4f}"
            f"{row.patient_weighted_full_ssim:>12.6f}{row.patient_weighted_full_psnr:>12.4f}"
            f"{row.selection_rank:>6}  {'<== SELECTED' if row.selected else ''}"
        )
    print("-" * (len(header) + 2))
    print(f"{'baseline':>6}{'-':>9}{baseline['body_ssim']:>12.6f}{baseline['body_psnr']:>12.4f}")
    print()

    best = ranked.iloc[0]
    beats = bool(best["patient_weighted_body_ssim"] > baseline["body_ssim"])
    print(
        f"selected        : clip_limit {winner.clip_limit:g}, "
        f"tile_grid_size [{winner.tile_rows}, {winner.tile_columns}]"
    )
    print(
        f"vs baseline     : body_ssim {best['delta_vs_degraded_body_ssim']:+.6f}  "
        f"-> beats no-restoration on the predeclared metric: {'YES' if beats else 'NO'}"
    )
    print(f"search table    : {search_csv}")

    write_frozen_config(winner, config_path, search_config_provenance(arguments.search_config))
    print(f"frozen config   : {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
