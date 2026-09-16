"""Audit the evaluation body mask on TRAINING slices only.

    uv run python scripts/audit_body_mask.py --root data/raw/chaos

The body mask decides which pixels the body-region metrics are computed over,
for every method in this benchmark. Before any score is produced, the rule has
to be shown to work on real data: it must generate a mask for every slice, no
mask may be empty, and no eroded SSIM interior may be empty.

Training slices only, deliberately
----------------------------------
Only ``split == train`` rows of the frozen slice manifest are read. Validation
is not touched until the evaluation policy is frozen, and test and stress stay
sealed until the final benchmark. Auditing the mask is an implementation check,
so training data answers it completely.

No metric is computed here. This audit measures the mask, not image quality.

Outputs
-------
``outputs/audit/evaluation_body_mask_train_summary.json``
    compact tracked summary, no pixel data, no timestamp, byte-identical when
    regenerated from an unchanged definition.
``outputs/audit/figures/evaluation_mask/``
    optional local QC panels, git-ignored, never committed.
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
from ct_restoration.data.dicom import load_ct_hu  # noqa: E402
from ct_restoration.evaluation import (  # noqa: E402
    EvaluationConfig,
    body_mask_from_hu,
    require_development_split,
    ssim_interior_mask,
)

#: The tracked, canonical summary. A partial audit must never land here, so the
#: debug-only --limit flag is refused whenever this is the output path.
CANONICAL_SUMMARY = Path("outputs/audit/evaluation_body_mask_train_summary.json")

#: Quantiles reported for the per-slice coverage distributions.
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

#: Decimal places for floats in the tracked JSON.
_ROUND = 8


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


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}


def check_output_policy(limit: int, summary_path: Path) -> None:
    """Refuse to write a partial audit over the tracked canonical summary.

    Raises:
        ValueError: ``limit`` is set and the output is the canonical path.
    """
    if limit and Path(summary_path).resolve() == CANONICAL_SUMMARY.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite the canonical tracked "
            f"summary {CANONICAL_SUMMARY.as_posix()} with a partial audit. "
            "Re-run without --limit, or pass a noncanonical --summary path."
        )


def manifest_rows(manifest_path: Path, split: str) -> pd.DataFrame:
    """Rows of one development split, in manifest order.

    Raises:
        HeldOutSplitError: the split is sealed.
        ValueError: the manifest holds no rows for that split.
    """
    require_development_split(split)
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    rows = manifest[manifest["split"] == split].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No split == {split} rows in {manifest_path}")
    return rows


def select_qc_slices(rows: pd.DataFrame) -> pd.DataFrame:
    """Deterministically pick a few slices for local visual QC.

    Same rule as the degradation audit, applied without looking at any image:
    for each (acquisition group, source archive) pair, the first subject in
    canonical numeric order, then that subject's middle slice by geometric
    index.
    """
    chosen = []
    pairs = sorted({(row.acquisition_group, row.source_archive) for row in rows.itertuples()})
    for group, archive in pairs:
        subset = rows[(rows["acquisition_group"] == group) & (rows["source_archive"] == archive)]
        subject = sorted(subset["subject_id"].unique(), key=int)[0]
        slices = subset[subset["subject_id"] == subject].sort_values("geometric_slice_index")
        chosen.append(slices.iloc[len(slices) // 2])
    return pd.DataFrame(chosen)


def render_qc(
    selection: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    config: EvaluationConfig,
    figures_dir: Path,
) -> list[Path]:
    """Write clean / body-mask / SSIM-interior panels locally. Never committed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ct_restoration.data.preprocessing import preprocess_ct_slice

    ensure_dir(figures_dir)
    written: list[Path] = []
    for row in selection.itertuples():
        image_hu = load_ct_hu(root / row.relative_dicom_path)
        size = tuple(preprocessing["image_size"])
        body = body_mask_from_hu(image_hu, size, config.body_mask)
        interior = ssim_interior_mask(body, config.ssim.win_size)
        clean = preprocess_ct_slice(
            image_hu,
            window_center=preprocessing["window_center"],
            window_width=preprocessing["window_width"],
            size=size,
            interpolation=preprocessing["interpolation"],
        )

        figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("clean reference")
        for axis, mask, title in (
            (axes[1], body, f"body mask ({body.mean() * 100:.1f} % of frame)"),
            (axes[2], interior, f"SSIM interior ({interior.mean() * 100:.1f} % of frame)"),
        ):
            axis.imshow(clean, cmap="gray", vmin=0, vmax=1)
            axis.imshow(
                np.ma.masked_where(~mask, np.ones_like(clean)),
                cmap="autumn",
                alpha=0.35,
                vmin=0,
                vmax=1,
            )
            axis.set_title(title)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"subject {row.subject_id}  group {row.acquisition_group}  "
            f"{row.source_archive}  slice {row.geometric_slice_index}  (train)"
        )
        figure.tight_layout()

        path = figures_dir / f"mask_subject{row.subject_id}_g{row.acquisition_group}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--evaluation-config", default="evaluation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--summary", default=CANONICAL_SUMMARY.as_posix())
    parser.add_argument("--figures-dir", default="outputs/audit/figures/evaluation_mask")
    parser.add_argument("--no-figures", action="store_true", help="skip local QC rendering")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="debug only: cap the slice count; refused when writing the canonical summary",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        check_output_policy(arguments.limit, Path(arguments.summary))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = Path(arguments.root)
    config = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    rows = manifest_rows(Path(arguments.manifest), "train")
    if arguments.limit:
        rows = rows.head(arguments.limit)

    subjects = sorted(rows["subject_id"].unique(), key=int)
    print(f"threshold_hu    : {config.body_mask.threshold_hu}")
    print(f"connectivity    : {config.body_mask.connectivity}")
    print(f"ssim win_size   : {config.ssim.win_size}")
    print(f"train subjects  : {len(subjects)}")
    print(f"train slices    : {len(rows)}")
    print("inspecting TRAINING slices only; validation/test/stress not read")
    print()

    body_fractions: list[float] = []
    interior_fractions: list[float] = []
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    failures: list[str] = []
    empty_masks: list[str] = []
    empty_interiors: list[str] = []
    generated = 0

    size = tuple(preprocessing["image_size"])
    for position, row in enumerate(rows.itertuples(), start=1):
        key = row.relative_dicom_path
        try:
            image_hu = load_ct_hu(root / key)
            body = body_mask_from_hu(image_hu, size, config.body_mask)
            interior = ssim_interior_mask(body, config.ssim.win_size)
        except Exception as error:  # noqa: BLE001 - the audit records, never hides
            failures.append(f"{key}: {type(error).__name__}: {error}")
            continue

        generated += 1
        shapes.add(body.shape)
        dtypes.add(str(body.dtype))
        if not body.any():
            empty_masks.append(key)
        if not interior.any():
            empty_interiors.append(key)
        body_fractions.append(float(body.mean()))
        interior_fractions.append(float(interior.mean()))

        if position % 500 == 0 or position == len(rows):
            print(f"  processed {position:>5} / {len(rows)}")

    summary = {
        "milestone": "5 - evaluation framework and degraded baseline",
        "audited_split": "train",
        "evaluation_config_path": Path("configs").joinpath(arguments.evaluation_config).as_posix(),
        "evaluation_config": config.as_dict(),
        "body_mask_rule": (
            "clean HU > threshold_hu, 8-connected components, keep the largest, fill enclosed "
            "holes, resize to the evaluation size with nearest-neighbour interpolation, cast "
            "to bool. Derived from the clean reference only; never from a degraded or restored "
            "image, and never supplied to a restoration method."
        ),
        "ssim_interior_rule": (
            "body mask eroded by an all-true win_size square, so that the complete SSIM "
            "neighbourhood of every retained centre lies inside the body. Outside-the-array is "
            "treated as background, which also removes the border rim scikit-image excludes "
            "from its own mean."
        ),
        "train_subjects": len(subjects),
        "train_slices": int(len(rows)),
        "train_subject_ids": subjects,
        "training_set_diagnostics": {
            "masks_generated": generated,
            "mask_generation_failures": len(failures),
            "empty_masks": len(empty_masks),
            "empty_ssim_interior_masks": len(empty_interiors),
            "mask_output_shapes": sorted(list(shape) for shape in shapes),
            "mask_dtypes": sorted(dtypes),
            "body_pixel_fraction_quantiles": _quantiles(body_fractions),
            "ssim_interior_fraction_quantiles": _quantiles(interior_fractions),
            "failure_examples": sorted(failures)[:8],
            "empty_mask_examples": sorted(empty_masks)[:8],
            "empty_ssim_interior_examples": sorted(empty_interiors)[:8],
        },
        "inspection_policy": {
            "training_images_inspected": True,
            "validation_images_inspected": False,
            "test_images_inspected": False,
            "stress_images_inspected": False,
            "note": (
                "All statistics here are TRAINING-SET diagnostics of the mask rule. No image "
                "quality metric is computed in this audit. The -500 HU threshold was fixed in "
                "advance and was not selected by comparing metric outcomes."
            ),
        },
        "scope_statement": (
            "The body mask is a crude deterministic silhouette separating the patient from air "
            "and table. It is an evaluation region, not clinical segmentation, and no "
            "anatomical claim is made about it."
        ),
    }

    summary_path = Path(arguments.summary)
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_round(summary), handle, indent=2, allow_nan=False)
        handle.write("\n")

    figures: list[Path] = []
    if not arguments.no_figures:
        figures = render_qc(
            select_qc_slices(rows), root, preprocessing, config, Path(arguments.figures_dir)
        )

    diagnostics = summary["training_set_diagnostics"]
    body_q = diagnostics["body_pixel_fraction_quantiles"]
    interior_q = diagnostics["ssim_interior_fraction_quantiles"]
    print()
    print(f"masks generated : {diagnostics['masks_generated']}")
    print(f"failures        : {diagnostics['mask_generation_failures']}")
    print(f"empty masks     : {diagnostics['empty_masks']}")
    print(f"empty interiors : {diagnostics['empty_ssim_interior_masks']}")
    print(f"mask shapes     : {diagnostics['mask_output_shapes']}")
    print(f"mask dtypes     : {diagnostics['mask_dtypes']}")
    print(
        f"body fraction   : min {body_q['q00']:.4f}  q25 {body_q['q25']:.4f}  "
        f"median {body_q['q50']:.4f}  q75 {body_q['q75']:.4f}  max {body_q['q100']:.4f}"
    )
    print(
        f"ssim interior   : min {interior_q['q00']:.4f}  q25 {interior_q['q25']:.4f}  "
        f"median {interior_q['q50']:.4f}  q75 {interior_q['q75']:.4f}  "
        f"max {interior_q['q100']:.4f}"
    )
    print()
    print(f"summary         : {summary_path}")
    for path in figures:
        print(f"qc figure       : {path}  (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
