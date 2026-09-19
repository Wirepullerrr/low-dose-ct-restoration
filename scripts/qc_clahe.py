"""Local visual QC of the frozen CLAHE configuration, on TRAINING slices only.

    uv run python scripts/qc_clahe.py --root data/raw/chaos

Run **after** the configuration has been selected numerically. Its purpose is
to confirm that the implementation does what CLAHE is supposed to do, not to
choose parameters. There is no ``--split`` option and no way to point it at
validation: looking at validation pictures and then adjusting anything would
turn the development estimate into a fitted one, and the selection is already
frozen in :file:`configs/clahe.yaml` regardless of what these panels show.

Selection of examples is deterministic and independent of image content: for
each (acquisition group, source archive) pair present in training, the first
subject in numeric order, then that subject's middle slice by geometric index.

Figures are written under a git-ignored directory. Rendered CT images are
never committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import manifest_rows  # noqa: E402
from ct_restoration.classical.clahe import ClaheConfig, apply_clahe  # noqa: E402
from ct_restoration.config import ensure_dir, load_config  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like  # noqa: E402
from ct_restoration.evaluation import EvaluationConfig, prepare_evaluation_slice  # noqa: E402

#: The split this command reads. Not configurable.
QC_SPLIT = "train"


def select_qc_slices(rows: pd.DataFrame) -> pd.DataFrame:
    """The same deterministic rule the earlier audits use."""
    chosen = []
    pairs = sorted({(row.acquisition_group, row.source_archive) for row in rows.itertuples()})
    for group, archive in pairs:
        subset = rows[(rows["acquisition_group"] == group) & (rows["source_archive"] == archive)]
        subject = sorted(subset["subject_id"].unique(), key=int)[0]
        slices = subset[subset["subject_id"] == subject].sort_values("geometric_slice_index")
        chosen.append(slices.iloc[len(slices) // 2])
    return pd.DataFrame(chosen)


def render(
    selection: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    evaluation: EvaluationConfig,
    degradation: DegradationConfig,
    config: ClaheConfig,
    figures_dir: Path,
) -> list[Path]:
    """Write clean / degraded / CLAHE / two difference panels. Never committed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figures_dir)
    written: list[Path] = []
    for row in selection.itertuples():
        key = row.relative_dicom_path
        clean, _, _ = prepare_evaluation_slice(root / key, preprocessing, evaluation)
        degraded = degrade_low_dose_like(clean, key, degradation)
        restored = apply_clahe(degraded, config)

        clean64 = clean.astype(np.float64)
        degraded64 = degraded.astype(np.float64)
        restored64 = restored.astype(np.float64)

        figure, axes = plt.subplots(1, 5, figsize=(20, 4.6))
        for axis, image, title in (
            (axes[0], clean, "clean reference"),
            (axes[1], degraded, "degraded (low-dose-like)"),
            (
                axes[2],
                restored,
                f"CLAHE clip {config.clip_limit:g} grid {config.tile_rows}x{config.tile_columns}",
            ),
        ):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=10)
        for axis, difference, title in (
            (axes[3], restored64 - degraded64, "CLAHE - degraded"),
            (axes[4], restored64 - clean64, "CLAHE - clean"),
        ):
            panel = axis.imshow(difference, cmap="coolwarm", vmin=-0.25, vmax=0.25)
            axis.set_title(f"{title}  (mean |d| {np.abs(difference).mean():.4f})", fontsize=10)
            figure.colorbar(panel, ax=axis, fraction=0.046)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"subject {row.subject_id}  group {row.acquisition_group}  "
            f"{row.source_archive}  slice {row.geometric_slice_index}  (TRAIN)"
        )
        figure.tight_layout()

        path = figures_dir / f"clahe_subject{row.subject_id}_g{row.acquisition_group}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--clahe-config", default="clahe.yaml")
    parser.add_argument("--evaluation-config", default="evaluation.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--figures-dir", default="outputs/audit/figures/clahe")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    root = Path(arguments.root)
    config = ClaheConfig.from_mapping(load_config(arguments.clahe_config))
    evaluation = EvaluationConfig.from_mapping(load_config(arguments.evaluation_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    rows = manifest_rows(Path(arguments.manifest), QC_SPLIT)
    selection = select_qc_slices(rows)

    print(
        f"config          : clip_limit {config.clip_limit:g}, "
        f"tile_grid_size [{config.tile_rows}, {config.tile_columns}]"
    )
    print(f"split           : {QC_SPLIT} only; validation/test/stress not rendered")
    print(f"examples        : {len(selection)} deterministic training slices")
    print()

    for path in render(
        selection,
        root,
        preprocessing,
        evaluation,
        degradation,
        config,
        Path(arguments.figures_dir),
    ):
        print(f"qc figure       : {path}  (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
