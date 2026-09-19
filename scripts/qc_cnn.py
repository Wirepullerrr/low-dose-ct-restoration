"""Render train-only visual QC panels for the selected CNN checkpoint.

    uv run python scripts/qc_cnn.py --root data/raw/chaos

Training slices only, by construction
--------------------------------------
There is no ``--split`` option. This reads the training split and nothing
else, and it runs **after** the checkpoint has already been selected
numerically. Looking at validation images and then changing something is how
a held-out estimate quietly becomes a fitted one; looking at training images
after the decision is made is just checking that the numbers describe what
they appear to describe.

What each panel shows
---------------------
clean | degraded | CNN restored | CNN - degraded (the correction) | CNN - clean
(the remaining error). The two difference panels are what make the figure
worth rendering: a plausible-looking restored image tells you very little,
while the correction and the residual error show what the network is actually
doing and where it is still wrong.

Output
------
Git-ignored PNGs under ``outputs/audit/figures/cnn/``. No CT image is ever
committed to this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.benchmark import manifest_rows  # noqa: E402
from ct_restoration.config import load_config  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig, degrade_low_dose_like  # noqa: E402
from ct_restoration.data.preprocessing import preprocess_ct_slice  # noqa: E402
from ct_restoration.models.adapter import restore_array  # noqa: E402
from ct_restoration.models.cnn import ResidualCnnConfig, build_model  # noqa: E402

#: The split this command reads. Not configurable: see the module docstring.
QC_SPLIT = "train"

#: Git-ignored output directory. Rendered CT images are never committed.
FIGURE_DIR = Path("outputs/audit/figures/cnn")

#: Training subjects rendered, chosen to span both acquisition groups and
#: both source archives. Fixed, so the panels are the same every run.
QC_SUBJECTS = ("2", "3", "21", "31")

#: Symmetric colour limit for the two difference panels, in normalized units.
DIFFERENCE_LIMIT = 0.08


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/cnn_seed2026_best.pt")
    parser.add_argument("--cnn-config", default="cnn.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--output-dir", default=FIGURE_DIR.as_posix())
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    root = Path(arguments.root)
    device = torch.device(arguments.device)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = ResidualCnnConfig.from_mapping(load_config(arguments.cnn_config))
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    payload = torch.load(Path(arguments.checkpoint), map_location="cpu", weights_only=True)
    model = build_model(model_config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # manifest_rows applies the hold-out gate; QC_SPLIT is not configurable.
    rows = manifest_rows(Path(arguments.manifest), QC_SPLIT)

    print(f"checkpoint      : epoch {payload['epoch']}, seed {payload['seed']}")
    print(f"split           : {QC_SPLIT} (train slices only, after checkpoint selection)")
    print("validation, test and stress image content is NOT read by this command")
    print()

    from ct_restoration.data.dicom import load_ct_hu

    written: list[Path] = []
    for subject_id in QC_SUBJECTS:
        subject_rows = rows[rows["subject_id"] == subject_id]
        if subject_rows.empty:
            print(f"  subject {subject_id} is not in {QC_SPLIT}; skipped")
            continue

        # The middle slice: deterministic, and more likely to show anatomy
        # than either end of the scan.
        row = subject_rows.iloc[len(subject_rows) // 2]
        key = str(row["relative_dicom_path"])
        group = str(row["acquisition_group"])

        clean = preprocess_ct_slice(
            load_ct_hu(root / key),
            window_center=preprocessing["window_center"],
            window_width=preprocessing["window_width"],
            size=tuple(preprocessing["image_size"]),
            interpolation=preprocessing["interpolation"],
        )
        degraded = degrade_low_dose_like(clean, key, degradation)
        restored = restore_array(model, degraded, device)

        panels = [
            ("clean reference", clean, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}),
            ("degraded input", degraded, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}),
            ("CNN restored", restored, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}),
            (
                "CNN - degraded (correction)",
                restored - degraded,
                {"cmap": "bwr", "vmin": -DIFFERENCE_LIMIT, "vmax": DIFFERENCE_LIMIT},
            ),
            (
                "CNN - clean (error)",
                restored - clean,
                {"cmap": "bwr", "vmin": -DIFFERENCE_LIMIT, "vmax": DIFFERENCE_LIMIT},
            ),
        ]

        figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6))
        for axis, (title, image, style) in zip(axes, panels, strict=True):
            handle = axis.imshow(np.asarray(image, dtype=np.float64), **style)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
            figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

        error = float(np.abs(restored - clean).mean())
        before = float(np.abs(degraded - clean).mean())
        figure.suptitle(
            f"subject {subject_id} (group {group}, train) - epoch {payload['epoch']} checkpoint"
            f"   |   full MAE {before:.5f} -> {error:.5f}",
            fontsize=11,
        )
        figure.tight_layout()

        path = output_dir / f"cnn_subject{subject_id}_g{group}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path)
        print(f"  {path}   full MAE {before:.6f} -> {error:.6f}")

    print()
    print(f"{len(written)} panel(s) written to {output_dir} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
