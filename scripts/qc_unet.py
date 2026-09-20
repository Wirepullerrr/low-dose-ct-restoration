"""Render train-only visual QC panels comparing the CNN and the U-Net.

    uv run python scripts/qc_unet.py --root data/raw/chaos

Training slices only, by construction
--------------------------------------
There is no ``--split`` option. This reads the training split and nothing
else, and it runs **after** both checkpoints have already been selected
numerically and the canonical metrics have already been written. Looking at
validation images and then changing something is how a held-out estimate
quietly becomes a fitted one; looking at training images after every decision
is made is just checking that the numbers describe what they appear to
describe.

What each panel shows
---------------------
clean | degraded | CNN restored | U-Net restored | the U-Net's predicted
correction | U-Net - clean (the remaining error).

The correction panel shows ``raw - degraded``, taken from the **unclamped**
output - what the network asked for. That is not the same image as
``restored - degraded``, which is only the part that survived the clamp, and
on this model the two differ across about half the frame. The panel label
and the image come from one shared helper so they cannot disagree. The MAE
figures in the title are computed from the **clamped** restoration, which is
what the benchmark scores.

The two difference panels are what make the figure worth rendering. A
plausible-looking restored image tells you very little; the correction and
the residual error show what the network is actually doing and where it is
still wrong. Putting the two models side by side is the point of this
milestone: the numbers say which scored better, the panels say whether they
fail in the same places.

Output
------
Git-ignored PNGs under ``outputs/audit/figures/unet/``. No CT image is ever
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
from ct_restoration.models.cnn import ResidualCnnConfig  # noqa: E402
from ct_restoration.models.cnn import build_model as build_cnn  # noqa: E402
from ct_restoration.models.diagnostics import predicted_correction_panel  # noqa: E402
from ct_restoration.models.unet import LightweightResidualUnetConfig  # noqa: E402
from ct_restoration.models.unet import build_model as build_unet  # noqa: E402

#: The split this command reads. Not configurable: see the module docstring.
QC_SPLIT = "train"

#: Git-ignored output directory. Rendered CT images are never committed.
FIGURE_DIR = Path("outputs/audit/figures/unet")

#: Training subjects rendered, chosen to span both acquisition groups and
#: both source archives. The same four the CNN QC used, so the two sets of
#: panels show the same anatomy. Fixed, so the panels are the same every run.
QC_SUBJECTS = ("2", "3", "21", "31")

#: Symmetric colour limit for the two difference panels, in normalized units.
#: The same limit the CNN panels used, so the two are visually comparable.
DIFFERENCE_LIMIT = 0.08


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--unet-checkpoint", default="outputs/checkpoints/unet_seed2026_best.pt")
    parser.add_argument("--cnn-checkpoint", default="outputs/checkpoints/cnn_seed2026_best.pt")
    parser.add_argument("--unet-config", default="unet.yaml")
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

    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    unet_payload = torch.load(
        Path(arguments.unet_checkpoint), map_location="cpu", weights_only=True
    )
    unet = build_unet(
        LightweightResidualUnetConfig.from_mapping(load_config(arguments.unet_config)), device
    )
    unet.load_state_dict(unet_payload["model_state_dict"])
    unet.eval()

    cnn_payload = torch.load(Path(arguments.cnn_checkpoint), map_location="cpu", weights_only=True)
    cnn = build_cnn(ResidualCnnConfig.from_mapping(load_config(arguments.cnn_config)), device)
    cnn.load_state_dict(cnn_payload["model_state_dict"])
    cnn.eval()

    # manifest_rows applies the hold-out gate; QC_SPLIT is not configurable.
    rows = manifest_rows(Path(arguments.manifest), QC_SPLIT)

    print(f"U-Net           : epoch {unet_payload['epoch']}, seed {unet_payload['seed']}")
    print(f"CNN             : epoch {cnn_payload['epoch']}, seed {cnn_payload['seed']}")
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
        # than either end of the scan. Same rule the CNN panels used.
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
        cnn_restored = restore_array(cnn, degraded, device)
        unet_restored = restore_array(unet, degraded, device)
        correction_label, unet_correction = predicted_correction_panel(unet, degraded, device)

        grey = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
        diverging = {"cmap": "bwr", "vmin": -DIFFERENCE_LIMIT, "vmax": DIFFERENCE_LIMIT}
        panels = [
            ("clean reference", clean, grey),
            ("degraded input", degraded, grey),
            ("CNN restored", cnn_restored, grey),
            ("U-Net restored", unet_restored, grey),
            (f"U-Net {correction_label}", unet_correction, diverging),
            ("U-Net - clean (error)", unet_restored - clean, diverging),
        ]

        figure, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.4))
        for axis, (title, image, style) in zip(axes, panels, strict=True):
            handle = axis.imshow(np.asarray(image, dtype=np.float64), **style)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
            figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

        before = float(np.abs(degraded - clean).mean())
        cnn_error = float(np.abs(cnn_restored - clean).mean())
        unet_error = float(np.abs(unet_restored - clean).mean())
        figure.suptitle(
            f"subject {subject_id} (group {group}, train)   |   full MAE  degraded {before:.5f}"
            f"  ->  CNN {cnn_error:.5f}  ->  U-Net {unet_error:.5f}",
            fontsize=11,
        )
        figure.tight_layout()

        path = output_dir / f"unet_subject{subject_id}_g{group}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path)
        print(f"  {path}   full MAE {before:.6f} -> CNN {cnn_error:.6f} -> U-Net {unet_error:.6f}")

    print()
    print(f"{len(written)} panel(s) written to {output_dir} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
