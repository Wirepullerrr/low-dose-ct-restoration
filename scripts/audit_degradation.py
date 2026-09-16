"""Technical audit of the frozen degradation, on TRAINING slices only.

    uv run python scripts/audit_degradation.py --root data/raw/chaos

Purpose: show that the canonical degradation behaves as specified before any
restoration method exists. It is not the quality benchmark. No PSNR, SSIM or
restoration error is computed here; those belong to the evaluation milestone,
after the methods exist.

Training slices only, deliberately
----------------------------------
Only ``split == train`` rows of the frozen slice manifest are read. Validation,
test and stress image content is not inspected. The degradation parameters were
fixed in advance and are not being tuned here, so there is nothing to gain from
looking at held-out data, and looking would start spending information that the
final comparison depends on.

What it does
------------
For each training slice: rebuild the clean reference with the Milestone 1
preprocessing, degrade it using the manifest's ``relative_dicom_path`` as the
sample key, and accumulate diagnostics. Nothing is written per slice. Degraded
images stay generated on demand, so no precomputed degradation dataset is
created anywhere.

Outputs
-------
``outputs/audit/degradation_train_summary.json``
    compact tracked summary, no pixel data, no timestamp, byte-identical when
    regenerated from an unchanged experiment definition.
``outputs/audit/figures/degradation/``
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
from ct_restoration.data.degradation import (  # noqa: E402
    DegradationConfig,
    degrade_low_dose_like,
    derive_sample_seed,
    noise_scale,
)
from ct_restoration.data.dicom import load_ct_hu  # noqa: E402
from ct_restoration.data.preprocessing import preprocess_ct_slice  # noqa: E402

#: The tracked, canonical summary. A partial audit must never land here, so
#: the debug-only --limit flag is refused whenever this is the output path.
CANONICAL_SUMMARY = Path("outputs/audit/degradation_train_summary.json")

#: Quantiles reported for the per-slice distributions.
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)

#: Bin count for the pooled clean-intensity histogram, from which the pooled
#: sigma-map median is derived. sigma is monotonic in intensity, so the sigma
#: quantile is sigma of the intensity quantile. 4096 bins over [0, 1] resolves
#: the median to about 2.4e-4 in intensity.
_HISTOGRAM_BINS = 4096

#: Decimal places for floats in the tracked JSON. Rounding keeps the file
#: readable and stable to inspect; it does not affect any computation.
_ROUND = 8


def _round(value: Any) -> Any:
    """Round floats for the tracked summary, recursively through containers."""
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
    """Named quantiles of a per-slice distribution."""
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in QUANTILES}


class Accumulator:
    """Running diagnostics over the training slices.

    Everything is accumulated in float64 running sums rather than by keeping
    272 million pixels in memory. Per-slice scalars are small enough to keep.
    """

    def __init__(self, config: DegradationConfig) -> None:
        self.config = config
        self.pixels = 0
        self.perturbation_sum = 0.0
        self.perturbation_square_sum = 0.0
        self.absolute_sum = 0.0
        self.at_clip_min = 0
        self.at_clip_max = 0
        self.clean_at_zero = 0
        self.clean_at_one = 0
        self.degraded_min = np.inf
        self.degraded_max = -np.inf
        self.sigma_min = np.inf
        self.sigma_max = -np.inf
        self.histogram = np.zeros(_HISTOGRAM_BINS, dtype=np.int64)
        self.slice_perturbation_std: list[float] = []
        self.slice_absolute_mean: list[float] = []
        self.shapes: set[tuple[int, ...]] = set()
        self.dtypes: set[str] = set()
        self.finite_failures: list[str] = []
        self.range_failures: list[str] = []

    def update(self, clean: np.ndarray, degraded: np.ndarray, key: str) -> None:
        clean64 = clean.astype(np.float64)
        degraded64 = degraded.astype(np.float64)
        perturbation = degraded64 - clean64

        self.shapes.add(degraded.shape)
        self.dtypes.add(str(degraded.dtype))

        if not np.all(np.isfinite(degraded64)):
            self.finite_failures.append(key)
        low, high = float(self.config.clip_min), float(self.config.clip_max)
        if float(degraded64.min()) < low or float(degraded64.max()) > high:
            self.range_failures.append(key)

        self.pixels += degraded64.size
        self.perturbation_sum += float(perturbation.sum())
        self.perturbation_square_sum += float(np.square(perturbation).sum())
        self.absolute_sum += float(np.abs(perturbation).sum())

        self.at_clip_min += int((degraded64 <= low).sum())
        self.at_clip_max += int((degraded64 >= high).sum())
        self.clean_at_zero += int((clean64 <= low).sum())
        self.clean_at_one += int((clean64 >= high).sum())

        self.degraded_min = min(self.degraded_min, float(degraded64.min()))
        self.degraded_max = max(self.degraded_max, float(degraded64.max()))

        sigma = noise_scale(clean64, self.config)
        self.sigma_min = min(self.sigma_min, float(sigma.min()))
        self.sigma_max = max(self.sigma_max, float(sigma.max()))

        self.histogram += np.bincount(
            np.clip((clean64.ravel() * _HISTOGRAM_BINS).astype(np.int64), 0, _HISTOGRAM_BINS - 1),
            minlength=_HISTOGRAM_BINS,
        )

        self.slice_perturbation_std.append(float(perturbation.std()))
        self.slice_absolute_mean.append(float(np.abs(perturbation).mean()))

    def _pooled_intensity_quantile(self, quantile: float) -> float:
        """Quantile of the pooled clean intensity, from the histogram."""
        cumulative = np.cumsum(self.histogram)
        target = quantile * cumulative[-1]
        index = int(np.searchsorted(cumulative, target, side="left"))
        return (index + 0.5) / _HISTOGRAM_BINS

    def report(self) -> dict[str, Any]:
        mean = self.perturbation_sum / self.pixels
        variance = max(self.perturbation_square_sum / self.pixels - mean * mean, 0.0)
        median_intensity = self._pooled_intensity_quantile(0.5)

        return {
            "output_shapes": sorted(list(shape) for shape in self.shapes),
            "output_dtypes": sorted(self.dtypes),
            "degraded_global_min": self.degraded_min,
            "degraded_global_max": self.degraded_max,
            "perturbation_mean": mean,
            "perturbation_std": float(np.sqrt(variance)),
            "perturbation_mean_absolute": self.absolute_sum / self.pixels,
            "per_slice_perturbation_std_quantiles": _quantiles(self.slice_perturbation_std),
            "per_slice_mean_absolute_perturbation_quantiles": _quantiles(self.slice_absolute_mean),
            "fraction_degraded_at_clip_min": self.at_clip_min / self.pixels,
            "fraction_degraded_at_clip_max": self.at_clip_max / self.pixels,
            "fraction_clean_at_clip_min": self.clean_at_zero / self.pixels,
            "fraction_clean_at_clip_max": self.clean_at_one / self.pixels,
            "sigma_map_min": self.sigma_min,
            "sigma_map_median": float(noise_scale(np.array([median_intensity]), self.config)[0]),
            "sigma_map_max": self.sigma_max,
            "sigma_map_median_source": (
                f"sigma of the pooled clean-intensity median, from a {_HISTOGRAM_BINS}-bin "
                "histogram; sigma is monotonic in intensity"
            ),
            "total_pixels": self.pixels,
            "finite_value_failures": len(self.finite_failures),
            "range_failures": len(self.range_failures),
            "finite_value_failure_keys": sorted(self.finite_failures)[:8],
            "range_failure_keys": sorted(self.range_failures)[:8],
        }


def check_output_policy(limit: int, summary_path: Path) -> None:
    """Refuse to write a partial audit over the tracked canonical summary.

    ``--limit`` exists for debugging and stops after a handful of slices. The
    resulting diagnostics describe those slices only, but they look exactly
    like a full audit once written to disk, and the tracked file is the record
    of what the frozen degradation does on the whole training split. Silently
    overwriting it with a truncated run would corrupt that record in a way
    nothing downstream could detect.

    Raises:
        ValueError: ``limit`` is set and the output is the canonical path.
    """
    if limit and Path(summary_path).resolve() == CANONICAL_SUMMARY.resolve():
        raise ValueError(
            f"--limit {limit} is debug-only and would overwrite the canonical tracked "
            f"summary {CANONICAL_SUMMARY.as_posix()} with a partial audit. "
            "Re-run without --limit, or pass a noncanonical --summary path."
        )


def training_slices(manifest_path: Path) -> pd.DataFrame:
    """Training rows of the frozen slice manifest, in manifest order.

    Raises:
        ValueError: the manifest holds no training rows, which would mean the
            frozen split is not the one this milestone was built against.
    """
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    training = manifest[manifest["split"] == "train"].reset_index(drop=True)
    if training.empty:
        raise ValueError(f"No split == train rows in {manifest_path}")
    return training


def clean_reference(dicom_path: Path, preprocessing: dict[str, Any]) -> np.ndarray:
    """Rebuild one clean reference with the established Milestone 1 pipeline."""
    return preprocess_ct_slice(
        load_ct_hu(dicom_path),
        window_center=preprocessing["window_center"],
        window_width=preprocessing["window_width"],
        size=tuple(preprocessing["image_size"]),
        interpolation=preprocessing["interpolation"],
    )


def select_qc_slices(training: pd.DataFrame) -> pd.DataFrame:
    """Deterministically pick a few training slices for local visual QC.

    Rule, fixed in advance and applied without looking at any image: for each
    (acquisition group, source archive) pair present in the training split,
    take the first subject in canonical numeric order, then that subject's
    middle slice by geometric index. Nothing is chosen because it looks good.
    """
    chosen = []
    pairs = sorted({(row.acquisition_group, row.source_archive) for row in training.itertuples()})
    for group, archive in pairs:
        subset = training[
            (training["acquisition_group"] == group) & (training["source_archive"] == archive)
        ]
        subject = sorted(subset["subject_id"].unique(), key=int)[0]
        slices = subset[subset["subject_id"] == subject].sort_values("geometric_slice_index")
        chosen.append(slices.iloc[len(slices) // 2])
    return pd.DataFrame(chosen)


def render_qc(
    selection: pd.DataFrame,
    root: Path,
    preprocessing: dict[str, Any],
    config: DegradationConfig,
    figures_dir: Path,
) -> list[Path]:
    """Write clean / degraded / difference panels locally. Never committed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(figures_dir)
    written: list[Path] = []
    for row in selection.itertuples():
        key = row.relative_dicom_path
        clean = clean_reference(root / key, preprocessing)
        degraded = degrade_low_dose_like(clean, key, config)
        difference = degraded.astype(np.float64) - clean.astype(np.float64)

        figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))
        axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("clean reference")
        axes[1].imshow(degraded, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("degraded (low-dose-like)")
        panel = axes[2].imshow(difference, cmap="coolwarm", vmin=-0.15, vmax=0.15)
        axes[2].set_title("difference")
        figure.colorbar(panel, ax=axes[2], fraction=0.046)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"subject {row.subject_id}  group {row.acquisition_group}  "
            f"{row.source_archive}  slice {row.geometric_slice_index}  "
            f"(train)  |  perturbation std {difference.std():.4f}"
        )
        figure.tight_layout()

        path = figures_dir / f"degradation_subject{row.subject_id}_g{row.acquisition_group}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--summary", default=CANONICAL_SUMMARY.as_posix())
    parser.add_argument("--figures-dir", default="outputs/audit/figures/degradation")
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
    config = DegradationConfig.from_mapping(load_config(arguments.degradation_config))
    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]

    training = training_slices(Path(arguments.manifest))
    if arguments.limit:
        training = training.head(arguments.limit)

    subjects = sorted(training["subject_id"].unique(), key=int)
    print(f"algorithm       : {config.algorithm}")
    print(f"global_seed     : {config.global_seed}")
    print(f"sigma(x)        : {config.sigma_floor} + {config.sigma_signal} * sqrt(x)")
    print(f"correlation     : {config.correlation_sigma_px} px, reflect boundary")
    print(f"train subjects  : {len(subjects)}")
    print(f"train slices    : {len(training)}")
    print("inspecting TRAINING slices only; validation/test/stress not read")
    print()

    accumulator = Accumulator(config)
    for position, row in enumerate(training.itertuples(), start=1):
        key = row.relative_dicom_path
        clean = clean_reference(root / key, preprocessing)
        degraded = degrade_low_dose_like(clean, key, config)
        accumulator.update(clean, degraded, key)
        if position % 500 == 0 or position == len(training):
            print(f"  processed {position:>5} / {len(training)}")

    diagnostics = accumulator.report()

    summary = {
        "milestone": "4 - synthetic low-dose-like degradation",
        "algorithm": config.algorithm,
        "config_path": Path("configs").joinpath(arguments.degradation_config).as_posix(),
        "config": config.as_dict(),
        "preprocessing": {
            "window_center": preprocessing["window_center"],
            "window_width": preprocessing["window_width"],
            "image_size": list(preprocessing["image_size"]),
            "interpolation": preprocessing["interpolation"],
        },
        "equation": "degraded = clip(x + (sigma_floor + sigma_signal*sqrt(x)) * eps, 0, 1)",
        "sigma_model": (
            "heuristic, not physical: sigma(x) = sigma_floor + sigma_signal*sqrt(x) uses local "
            "normalized intensity only, to make the corruption spatially non-uniform"
        ),
        "noise_field": (
            "zero-mean unit-variance Gaussian field, spatially correlated with a Gaussian "
            f"filter of sigma {config.correlation_sigma_px} px under reflect boundary handling, "
            "renormalized to zero mean and unit standard deviation after filtering"
        ),
        "audited_split": "train",
        "train_subjects": len(subjects),
        "train_slices": int(len(training)),
        "train_subject_ids": subjects,
        "training_set_diagnostics": diagnostics,
        "inspection_policy": {
            "training_images_inspected": True,
            "validation_images_inspected": False,
            "test_images_inspected": False,
            "stress_images_inspected": False,
            "note": (
                "All empirical statistics in training_set_diagnostics are TRAINING-SET "
                "diagnostics. Validation, test and stress image content was not read while "
                "establishing this degradation. The v1 parameters were fixed in advance and "
                "were not tuned against any image."
            ),
        },
        "determinism_policy": (
            "The degraded image is a pure function of the clean reference, this config and the "
            "sample key, which is the POSIX-style relative_dicom_path from the frozen slice "
            "manifest. Per-slice seeds come from SHA-256 over (algorithm, global_seed, "
            "canonical key) and drive a private PCG64 generator, so results do not depend on "
            "call order, batching, DataLoader workers, global NumPy random state, time, "
            "hostname, absolute path or process id. Path separators are canonicalized, so "
            "Windows and POSIX spellings of one slice agree."
        ),
        "scope_statement": (
            "Controlled image-domain synthetic low-dose-like corruption of an already windowed "
            "and normalized CT representation. Not a physical dose simulation: no mAs, no dose "
            "reduction percentage, no photon count, no scanner-equivalent low-dose "
            "reconstruction and no clinical validation is claimed or implied."
        ),
        "known_limitations": [
            "The fixed 40/400 HU window leaves large regions at exactly 0 or 1, and this "
            "degradation adds noise there too. That does not reproduce the physics of air, "
            "out-of-field padding or saturated dense structures.",
            "Clipping at the output bounds makes the realized perturbation asymmetric wherever "
            "the clean image is already saturated, so the realized noise std in those regions "
            "is below the configured sigma.",
            "The HU interpretation of a normalized sigma holds only inside the linear portion "
            "of the window and breaks down for pixels the window already clipped.",
            "Only increased intensity-dependent correlated noise is modelled. Real low-dose "
            "reconstructions also differ in resolution, streak artefacts and reconstruction "
            "behaviour, none of which is represented here.",
            "sigma(x) uses local normalized intensity as a simple heuristic for making the "
            "corruption spatially non-uniform, as reconstructed low-dose CT noise also is. It "
            "is not derived from projection-domain photon counts, line-integral attenuation, "
            "scanner calibration or any physical noise model; a voxel intensity is not a "
            "detected photon count, and real image-domain noise at a point depends on every "
            "ray path crossing it and on the reconstruction algorithm. No claim is made that a "
            "brighter pixel physically receives the configured amount of noise.",
        ],
        "no_precomputed_dataset": (
            "No degraded image is written to disk by this audit. Degradation is regenerated on "
            "demand from the clean reference and the sample key."
        ),
    }

    summary_path = Path(arguments.summary)
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_round(summary), handle, indent=2)
        handle.write("\n")

    figures: list[Path] = []
    if not arguments.no_figures:
        figures = render_qc(
            select_qc_slices(training), root, preprocessing, config, Path(arguments.figures_dir)
        )

    print()
    print(f"output shapes   : {diagnostics['output_shapes']}")
    print(f"output dtypes   : {diagnostics['output_dtypes']}")
    print(
        f"degraded range  : [{diagnostics['degraded_global_min']:.6f}, "
        f"{diagnostics['degraded_global_max']:.6f}]"
    )
    print(f"perturb mean    : {diagnostics['perturbation_mean']:+.6f}")
    print(f"perturb std     : {diagnostics['perturbation_std']:.6f}")
    print(f"perturb mean|.| : {diagnostics['perturbation_mean_absolute']:.6f}")
    per_slice = diagnostics["per_slice_perturbation_std_quantiles"]
    print(
        f"per-slice std   : min {per_slice['q00']:.6f}  median {per_slice['q50']:.6f}  "
        f"max {per_slice['q100']:.6f}"
    )
    per_slice_abs = diagnostics["per_slice_mean_absolute_perturbation_quantiles"]
    print(
        f"per-slice |.|   : min {per_slice_abs['q00']:.6f}  median {per_slice_abs['q50']:.6f}  "
        f"max {per_slice_abs['q100']:.6f}"
    )
    print(f"degraded at 0   : {diagnostics['fraction_degraded_at_clip_min'] * 100:.3f} %")
    print(f"degraded at 1   : {diagnostics['fraction_degraded_at_clip_max'] * 100:.3f} %")
    print(f"clean at 0      : {diagnostics['fraction_clean_at_clip_min'] * 100:.3f} %")
    print(f"clean at 1      : {diagnostics['fraction_clean_at_clip_max'] * 100:.3f} %")
    print(
        f"sigma map       : min {diagnostics['sigma_map_min']:.6f}  "
        f"median {diagnostics['sigma_map_median']:.6f}  max {diagnostics['sigma_map_max']:.6f}"
    )
    print(f"finite failures : {diagnostics['finite_value_failures']}")
    print(f"range failures  : {diagnostics['range_failures']}")
    print()
    first_key = training.iloc[0].relative_dicom_path
    print(f"example key     : {first_key}")
    print(f"example seed    : {derive_sample_seed(config.global_seed, first_key)}")
    print(f"summary         : {summary_path}")
    for path in figures:
        print(f"qc figure       : {path}  (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
