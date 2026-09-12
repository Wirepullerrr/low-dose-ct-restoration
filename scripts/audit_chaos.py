"""Audit a real CHAOS CT download before it is used for any experiment.

Run after extracting the CT portions of the CHAOS archives::

    uv run python scripts/audit_chaos.py --root data/raw/chaos

The script is read-only with respect to the dataset and is safe to re-run after
a fresh download. It answers the questions that have to be settled before any
split, degradation or metric is defined:

* how many subjects, series and slices actually exist, and which archive each
  subject came from;
* whether every slice converts to Hounsfield Units, and with which
  ``RescaleSlope`` / ``RescaleIntercept``;
* whether pixel padding is declared, and how much of a frame is background;
* how much of each slice the current 40 / 400 HU window clips;
* whether slice geometry supports a reliable anatomical ordering, using the
  slice normal from *ImageOrientationPatient* rather than assuming that slices
  advance along the patient z axis;
* where non-anatomical outlier pixels are, exhaustively and per coordinate.

Patient-identifying elements are never read or written. Subjects are identified
by the CHAOS folder name, and DICOM instance UIDs are written out only as short
deterministic hashes, which keeps them usable for grouping and counting without
republishing dataset metadata.

Two instruments in this file are deliberately audit-only and are not part of
the production preprocessing path: the crude body mask behind the in-body
window statistics, and the outlier accounting, which records extreme pixels but
never alters them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.config import ensure_dir, load_config  # noqa: E402
from ct_restoration.data.chaos import ChaosCtSubject, find_ct_subjects  # noqa: E402
from ct_restoration.data.dicom import (  # noqa: E402
    describe_ct_dataset,
    get_stored_pixel_array,
    load_dicom,
    pixel_padding_mask,
    to_hounsfield_units,
)

#: HU below this is treated as air/background for the audit only.
AIR_THRESHOLD_HU = -900.0
#: HU above this is treated as "inside the body" by the crude audit mask.
BODY_THRESHOLD_HU = -300.0
#: A generous ceiling for physically plausible CT values. Anything above it is
#: recorded as a non-anatomical outlier rather than interpreted as tissue.
OUTLIER_CEILING_HU = 4000.0
#: Most outlier coordinates recorded per slice, to bound the CSV cell size.
MAX_RECORDED_OUTLIERS = 16


def slice_normal(orientation: Any) -> np.ndarray | None:
    """Unit normal of the image plane from *ImageOrientationPatient*.

    The attribute holds the direction cosines of the image row axis followed by
    those of the column axis. Their cross product is the slice normal, which is
    the axis along which slices actually advance. Projecting
    *ImagePositionPatient* onto it gives a scalar position that is correct for
    any orientation, not only for acquisitions aligned with the patient z axis.
    """
    if orientation is None or len(orientation) != 6:
        return None
    row_cosines = np.asarray(orientation[:3], dtype=np.float64)
    column_cosines = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row_cosines, column_cosines)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return None
    return normal / norm


def short_hash(value: Any) -> str | None:
    """Deterministic 12-hex-character digest, so UIDs group without publishing."""
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def crude_body_mask(image_hu: np.ndarray) -> np.ndarray:
    """Largest connected above-air region with interior holes filled.

    Audit instrument only. It is intentionally simple: threshold, keep the
    largest component, fill enclosed holes. It is not anatomically validated
    and must not be used to compute reported metrics without a separate
    decision.
    """
    mask = (image_hu > BODY_THRESHOLD_HU).astype(np.uint8)
    if not mask.any():
        return mask.astype(bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask.astype(bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    body = (labels == largest).astype(np.uint8)

    # Flood fill the outside, then anything still unset is an interior hole.
    flooded = body.copy()
    height, width = body.shape
    cv2.floodFill(flooded, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 1)
    return (body | (1 - flooded)).astype(bool)


def fraction(condition: np.ndarray) -> float:
    return float(np.count_nonzero(condition)) / float(condition.size)


def audit_slice(path: Path, subject: ChaosCtSubject, lower: float, upper: float) -> dict[str, Any]:
    """Read one slice and return a flat record of technical facts about it."""
    record: dict[str, Any] = {
        "subject_id": subject.subject_id,
        "source": subject.source,
        "file": path.name,
        "error": None,
    }

    try:
        dataset = load_dicom(path)
        description = describe_ct_dataset(dataset)
        stored = get_stored_pixel_array(dataset)
        image_hu = to_hounsfield_units(stored, dataset)
    except Exception as error:  # noqa: BLE001 - the audit must survive bad files
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    record.update(
        {
            "modality": description["modality"],
            "rows": description["rows"],
            "columns": description["columns"],
            "samples_per_pixel": getattr(dataset, "SamplesPerPixel", None),
            "photometric_interpretation": description["photometric_interpretation"],
            "bits_allocated": getattr(dataset, "BitsAllocated", None),
            "bits_stored": description["bits_stored"],
            "pixel_representation": description["pixel_representation"],
            "rescale_slope": description["rescale_slope"],
            "rescale_intercept": description["rescale_intercept"],
            "has_modality_lut": description["has_modality_lut"],
            "pixel_padding_value": description["pixel_padding_value"],
            "pixel_padding_range_limit": description["pixel_padding_range_limit"],
            "display_window_center": _first(description["display_window_center"]),
            "display_window_width": _first(description["display_window_width"]),
            "series_uid_hash": short_hash(getattr(dataset, "SeriesInstanceUID", None)),
            "study_uid_hash": short_hash(getattr(dataset, "StudyInstanceUID", None)),
            "instance_number": _as_int(getattr(dataset, "InstanceNumber", None)),
            "stored_min": int(stored.min()),
            "stored_max": int(stored.max()),
            "stored_dtype": str(stored.dtype),
        }
    )

    spacing = getattr(dataset, "PixelSpacing", None)
    record["pixel_spacing_row"] = float(spacing[0]) if spacing is not None else None
    record["pixel_spacing_col"] = float(spacing[1]) if spacing is not None else None
    record["slice_thickness"] = _as_float(getattr(dataset, "SliceThickness", None))
    record["spacing_between_slices"] = _as_float(getattr(dataset, "SpacingBetweenSlices", None))

    position = getattr(dataset, "ImagePositionPatient", None)
    record["has_image_position"] = position is not None
    record["position_z"] = float(position[2]) if position is not None else None

    orientation = getattr(dataset, "ImageOrientationPatient", None)
    record["has_image_orientation"] = orientation is not None
    record["orientation"] = (
        ",".join(f"{float(value):g}" for value in orientation) if orientation is not None else None
    )
    normal = slice_normal(orientation)
    if normal is not None:
        record["normal"] = ",".join(f"{value:g}" for value in normal)
        # Axial acquisition means the slice normal is the patient z axis.
        record["is_axial"] = bool(np.allclose(np.abs(normal), (0.0, 0.0, 1.0), atol=1e-6))
        if position is not None:
            projected = float(np.dot(np.asarray(position, dtype=np.float64), normal))
            record["projected_position"] = projected
            record["projected_minus_z"] = projected - float(position[2])
    else:
        record["normal"] = None
        record["is_axial"] = None
        record["projected_position"] = None
        record["projected_minus_z"] = None

    record.update(
        {
            "hu_min": float(image_hu.min()),
            "hu_max": float(image_hu.max()),
            "hu_mean": float(image_hu.mean()),
            "hu_p01": float(np.percentile(image_hu, 1)),
            "hu_p99": float(np.percentile(image_hu, 99)),
            "frac_below_window": fraction(image_hu < lower),
            "frac_in_window": fraction((image_hu >= lower) & (image_hu <= upper)),
            "frac_above_window": fraction(image_hu > upper),
            "frac_air": fraction(image_hu < AIR_THRESHOLD_HU),
        }
    )

    padding = pixel_padding_mask(dataset, stored)
    record["declares_padding"] = padding is not None
    record["frac_padding"] = fraction(padding) if padding is not None else None
    if padding is not None and padding.any():
        record["padding_hu_min"] = float(image_hu[padding].min())
        record["padding_hu_max"] = float(image_hu[padding].max())
        record["padding_below_window"] = bool((image_hu[padding] < lower).all())
    else:
        record["padding_hu_min"] = None
        record["padding_hu_max"] = None
        record["padding_below_window"] = None

    body = crude_body_mask(image_hu)
    record["frac_body"] = fraction(body)
    if body.any():
        inside = image_hu[body]
        record["body_frac_below_window"] = float(np.count_nonzero(inside < lower)) / inside.size
        record["body_frac_in_window"] = (
            float(np.count_nonzero((inside >= lower) & (inside <= upper))) / inside.size
        )
        record["body_frac_above_window"] = float(np.count_nonzero(inside > upper)) / inside.size
    else:
        record["body_frac_below_window"] = None
        record["body_frac_in_window"] = None
        record["body_frac_above_window"] = None

    # Non-anatomical outliers. Recorded, never modified: windowing already
    # bounds their effect, and any cleaning policy is a separate decision.
    outliers = image_hu > OUTLIER_CEILING_HU
    outlier_count = int(np.count_nonzero(outliers))
    record["n_outliers"] = outlier_count
    if outlier_count:
        coordinates = np.argwhere(outliers)[:MAX_RECORDED_OUTLIERS]
        record["outlier_coords"] = ";".join(
            f"{int(row)}x{int(column)}={image_hu[row, column]:.0f}" for row, column in coordinates
        )
        record["outlier_hu_max"] = float(image_hu[outliers].max())
        record["outliers_all_outside_body"] = bool(not body[outliers].any())
    else:
        record["outlier_coords"] = None
        record["outlier_hu_max"] = None
        record["outliers_all_outside_body"] = None

    return record


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_int(value: Any) -> int | None:
    return None if value is None else int(value)


def geometry_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Per-subject slice ordering and spacing facts.

    Positions come from projecting *ImagePositionPatient* onto the slice
    normal, which is the orientation-independent way to order slices. The
    report also states how far that projection differs from simply reading the
    z component, so the simpler rule can be justified or rejected on evidence.
    """
    usable = frame[frame["error"].isna()]
    report: dict[str, Any] = {
        "slices": int(len(usable)),
        "has_image_position_fraction": float(usable["has_image_position"].mean()),
        "has_image_orientation_fraction": float(usable["has_image_orientation"].mean()),
        "has_instance_number_fraction": float(usable["instance_number"].notna().mean()),
    }

    orientations = usable["orientation"].dropna().unique()
    report["distinct_orientations_in_series"] = int(len(orientations))
    report["orientation"] = str(orientations[0]) if len(orientations) == 1 else None
    report["orientation_constant"] = bool(len(orientations) == 1)
    if usable["is_axial"].notna().any():
        report["all_slices_axial"] = bool(usable["is_axial"].dropna().all())

    if usable["projected_position"].notna().all() and len(usable) > 1:
        by_position = usable.sort_values("projected_position")
        positions = by_position["projected_position"].to_numpy()
        differences = np.abs(np.diff(positions))

        report["position_range"] = [float(positions.min()), float(positions.max())]
        report["median_step"] = float(np.median(differences))
        report["max_step"] = float(np.max(differences))
        report["min_step"] = float(np.min(differences))
        report["uniform_spacing"] = bool(
            np.allclose(differences, np.median(differences), atol=1e-3)
        )
        report["duplicate_positions"] = int(len(usable) - usable["projected_position"].nunique())
        report["max_abs_projected_minus_z"] = float(usable["projected_minus_z"].abs().max())

        if usable["instance_number"].notna().all():
            by_instance = usable.sort_values("instance_number")
            report["filename_order_matches_position_order"] = bool(
                list(usable["file"]) == list(by_position["file"])
            )
            report["instance_order_matches_position_order"] = bool(
                list(by_instance["file"]) == list(by_position["file"])
            )
            report["instance_order_matches_reversed_position_order"] = bool(
                list(by_instance["file"]) == list(by_position["file"])[::-1]
            )

    return report


def outlier_pixel_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (subject, pixel coordinate) that ever exceeds the ceiling.

    Answers, exhaustively and reproducibly, where the non-anatomical outliers
    are, whether they sit at the same coordinate on every slice of a subject,
    and whether the crude body mask excludes them.
    """
    columns = [
        "subject_id",
        "source",
        "coordinate",
        "slices_with_outlier_here",
        "subject_slices",
        "present_in_every_slice",
        "min_hu",
        "max_hu",
        "always_outside_body",
    ]

    affected = frame[frame["n_outliers"].fillna(0) > 0].copy()
    if affected.empty:
        return pd.DataFrame(columns=columns)

    affected["entry"] = affected["outlier_coords"].str.split(";")
    exploded = affected.explode("entry")
    exploded["coordinate"] = exploded["entry"].str.split("=").str[0]
    exploded["hu"] = exploded["entry"].str.split("=").str[1].astype(float)

    slices_per_subject = frame.groupby("subject_id")["file"].count()

    table = (
        exploded.groupby(["subject_id", "source", "coordinate"])
        .agg(
            slices_with_outlier_here=("file", "count"),
            min_hu=("hu", "min"),
            max_hu=("hu", "max"),
            always_outside_body=("outliers_all_outside_body", "all"),
        )
        .reset_index()
    )
    table["subject_slices"] = table["subject_id"].map(slices_per_subject)
    table["present_in_every_slice"] = table["slices_with_outlier_here"] == table["subject_slices"]
    return table[columns]


def summarise_column(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = frame[column].dropna()
    if values.empty:
        return {"unique_values": [], "count": 0}
    counts = values.value_counts()
    return {
        "unique_values": [_jsonable(value) for value in counts.index[:10]],
        "value_counts": {str(_jsonable(k)): int(v) for k, v in counts.items()},
        "count": int(values.size),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def render_figures(
    subjects: Iterable[ChaosCtSubject],
    figure_dir: Path,
    center: float,
    width: float,
) -> list[str]:
    """Save one audit panel per selected subject. Local inspection only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ct_restoration.data.preprocessing import window_ct

    ensure_dir(figure_dir)
    written: list[str] = []

    for subject in subjects:
        middle = subject.dicom_paths[len(subject.dicom_paths) // 2]
        dataset = load_dicom(middle)
        image_hu = to_hounsfield_units(get_stored_pixel_array(dataset), dataset)

        display = describe_ct_dataset(dataset)
        display_center = _first(display["display_window_center"])
        display_width = _first(display["display_window_width"])

        panels = [
            ("HU, broad view [-1000, 1000]", window_ct(image_hu, 0, 2000)),
            (f"project window {center:g}/{width:g}", window_ct(image_hu, center, width)),
        ]
        if display_center is not None and display_width is not None and display_width > 0:
            panels.append(
                (
                    f"DICOM display preset {display_center:g}/{display_width:g}",
                    window_ct(image_hu, display_center, display_width),
                )
            )
        panels.append(("crude audit body mask", crude_body_mask(image_hu).astype(np.float32)))

        figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
        for axis, (title, image) in zip(axes, panels, strict=True):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        figure.suptitle(
            f"CHAOS CT subject {subject.subject_id}, slice {middle.name} "
            f"({subject.slice_count} slices)",
            fontsize=10,
        )
        figure.tight_layout()

        out_path = figure_dir / f"chaos_ct_subject_{subject.subject_id}.png"
        figure.savefig(out_path, dpi=110)
        plt.close(figure)
        written.append(str(out_path))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--config", default="baseline.yaml", help="preprocessing config to audit")
    parser.add_argument("--out", default="outputs/audit", help="directory for CSV/JSON outputs")
    parser.add_argument(
        "--figure-dir",
        default="outputs/audit/figures",
        help="directory for local-only audit figures (git-ignored)",
    )
    parser.add_argument("--figure-subjects", type=int, default=4, help="how many panels to render")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()

    settings = load_config(arguments.config)["preprocessing"]
    center = float(settings["window_center"])
    width = float(settings["window_width"])
    lower, upper = center - width / 2, center + width / 2

    print(f"CHAOS root      : {Path(arguments.root).resolve()}")
    print(
        f"audited window  : center {center:g} HU, width {width:g} HU -> [{lower:g}, {upper:g}] HU"
    )

    subjects = find_ct_subjects(arguments.root)
    if not subjects:
        print("No CT subjects found.", file=sys.stderr)
        return 1
    total_slices = sum(subject.slice_count for subject in subjects)
    print(f"subjects        : {len(subjects)}")
    print(f"slices          : {total_slices}")
    print()

    records: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, start=1):
        print(
            f"  [{index:>2}/{len(subjects)}] subject {subject.subject_id:>3}: "
            f"{subject.slice_count:>4} slices",
            flush=True,
        )
        for path in subject.dicom_paths:
            records.append(audit_slice(path, subject, lower, upper))

    frame = pd.DataFrame.from_records(records)
    failures = frame[frame["error"].notna()]
    usable = frame[frame["error"].isna()]

    out_dir = ensure_dir(arguments.out)
    frame.to_csv(out_dir / "chaos_slice_metrics.csv", index=False)

    per_subject = (
        usable.groupby("subject_id")
        .agg(
            source=("source", "first"),
            slices=("file", "count"),
            series=("series_uid_hash", "nunique"),
            rows=("rows", "first"),
            columns=("columns", "first"),
            rescale_slope=("rescale_slope", "first"),
            rescale_intercept=("rescale_intercept", "first"),
            pixel_spacing_row=("pixel_spacing_row", "first"),
            slice_thickness=("slice_thickness", "first"),
            hu_min=("hu_min", "min"),
            hu_max=("hu_max", "max"),
            mean_frac_below_window=("frac_below_window", "mean"),
            mean_frac_in_window=("frac_in_window", "mean"),
            mean_frac_above_window=("frac_above_window", "mean"),
            mean_frac_air=("frac_air", "mean"),
            mean_frac_body=("frac_body", "mean"),
            mean_body_frac_in_window=("body_frac_in_window", "mean"),
            mean_body_frac_above_window=("body_frac_above_window", "mean"),
            declares_padding=("declares_padding", "any"),
            slices_with_outliers=("n_outliers", lambda values: int((values > 0).sum())),
            max_outliers_in_a_slice=("n_outliers", "max"),
        )
        .reset_index()
    )
    per_subject["subject_sort"] = per_subject["subject_id"].astype(int)
    per_subject = per_subject.sort_values("subject_sort").drop(columns="subject_sort")
    per_subject.to_csv(out_dir / "chaos_series_summary.csv", index=False)

    outliers = outlier_pixel_table(frame)
    outliers.to_csv(out_dir / "chaos_outlier_pixels.csv", index=False)

    geometry = {
        subject_id: geometry_report(group) for subject_id, group in frame.groupby("subject_id")
    }

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "CHAOS Train_Sets CT",
        "root": str(Path(arguments.root).resolve()),
        "audited_window": {"center_hu": center, "width_hu": width, "range_hu": [lower, upper]},
        "counts": {
            "subjects": int(len(subjects)),
            "slices_found": int(total_slices),
            "slices_read": int(len(usable)),
            "slices_failed": int(len(failures)),
            "distinct_series": int(usable["series_uid_hash"].nunique()),
            "distinct_studies": int(usable["study_uid_hash"].nunique()),
            "subjects_with_one_series": int(
                (usable.groupby("subject_id")["series_uid_hash"].nunique() == 1).sum()
            ),
            "subjects_by_source_archive": {
                str(source): int(count)
                for source, count in usable.groupby("source")["subject_id"].nunique().items()
            },
            "slices_by_source_archive": {
                str(source): int(count)
                for source, count in usable.groupby("source")["file"].count().items()
            },
        },
        "failures": failures[["subject_id", "file", "error"]].to_dict("records"),
        "metadata_distributions": {
            column: summarise_column(usable, column)
            for column in (
                "modality",
                "rows",
                "columns",
                "samples_per_pixel",
                "photometric_interpretation",
                "bits_allocated",
                "bits_stored",
                "pixel_representation",
                "rescale_slope",
                "rescale_intercept",
                "has_modality_lut",
                "stored_dtype",
                "pixel_padding_value",
                "pixel_padding_range_limit",
                "display_window_center",
                "display_window_width",
                "pixel_spacing_row",
                "slice_thickness",
                "spacing_between_slices",
            )
        },
        "hounsfield_units": {
            "min_over_all_slices": float(usable["hu_min"].min()),
            "max_over_all_slices": float(usable["hu_max"].max()),
            "median_slice_min": float(usable["hu_min"].median()),
            "median_slice_max": float(usable["hu_max"].median()),
            "slices_with_negative_hu": int((usable["hu_min"] < 0).sum()),
            "slices_with_hu_below_minus_2000": int((usable["hu_min"] < -2000).sum()),
        },
        "padding": {
            "slices_declaring_pixel_padding_value": int(usable["declares_padding"].sum()),
            "slices_declaring_range_limit": int(usable["pixel_padding_range_limit"].notna().sum()),
            "mean_padding_fraction": _optional_mean(usable["frac_padding"]),
            "padding_always_below_window": _optional_all(usable["padding_below_window"]),
        },
        "background": {
            "mean_air_fraction": float(usable["frac_air"].mean()),
            "median_air_fraction": float(usable["frac_air"].median()),
            "max_air_fraction": float(usable["frac_air"].max()),
            "mean_body_fraction": float(usable["frac_body"].mean()),
            "median_body_fraction": float(usable["frac_body"].median()),
        },
        "window_clipping": {
            "whole_frame": {
                "mean_fraction_below": float(usable["frac_below_window"].mean()),
                "mean_fraction_inside": float(usable["frac_in_window"].mean()),
                "mean_fraction_above": float(usable["frac_above_window"].mean()),
            },
            "inside_crude_body_mask": {
                "mean_fraction_below": _optional_mean(usable["body_frac_below_window"]),
                "mean_fraction_inside": _optional_mean(usable["body_frac_in_window"]),
                "mean_fraction_above": _optional_mean(usable["body_frac_above_window"]),
            },
        },
        "slice_geometry": {
            "slices_with_image_orientation": int(usable["has_image_orientation"].sum()),
            "slices_with_image_position": int(usable["has_image_position"].sum()),
            "distinct_orientations_in_cohort": summarise_column(usable, "orientation"),
            "all_slices_axial": bool(usable["is_axial"].dropna().all()),
            "max_abs_projected_position_minus_z": float(usable["projected_minus_z"].abs().max()),
            "note": (
                "Positions are ImagePositionPatient projected onto the slice normal "
                "derived from ImageOrientationPatient. max_abs_projected_position_minus_z "
                "shows how far that differs from using the z component alone."
            ),
        },
        "non_anatomical_outliers": {
            "ceiling_hu": OUTLIER_CEILING_HU,
            "slices_with_outliers": int((usable["n_outliers"] > 0).sum()),
            "subjects_with_outliers": sorted(
                usable.loc[usable["n_outliers"] > 0, "subject_id"].unique().tolist(),
                key=int,
            ),
            "max_outliers_in_any_slice": int(usable["n_outliers"].max()),
            "max_outlier_hu": _optional_max(usable["outlier_hu_max"]),
            "always_outside_crude_body_mask": _optional_all(usable["outliers_all_outside_body"]),
            "note": (
                "Persistent non-anatomical outlier pixels of unknown provenance. They "
                "are recorded, never modified. Per-coordinate detail is in "
                "chaos_outlier_pixels.csv."
            ),
        },
        "geometry_by_subject": geometry,
    }

    with (out_dir / "chaos_metadata_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)

    print()
    print(f"read {len(usable)} slices, {len(failures)} failures")
    print(f"wrote {out_dir / 'chaos_series_summary.csv'}")
    print(f"wrote {out_dir / 'chaos_metadata_summary.json'}")
    print(f"wrote {out_dir / 'chaos_slice_metrics.csv'}")
    print(f"wrote {out_dir / 'chaos_outlier_pixels.csv'} ({len(outliers)} outlier coordinates)")

    if not arguments.no_figures:
        chosen = subjects[:: max(1, len(subjects) // arguments.figure_subjects)][
            : arguments.figure_subjects
        ]
        written = render_figures(chosen, Path(arguments.figure_dir), center, width)
        print(f"wrote {len(written)} audit figures to {arguments.figure_dir}")

    return 0


def _optional_mean(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.mean()) if not values.empty else None


def _optional_max(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.max()) if not values.empty else None


def _optional_all(series: pd.Series) -> bool | None:
    values = series.dropna()
    return bool(values.all()) if not values.empty else None


if __name__ == "__main__":
    raise SystemExit(main())
