"""Tests for the pure helpers inside the CHAOS audit script.

The audit itself needs the real download, so it is a command rather than a
test. These cover its reusable logic on synthetic arrays and frames, so a
mistake in the body mask, the slice geometry or the outlier accounting is
caught without any dataset present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_chaos.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_chaos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_chaos"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()

AXIAL = "1,0,0,0,1,0"  # rows along +x, columns along +y, so the normal is +z


def phantom(size: int = 128) -> np.ndarray:
    """Air background, soft-tissue ellipse, an internal air pocket, a bone spot."""
    rows, columns = np.mgrid[0:size, 0:size]
    image = np.full((size, size), -1000.0)
    body = ((rows - size / 2) ** 2 / 40**2 + (columns - size / 2) ** 2 / 50**2) <= 1
    image[body] = 40.0
    image[(rows - 55) ** 2 + (columns - 55) ** 2 <= 100] = -850.0
    image[(rows - 80) ** 2 + (columns - 80) ** 2 <= 25] = 900.0
    return image


# --------------------------------------------------------------------------
# Crude audit body mask
# --------------------------------------------------------------------------


def test_body_mask_covers_the_body_and_excludes_the_background() -> None:
    image = phantom()
    rows, columns = np.mgrid[0:128, 0:128]
    body = ((rows - 64) ** 2 / 40**2 + (columns - 64) ** 2 / 50**2) <= 1

    mask = audit.crude_body_mask(image)

    assert mask[body].all()
    assert not mask[0, 0]


def test_body_mask_fills_internal_air_pockets() -> None:
    image = phantom()
    rows, columns = np.mgrid[0:128, 0:128]
    pocket = (rows - 55) ** 2 + (columns - 55) ** 2 <= 100

    assert audit.crude_body_mask(image)[pocket].all()


def test_body_mask_on_an_all_air_slice_is_empty() -> None:
    assert not audit.crude_body_mask(np.full((32, 32), -1000.0)).any()


def test_body_mask_excludes_an_isolated_corner_outlier() -> None:
    """The outliers seen in the real data must not be counted as anatomy."""
    image = phantom()
    image[0, 1] = 49944.0

    assert not audit.crude_body_mask(image)[0, 1]


def test_fraction_counts_the_share_of_true_pixels() -> None:
    condition = np.zeros((10, 10), dtype=bool)
    condition[:2] = True

    assert audit.fraction(condition) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# UID hashing
# --------------------------------------------------------------------------


def test_uid_hash_is_deterministic_and_short() -> None:
    digest = audit.short_hash("1.2.840.113619.2.55.3")

    assert digest == audit.short_hash("1.2.840.113619.2.55.3")
    assert len(digest) == 12
    assert digest != audit.short_hash("1.2.840.113619.2.55.4")


def test_uid_hash_passes_through_missing_values() -> None:
    assert audit.short_hash(None) is None


# --------------------------------------------------------------------------
# Slice normal from direction cosines
# --------------------------------------------------------------------------


def test_slice_normal_of_a_standard_axial_acquisition_is_the_z_axis() -> None:
    normal = audit.slice_normal([1, 0, 0, 0, 1, 0])

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-12)


def test_slice_normal_of_a_coronal_acquisition_is_not_the_z_axis() -> None:
    normal = audit.slice_normal([1, 0, 0, 0, 0, -1])

    np.testing.assert_allclose(np.abs(normal), [0.0, 1.0, 0.0], atol=1e-12)


def test_slice_normal_is_a_unit_vector_for_a_tilted_plane() -> None:
    normal = audit.slice_normal([1, 0, 0, 0, 0.7071067811865476, 0.7071067811865476])

    assert float(np.linalg.norm(normal)) == pytest.approx(1.0)


def test_projection_differs_from_the_z_component_on_a_tilted_plane() -> None:
    """This is why the audit projects instead of reading ImagePositionPatient[2]."""
    normal = audit.slice_normal([1, 0, 0, 0, 0.7071067811865476, 0.7071067811865476])
    position = np.array([0.0, 10.0, 10.0])

    projected = float(np.dot(position, normal))

    assert projected != pytest.approx(float(position[2]))


def test_projection_equals_the_z_component_on_an_axial_plane() -> None:
    normal = audit.slice_normal([1, 0, 0, 0, 1, 0])
    position = np.array([-250.0, -250.0, 37.5])

    assert float(np.dot(position, normal)) == pytest.approx(float(position[2]))


@pytest.mark.parametrize("orientation", [None, [1, 0, 0], [0, 0, 0, 0, 0, 0]])
def test_slice_normal_returns_none_for_unusable_orientation(orientation) -> None:
    assert audit.slice_normal(orientation) is None


# --------------------------------------------------------------------------
# Slice ordering / geometry report
# --------------------------------------------------------------------------


def _geometry_frame(positions, instance_numbers, files, orientation=AXIAL) -> pd.DataFrame:
    orientations = orientation if isinstance(orientation, list) else [orientation] * len(files)
    normals = [
        audit.slice_normal([float(value) for value in item.split(",")]) if item else None
        for item in orientations
    ]
    # Every orientation used here is axis-aligned, so the projection is the z component.
    projected = [None if position is None else float(position) for position in positions]
    return pd.DataFrame(
        {
            "error": [None] * len(files),
            "file": files,
            "position_z": positions,
            "projected_position": projected,
            "projected_minus_z": [None if position is None else 0.0 for position in positions],
            "instance_number": instance_numbers,
            "has_image_position": [position is not None for position in positions],
            "has_image_orientation": [item is not None for item in orientations],
            "orientation": orientations,
            "is_axial": [
                None if normal is None else bool(abs(float(normal[2])) == 1.0) for normal in normals
            ],
        }
    )


def test_geometry_report_detects_uniform_spacing() -> None:
    frame = _geometry_frame([0.0, 3.0, 6.0, 9.0], [1, 2, 3, 4], ["a", "b", "c", "d"])

    report = audit.geometry_report(frame)

    assert report["slices"] == 4
    assert report["median_step"] == pytest.approx(3.0)
    assert report["uniform_spacing"] is True
    assert report["duplicate_positions"] == 0
    assert report["orientation_constant"] is True
    assert report["all_slices_axial"] is True


def test_geometry_report_flags_non_uniform_spacing() -> None:
    frame = _geometry_frame([0.0, 3.0, 20.0], [1, 2, 3], ["a", "b", "c"])

    assert audit.geometry_report(frame)["uniform_spacing"] is False


def test_geometry_report_detects_instance_order_reversed_against_position() -> None:
    """A descending-position series is normal in CT and must not read as disorder."""
    frame = _geometry_frame([9.0, 6.0, 3.0, 0.0], [1, 2, 3, 4], ["a", "b", "c", "d"])

    report = audit.geometry_report(frame)

    assert report["instance_order_matches_position_order"] is False
    assert report["instance_order_matches_reversed_position_order"] is True


def test_geometry_report_detects_filenames_disagreeing_with_geometry() -> None:
    frame = _geometry_frame([6.0, 0.0, 3.0], [3, 1, 2], ["a", "b", "c"])

    report = audit.geometry_report(frame)

    assert report["filename_order_matches_position_order"] is False
    assert report["instance_order_matches_position_order"] is True


def test_geometry_report_counts_duplicate_positions() -> None:
    frame = _geometry_frame([0.0, 3.0, 3.0, 6.0], [1, 2, 3, 4], ["a", "b", "c", "d"])

    assert audit.geometry_report(frame)["duplicate_positions"] == 1


def test_geometry_report_reports_missing_geometry() -> None:
    frame = _geometry_frame([None, None], [1, 2], ["a", "b"])

    report = audit.geometry_report(frame)

    assert report["has_image_position_fraction"] == 0.0
    assert "median_step" not in report


def test_geometry_report_flags_orientation_changing_within_a_series() -> None:
    frame = _geometry_frame([0.0, 3.0], [1, 2], ["a", "b"], orientation=[AXIAL, "1,0,0,0,0,-1"])

    report = audit.geometry_report(frame)

    assert report["orientation_constant"] is False
    assert report["distinct_orientations_in_series"] == 2
    assert report["all_slices_axial"] is False


# --------------------------------------------------------------------------
# Non-anatomical outlier pixels
# --------------------------------------------------------------------------


def _outlier_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": subject,
                "source": "Train_Sets",
                "file": file,
                "n_outliers": count,
                "outlier_coords": coordinates,
                "outliers_all_outside_body": outside,
            }
            for subject, file, count, coordinates, outside in rows
        ]
    )


def test_outlier_table_is_empty_when_no_slice_has_outliers() -> None:
    frame = _outlier_frame([("1", "a.dcm", 0, None, None), ("1", "b.dcm", 0, None, None)])

    assert audit.outlier_pixel_table(frame).empty


def test_outlier_table_reports_a_coordinate_present_in_every_slice() -> None:
    frame = _outlier_frame(
        [
            ("10", "a.dcm", 1, "0x1=49944", True),
            ("10", "b.dcm", 1, "0x1=49944", True),
            ("10", "c.dcm", 1, "0x1=49944", True),
        ]
    )

    table = audit.outlier_pixel_table(frame)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["coordinate"] == "0x1"
    assert row["slices_with_outlier_here"] == 3
    assert row["subject_slices"] == 3
    assert bool(row["present_in_every_slice"]) is True
    assert row["max_hu"] == pytest.approx(49944.0)
    assert bool(row["always_outside_body"]) is True


def test_outlier_table_separates_coordinates_and_subjects() -> None:
    frame = _outlier_frame(
        [
            ("10", "a.dcm", 2, "0x0=16658;0x1=49944", True),
            ("10", "b.dcm", 2, "0x0=16600;0x1=49944", True),
            ("14", "a.dcm", 1, "0x1=40680", True),
        ]
    )

    table = audit.outlier_pixel_table(frame)

    assert len(table) == 3
    subject_10 = table[table.subject_id == "10"]
    assert set(subject_10.coordinate) == {"0x0", "0x1"}
    assert subject_10[subject_10.coordinate == "0x0"].iloc[0]["min_hu"] == pytest.approx(16600.0)
    assert table[table.subject_id == "14"].iloc[0]["max_hu"] == pytest.approx(40680.0)


def test_outlier_table_marks_a_coordinate_missing_from_some_slices() -> None:
    frame = _outlier_frame([("3", "a.dcm", 1, "5x5=9000", False), ("3", "b.dcm", 0, None, None)])

    table = audit.outlier_pixel_table(frame)

    assert bool(table.iloc[0]["present_in_every_slice"]) is False
    assert bool(table.iloc[0]["always_outside_body"]) is False
