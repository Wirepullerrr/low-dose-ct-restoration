# Audit outputs

These files are technical summaries derived from the CHAOS CT dataset,
produced by [`scripts/audit_chaos.py`](../../scripts/audit_chaos.py), except
`degradation_train_summary.json`, from
[`scripts/audit_degradation.py`](../../scripts/audit_degradation.py), and
`evaluation_body_mask_train_summary.json`, from
[`scripts/audit_body_mask.py`](../../scripts/audit_body_mask.py). Re-running
those scripts over a fresh download regenerates all of them.

| File | Contents |
| --- | --- |
| `chaos_slice_metrics.csv` | one row per CT slice: encoding attributes, HU calibration, geometry, HU statistics, window-clipping fractions, background and outlier counts |
| `chaos_series_summary.csv` | one row per subject, aggregated from the per-slice table |
| `chaos_metadata_summary.json` | dataset-level distributions, counts and per-subject geometry findings |
| `chaos_outlier_pixels.csv` | one row per (subject, pixel coordinate) exceeding the audit's 4000 HU ceiling |
| `chaos_split_summary.json` | counts, balance and checksum for the frozen patient split; carries no timestamp, so regenerating an unchanged split rewrites it byte for byte |
| `degradation_train_summary.json` | technical diagnostics of the frozen degradation, computed on TRAINING slices only; no timestamp, so regenerating an unchanged definition rewrites it byte for byte |
| `evaluation_body_mask_train_summary.json` | diagnostics of the frozen evaluation body mask, computed on TRAINING slices only; no timestamp, so regenerating an unchanged definition rewrites it byte for byte |
| `figures/` | rendered inspection panels, git-ignored |

## What is and is not in them

They contain **measurements and counts, not image pixels**. No DICOM file, no
pixel array, and no rendered image is stored in the tracked files here. The
`figures/` subdirectory does hold rendered images and is excluded from Git.

**Patient-identifying fields are not exported.** Subjects are identified by the
CHAOS folder name, which is the dataset's own anonymized case label. No DICOM
patient attribute is read for identity. Series and study instance UIDs appear
only as short deterministic hashes, which keeps them usable for grouping and
counting without reproducing the identifiers themselves.

`degradation_train_summary.json` and
`evaluation_body_mask_train_summary.json` report only training-split
diagnostics.
Validation, test and stress image content was not inspected while the
degradation was being established, and the file records that explicitly.

The per-slice table is kept deliberately, despite its size, because it is the
evidence behind every measured claim in the repository README. Keeping it means
those numbers can be re-derived without downloading and re-reading the imaging
data.

## Terms

These are CHAOS-derived artifacts. Use and redistribution of CHAOS data and of
artifacts derived from it remain subject to the applicable CHAOS dataset terms,
which are recorded with the dataset provenance in
[`data/README.md`](../../data/README.md).

Those dataset terms are conceptually separate from the license covering this
repository's source code. Nothing here states or implies terms beyond what the
dataset's own documented license says.
