# Audit outputs

These files are technical summaries derived from the CHAOS CT dataset,
produced by [`scripts/audit_chaos.py`](../../scripts/audit_chaos.py), except
`degradation_train_summary.json`, from
[`scripts/audit_degradation.py`](../../scripts/audit_degradation.py),
`evaluation_body_mask_train_summary.json`, from
[`scripts/audit_body_mask.py`](../../scripts/audit_body_mask.py), and
`dataset_dataloader_summary.json`, from
[`scripts/audit_dataset.py`](../../scripts/audit_dataset.py). Re-running those
scripts over a fresh download regenerates all of them.

| File | Contents |
| --- | --- |
| `chaos_slice_metrics.csv` | one row per CT slice: encoding attributes, HU calibration, geometry, HU statistics, window-clipping fractions, background and outlier counts |
| `chaos_series_summary.csv` | one row per subject, aggregated from the per-slice table |
| `chaos_metadata_summary.json` | dataset-level distributions, counts and per-subject geometry findings |
| `chaos_outlier_pixels.csv` | one row per (subject, pixel coordinate) exceeding the audit's 4000 HU ceiling |
| `chaos_split_summary.json` | counts, balance and checksum for the frozen patient split; carries no timestamp, so regenerating an unchanged split rewrites it byte for byte |
| `degradation_train_summary.json` | technical diagnostics of the frozen degradation, computed on TRAINING slices only; no timestamp, so regenerating an unchanged definition rewrites it byte for byte |
| `evaluation_body_mask_train_summary.json` | diagnostics of the frozen evaluation body mask, computed on TRAINING slices only; no timestamp, so regenerating an unchanged definition rewrites it byte for byte |
| `dataset_dataloader_summary.json` | contract checks, sampling counts and sequence hashes for the Dataset / DataLoader layer, on TRAIN and VALIDATION only; no timestamp, so regenerating an unchanged definition rewrites it byte for byte |
| `figures/` | rendered inspection panels, git-ignored - including `figures/cnn/` and `figures/unet/`, the post-selection visual QC of the two learned checkpoints, rendered from TRAINING slices only |

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

`figures/cnn/` and `figures/unet/` are the same discipline applied to the
learned methods. Neither `scripts/qc_cnn.py` nor `scripts/qc_unet.py` has a
`--split` option: both render **training** slices only, and both run *after*
the checkpoints have already been selected numerically and the canonical
metrics written. The U-Net panels put both models side by side on the same
slices, so the numeric comparison can be sanity-checked visually without
looking at held-out data. No validation, test or stress image has been looked
at.

## `dataset_dataloader_summary.json` in particular

It is the evidence behind the Dataset / DataLoader claims in the repository
README, and it carries:

* **no image pixels** - only counts, flags and SHA-256 digests of *sample-key
  sequences*, never of image content;
* **no test or stress anything** - no metric, no count, no sample key. The
  audit reads train and validation, and the helper it builds datasets with
  refuses the sealed splits. The file records `test_images_read: 0` and
  `stress_images_read: 0`;
* **dataset contract checks** - per split, every slice opened once and checked
  for tensor shape, dtype, finiteness and [0, 1] range. A smaller set of
  deterministic probes per split (8 at present, recorded as `probes`) also
  covers repeated-access equality, independence from the global NumPy and
  PyTorch RNGs, and byte-equality against the frozen preprocessing and
  degradation functions called independently. The probe counts are in the
  file: read the scan counts and the probe counts as the different scopes
  they are;
* **patient-sampling counts** - realized draws per patient for three epochs of
  the patient-balanced sampler at the audit seed, with the rotation of the
  extra-quota patients over eight epochs;
* **deterministic sequence hashes** - of the canonical per-split order, of
  each sampled epoch, and of what each DataLoader actually yielded, so
  reproducibility and batch-size invariance are checkable rather than
  asserted.

It contains no timestamp and no absolute filesystem path, so two runs over an
unchanged definition produce byte-identical files.

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
