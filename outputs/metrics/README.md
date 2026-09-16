# Metric outputs

Tracked image-quality results, produced by the evaluation commands in
[`scripts/`](../../scripts/) using the frozen policy in
[`configs/evaluation.yaml`](../../configs/evaluation.yaml).

| File | Contents |
| --- | --- |
| `degraded_baseline_validation_slices.csv` | one row per validation slice: full-frame and body-region MAE, MSE, PSNR, SSIM, plus mask coverage |
| `degraded_baseline_validation_patients.csv` | one row per validation patient, the mean of that patient's slices |
| `degraded_baseline_validation_summary.json` | the PRIMARY patient-weighted split result, a secondary slice-weighted figure, and the configs that produced them |

## What is and is not in them

**No image pixels, no DICOM UIDs, no patient-identifying fields, no
timestamps.** Slices are identified by the same `relative_dicom_path` sample
key the frozen manifest uses. Regenerating from an unchanged experiment
definition rewrites every file byte for byte.

## Reading them

MAE and MSE are lower-is-better; PSNR, in decibels, and SSIM are
higher-is-better. The **primary** figure for a split is the patient-weighted
one: per-slice values are averaged within a patient, then patients are averaged
with equal weight, because slices from one patient are not independent
observations. The slice-weighted value is retained only as a secondary
descriptive diagnostic and is never the headline number.

These are relative figures on a synthetic benchmark. No value here indicates
clinical adequacy, and a PSNR difference is a difference in decibels, never a
percentage.

## Terms

These are CHAOS-derived artifacts. Use and redistribution of CHAOS data and of
artifacts derived from it remain subject to the applicable CHAOS dataset terms,
recorded with the dataset provenance in [`data/README.md`](../../data/README.md).
Those dataset terms are separate from the license covering this repository's
source code.
