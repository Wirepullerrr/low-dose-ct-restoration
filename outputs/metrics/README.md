# Metric outputs

Tracked image-quality results, produced by the evaluation commands in
[`scripts/`](../../scripts/) using the frozen policy in
[`configs/evaluation.yaml`](../../configs/evaluation.yaml).

| File | Contents |
| --- | --- |
| `degraded_baseline_validation_slices.csv` | one row per validation slice: full-frame and body-region MAE, MSE, PSNR, SSIM, plus mask coverage |
| `degraded_baseline_validation_patients.csv` | one row per validation patient, the mean of that patient's slices |
| `degraded_baseline_validation_summary.json` | the PRIMARY patient-weighted split result, a secondary slice-weighted figure, and the configs that produced them |
| `clahe_validation_search.csv` | one row per CLAHE candidate in the predeclared 12-point grid: patient-weighted metrics, deltas against the degraded baseline, selection rank, and which one won |
| `clahe_validation_slices.csv` | one row per validation slice for the frozen CLAHE configuration, same schema and same canonical row order as the baseline table |
| `clahe_validation_patients.csv` | one row per validation patient for frozen CLAHE |
| `clahe_vs_degraded_baseline_validation_patient_deltas.csv` | the six paired per-patient deltas, with both methods' values beside each one |
| `clahe_validation_summary.json` | frozen CLAHE's patient-weighted result, the paired comparison, technical diagnostics, and an explicit verdict on the predeclared metric |

## The CLAHE files in particular

**The search is validation-only.** `scripts/tune_clahe.py` has no `--split`
option: the 12-candidate sweep reads validation and nothing else. Test and
stress stay sealed.

**The selected configuration is frozen** in
[`configs/clahe.yaml`](../../configs/clahe.yaml), written by the sweep rather
than hand-transcribed. The tuning command refuses to overwrite it without an
explicit `--overwrite`, so CLAHE cannot be quietly retuned after later results
exist.

**Candidate ranking is patient-weighted.** A candidate's score is the mean
over six patients, each weighted equally, not a pool of 885 correlated slices.

**Deltas are paired against the exact Milestone 5 degraded baseline** — the
same patients, the same 885 sample keys in the same order, the same masks and
the same metric code. The sign convention is uniform: `delta = CLAHE -
baseline`, so positive is an improvement for PSNR and SSIM and negative is an
improvement for MAE and MSE.

## What is and is not in them

**No image pixels, no DICOM UIDs, no patient-identifying fields, no
timestamps.** Slices are identified by the same `relative_dicom_path` sample
key the frozen manifest uses. Regenerating from an unchanged experiment
definition rewrites every file byte for byte.

## Reading them

All of these are **validation development results**, not final benchmark
results. The final comparison happens on the held-out test split once every
method decision is frozen.

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
