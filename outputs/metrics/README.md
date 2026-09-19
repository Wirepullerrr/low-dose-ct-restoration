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
| `cnn_validation_slices.csv` | one row per validation slice for the selected CNN checkpoint, same schema and same canonical row order as the baseline table |
| `cnn_validation_patients.csv` | one row per validation patient for the selected CNN checkpoint |
| `cnn_vs_degraded_baseline_validation_patient_deltas.csv` | the six paired per-patient deltas against no restoration, with both methods' values beside each one |
| `cnn_vs_clahe_validation_patient_deltas.csv` | the same six patients against frozen CLAHE; descriptive context, not the bar the method has to clear |
| `cnn_validation_summary.json` | the selected CNN's patient-weighted result, both paired comparisons, clamp and correction diagnostics, the checkpoint-provenance and sample-alignment verification records, and an explicit verdict on all eight metrics |

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

## The CNN files in particular

**One seed.** These are a single training run. The number is what that run
produced; it is not evidence that the architecture reaches it reliably.
Multi-seed stability is a later milestone.

**The checkpoint was selected on one predeclared metric** - lowest
patient-weighted validation full-frame MAE - declared in
[`configs/cnn.yaml`](../../configs/cnn.yaml) before training and identified in
the summary by that config's SHA-256. Every other metric in these files was
computed once, afterwards, for the already-selected checkpoint.

**Scored through the same harness as every other method.** The CNN adapter
lives in `src/ct_restoration/models/adapter.py` and plugs into the existing
`benchmark.py` run harness, which was not modified for it. The baseline and
CLAHE artifacts are byte-identical to the ones produced before the CNN
existed.

**Two hard gates run before any canonical artifact is written.** Neither is
advisory: each one refuses the evaluation and writes nothing.

*Checkpoint provenance*, verified **before a single validation image is
opened**. Fifteen conditions, each checked against something computed
independently of the checkpoint - the frozen config's bytes re-hashed, the
checkpoint file's bytes re-hashed, and the tracked training history re-run
through the predeclared selection rule to re-derive the selected epoch rather
than trusting the one the file claims. A checkpoint that cannot be shown to
be the frozen run's is not scored.

*Sample alignment*, verified **before anything is written**. 885 rows, 885
unique sample keys, 0 duplicates, and 0 missing, 0 extra and 0 order
mismatches against both the baseline and the CLAHE slice tables. A paired
delta is only paired if both sides ran on the same slice in the same
position; reordered keys pass every set-based check and are still wrong row
by row, so order is checked explicitly.

Both verification records are written into the summary, so a reader can see
what was checked rather than take it on trust.

**Correction diagnostics name two different quantities.** The *predicted
correction* is measured on the raw, unclamped output (`raw - degraded`); the
*post-clamp change* is what survived into the scored image
(`clamp(raw, 0, 1) - degraded`). They differ wherever the raw output leaves
[0, 1]. The benchmark metrics themselves always use the clamped restoration.

**The weights are not here.** `outputs/checkpoints/` is git-ignored; the
checkpoint's SHA-256 is recorded in
[`outputs/runs/`](../runs/README.md) instead.

## What is and is not in them

**No image pixels, no DICOM UIDs, no patient-identifying fields, no
timestamps.** Slices are identified by the same `relative_dicom_path` sample
key the frozen manifest uses. Regenerating from an unchanged experiment
definition rewrites every file byte for byte.

## Reading them

All of these are **validation development results**, not final benchmark
results, and the CNN's are additionally a **single-seed** result. The final
comparison happens on the held-out test split once every method decision is
frozen.

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
