# Metric outputs

Tracked image-quality results, produced by the evaluation commands in
[`scripts/`](../../scripts/) using the frozen policy in
[`configs/evaluation.yaml`](../../configs/evaluation.yaml). The table below
lists the validation files; the one-shot held-out test tables are in
`holdout/test/`, described [at the end](#the-held-out-test-tables).

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
| `unet_validation_slices.csv` | one row per validation slice for the selected U-Net checkpoint, same schema and same canonical row order as every other method's |
| `unet_validation_patients.csv` | one row per validation patient for the selected U-Net checkpoint |
| `unet_vs_degraded_baseline_validation_patient_deltas.csv` | the six paired per-patient deltas against no restoration |
| `unet_vs_clahe_validation_patient_deltas.csv` | the same six patients against frozen CLAHE; descriptive context |
| `unet_vs_cnn_validation_patient_deltas.csv` | **the architecture comparison**: U-Net minus residual CNN, per patient, both trained under an identical policy |
| `unet_validation_summary.json` | the selected U-Net's patient-weighted result, all three paired comparisons, the explicit CNN-versus-U-Net verdict per metric, clamp and correction diagnostics, and both verification records |

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

## The learned-method files in particular

Everything in this section applies to both `cnn_*` and `unet_*`.

**One seed each, in this directory.** Each file here is a single training
run at seed 2026. The number is what that run produced; on its own it is not
evidence that the architecture reaches it reliably, so
`unet_vs_cnn_validation_patient_deltas.csv` should be read as "this U-Net run
scored slightly better than this CNN run", not as an architecture ranking.

The run-to-run spread is no longer unmeasured. Milestone 10 trained five
seeds per architecture and aggregated them in
`multiseed/multiseed_summary.json`; the architecture comparison belongs
there, across all five seeds, rather than in the single pair of files here.

**The CNN and U-Net were trained under an identical policy** - same data,
same corruption, same sampler, same loss, same optimizer, same seed, same
epochs, same batch size, same checkpoint criterion - so the intended
difference between them is the architecture. Neither model's recipe was ever
hyperparameter-tuned - the CNN's values were one predeclared development
configuration - so the accurate limitation is that the U-Net inherits the CNN
benchmark's predeclared training recipe rather than receiving
architecture-specific tuning.

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
mismatches against every earlier method's slice table - the baseline and
CLAHE for the CNN, and the baseline, CLAHE **and** the CNN for the U-Net. A
paired delta is only paired if both sides ran on the same slice in the same
position; reordered keys pass every set-based check and are still wrong row
by row, so order is checked explicitly. All four methods scored the same 885
slices in the same order.

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

Two exceptions, both in `holdout/test/`. Its opening record, log and
execution receipt carry the opening and completion timestamps, by design.
And its tables are never regenerated, because doing so would reopen the
test split.

## Reading them

Everything outside `holdout/` is a **validation development result**, not
a final benchmark result, and the learned tables in this directory - the CNN's and the
U-Net's - are **single-seed** results: one training seed each, which shows
what those runs did. How much the figure of either architecture varied
across five training seeds is described in `multiseed/`, not here; five
seeds describe that spread, they do not bound it. The final comparison was
made once, on the held-out test split, after every method decision was
frozen; its tables are in `holdout/test/`.

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

## Where the additional statistical seeds wrote

The tables in this directory are the canonical single-seed results. Each
further seed wrote into its own directory instead, and seeds 2027 through
2030 are now present:

```
outputs/metrics/multiseed/seed<SEED>/
    cnn_validation_slices.csv       unet_validation_slices.csv
    cnn_validation_patients.csv     unet_validation_patients.csv
    cnn_validation_summary.json     unet_validation_summary.json
    cnn_vs_*_patient_deltas.csv     unet_vs_*_patient_deltas.csv
```

so no later seed can overwrite a number reported for Milestone 8 or 9. The
eight new runs confirmed this in practice: every file in this directory is
byte-identical to what it was before they ran.

The aggregate across all five seeds is
`multiseed/multiseed_summary.json`, written by
[`scripts/summarize_multiseed.py`](../../scripts/summarize_multiseed.py). It
records the SHA-256 of the plan it was produced under, every seed's value
for all eight metrics, per-architecture means with sample standard
deviations (ddof = 1), and the per-seed paired deltas in both their raw and
their orientation-corrected form. The summarizer refuses to write it unless
all ten planned runs are present and no unplanned seed exists on disk, so a
partial or quietly extended experiment cannot be summarized at all.

Two directories, because two kinds of reference behave differently:

* `--output-dir` is where *this seed's* learned artifacts are written.
* `--reference-dir` holds the frozen non-learned references - the degraded
  baseline and CLAHE. Those are measured once, not per seed, so a seed reads
  them from here and no copy is made into each seed directory.

The CNN is the exception among comparisons: it is a *learned, per-seed*
result, so when the U-Net is scored it reads the CNN from its **own**
`--output-dir`, not from the reference directory. Pairing a seed-2027 U-Net
against the seed-2026 CNN would compare two different experiments while
looking exactly like a valid delta.

With both options left at their defaults the layout is the canonical one and
nothing changes.

## The held-out test tables

`holdout/test/` holds the one-shot Milestone 11 measurement on the six
held-out test patients (941 slices). It was written once, on 2026-09-25, by
[`scripts/run_holdout_test.py`](../../scripts/run_holdout_test.py), under the
protocol frozen in
[`configs/holdout/test_plan.yaml`](../../configs/holdout/test_plan.yaml)
before the test split was opened, and it was audited independently
afterwards. The runner refuses to write it again, and it is not to be
regenerated: the test split has been read once and is now spent.

The layout is exactly the one frozen in the plan, with no file missing and
none added:

```
holdout/test/
  degraded_test_{slices,patients}.csv, degraded_test_summary.json
  clahe_test_{slices,patients}.csv, clahe_test_summary.json
  clahe_vs_degraded_test_patient_deltas.csv
  deterministic_methods.csv    degraded and CLAHE, measured once, no seed spread
  seed_level_metrics.csv       each learned architecture at each of five seeds
  paired_seed_deltas.csv       U-Net - CNN by seed, raw and oriented
  learned_vs_degraded.csv      every learned seed against the one degraded baseline
  learned_vs_clahe.csv         every learned seed against the one CLAHE result
  validation_to_test.csv       secondary, descriptive only
  holdout_test_summary.json    every predeclared comparison
  opening_record.json          written before the first test file is opened
  execution_log.txt            progress lines only; no metric is printed
  execution_receipt.json       what ran and what was read; no metric value
  seed2026/ ... seed2030/
    {cnn,unet}_test_{slices,patients}.csv, {cnn,unet}_test_summary.json
    {cnn,unet}_vs_{degraded,clahe}_test_patient_deltas.csv
    unet_vs_cnn_test_patient_deltas.csv
```

The aggregate files, all patient-weighted and all descriptive:

* `deterministic_methods.csv` - no restoration and CLAHE, measured once each,
  with the CLAHE-minus-degraded change per metric. No seed spread exists for
  them.
* `seed_level_metrics.csv` - both learned architectures at each of the five
  seeds, all eight metrics.
* `paired_seed_deltas.csv` - U-Net minus CNN at each seed, raw and oriented
  so that positive means the U-Net did better.
* `learned_vs_degraded.csv`, `learned_vs_clahe.csv` - every learned seed
  against the one deterministic reference, raw and oriented.
* `validation_to_test.csv` - held-out minus validation per method and
  metric. Secondary and descriptive only; not a generalization-error
  estimate.
* `holdout_test_summary.json` - every predeclared comparison with its five
  seed values, mean, sample SD (ddof = 1), minimum, maximum and win counts,
  and the directional statements the frozen rule permits.

The execution record:

* `opening_record.json` - written before the first test file was opened.
* `execution_log.txt` - progress lines only; no metric was printed.
* `execution_receipt.json` - the commit, the clean-tree state, every SHA-256
  that was verified, and the file reads the audit hook measured. It holds no
  metric value.

Tables and JSON only - no image, figure, array or model output of any kind.
Every per-slice table has the Milestone 5-10 columns plus one,
`degraded_input_sha256`: the hash of the single degraded input every method
scored on that slice, so a reader can check that all twelve tables measured
the same corruption.

The staging directory `holdout/test.incomplete/` existed only while the run
was in progress, and was renamed into place when the run completed. An
interrupted run would have left it behind, with a failure record, for an
integrity review; none was left.

The stress set has no directory here. It remains sealed, and no stress image
has been read since the split was frozen. The numbers are written up in the
[README](../../README.md#milestone-11b-the-held-out-test-result).
