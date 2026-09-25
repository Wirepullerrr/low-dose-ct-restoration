# Low-Dose-like CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public abdominal CT data.

> **Status: in progress (Milestone 10 of 15 complete - all four methods are
> measured on validation, and both learned methods are measured across five
> training seeds).**
>
> **CLAHE scored worse than doing nothing on every metric and every
> patient.** Both learned methods beat no restoration on all eight reported
> metrics and on all six validation patients, at every one of the five
> training seeds. The full-frame PSNR gain over no restoration was +3.08 to
> +3.19 dB across the five seeds of the 28,353-parameter residual CNN, and
> +3.37 to +3.40 dB across those of the 116,753-parameter lightweight U-Net
> (seed 2026: +3.18 and +3.38 dB).
>
> **U-Net versus CNN, across five predeclared training seeds each:** the
> U-Net used 4.12x the trainable parameters, and all eight metrics favoured
> it at each of the five seeds. Across the five seeds, the paired
> U-Net-minus-CNN full-frame PSNR difference averaged **+0.228 dB** (sample
> SD 0.050 dB; range +0.195 to +0.315 dB), with all five seeds favouring the
> U-Net; the body-region PSNR difference averaged +0.234 dB (sample SD
> 0.041 dB).
>
> The smallest observed full-PSNR difference (+0.195 dB) exceeded either
> architecture's observed five-seed full-PSNR range. This is a
> **descriptive** validation-development finding from five training seeds
> on one dataset and one synthetic degradation. No p-value, confidence
> interval or significance claim is reported, and five observed seeds do
> not bound what further seeds could produce.
>
> **Milestone 10 executed as pre-registered.** The design was committed in
> [configs/multiseed/plan.yaml](configs/multiseed/plan.yaml) *before* any of
> the eight new runs existed, and the eight ran afterwards without a single
> change to it: five training seeds per architecture (2026, 2027, 2028,
> 2029, 2030), seed 2026 reused from Milestones 8 and 9 rather than
> retrained. No seed was added, dropped, rerun or selected, and every run
> kept the checkpoint its own frozen rule chose.
>
> Every measured number here is a **validation** development result. No test
> or stress number exists, and no test or stress image content has been read
> since the split was frozen.
>
> Every number reported here is read from a tracked file under `outputs/`.
> The degraded-baseline and CLAHE numbers are re-derivable from the committed
> configs and the imaging data alone. Every learned-method number - the CNN
> and U-Net at seed 2026 and all eight Milestone 10 runs - additionally
> requires re-running a training command, because every checkpoint is
> git-ignored; each checkpoint's SHA-256 and selection evidence are recorded
> in tracked run and metric artifacts, so a regenerated checkpoint can be
> checked against them. The two seed-2026 training runs were each executed
> twice and reproduced a byte-identical training history and checkpoint on
> the same machine; the eight Milestone 10 runs were each executed once and
> were **not** independently bitwise-repeated.

## Research question

Given a clean reference CT slice and a controlled low-dose-like degradation of
it, how well do classical and deep-learning restoration methods recover image
quality, and what does each method cost in inference latency?

## Methods to be compared

| Method | Type | Status |
| --- | --- | --- |
| Degraded input (no restoration) | mandatory reference baseline | implemented; [measured on validation](#validation-degraded-baseline) |
| CLAHE | classical local contrast enhancement | implemented, validation-tuned and frozen; [worse than no restoration](#validation-result-clahe-versus-no-restoration) |
| Small residual CNN | deep learning | implemented and trained, **five seeds**; [beat no restoration on all 8 metrics](#validation-result-cnn-versus-no-restoration) |
| Lightweight U-Net | deep learning | implemented and trained, **five seeds**; [favoured over the CNN on all 8 metrics at each of 5 seeds, on validation](#milestone-10-the-multi-seed-result) |

All four have now been evaluated on identical patients, identical clean
targets and identical degraded inputs, through the same frozen metric code,
using MAE, MSE, PSNR and SSIM. **Inference latency has not been measured for
any of them**, so the cost half of the research question is still open.

The final comparison will be made on the held-out **test** split, once every
method decision is frozen. Nothing has been evaluated on test or stress yet:
every figure in this document is a validation number, used to develop and
sanity-check the measurement and to select among candidates, not a final
result.

## Planned pipeline

```
CHAOS abdominal CT DICOM
      -> HU conversion -> CT windowing -> normalized [0,1] reference image
      -> controlled synthetic low-dose-like degradation -> degraded input
      -> {CLAHE | CNN | U-Net} -> restored image
      -> MAE / MSE / PSNR / SSIM / latency -> engineering conclusion
```

Train, validation, and test splits are made **at the patient level**, never at
the slice level, because adjacent slices from one patient are highly correlated
and slice-level splitting would leak information into the test estimate.

## Dataset

The planned primary data source is **CHAOS (Combined CT-MR Healthy Abdominal
Organ Segmentation)**, using its abdominal CT DICOM series. The following was
checked against the Zenodo record and the challenge site on 2026-09-10:

| Fact | Value |
| --- | --- |
| Zenodo record published | 11 April 2019 |
| Concept DOI | [10.5281/zenodo.3362844](https://doi.org/10.5281/zenodo.3362844) |
| License stated on the record | Creative Commons Attribution Non Commercial Share Alike 4.0 International |
| CT subjects | potential liver donors with healthy livers |
| CT acquisition | upper abdomen, portal venous phase after contrast injection |

CHAOS was assembled for organ segmentation, and the challenge is described in
Kavur et al., "CHAOS Challenge - combined (CT-MR) healthy abdominal organ
segmentation", *Medical Image Analysis* vol. 69, 2021.

### Measured cohort

Both CHAOS archives were downloaded, checksum-verified, and audited by
[scripts/audit_chaos.py](scripts/audit_chaos.py). Their CT subject folders are
disjoint, so pooling them gives one cohort of 40 patients. These numbers come
from the files themselves, not from the dataset description:

| Measured | Value |
| --- | --- |
| CT subjects | 40, each with exactly one series |
| CT slices | 6407, between 78 and 294 per subject |
| Image size | 512x512 for every slice, MONOCHROME2, 16 bits stored |
| `RescaleSlope` | 1.0 for every slice |
| `RescaleIntercept` | varies by subject: -1024, -1000, -1200, or 0 |
| Pixel representation | unsigned for 39 subjects, signed for one |
| `SliceThickness` | 2.0, 3.0 or 3.2 mm depending on subject |
| In-plane pixel spacing | 0.541 to 0.791 mm |
| Slices failing HU conversion | 0 |

The subjects fall into four internally consistent **acquisition and
reconstruction parameter groups**, distinguished by their combination of
rescale intercept, slice thickness and stored display preset. The DICOM
`Manufacturer` and `ManufacturerModelName` elements are present but empty in
every file, so these groups cannot be mapped to specific scanner models from
the data, and this repository does not claim such a mapping. Every statistic
reported here is reproducible from `outputs/audit/`.

| Group | Subjects | Intercept | Thickness | Stored display preset |
| --- | --- | --- | --- | --- |
| A | 16 | -1024 | 2.0 mm | 60 / 360 |
| B | 21 | -1000 | 3.2 mm | 50 / 400 |
| C | 2 | -1200 | 3.0 mm | 45 / 350 |
| D | 1 | 0, signed pixels | 2.0 mm | 40 / 400 |

Note that 6407 slices is not 6407 independent samples. There are **40
independent patients**, and adjacent slices of one patient are highly
correlated, which is why splitting happens at the patient level.

The expanded 40-subject cohort supports a substantially stronger patient-level
split than the 20 subjects of the training archive alone. Because the four
parameter groups hold 16, 21, 2 and 1 subjects, the two rare groups cannot be
represented in every partition, so the assignment needed an explicit design
decision rather than a fixed ratio. That decision is recorded under
[Experimental partitioning](#experimental-partitioning).

**CHAOS contains no paired low-dose scans.** Its CT series are the clean
reference images in this project. Every degraded image in this benchmark is
produced by a synthetic, image-domain degradation that this repository applies
itself. Nothing here is a real low-dose acquisition.

Raw DICOM files are never committed to this repository. Anyone reproducing the
work downloads CHAOS themselves under its own license terms.

## CT preprocessing

Implemented in [src/ct_restoration/data/](src/ct_restoration/data/). The order
of operations is deliberate.

1. **Read stored pixel values** from DICOM with pydicom. These integers are not
   Hounsfield Units.
2. **Convert to Hounsfield Units** with the modality rescale,
   `HU = stored * RescaleSlope + RescaleIntercept`, or with a Modality LUT when
   the file carries one and declares that the LUT outputs Hounsfield Units. A
   modality LUT may instead output optical density, electron density or
   unspecified units, so its output is not assumed to be HU. If no HU
   calibration is available the loader raises rather than assuming defaults,
   because silently untransformed values would look like HU without being HU.
   Reading is CT-only, since Hounsfield Units are defined by the CT scale.
3. **Window in HU space.** A window is defined in Hounsfield Units, so it is
   only meaningful after calibration.
4. **Normalize to [0, 1]** using the window boundaries, so that a given HU
   value maps to the same model input in every slice.
5. **Resize to 256x256** last. Not because interpolation needs calibrated
   values: the modality rescale is affine and linear interpolation is a
   weighted average, so the two commute up to floating-point error. The reason
   is that **windowing clips, and clipping does not commute with
   interpolation**. Windowing first bounds how far any single pixel can pull
   its neighbours, which matters most for padding and other sentinel values at
   extreme HU. Resizing first would let a sentinel drag genuine tissue values
   with it and leave a rim of pixels corresponding to no real tissue. Keeping
   one fixed order also means every method and every image travels the same
   path, so differences in the final metrics come from the restoration method.

The default window is `center = 40 HU, width = 400 HU`, covering HU
`[-160, 240]`. This is a commonly used abdominal soft-tissue window, chosen
because it emphasizes the abdominal soft tissues relevant to the planned CHAOS
experiment. It is not a universally optimal setting, and dense structures such
as bone and strongly opacified vessels lie above its upper bound. Both numbers
are configurable and are recorded in
[configs/baseline.yaml](configs/baseline.yaml).

Three limitations follow directly from this design.

* **Windowing discards information on purpose.** Everything below -160 HU and
  above 240 HU is clipped to the range boundary, so lung, air, bone and metal
  collapse to a single value. The models in this project restore a windowed 2D
  view of a scan. They do not recover the full CT dynamic range.
* **Our windowing is an ML preprocessing convention.** It is a linear ramp
  between the window boundaries. It does not reproduce the VOI LUT or sigmoid
  presentation that a particular DICOM viewer may apply, and it deliberately
  ignores the `WindowCenter` / `WindowWidth` display presets stored in the
  files, which vary between series and are not an experiment definition.
* **Background is large, and it shapes the metrics.** No CHAOS CT slice
  declares `PixelPaddingValue` or `PixelPaddingRangeLimit`, so there is no
  formally declared padding to mask. Even so, about 45 percent of an average
  frame is air below -900 HU, and one subject carries an additional uniform
  -2048 HU region outside the reconstruction circle, consistent with
  out-of-field or reconstruction-background padding. All of it clips to 0.0
  after windowing, leaving roughly half the frame flat.

  The metric-region question this raised is **now settled: both full-frame and
  body-region metrics are reported for every method**, see
  [Two regions, both reported](#two-regions-both-reported). What the later
  milestones measured is that the background is not a neutral filler — it
  pulls the two metric families in opposite directions, easing MAE/MSE/PSNR
  while depressing SSIM. The first method measured there, CLAHE, redistributes
  regional brightness and carries no denoising mechanism; it loses on both
  families. How a *learned* method behaves there is still unknown.

## Real-data audit

`scripts/audit_chaos.py` reads all 6407 slices and writes its findings to
`outputs/audit/`: a per-slice metric table, a per-subject summary, a
dataset-level JSON summary, and an exhaustive per-coordinate table of extreme
pixels. Re-running it after a fresh download reproduces all of them. Four
results shape later milestones.

**The 40 / 400 HU window holds up on the real images.** Inside a crude body
mask, 84.0 percent of pixels fall within the window on average, 12.5 percent
below it and 3.5 percent above. Independently, the four parameter groups carry
their own stored soft-tissue display presets of 60/360, 50/400, 45/350 and
40/400, so the project window sits inside the same family as the presets the
files themselves carry. The window was not changed after looking at the images.

**The HU calibration is not uniform, and assuming it would have been wrong.**
`RescaleSlope` is 1.0 everywhere, but `RescaleIntercept` is -1024 for 16
subjects, -1000 for 21, -1200 for 2, and 0 for one. That last subject also
stores **signed** pixels, so its stored values already are Hounsfield Units and
run down to -2048. Hard-coding the common -1024 would have miscalibrated 24 of
40 subjects; assuming unsigned storage would have wrapped that subject's
negative values into large positive ones. Reading both the calibration and the
pixel representation from each file is what keeps the HU scale comparable
across patients.

**Two separate populations of extreme pixels exist, and they are not the same
phenomenon.** Above a generous 4000 HU ceiling the audit finds:

* *Fixed-coordinate corner outliers* in seven subjects (4, 9, 10, 13, 14, 17,
  19). Exactly the two pixels at image positions (0,0) and (0,1), on every
  slice of those subjects, reaching 49944 HU, always outside the body mask.
  These are persistent non-anatomical outlier pixels of **unknown provenance**;
  nothing in the metadata explains them, and this repository does not guess.
* *A localized in-body high-density region with associated artifact* in subject
  39 alone: 69 distinct coordinates confined to rows 283-423 and columns
  240-278, appearing on only 11 of 237 slices, reaching 10318 HU, entirely
  **inside** the body mask. Its spatial extent and per-slice variation
  distinguish it from the fixed-coordinate population above. Its composition is
  not established by anything in the data, and this repository does not
  speculate about it.

Nothing is modified. Windowing already bounds the effect of both, and any
cleaning policy is a later, explicit decision. They are also a live
demonstration of why windowing precedes resizing: clipping pins these values to
the window ceiling first, so their influence stays inside one output pixel
instead of smearing into neighbouring tissue during interpolation.

**Slice order comes from DICOM geometry, not from the z coordinate alone.**
Every slice carries `ImageOrientationPatient`, and the correct position is
`ImagePositionPatient` projected onto the slice normal obtained by crossing the
row and column direction cosines. Measured on this cohort, the orientation is
the canonical axial `1,0,0,0,1,0` on all 6407 slices, and the projected
position differs from the bare z component by exactly 0.0, so the simpler rule
happens to be equivalent here. The audit uses the projection regardless,
because that equivalence is a property of this dataset rather than of the rule.
Spacing is uniform within every subject, at a median step of 1.0 to 2.0 mm
depending on subject, and no two slices share a position. `InstanceNumber`
ascends as the projected position descends in all 40 subjects, which is
ordinary for CT. Filename order agrees with `InstanceNumber` in all 40 as well,
but that is a property of how the archives were written, not a guarantee, so
anything order-dependent sorts by geometry with the direction recorded.

## Experimental partitioning

The independent experimental unit is the **patient**, never the slice. Every
slice of a subject belongs to exactly one partition, so no anatomy from an
evaluation patient can ever have been seen during training. The assignment is
generated by [scripts/create_splits.py](scripts/create_splits.py) and frozen in
[data/splits/chaos_patient_split.csv](data/splits/chaos_patient_split.csv).

| Partition | Patients | Slices | A | B | C | D | From `Train_Sets` | From `Test_Sets` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 25 | 4160 | 10 | 15 | 0 | 0 | 12 | 13 |
| validation | 6 | 885 | 3 | 3 | 0 | 0 | 3 | 3 |
| test | 6 | 941 | 3 | 3 | 0 | 0 | 3 | 3 |
| stress | 3 | 421 | 0 | 0 | 2 | 1 | 2 | 1 |
| **total** | **40** | **6407** | **16** | **21** | **2** | **1** | **20** | **20** |

Patient overlap between partitions is zero.

### Why two rare groups are held out separately

Groups C and D contain two subjects and one subject. A group that small cannot
be present in train *and* validation *and* test at once, and a one-patient
share of a partition is not an estimate of anything. Rather than imply balanced
acquisition coverage that the cohort cannot support, every subject from those
groups is withheld as a **small held-out rare-acquisition robustness stress
set**.

That stress set is not a second statistically powered benchmark, is not
evidence of broad generalization across acquisition settings, and is not
validation data. It is three patients. No robustness claim is made here: no
model has been evaluated on it, and nothing in this document is a
robustness result.

The primary test set is the held-out patient-level benchmark drawn from the
adequately represented A and B acquisition groups, and results from it should
be described that way rather than as performance on CHAOS as a whole.

### What each partition is for

| Partition | Permitted use |
| --- | --- |
| train | parameter fitting only |
| validation | every empirical decision: hyperparameters, degradation settings, CLAHE parameters, checkpoint and model selection |
| test | the final primary benchmark, evaluated once decisions are frozen |
| stress | rare-acquisition robustness inspection, after the same freeze |

Test and stress must not be inspected repeatedly during development. Selecting
anything against them would turn them into validation data and the final
numbers would no longer be held-out estimates.

### How the assignment was chosen

No random number generator is involved, and nothing about image content
influences it. Three facts about each subject are used, each for one purpose:

| Input | Role |
| --- | --- |
| subject identity | deterministic canonical ordering and tie-breaking only |
| acquisition group | constrains the split: rare groups withheld, primary groups allocated |
| slice count | balances image counts across partitions |

`source_archive` is **not** among them. It is retained as provenance, written
into the outputs, and audited after the assignment has been made, but no step
below reads it. The archive mixing reported in the table above is therefore an
observed property of the resulting split, not something the algorithm was asked
to achieve.

1. Subjects are classified into acquisition groups by their measured
   `RescaleIntercept`, `SliceThickness` and `PixelRepresentation`. Membership
   is derived from the audit, never typed by hand.
2. All rare-group subjects go to the stress set.
3. The group allocation across partitions is chosen by scoring every feasible
   integer option, 25 of them here. The selected 10A/15B train, 3A/3B
   validation and 3A/3B test allocation has the lowest total deviation from the
   primary cohort's 43.2 percent A share among the feasible allocations, 0.168
   against 0.174 for the runner-up. It also gives validation and test identical
   A/B patient counts. That symmetry is not why it wins: five of the 25
   allocations are symmetric, and symmetry only acts as a tie-break between
   options that already score equally. Matching A/B counts removes one known
   source of compositional mismatch between validation and test. It does not
   make six validation patients representative of six different test patients.
4. Subjects are then placed heaviest first into whichever partition is furthest
   below its slice quota, followed by an exhaustive improving-swap pass between
   validation and test. Equal patient counts do not mean equal image counts,
   because subjects hold between 78 and 294 slices. The result leaves
   validation and test within 56 slices of each other, about 6 percent, which
   is the best achievable under same-group swaps.

The CHAOS `Train_Sets` and `Test_Sets` archives are provenance only. They are
the challenge's split of its own segmentation task, which this project does not
perform. Both happen to appear on both sides of every partition here, which is
a welcome outcome rather than a constraint that was imposed; a test asserts
that relabelling or uniformly flattening the archive labels leaves the
assignment unchanged.

### Freeze

`data/splits/chaos_patient_split.csv` is **frozen**. Its SHA-256 is

```
4a11d080c80e4096c052436751ba948eb450539e5f83470b5ab2cf49722b3866
```

recorded in [outputs/audit/chaos_split_summary.json](outputs/audit/chaos_split_summary.json)
and asserted by the test suite.

The checksum covers the **entire canonical CSV**, not just its `split` column.
Read it precisely:

* **The same SHA-256 proves the file is byte-identical** to the one any earlier
  result was computed against. Cohort membership, every recorded acquisition
  attribute, and every partition assignment are unchanged.
* **A different SHA-256 proves the file changed, and that is all it proves.**
  It does not by itself mean the partitioning changed. A corrected
  `slice_thickness`, a reordered column, or a different line ending would move
  the hash while leaving every patient in the same partition. A hash mismatch
  is a signal to investigate and to say what changed, not a conclusion.
* **If the investigation finds that cohort membership or partition assignment
  changed, then previously reported final-test comparisons are no longer
  directly equivalent** and must be recomputed or explicitly labelled as
  measured against a different split.

Changing the split after seeing model results requires an explicit
methodological reason, recorded alongside the new checksum. Regenerating from
the same audit reproduces this file byte for byte, in any input row order, as
it does the slice manifest and the split summary: none of the three carries a
timestamp or any other run-varying field, so regenerating an unchanged
experiment definition is a no-op rather than a diff.

[data/splits/chaos_slice_manifest.csv](data/splits/chaos_slice_manifest.csv)
lists all 6407 slices, each inheriting its patient's partition. Slices carry a
geometric position and index computed from `ImagePositionPatient` projected onto
the slice normal, so nothing downstream has to treat filenames as anatomical
order.

## Synthetic low-dose-like degradation

The corruption that creates every model input is defined and frozen in
[configs/degradation.yaml](configs/degradation.yaml) and implemented in
[src/ct_restoration/data/degradation.py](src/ct_restoration/data/degradation.py).
It was fixed **before any restoration method existed**, so no method could
influence what it has to undo.

### Why "low-dose-like" and not "simulated low-dose CT"

CHAOS distributes reconstructed CT images. It does not distribute raw
projection data, and it carries no scanner dose or noise calibration. A genuine
dose simulation works on the projections: it reduces the photon count, passes
the result through the scanner's own reconstruction, and needs calibration data
this dataset does not contain. Nothing here does that.

So this project makes **no claim** about a particular mAs, a percentage dose
reduction, a photon count, a scanner-equivalent low-dose reconstruction, or
clinical validity. What it does claim is narrower and defensible: a controlled,
reproducible, image-domain corruption whose dominant characteristic is
qualitatively the kind of difference that separates a lower-dose reconstruction
from a higher-dose one, namely increased, spatially varying, spatially
correlated noise. That is enough to benchmark restoration methods against each
other. It is not enough to say anything about real low-dose acquisition.

### Where it applies

After the fixed preprocessing above, never before it.

```
stored DICOM pixels -> HU -> 40/400 HU window -> [0,1] -> 256x256
      -> DEGRADATION          <- here
      -> {degraded baseline | CLAHE | CNN | U-Net}
```

The restoration target is therefore the windowed, normalized 2-D
representation, not the full original CT dynamic range.

### The algorithm

For a clean image `x` in [0, 1]:

```
sigma(x) = 0.015 + 0.035 * sqrt(x)
epsilon  = zero-mean, unit-variance Gaussian field,
           spatially correlated with a Gaussian filter of sigma 0.6 px
           (reflect boundary), renormalized to zero mean and unit std
degraded = clip(x + sigma(x) * epsilon, 0, 1)
```

**Heteroscedastic.** The noise standard deviation varies across the image
instead of being one constant for the frame. The motivation is that noise in a
reconstructed low-dose CT image is not generally spatially uniform either.

The motivation stops there, and the wording matters. `sigma(x)` is **a simple
heuristic** for making the corruption non-uniform, built from the one quantity
available at this point in the pipeline: the local normalized intensity. The
`sqrt` shape is a modelling convention, picked so that noise grows with
intensity but sub-linearly.

It is **not** derived from projection-domain photon counts, from line-integral
attenuation, from scanner calibration, or from any physical noise model. A
voxel's intensity is not the number of photons detected anywhere, and real
image-domain noise at a point depends on every ray path crossing it and on the
reconstruction algorithm, none of which is modelled here. Nothing in this
project claims that a brighter pixel physically receives the amount of noise
configured for it. The claim is only that the corruption is non-uniform in a
fixed, declared and reproducible way.

`sigma_floor = 0.015` keeps noise present in the darkest regions, where a pure
`sqrt` term would vanish and leave the background implausibly clean.

**Spatially correlated.** Real reconstruction noise is not independent from
pixel to pixel; filtered back-projection and iterative reconstruction both
impose a texture with a characteristic grain. A 0.6 px Gaussian correlation
gives the noise a mild grain instead of a pure per-pixel speckle, which a
convolutional model would find unrealistically easy to average away. The field
is renormalized to unit variance after filtering, because smoothing reduces
variance; without that step, changing the correlation width would silently
change the noise amplitude too.

**Noise only.** No blur, contrast compression, sharpening, streak artefact,
synthetic motion, gamma or histogram operation is applied. Bundling several
phenomena into one corruption would make any later result much harder to
attribute: a method that scored better could be undoing any one of them, or
trading one against another, and the benchmark could not tell which.

Magnitude, with its caveat: the window is 400 HU wide, so inside the linear
part of the window a normalized sigma of 0.015 to 0.050 corresponds to roughly
6 to 20 HU. That is a rough sense of perturbation size in normalized space, not
measured scanner noise and not calibrated physical HU noise. The mapping also
breaks down wherever the window already clipped a pixel to 0 or 1, since such a
pixel no longer stands for a single HU value.

### Deterministic per-slice seeding

Every slice gets its own independent, stable seed:

```
seed = SHA-256(algorithm_version, global_seed, canonical_sample_key)
```

taken as a 64-bit integer and used to construct a `PCG64` generator private to
that call. The canonical key is the POSIX-style `relative_dicom_path` from the
frozen slice manifest, with backslashes normalized, so a Windows and a POSIX
spelling of one slice derive the same seed.

Python's built-in `hash()` is deliberately **not** used: it is randomized per
process for strings, so the same slice would receive different noise in
different runs and nothing would be reproducible. The global NumPy random state
is never read or written either, so the noise on a slice cannot depend on how
many random numbers the rest of the program happened to draw first.

The result is that the degraded image is a pure function of the clean
reference, this config and the sample key. It does not depend on call order,
batching, DataLoader worker order, time, hostname, absolute path or process id.
Tests assert each of these.

### Same corruption for every method

The degraded baseline, CLAHE, the residual CNN and the U-Net all receive inputs
generated from the same config, the same sample keys and the same clean
references. No method gets its own realization of the noise, so any difference
between them comes from the method.

Degradation is generated on demand and never precomputed to disk. There is no
degraded image dataset to fall out of sync with the definition.

### Freeze

`configs/degradation.yaml` is **frozen** before any model benchmarking. Changing
it defines a different benchmark: results measured against the old definition
are not comparable to results measured against the new one, and would have to
be relabelled or recomputed rather than quietly carried over. The algorithm
version string is hashed into every seed, so a v2 would produce entirely
different realizations by construction, and code implementing v1 refuses to run
a config that names anything else.

### Training-only audit

`scripts/audit_degradation.py` applies the degradation to all 4160 training
slices and writes
[outputs/audit/degradation_train_summary.json](outputs/audit/degradation_train_summary.json).
It reports no PSNR, SSIM or restoration error; those belong to the evaluation
milestone.

**No validation, test or stress image was inspected.** The v1 parameters were
fixed in advance and were not tuned by looking at pictures. Holding out those
images is the whole point of the frozen split, and there was nothing to gain
from spending them here.

Measured on the training split, 25 subjects and 4160 slices, all 256x256
`float32` and all within [0, 1], with zero non-finite and zero out-of-range
results:

| Training-set diagnostic | Value |
| --- | --- |
| perturbation mean | +0.00295 |
| perturbation std | 0.02643 |
| per-slice perturbation std, min / median / max | 0.0193 / 0.0263 / 0.0322 |
| per-slice mean absolute perturbation, min / median / max | 0.0110 / 0.0167 / 0.0213 |
| clean pixels at 0 / at 1 | 54.0 % / 1.1 % |
| degraded pixels at 0 / at 1 | 27.1 % / 0.8 % |
| sigma map min / median / max | 0.0150 / 0.0154 / 0.0500 |

Two of those numbers deserve a word, because they look odd until the clipped
window is taken into account.

The **perturbation mean is not zero** but +0.003, although the noise field is
zero-mean by construction. More than half the clean pixels sit at exactly 0
after windowing, and clipping the output to [0, 1] removes the negative half of
the noise there while keeping the positive half. The bias is an artefact of
adding noise to an already saturated background, and it is also why the
fraction of pixels sitting at 0 falls from 54 % to 27 %.

The **pooled sigma median, 0.0154, is near the floor** because the median pixel
of a CT slice is background, not tissue. Inside the body the sigma map runs far
higher, up to 0.0500 at full intensity.

### The clipped-background caveat

The fixed CT window leaves large regions at exactly 0 or 1, and this
degradation adds noise to them along with everything else. That does **not**
reproduce the physics of air, out-of-field padding or saturated dense
structures; it is a consequence of operating on a windowed representation.

Together with the background dominance already noted for the clean images, this
is a second reason the evaluation milestone may need both full-frame and
body-region metrics, and must state which it is reporting. No metric mask is
defined here, and no subject is special-cased.

## Evaluation framework and the degraded baseline

The measuring apparatus is defined and frozen in
[configs/evaluation.yaml](configs/evaluation.yaml), implemented in
[src/ct_restoration/metrics.py](src/ct_restoration/metrics.py) and
[src/ct_restoration/evaluation.py](src/ct_restoration/evaluation.py), and fixed
**before any method exists**. CLAHE and the CNN call those same functions on
the same slices with the same masks, and so does the U-Net, so a comparison
between them reflects the methods rather than the measurement.

### The degraded baseline is "no restoration"

| | |
| --- | --- |
| reference | the preprocessed clean CT slice |
| input | the frozen synthetic low-dose-like degraded image |
| "restored" output | the degraded image, unchanged |

No enhancement and no model is involved. The numbers below are therefore the
image-quality cost of the degradation itself, and they are the floor that every
later method has to beat on exactly this benchmark.

### Four metrics

| Metric | Definition | Direction |
| --- | --- | --- |
| MAE | `mean(abs(prediction - reference))` | lower is better |
| MSE | `mean((prediction - reference)^2)` | lower is better |
| PSNR | `10 * log10(data_range^2 / MSE)`, in dB | higher is better |
| SSIM | `skimage.metrics.structural_similarity` | higher is better, max 1 |

`data_range = 1.0`, fixed rather than inferred per image. The images being
compared are the normalized [0, 1] representation produced by the fixed 40/400
HU window, so 1.0 is the actual dynamic range. Inferring it per slice would
make PSNR mean something different on every slice and destroy comparability,
which is the one thing a benchmark cannot lose.

SSIM settings are pinned explicitly rather than left to library defaults, so a
scikit-image upgrade cannot silently redefine the benchmark:

```
win_size              11
gaussian_weights      true
sigma                 1.5
use_sample_covariance false
data_range            1.0
channel_axis          None
```

`sigma = 1.5` with scikit-image's gaussian weighting produces exactly an 11-tap
window, so `win_size = 11` agrees with it rather than competing with it.

### Two regions, both reported

The audit found that about 54 % of clean training pixels are exactly 0 after
the fixed CT window. A full-frame metric therefore assigns that region
substantial weight, and the background can materially affect the score.

Not because the background is trivially correct. The frozen degradation uses
`sigma_floor = 0.015`, so clipped background pixels are perturbed too, and the
fraction of pixels sitting at exactly 0 falls from 54 % to 27 % after
degradation. The background is not a region every method gets right for free.

What it does instead is affect the two metric families in **opposite
directions**:

* under the intensity-dependent degradation, background corruption is smaller
  than much of the body's, so including it makes full-frame **MAE, MSE and
  PSNR** look easier;
* **SSIM** reacts the other way. Adding noise to a low-variance flat region
  changes its local luminance, contrast and variance statistics sharply, so
  SSIM over background is *lower* than over textured tissue.

The measured baseline shows both effects at once: body PSNR is lower than full
PSNR, while body SSIM is *higher* than full SSIM.

Neither region alone tells the whole story, which is exactly why both are
retained and reported for every method.

### The evaluation body mask

Applied to the clean HU slice **before** windowing:

1. threshold `HU > -500`;
2. label 2-D connected components with 8-connectivity;
3. keep the largest component;
4. fill enclosed holes;
5. resize to 256x256 with **nearest-neighbour** interpolation;
6. cast to `bool`.

Step 3 is what removes the scanner table: the threshold does pick the table up,
but it is a separate component from the patient and therefore discarded. Step 4
keeps low-HU structures *inside* the body — bowel gas, lung base — in the
evaluation region instead of punching holes in it. Step 5 is the one place this
project uses nearest-neighbour: any interpolating method would produce
fractional values along the boundary of a binary mask, which is not a mask.

`-500 HU` sits between air, around -1000 HU, and soft tissue, around 0 to 60
HU. It is far from both, so the silhouette does not hinge on a finely tuned
cut. It was fixed in advance and **never** selected by comparing metrics;
choosing a threshold because it produced a nicer PSNR would be tuning the
evaluation against the scores it is supposed to judge.

Three properties matter more than the mask's anatomical accuracy:

* it comes from the **clean reference**, never from a degraded or restored
  image, so a method cannot influence the region it is judged on;
* it is **never given to a restoration method**, so it is not a hidden input;
* it is **identical for every method** on a given slice.

This is a crude deterministic silhouette separating patient from air and table.
It is **not clinical segmentation** and no anatomical claim is made about it.

### Body SSIM is a masked map mean, not a new SSIM

MAE, MSE and PSNR can be computed directly over masked pixels, because each is
an average of independent per-pixel terms.

SSIM cannot. It is defined on a local neighbourhood, so flattening the masked
pixels into a vector would destroy the very structure it measures. Instead:

1. compute **one** SSIM map with the frozen settings;
2. erode the body mask by an all-true 11x11 element, keeping only centres whose
   complete SSIM neighbourhood lies inside the body;
3. average that same map over the eroded mask.

`body_ssim` is therefore **the mean of the standard local SSIM map over
interior body pixels**, not a separate definition of SSIM, and must not be
described as one. Erosion treats outside-the-array as background, so it also
drops the 5-pixel image rim that scikit-image excludes from its own mean.

An empty eroded mask fails loudly rather than averaging nothing.

### The patient is the experimental unit

The 885 validation slices are **not** 885 independent observations. Neighbouring
slices are 1 to 2 mm apart, from one acquisition, one reconstruction and one
anatomy, and validation patients hold between 94 and 240 slices.

So the aggregation is two-stage:

1. average a patient's per-slice metrics into one number per patient;
2. average the patients with **equal weight**.

A slice-weighted figure is also computed, labelled **secondary descriptive
summary**, purely to show how much unequal slice counts would move the result.
It is never the headline number.

### Training-only body-mask audit

`scripts/audit_body_mask.py` ran the rule over all 4160 training slices before
any validation number was computed, and
[outputs/audit/evaluation_body_mask_train_summary.json](outputs/audit/evaluation_body_mask_train_summary.json)
records the result.

| Training-set mask diagnostic | Value |
| --- | --- |
| masks generated | 4160 / 4160 |
| generation failures | 0 |
| empty masks | 0 |
| empty SSIM-interior masks | 0 |
| output shape / dtype | 256x256 / `bool` |
| body pixel fraction, min / median / max | 0.306 / 0.496 / 0.705 |
| SSIM-interior fraction, min / median / max | 0.257 / 0.435 / 0.634 |

With zero failures the policy was frozen, and only then was validation touched.

### Validation degraded baseline

**PRIMARY result** — 6 patients, 885 slices, every patient weighted equally:

| Region | MAE | MSE | PSNR (dB) | SSIM |
| --- | --- | --- | --- | --- |
| full frame | 0.016359 | 0.00072130 | 31.4813 | 0.781346 |
| body region | 0.028187 | 0.00141527 | 28.5620 | 0.810430 |

Spread across the six patients, as standard deviation and range:

| Metric | std | min | max |
| --- | --- | --- | --- |
| full PSNR | 0.4636 | 30.7432 | 32.1612 |
| body PSNR | 0.3401 | 28.0116 | 28.8229 |
| full SSIM | 0.007361 | 0.768523 | 0.790428 |
| body SSIM | 0.017644 | 0.795665 | 0.842447 |

This spread describes **how much the six patients differ from one another at
baseline**. It is not a yardstick for judging a later method's improvement,
and it must not be used as one: baseline cross-patient variation and the
variation of a method's improvement are different quantities, and neither
bounds the other. A method could lift every patient by a consistent 0.3 dB
while baseline levels differ by 1.4 dB, or swing wildly between patients whose
baselines are nearly identical.

The right descriptive object for judging a method is the **paired per-patient
delta**, `delta_i = method_metric_i - baseline_metric_i`, computed on the same
patient and the same slices, with the sign read according to the metric's
direction: positive is an improvement for PSNR and SSIM, negative is an
improvement for MAE and MSE. That is exactly how CLAHE is compared against
this baseline in
[the paired per-patient deltas](#the-paired-per-patient-deltas); nothing of
the kind is implied by the spread in the table above.

**SECONDARY slice-weighted descriptive summary**, not the benchmark result:

| Region | MAE | MSE | PSNR (dB) | SSIM |
| --- | --- | --- | --- | --- |
| full frame | 0.016294 | 0.00071373 | 31.5176 | 0.781470 |
| body region | 0.027858 | 0.00138778 | 28.6452 | 0.809175 |

The two differ by under 0.09 dB here, which says the validation patients happen
to be similar enough that unequal slice counts do not move this particular
number much. That is a fact about this split, not a reason to stop reporting
the patient-weighted figure: the guarantee has to hold for methods whose errors
may vary far more between patients than the degradation's do.

Descriptive acquisition-group breakdown, three patients each, useful only for
spotting a glaring imbalance:

| Group | full PSNR | body PSNR | full SSIM | body SSIM |
| --- | --- | --- | --- | --- |
| A | 31.5202 | 28.7734 | 0.781095 | 0.803803 |
| B | 31.4423 | 28.3507 | 0.781596 | 0.817057 |

No significance test is run, no claim is made about acquisition settings in
general, and nothing is concluded from these small differences. With n = 3 per
group there is nothing to conclude.

### Reading these numbers

MAE and MSE are lower-is-better; PSNR and SSIM are higher-is-better. Together
they say how much damage the frozen degradation does, and nothing else. Later
methods are judged by the paired per-patient improvement over *these exact
figures*, measured by the same code on the same slices and the same masks.

One caution about aggregate PSNR. For a single slice in a single region, PSNR
is a strictly decreasing transform of MSE, so the two always rank the same
way. That equivalence does not survive averaging: the logarithm is nonlinear,
so a mean of per-slice PSNRs is not a monotone function of the mean of the
per-slice MSEs, and two methods can be ordered one way by mean MSE and the
other way by mean PSNR. Read the aggregate PSNR as its own figure rather than
as a restatement of the aggregate MSE.

The two regions do **not** move together, and it is worth being precise about
which way each one goes:

| Metric | Full frame | Body region | Body is |
| --- | --- | --- | --- |
| MAE | 0.016359 | 0.028187 | worse (higher) |
| MSE | 0.00072130 | 0.00141527 | worse (higher) |
| PSNR | 31.4813 | 28.5620 | worse (lower) |
| SSIM | 0.781346 | 0.810430 | **better (higher)** |

The three pixel-error metrics agree that the body carries larger absolute
corruption, which follows from an intensity-dependent noise scale: more of the
perturbation lands on tissue than on clipped background. SSIM goes the other
way, because a flat low-variance region is where added noise disturbs local
contrast and structure statistics most.

So the honest summary is that **the choice of region affects different metric
families differently**, not that the body is objectively harder or easier
overall. These are two different questions — "how large is the pixel error
here" and "how much local structure survives here" — and they have different
answers in the two regions. That is the reason both regions are reported
rather than one being chosen as the real one.

Two things this section deliberately does not say. No PSNR or SSIM value here
is "good", "acceptable" or "clinically adequate": these are relative figures on
a synthetic benchmark. And PSNR is measured in **decibels**, a logarithmic
unit, so a difference between two PSNRs is a difference in dB and never a
percentage improvement.

### What has and has not been looked at

Training image content was read, to audit the mask rule. Validation image
content was evaluated numerically to produce the baseline above; no validation
image was inspected visually, and no validation number influenced the
evaluation policy, which was frozen first.

The hold-out claim has to be stated carefully, because the accurate version is
narrower than "never read". **Before the patient split was frozen**, all 40
subjects were included in dataset-level technical QC and cohort
characterization — the Milestone 2 audit read every one of the 6407 slices.
That was dataset audit, not model selection: it established HU calibration,
geometry, padding and outlier facts, and it happened before any partition, any
degradation or any metric existed.

**Since the experimental split was frozen**, test and stress image content has
not been used for degradation design, evaluation-policy development, model
selection, or benchmark scoring. The development commands written after the
freeze refuse those splits outright rather than merely discouraging them, and
tests assert the refusal:

* `audit_body_mask.py` reads train only — it has no `--split` option at all;
* `evaluate_degraded_baseline.py` accepts train and validation, and produced
  the canonical baseline above from validation;
* neither command can read test or stress.

Those splits stay sealed for all post-split experimental development. The final
benchmark will be a separate, explicitly final command, run after every method
decision is frozen.

## CLAHE: the classical comparison method

Contrast Limited Adaptive Histogram Equalization, via OpenCV, implemented in
[src/ct_restoration/classical/clahe.py](src/ct_restoration/classical/clahe.py).
It is the classical reference point the deep-learning methods are measured
against. It is **not** a denoiser and carries no trained parameters.

The headline finding, stated first because the rest of this section explains
it: **every CLAHE configuration tried scored worse than doing nothing.**

### What CLAHE sees

Exactly the frozen degraded image, and nothing else.

```
clean reference -> frozen degradation -> degraded image -> CLAHE -> output -> evaluator
```

No clean reference, no HU slice, no body mask, no acquisition group, no source
archive, no patient identity. The restoration contract is a one-argument
callable for precisely this reason. The body mask in particular is
evaluation-only: applying CLAHE inside it would feed the method a region
derived from the clean reference, which a deployed method would never have.

### The 8-bit conversion is part of the method

OpenCV's CLAHE builds per-tile histograms and needs an integer single-channel
image, so the [0, 1] benchmark representation is quantized, enhanced, and
mapped back:

```
uint8 = round(x * 255)   ->   CLAHE   ->   float32 = uint8 / 255
```

The mapping is fixed to the benchmark range, never to the image's own minimum
and maximum: per-image normalization would give a different mapping to every
slice and destroy the comparability the fixed 40/400 HU window exists to
provide. This quantization is declared in the config as
`input_quantization_bits` and is a defining part of the method, not an
incidental detail — a different depth would be a different method, and the
config refuses one.

Its cost, measured rather than assumed: the round-trip error is at most
`0.5 / 255 ≈ 0.00196` in normalized units. Spread over the 400 HU window that
is about 0.8 HU per step, against a degradation whose per-slice perturbation
standard deviation is about 0.026. Small, but an approximation all the same.
This is not a claim that 8 bits is clinically lossless.

### The predeclared search space

Fixed in [configs/clahe_search.yaml](configs/clahe_search.yaml) **before any
CLAHE candidate was scored**: 4 clip limits × 3 tile grids = 12 candidates.
(The Milestone 5 degraded-baseline numbers already existed at that point; what
was fixed in advance is the CLAHE grid and the rule for choosing within it.)

| Parameter | Values |
| --- | --- |
| `clip_limit` | 0.5, 1.0, 2.0, 4.0 |
| `tile_grid_size` | [4, 4], [8, 8], [16, 16] |

On the 256×256 benchmark image those grids give 64×64, 32×32 and 16×16 pixel
tiles. `clip_limit` bounds how far a single histogram bin may rise before the
excess is redistributed across the tile — the "contrast limited" part, and the
thing that stops a near-uniform tile from having its noise stretched across
the full output range.

The grid was **not** widened after the results came in. Trying extra values
until something beats the baseline is how a validation split stops being an
honest development estimate.

### The predeclared selection rule

Also fixed in advance. Primary metric: **patient-weighted mean body SSIM**,
higher is better. CLAHE is a local contrast and structure method, so a local
structural measure inside the anatomy is the metric most aligned with what it
attempts; choosing it beforehand avoids picking whichever metric the method
happened to win on. Ties break on body PSNR, then full SSIM, then full PSNR,
then the lower clip limit, then the smaller grid — a total order, so exactly
one candidate wins and the winner never depends on row order.

Candidates are ranked on six patients, not 885 slices: per-slice metrics are
averaged within a patient first, then patients are averaged with equal weight.

### The 12-candidate validation sweep

All 885 validation slices, each loaded and degraded once and then shown to all
12 candidates, scored with the same Milestone 5 metric code.
[outputs/metrics/clahe_validation_search.csv](outputs/metrics/clahe_validation_search.csv)
holds the full table.

| clip | grid | body SSIM | body PSNR | full SSIM | full PSNR | rank |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 4×4 | 0.768085 | 26.4831 | 0.620769 | 29.1561 | **1 selected** |
| 0.5 | 8×8 | 0.755407 | 25.0473 | 0.604626 | 27.8322 | 2 |
| 1.0 | 4×4 | 0.722111 | 24.0467 | 0.530833 | 26.6919 | 3 |
| 1.0 | 8×8 | 0.703800 | 22.8213 | 0.511055 | 25.5663 | 4 |
| 0.5 | 16×16 | 0.664602 | 16.7047 | 0.478881 | 19.9083 | 5 |
| 1.0 | 16×16 | 0.664602 | 16.7047 | 0.478881 | 19.9083 | 6 |
| 2.0 | 4×4 | 0.658124 | 21.2925 | 0.443808 | 23.7825 | 7 |
| 2.0 | 8×8 | 0.625283 | 19.7940 | 0.422941 | 22.5031 | 8 |
| 4.0 | 4×4 | 0.614461 | 19.3077 | 0.382250 | 21.2416 | 9 |
| 2.0 | 16×16 | 0.607239 | 18.4132 | 0.403766 | 21.2020 | 10 |
| 4.0 | 8×8 | 0.554888 | 17.2304 | 0.353577 | 19.6415 | 11 |
| 4.0 | 16×16 | 0.514593 | 16.1574 | 0.327203 | 18.6425 | 12 |
| — | *no restoration* | *0.810430* | *28.5620* | *0.781346* | *31.4813* | *baseline* |

Two things in that table are worth naming.

Ranks 5 and 6 are **identical to the last digit**, and that is not a
coincidence. OpenCV converts `clip_limit` into an integer per-bin threshold,
`max(int(clip_limit * tile_area / 256), 1)`. At a 16×16 grid on a 256×256
image the tile area is exactly 256, so clip limit 0.5 gives `int(0.5) = 0`
and clip limit 1.0 gives `int(1.0) = 1`; the `max(..., 1)` then raises both to
a threshold of 1, and the two select the *same* effective operator. The search space
still contains **12 declared parameter tuples**, but two of them map to one
operator under the pinned OpenCV build, so the sweep covers **11 distinct
effective operators**. The tie-breaker resolved the duplicate pair
deterministically in favour of the lower clip limit. Neither declared
candidate was removed after the result was seen, and the grid was left as
declared rather than quietly repaired.

Within the predeclared grid the ranking is **monotone in both parameters**:
the lower tested clip limits and the coarser tested tile grids scored better.
The winner therefore lies at the boundary of the tested parameter space —
clip limit 0.5 with 4×4 tiles is the mildest corner of the grid, its lowest
clip limit and coarsest tiling — so the search did not identify an interior
optimum, and the sweep does **not** establish what would happen outside that
space —
neither that a still lower clip limit would keep improving, nor what OpenCV's
integer clip threshold would do there. The grid was intentionally not expanded
after the validation results were seen.

Stated separately, because it is the comparison that matters: **the
no-restoration identity baseline outperformed every one of the 12 tested
parameter combinations**, on all four metrics in the table above.

### The frozen configuration

[configs/clahe.yaml](configs/clahe.yaml), written by the sweep, not hand-
transcribed:

```yaml
clahe:
  algorithm: opencv_clahe_v1
  input_quantization_bits: 8
  clip_limit: 0.5
  tile_grid_size: [4, 4]
  selected_by:
    split: validation
    primary_metric: body_ssim
    aggregation: patient_weighted
    search_config: configs/clahe_search.yaml
```

Frozen. `scripts/tune_clahe.py` refuses to overwrite it without an explicit
`--overwrite`, so CLAHE could not be quietly retuned once the CNN result
existed, and could not be once the U-Net result did either.

### Validation result: CLAHE versus no restoration

Same 6 patients, same 885 slices, same masks, same metric code. Patient-
weighted:

| Region | Metric | No restoration | CLAHE | Delta |
| --- | --- | --- | --- | --- |
| full | MAE | 0.016359 | 0.023570 | +0.007211 |
| full | MSE | 0.00072130 | 0.00123279 | +0.00051149 |
| full | PSNR | 31.4813 | 29.1561 | −2.3251 |
| full | SSIM | 0.781346 | 0.620769 | −0.160577 |
| body | MAE | 0.028187 | 0.036281 | +0.008094 |
| body | MSE | 0.00141527 | 0.00229926 | +0.00088399 |
| body | PSNR | 28.5620 | 26.4831 | −2.0789 |
| body | SSIM | 0.810430 | 0.768085 | −0.042345 |

**Did CLAHE beat the degraded baseline on the predeclared primary metric,
patient-weighted body SSIM? No.** It lost by 0.042345.

It also lost on body PSNR (−2.08 dB), full SSIM (−0.161), full PSNR
(−2.33 dB), and on MAE and MSE in both regions, where higher is worse. **No
metric moved in CLAHE's favour.** There is no qualification to add and no
angle from which this is an improvement.

### The paired per-patient deltas

The split means above say which number is larger. The paired deltas say
whether that holds patient by patient, and here they are unanimous:

| Subject | Group | Δ body PSNR | Δ body SSIM | Δ full PSNR | Δ full SSIM |
| --- | --- | --- | --- | --- | --- |
| 4 | B | −1.519655 | −0.021084 | −1.772764 | −0.140621 |
| 14 | B | −1.859560 | −0.040561 | −2.153464 | −0.171720 |
| 17 | B | −1.791796 | −0.040447 | −2.185182 | −0.175813 |
| 23 | A | −2.232165 | −0.057301 | −2.422420 | −0.156846 |
| 24 | A | −2.303587 | −0.049855 | −2.481504 | −0.151718 |
| 34 | A | −2.766645 | −0.044825 | −2.935498 | −0.166744 |
| **mean** | | **−2.078901** | **−0.042345** | **−2.325139** | **−0.160577** |
| **improved** | | **0 / 6** | **0 / 6** | **0 / 6** | **0 / 6** |

Zero patients improved on any metric. That unanimity is more informative than
the mean alone would be: a method that helped three patients and harmed three
could average to the same place while meaning something quite different. No
significance test is run and no p-value is computed — six patients is a small
descriptive sample, and this is model selection, not inference.

### Why CLAHE loses here

At clip 0.5 with 4×4 tiles CLAHE makes a smooth, low-amplitude, tile-scale
change to the image. Measured across all 885 validation slices, the mean
absolute change it makes to the degraded image is **0.012361** in normalized
units, with per-slice values of 0.009511 / 0.011743 / 0.018227 at the 5th,
50th and 95th percentiles
([clahe_validation_summary.json](outputs/metrics/clahe_validation_summary.json)).

What CLAHE has no mechanism for is noise. It performs no noise estimation and
no denoising: it has no model of what in a tile is structure and what is
perturbation. The degradation's noise is simply part of every local histogram,
and the histogram remapping redistributes it along with the structure. That
remapping is nonlinear, so the noise realization in the output is not the
input noise carried through numerically unchanged — it is altered, and at
harsher settings amplified, which is consistent with every harsher candidate
in the table scoring worse. Whatever the detailed mechanism, the measured
outcome is unambiguous: the output moved farther from the clean reference on
all eight reported metrics.

That is not a flaw in CLAHE, which is a contrast enhancement method and was
never a denoiser. The frozen degradation here is a **noise-only image-domain
corruption** — a signal-dependent (heteroscedastic), spatially correlated
Gaussian perturbation followed by clipping — with no blur, no contrast
compression and no streak artifacts of the kind a contrast method might have
something to work against.

The conclusion, kept to what was measured: **the tested CLAHE baseline did not
improve this frozen benchmark.** That is one algorithm over one predeclared
search grid on one degradation, not a statement about classical contrast
methods in general. It still sets a meaningful bar for the deep-learning
methods: they must beat no restoration, not merely beat CLAHE.

### What was and was not looked at

Tuning read the **validation** split only — `scripts/tune_clahe.py` has no
`--split` option at all. Visual QC read **training** slices only, after the
configuration had already been selected numerically, and nothing was adjusted
afterwards. No validation image was inspected visually.

**No test or stress image content was read during this milestone.** Every M6
command refuses those splits, and tests assert the refusal.

These are validation development results. They are not a final benchmark
result, and the final comparison on the held-out test split happens only once
every method decision is frozen.

## The learned-method data pipeline

Everything both learned methods trained on is defined here and nowhere
else: what one supervised sample is, how the pair is built, how patients are
weighted during training, and how validation is traversed. No architecture, no
optimizer, no loss, no learning rate. Implemented in
[src/ct_restoration/data/dataset.py](src/ct_restoration/data/dataset.py),
[src/ct_restoration/data/sampling.py](src/ct_restoration/data/sampling.py) and
[src/ct_restoration/data/loaders.py](src/ct_restoration/data/loaders.py), with
the policy recorded in
[configs/data_loader.yaml](configs/data_loader.yaml).

### One sample

One row of the frozen slice manifest:

| Field | Type | Role |
| --- | --- | --- |
| `degraded` | `float32` tensor `[1, 256, 256]`, in [0, 1] | **model input** |
| `clean` | `float32` tensor `[1, 256, 256]`, in [0, 1] | **supervised target** |
| `subject_id` | `str` | auditing and grouping |
| `sample_key` | `str` | the stable portable slice identity; one input to the derivation of that slice's degradation seed |
| `source_archive` | `str` | auditing |
| `acquisition_group` | `str` | auditing |
| `geometric_slice_index` | `int` | auditing |

The metadata is for auditing and grouping. A model receives `degraded` and
nothing else.

What the sample does **not** contain is as much of the contract as what it
does: no body mask, no SSIM interior mask, no clean HU array, no segmentation,
no DICOM UID, no absolute filesystem path. The evaluation body mask is
computed from the clean reference, so handing it to a model would feed the
method a region derived from the very thing it is being asked to predict. The
clean image reaches the model only as the target. A test asserts the exact
field list, and the audit re-checks it on every one of the 5045 train and
validation slices.

### The pair is a pure function of the slice

```
DICOM -> HU -> frozen M1 preprocessing      -> clean
clean + frozen M4 degradation + sample key  -> degraded
```

Both halves are rebuilt on demand by calling the existing preprocessing and
degradation code. The formulas are not reimplemented, and no new noise
realization is created. The per-slice seed is derived by SHA-256 over the
algorithm version, the global seed and the sample key — the key is the slice's
stable portable identity and one input to that derivation, not the numeric
seed itself.

This is the part that matters most for training. **Sampling order changes from
epoch to epoch; the corruption does not.** A given slice carries the same
frozen noise the tenth time it is drawn as the first. Nothing depends on the
epoch, the batch, the worker, the access count, the sampler seed, or the
global NumPy or PyTorch RNG.

The reason is a scoping decision, not a claim that the alternative is
invalid. This benchmark defines exactly **one** degraded counterpart per clean
slice. Holding that pair fixed makes the training inputs reproducible and lets
the CNN and the U-Net inherit an identical input-target mapping, so a
difference between them cannot come from the training pairs: it comes from
the model side, meaning the architecture together with the training run of
that model. Re-drawing the
corruption every epoch is a perfectly legitimate way to train a denoiser — it
is a standard stochastic augmentation — but it is a different experiment: it
changes the training distribution and introduces a second stochastic policy to
account for. That is worth studying on its own, and it is deliberately not an
experimental factor in the first learned benchmark here.

It also ties training to evaluation. The Dataset calls the same frozen
preprocessing and degradation functions the evaluation path calls, for every
item; the audit additionally rebuilds eight deterministic probes per split
through those functions independently and compares bytes, with zero clean or
degraded mismatches. The Dataset's canonical order, natural subject order then
geometric slice index, is the same order the committed per-slice metric tables
already use.

No cache is written. No `.npy`, `.pt`, PNG, LMDB, HDF5 or WebDataset archive
exists: a cache would be a second representation of the benchmark data, and
every claim made about it would have to be re-proved. If DICOM decoding later
turns out to be a material training bottleneck, it can be optimized then,
against this contract.

### Why training is patient-balanced

The 25 training patients hold **78 to 294 slices each**. Visiting every slice
once per epoch — ordinary `shuffle=True` — would give the longest scan almost
four times as many training draws as the shortest, purely because of how long
that patient's scan was. Scan length is an acquisition-protocol fact, not a
statement about how much a patient should matter.

Patient is already the experimental unit everywhere else here: the split is
patient-level, and every reported metric averages slices within a patient
before averaging patients. Slice-weighted training would be the one place the
experiment quietly switched units.

So the canonical training policy approximately equalizes **per-patient
sampling exposure**: within an epoch every training patient is drawn 166 or
167 times, via `PatientBalancedSampler` (`patient_balanced_v1`).

Three things this does not mean. It does **not** equalize optimizer
influence: equal draw counts are not equal gradient contributions, because
how far one draw moves the weights depends on that slice's content and on
the current model, not only on how often the patient is drawn. It does
**not** make slices statistically independent — adjacent slices of one CT
scan remain highly correlated, and nothing here changes that; it only stops
scan length from directly determining how often a patient is drawn. And it
does **not** balance acquisition groups: training holds 10 group-A and 15
group-B patients, so equal per-patient draw counts leave group B with 15/25
of the draws, which is the cohort's own composition. Forcing A and B to 50/50 would be a second
intervention the frozen cohort definition does not justify. Source archive and
slice position are likewise left alone.

`torch.utils.data.WeightedRandomSampler` is deliberately not used. It balances
patients only *in expectation*, so any single epoch can over- or under-draw a
patient and an audit could only report what happened afterwards. The counts
here are enforced exactly.

### 4160 draws, 166 or 167 per patient

One epoch is exactly the training set size, 4160 draws — not
`25 × max_patient_slices`, and not a new invented number.

```
divmod(4160, 25) == (166, 10)
10 patients x 167  +  15 patients x 166  ==  4160
```

Patient exposure therefore differs **by at most one sample** inside an epoch.

The ten extra draws rotate. The canonical patient list is cycled by
`(epoch × remainder) % n_patients` positions and the first `remainder`
patients of that rotation take the extra, so the start walks 0, 10, 20, 5, 15
and then repeats: over any five consecutive epochs every patient takes the
extra draw exactly twice. Nothing depends on the order the manifest CSV
happened to be written in — the patient list is derived from the subject ids.

Within a patient, draws come from shuffled full passes: permute all of that
patient's slices, consume the permutation, reshuffle for another pass if more
are needed.

* **A short scan repeats slices, but only after covering all of them.** The
  shortest training patient has 78 slices against a quota of 166 or 167, so it
  takes three shuffled passes; the audit confirms the first 78 draws are a
  full permutation, with zero repeats before full coverage.
* **A long scan does not use every slice in one epoch.** A 294-slice patient
  contributes 166 or 167 of them, drawn without repetition within the epoch. A
  new deterministic permutation is derived for each epoch, so the subsets
  generally change and coverage broadens across epochs. The v1 sampler keeps
  no cross-epoch cursor, so it does **not** guarantee that every slice has
  appeared by any particular finite epoch.

Measured at audit seed 2026 on the real training split:

| Epoch | Draws | Patients | Per patient | Unique slices | Repeated draws | Slices not drawn |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 4160 | 25 | 166 x15, 167 x10 | 3167 | 993 | 993 |
| 1 | 4160 | 25 | 166 x15, 167 x10 | 3172 | 988 | 988 |
| 2 | 4160 | 25 | 166 x15, 167 x10 | 3172 | 988 | 988 |

Finally the whole epoch sequence is shuffled with a local RNG, so batches
intermix patients instead of arriving in contiguous per-patient blocks.

Every draw comes from a private generator seeded by
`SHA-256(algorithm_version, seed, epoch, stream)`. The global NumPy RNG is
never read or written and the sampler uses no PyTorch randomness at all, so
the sampler's randomness and the per-slice degradation's randomness are fully
independent: changing which slice is drawn is allowed, changing what
corruption that slice carries is not.

`sampler.set_epoch(n)` selects the epoch. Same seed and epoch reproduce the
sequence exactly; a different epoch or a different seed gives a different one;
returning to epoch 0 reproduces epoch 0, because the sequence is derived from
the epoch number rather than accumulated in sampler state. The seed is a
constructor argument, not a frozen constant — a training run owns its seed, and
the sampler must not silently pin the eventual network training seed.

### Validation is never resampled

Validation exists to produce a comparable number, so it visits **all 885
slices exactly once**, in the frozen canonical manifest order, with
`shuffle=False`, `drop_last=False` and no patient balancing. Balancing it
would change which slices contribute and make the figure incomparable with the
degraded-baseline and CLAHE numbers already measured.

`drop_last=False` matters on the training side too: dropping a partial final
batch would silently discard whichever patients landed at the end of the
shuffled epoch, so the realized per-patient counts would stop matching the
audited ones.

Batch size is a caller argument, not a benchmark parameter — the CNN and the
U-Net each declare their own in their frozen config, and both chose 32. **Batching changes grouping, not
sampling.** The audit flattens both loaders at batch sizes 1 and 17 (neither
divides 885 or 4160, so a ragged final batch is exercised) and gets identical
ordered sample-key sequences; validation matches the canonical manifest order
and covers all 885 exactly once.

Worker count does not change anything either: pairs are deterministic and the
sampler runs in the main process. On this Windows environment a comparison at
`num_workers=0` against `num_workers=2` gave identical sample keys and
byte-identical tensors.

### No augmentation, no model-specific normalization

Neither is used, on purpose. Both learned methods were trained without
either.

No flips, rotations, crops, intensity jitter, extra noise, mixup or CutMix.
The first learned benchmark should establish whether a model can learn the
frozen restoration problem at all; augmentation would add another experimental
factor to attribute the result to.

No transform to [-1, 1] and no z-scoring either. The learned models receive
the same frozen [0, 1] windowed representation every current method already
uses. An architecture that needs a different internal normalization must
declare it as part of that model.

### What was and was not read

The Dataset construction helper accepts **train** and **validation** and
refuses **test** and **stress** with the same hold-out error the evaluation
commands use; tests assert the refusal, including before any file is opened.
PyTorch existing is not a reason to loosen the gate — the final benchmark will
build its held-out datasets explicitly, in the milestone that runs it.

`scripts/audit_dataset.py` has no `--split` option. It opened all 4160
training and 885 validation slices and wrote
[outputs/audit/dataset_dataloader_summary.json](outputs/audit/dataset_dataloader_summary.json).
Two different checks, at two different scopes, and the distinction matters:

* **Every one of the 5045 slices** was scanned for the tensor and sample
  contract — 0 shape, dtype, finiteness and range failures on either split, 0
  forbidden fields, 0 non-portable sample keys.
* **Eight deterministic probes per split** were also rebuilt independently
  through the frozen preprocessing and degradation functions and compared
  byte for byte: 0 clean and 0 degraded mismatches, plus 0 mismatches on
  repeated access and under a reseeded global NumPy and PyTorch RNG. The
  probes are a spot check; what covers every item is that the Dataset calls
  those same frozen functions for all of them.

**No test or stress image content was read.** Regenerating the summary is
byte-identical.

Everything above describes the data layer only. The first model trained on it
is the subject of the next section.

## The small residual CNN

The first learned method: a deliberately small convolutional network,
implemented in
[src/ct_restoration/models/cnn.py](src/ct_restoration/models/cnn.py), trained
by [scripts/train_cnn.py](scripts/train_cnn.py) and scored by
[scripts/evaluate_cnn.py](scripts/evaluate_cnn.py) through the same frozen
benchmark harness the degraded baseline and CLAHE went through.

### What the model is

| | |
| --- | --- |
| algorithm | `residual_cnn_v1` |
| layers | 5 × `Conv2d` 3×3, stride 1, padding 1 |
| hidden channels | 32 |
| activation | ReLU between convolutions, **not** after the last one |
| normalization | none — no batch norm, no dropout |
| trainable parameters | **28,353** |
| receptive field | **11 pixels** |
| input | the degraded slice only, `[B, 1, 256, 256]` in [0, 1] |
| output | `clamp(degraded + correction, 0, 1)` |

The receptive field is arithmetic, not a claim about anatomy: five stacked
3×3 stride-1 convolutions see `1 + 5 × 2 = 11` pixels. Every output pixel is
therefore a function of an 11×11 neighbourhood of the input and of nothing
further away. That is a real constraint on what this architecture can do —
it cannot use context beyond that window, whatever the context might be worth
— and it is the main structural difference between this model and the
[lightweight U-Net](#the-lightweight-u-net), whose deepest path sees 44x44.

There is no activation after the final convolution. A sigmoid or tanh there
would bound the output, but it would also bound the *correction*, place the
easy answer (predict nothing) at a saturating point of the nonlinearity, and
make the identity initialization below impossible. The output range is
enforced by an explicit clamp instead, which is a declared post-processing
step rather than a property of the architecture.

### Why it predicts a correction, not an image

The last convolution is **initialized to exactly zero** — zero weights, zero
bias. At initialization the network's correction is identically 0, so its
output is `clamp(degraded + 0, 0, 1)`, which is the degraded image unchanged.

Two things follow, and both matter more than they might look.

**The model starts as the baseline and has to earn every departure from it.**
Predicting the clean image from scratch means learning to reproduce the
anatomy as well as remove the perturbation; predicting a correction means the
anatomy is already there and only the difference has to be learned. The
restoration problem here is a small perturbation of the identity, so that is
the cheaper parameterization by a wide margin.

**It gives a falsifiable plumbing check.** A zero-initialized model is
feeding the degraded image straight through, so it must score what the
degraded baseline scored, up to floating-point reduction order. If it does
not, the learned-method evaluation path and the frozen benchmark disagree
about something — the slice set, the mask, the aggregation — and every later
number is suspect. That check runs before training starts, below.

### The training recipe, frozen before the run

Written into [configs/cnn.yaml](configs/cnn.yaml)
(SHA-256 `ac4bf06c3957b799…`) before the first training run, and the config's
hash is recorded in every artifact the run produced.

| | |
| --- | --- |
| loss | full-frame **L1** on the **raw, unclamped** restoration |
| optimizer | Adam, lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| schedule | none — constant learning rate |
| epochs | 30, all of them, no early stopping |
| batch size | 32 |
| sampling | `patient_balanced_v1`, 4160 draws per epoch |
| augmentation | none |
| gradient clipping | none |
| mixed precision | none |
| seed | 2026 |

Three of these are worth stating plainly rather than leaving in a table.

**The loss is computed on the raw output, before the clamp.** Clamping first
would zero the gradient everywhere the prediction had left [0, 1], so the
optimizer would get no signal precisely where the model is most wrong. The
clamp belongs to evaluation, not to the objective.

**The loss is full-frame, and that is a choice with consequences.** It weights
every pixel equally, including the large clipped-background region, which is
not what a radiologist would weight. It was chosen because it is the simplest
objective that is not tuned to the metric it will be judged on, and it is not
claimed to be clinically optimal. Body-region and perceptual objectives are
exactly the kind of thing a later milestone could compare — but comparing them
is a search, and this milestone deliberately runs none.

**No search was run over any of it.** One architecture, one learning rate, one
batch size, one loss, one seed. Everything above was declared before the first
gradient step and nothing was changed after seeing a validation number. That
is the whole reason the result below can be read at face value; it is also why
it is a weaker result than a tuned one would look.

### One predeclared checkpoint criterion

**Lowest patient-weighted validation full-frame MAE, computed from the clamped
prediction, over epochs 1..30, ties broken towards the earlier epoch.** One
metric, declared in the config before training.

Epoch 0 is recorded but never eligible: it is the identity check, and a
selection rule that could return it would be able to conclude "do nothing" by
accident.

Selecting on one predeclared number is the point. Scanning across MAE, MSE,
PSNR, SSIM, full frame and body and then choosing the epoch that looks best
somewhere would be eight chances to find a favourable epoch and no way to say
afterwards which one the method actually committed to. The other seven metrics
are computed **once, afterwards, for the already-selected checkpoint**. They
are reported, not optimized against.

### The epoch-0 identity check

Before any training, the zero-initialized model was run through the full
validation evaluation path:

| | |
| --- | --- |
| zero-initialized CNN, patient-weighted validation full MAE | 0.016358921897 |
| committed degraded baseline, same figure | 0.016358921877 |
| absolute difference | **2.03e-11** |
| tolerance | 1e-6 |

The baseline value is **read from the committed
[degraded_baseline_validation_patients.csv](outputs/metrics/degraded_baseline_validation_patients.csv)**,
not from a number typed into the script, so the comparison cannot pass by
agreeing with a stale literal. The training command exits non-zero and refuses
to train if this check fails.

The difference is not exactly zero and is not expected to be: the benchmark
computes its figure with NumPy metric code and the training script with a
torch reduction, so the two differ by floating-point summation order. A real
plumbing mismatch — a different mask, a different slice set, a different
normalization — would be orders of magnitude larger than 2e-11. The summary
records the exact value as a string as well as a rounded float, because the
tracked JSON rounds to 10 decimal places and would otherwise print this as a
flat `0.0` and read as exact equality.

### The training run

30 epochs on one RTX 5070 Ti, 4160 patient-balanced draws per epoch, validation
evaluated in full after every epoch. The per-epoch record is
[outputs/runs/cnn_seed2026/training_history.csv](outputs/runs/cnn_seed2026/training_history.csv).

| epoch | train L1 (raw) | validation patient-weighted full MAE | |
| --- | --- | --- | --- |
| 0 | — | 0.01635892 | identity check, ineligible |
| 1 | 0.01614127 | 0.01225671 | |
| 5 | 0.01126369 | 0.01003056 | |
| 10 | 0.01097686 | 0.00990126 | |
| 15 | 0.01081030 | 0.00962256 | |
| 20 | 0.01069439 | 0.00966141 | |
| 25 | 0.01060133 | 0.00938976 | |
| **29** | **0.01053796** | **0.00935309** | **selected** |
| 30 | 0.01052905 | 0.00953269 | |

Training loss fell monotonically apart from small increases at epochs 8, 24
and 28. The validation metric oscillated by roughly ±0.0002 from epoch to
epoch while trending downwards throughout, and epoch 30 sits on an upward
wobble rather than at the bottom — which is why the criterion selects 29 and
why "use the last epoch" would have been a slightly worse rule here by luck as
much as by anything else.

**No overfitting is visible in this run**, in the narrow sense that the
validation curve never turned around and rose while training loss kept
falling. Thirty epochs of a 28k-parameter model on 4160 slices is not a
regime where overfitting would be the expected failure mode, and the absence
of it in one run is a description of that run, not a general property.

### Validation result: CNN versus no restoration

Same 6 patients, same 885 slices, same masks, same metric code, same frozen
degradation. Patient-weighted, every patient equally weighted:

| Region | Metric | No restoration | CLAHE | **CNN** | CNN − no restoration |
| --- | --- | --- | --- | --- | --- |
| full | MAE | 0.016359 | 0.023570 | **0.009353** | −0.007006 |
| full | MSE | 0.00072130 | 0.00123279 | **0.00034916** | −0.00037215 |
| full | PSNR | 31.4813 | 29.1561 | **34.6583** | **+3.1771** |
| full | SSIM | 0.781346 | 0.620769 | **0.953860** | **+0.172515** |
| body | MAE | 0.028187 | 0.036281 | **0.019940** | −0.008247 |
| body | MSE | 0.00141527 | 0.00229926 | **0.00074864** | −0.00066662 |
| body | PSNR | 28.5620 | 26.4831 | **31.3631** | **+2.8011** |
| body | SSIM | 0.810430 | 0.768085 | **0.897097** | **+0.086667** |

**Did the CNN beat no restoration? Yes, on all eight metrics.** Full-frame
PSNR improved by 3.18 dB and body PSNR by 2.80 dB; full-frame SSIM rose from
0.781 to 0.954.

The selection metric deserves one note. Full MAE here, 0.009353093, is the
same quantity the checkpoint was chosen on, 0.0093530865 — they agree to
6.5e-9, again a summation-order difference between two code paths. The other
seven columns were never used to choose anything.

Spread across the six patients, and the secondary slice-weighted figure:

| Metric | patient std | min | max | slice-weighted |
| --- | --- | --- | --- | --- |
| full PSNR | 0.5266 | 33.7300 | 35.1811 | 34.7484 |
| body PSNR | 0.4697 | 30.7315 | 31.7839 | 31.5004 |
| full SSIM | 0.008246 | 0.944020 | 0.968284 | 0.954181 |
| body SSIM | 0.013115 | 0.882705 | 0.917179 | 0.898668 |

The slice-weighted column is descriptive only and is never the headline
number.

Descriptive acquisition-group breakdown, three patients each:

| Group | full PSNR | body PSNR | full SSIM | body SSIM |
| --- | --- | --- | --- | --- |
| A | 34.8704 | 31.7581 | 0.953482 | 0.899947 |
| B | 34.4463 | 30.9681 | 0.954238 | 0.894248 |

With n = 3 per group there is nothing to conclude, no significance test is
run, and no claim is made about acquisition settings.

### The paired per-patient deltas

The split means say which number is larger. The paired deltas say whether it
holds patient by patient:

| Subject | Group | Δ body PSNR | Δ body SSIM | Δ full PSNR | Δ full SSIM |
| --- | --- | --- | --- | --- | --- |
| 4 | B | +2.719899 | +0.070557 | +2.986792 | +0.158183 |
| 14 | B | +2.621091 | +0.086285 | +3.005278 | +0.181888 |
| 17 | B | +2.511159 | +0.074731 | +3.019941 | +0.177856 |
| 23 | A | +2.919770 | +0.100726 | +3.271632 | +0.171625 |
| 24 | A | +3.064383 | +0.100431 | +3.406450 | +0.170690 |
| 34 | A | +2.970161 | +0.087273 | +3.372391 | +0.174846 |
| **mean** | | **+2.801077** | **+0.086667** | **+3.177081** | **+0.172515** |
| **improved** | | **6 / 6** | **6 / 6** | **6 / 6** | **6 / 6** |

Every patient improved on every one of the eight metrics; the full table
including MAE and MSE is
[cnn_vs_degraded_baseline_validation_patient_deltas.csv](outputs/metrics/cnn_vs_degraded_baseline_validation_patient_deltas.csv).
Unanimity across patients is worth more than the mean alone, because a method
that helped four patients and harmed two could average to a similar place
while behaving quite differently.

**No significance test is run and no p-value is computed.** Six patients is a
small descriptive sample, these are validation numbers used for development,
and one seed is one seed.

### CNN versus CLAHE

Descriptive context only — the bar a restoration method has to clear is *no
restoration*, not CLAHE:

| Region | Metric | Δ (CNN − CLAHE) | improved |
| --- | --- | --- | --- |
| full | PSNR | +5.5022 | 6 / 6 |
| full | SSIM | +0.333091 | 6 / 6 |
| body | PSNR | +4.8800 | 6 / 6 |
| body | SSIM | +0.129013 | 6 / 6 |

This gap is large, and it is largely a statement about the problem rather than
about the two methods' relative sophistication. The frozen degradation is a
noise-only corruption; CLAHE is a contrast method with no noise model, applied
to a problem with nothing for it to correct, and it damaged the image. The CNN
was trained on exactly this corruption. A comparison arranged that way is not
a general ranking of learned against classical methods.

### What the clamp is doing

The reported metrics come from the clamped output, so the clamp is part of the
method and its behaviour is worth measuring rather than assuming.

Two quantities have to be kept apart here. They are easy to conflate and they
are not the same number:

* the **predicted correction**, `raw − degraded` — what the network actually
  asked for, measured on the **unclamped** output;
* the **post-clamp change**, `restored − degraded` — the part of that request
  which survived into the image the metrics were computed on.

From
[cnn_validation_summary.json](outputs/metrics/cnn_validation_summary.json),
over all 885 validation slices:

| | |
| --- | --- |
| raw minimum, before clamping | −0.025799 |
| raw maximum, before clamping | 1.056146 |
| fraction of pixels raw < 0 | **0.508982** |
| fraction of pixels raw > 1 | 0.005719 |
| fraction changed by the clamp | 0.514702 |
| **mean absolute predicted correction**, from the raw output | **0.012068** |
| per-slice, q05 / q50 / q95 | 0.010319 / 0.011903 / 0.014088 |
| mean absolute post-clamp change vs degraded | 0.011788 |
| per-slice, q05 / q50 / q95 | 0.009980 / 0.011617 / 0.013836 |
| non-finite outputs | 0 |

Half the frame is modified by the clamp, which sounds alarming until the
magnitudes are read alongside it: the raw output never goes below −0.026 or
above 1.056, so these are small overshoots, not wild predictions.

The two correction figures make that precise. Because the degraded image
always lies inside [0, 1], the gap between them is exactly the overshoot the
clamp discarded: 0.012068 − 0.011788 = **0.000280** per pixel across the
whole frame, or about 0.00054 spread over the 51.5% of pixels that were
clamped at all. So the clamp touches half the frame and removes very little
from it — which is the honest reading, and it is only visible because the two
quantities are now reported separately.

The mean body-pixel coverage of these slices is 0.4629, so roughly 53.7% of
the frame lies outside the body mask — a fraction of similar size to the
50.9% of pixels that are raw-negative. That similarity is **consistent with**
the clamp acting mostly on background, but the tracked summary does not break
the below-zero pixels down by region, so this is a consistency observation and
nothing more. It would take a per-region diagnostic to establish it, and none
was run.

These numbers are descriptive. **None of them is an optimization objective**,
and no model, loss or config decision was made on the basis of them.

### Reproducibility of this run

The claim is scoped: **same repository, same config, same environment, same
hardware and same seed reproduce this run.** Bitwise identity across different
GPUs, drivers or PyTorch builds is **not** claimed, because floating-point
reduction order in cuDNN kernels depends on the hardware and on the algorithm
chosen.

Within that scope it was checked rather than asserted:

* The training run was executed **twice**, start to finish. Both runs produced
  a byte-identical `training_history.csv`
  (`fb9546cf26c4188c…`) and a byte-identical checkpoint
  (`fe3cdc42c72dea16…`). The second run used a script that additionally
  performs the round-trip check below and records the identity difference at
  full precision; nothing in the training path differed, and the identical
  hashes are the evidence that nothing did.
* The evaluation has been run repeatedly against that checkpoint, and all
  four metric tables — the per-slice, per-patient and two paired-delta CSVs —
  came out byte-identical every time. The summary JSON changed exactly once,
  when the correction diagnostics were corrected and the two verification
  records below were added; no metric in it moved.
* `torch.use_deterministic_algorithms(True)` was enabled in strict mode — not
  `warn_only` — with `cudnn.benchmark=False`, `cudnn.deterministic=True` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Python, NumPy, torch CPU and torch CUDA
  RNGs are all seeded from 2026.
* CUDA is required. The training command refuses to run on CPU rather than
  silently producing a canonical result on different hardware.
* After selection, the saved checkpoint was reloaded in-process and compared
  against the model still in memory: **10 state-dict tensors, 0 mismatches**,
  and **0 differing pixels** on four deterministic probe images. A checkpoint
  that cannot reproduce the model it was written from is a failed run, however
  good the validation curve looked, because every number above comes from
  reloading that file.

Two further conditions are enforced by the evaluation command rather than
checked by hand, because a report nobody acts on is not a check:

* **Checkpoint provenance, before a single validation image is opened.** The
  command refuses to score a checkpoint it cannot prove is the frozen run's.
  Fifteen conditions are checked, each against something computed
  independently of the checkpoint: the frozen config's bytes are re-hashed,
  the checkpoint file's bytes are re-hashed, and the tracked training history
  is re-run through the predeclared selection rule to re-derive epoch 29
  rather than trusting the epoch the file claims. Any mismatch exits without
  reading an image.
* **Sample alignment, before anything is written.** 885 rows, 885 unique
  sample keys, 0 duplicates, and 0 missing, 0 extra and 0 order mismatches
  against both the degraded-baseline and CLAHE slice tables. A paired delta
  is only paired if both sides ran on the same slice; if any of that fails
  the command writes no artifact at all.

The checkpoint binary itself is **git-ignored**; its SHA-256 is tracked in
[run_summary.json](outputs/runs/cnn_seed2026/run_summary.json) instead. It
holds plain tensors and scalars only, so it loads with `weights_only=True` and
restoring it never executes code from the file.

### What this result is not

**It is one seed.** The single most important limitation *of this section*.
One training run shows what that run did, so no statement here should be
read as "the residual CNN achieves 34.66 dB".

The run-to-run spread was unmeasured when this section was written, and
[Milestone 10](#milestone-10-the-multi-seed-result) has since measured it:
across five predeclared training seeds the CNN's full-frame PSNR ranged from
34.5617 to 34.6668 dB (sample SD 0.0429 dB). This run, seed 2026, scored
34.658340 dB, the second highest of the five. Seed 2026 was the project's
canonical seed, fixed before any training run existed; it was not chosen for
its result. The figure above is left unedited because it is what the
Milestone 8 run produced.

**It is validation, not test.** These numbers were computed on the split the
checkpoint was selected on. Selecting one of 30 epochs on validation makes a
validation figure optimistic, even with a single predeclared criterion. The
test split is still sealed.

**It is not a denoising result in general.** The model was trained on one
frozen synthetic corruption and measured on the same one. Nothing here
indicates how it behaves on real low-dose CT noise, on a different
reconstruction kernel, or on anatomy outside this cohort.

**No metric here indicates clinical adequacy.** PSNR and SSIM improvements are
not diagnostic-quality improvements, a PSNR difference is in decibels and
never a percentage, and no reader study, no lesion-detection task and no
clinical evaluation of any kind has been performed.

**No latency has been measured.** The research question asks about inference
cost and that part is unanswered for every method.

### What was and was not looked at

Training and validation image content was read numerically. **No test or
stress image content was read** — `scripts/train_cnn.py` and
`scripts/evaluate_cnn.py` both refuse those splits through the same hold-out
gate every earlier command uses, and both summaries record
`test_images_read: 0` and `stress_images_read: 0`.

**No validation image was inspected visually at any point.** Visual QC
(`scripts/qc_cnn.py`) has no `--split` option: it reads **training** slices
only, and it was run *after* the checkpoint had already been selected
numerically. Its panels — clean, degraded, restored, the correction, and the
remaining error — are written to a git-ignored directory, and nothing about
the model was changed after looking at them. Looking at held-out images and
then adjusting something is how a held-out estimate quietly becomes a fitted
one.

Nothing in the architecture, the loss, the optimizer, the learning rate, the
epoch count or the batch size was changed after the first validation number
existed. The one change made to the training script after its first complete
run added a checkpoint round-trip check and recorded the identity difference
at full precision; the run was then repeated from scratch rather than patched
after the fact, and no scientific hyperparameter moved.

These are validation development results for one seed; the spread across five
seeds is reported in [Milestone 10](#milestone-10-the-multi-seed-result). The
final comparison on the held-out test split happens only once every method
decision is frozen.

## The lightweight U-Net

The second learned method, and the one architectural question this benchmark
was built to ask: does multi-scale context help? Implemented in
[src/ct_restoration/models/unet.py](src/ct_restoration/models/unet.py),
trained by [scripts/train_unet.py](scripts/train_unet.py) and scored by
[scripts/evaluate_unet.py](scripts/evaluate_unet.py) through the same frozen
benchmark every other method went through.

### What the model is

| | |
| --- | --- |
| algorithm | `lightweight_residual_unet_v1` |
| levels | 2 downsampling steps |
| base channels | 16, doubling per level: 16 → 32, bottleneck 64 |
| block | 2 × `Conv2d` 3×3, ReLU after each |
| downsampling | `MaxPool2d` 2×2 stride 2 |
| upsampling | `ConvTranspose2d` 2×2 stride 2 |
| skip connections | 2, **concatenation** along channels |
| normalization | none — no batch norm, no dropout, no attention |
| head | `Conv2d` 1×1, zero-initialized |
| trainable parameters | **116,753** |
| maximum deepest-path receptive field | **44 × 44 pixels** |
| output | `clamp(degraded + correction, 0, 1)` |

Eleven `Conv2d` and two `ConvTranspose2d` layers, all with bias. The shape:

```
256²×1  --conv,conv-->  256²×16  ------------------ skip 1 ------------------+
                |                                                            |
            maxpool 2×2                                                      |
                v                                                            |
128²×16 --conv,conv-->  128²×32  ------ skip 2 ------+                       |
                |                                     |                      |
            maxpool 2×2                                v                     v
                v                                  concat 64             concat 32
 64²×32 --conv,conv-->   64²×64  --up--> 128²×32 ------+--conv,conv--> ... --+--conv,conv--> 256²×16 --1×1--> correction
                        (bottleneck)
```

### Encoder, bottleneck, decoder

The **encoder** halves the spatial resolution twice while doubling the
channel count. That trade is the whole point of the shape: after pooling, one
feature-map pixel summarises a larger patch of the original image, so the
same 3×3 convolution now relates things that were further apart. The extra
channels are the capacity to describe what those larger patches contain.

The **bottleneck** is where the representation is coarsest — 64×64 at 64
channels — and therefore where a single convolution reaches furthest across
the image.

The **decoder** brings the resolution back with transposed convolutions,
which learn the upsampling rather than interpolating it, and which here are
2×2 stride 2 and so exactly non-overlapping: each output pixel comes from one
input pixel.

### Skip connections, and why concatenation

Pooling coarsens the representation, and fine spatial detail is what a
restoration task most needs. The skip connections carry higher-resolution
encoder features around that bottleneck: each encoder block's output is saved
and joined to the matching decoder stage, so the decoder sees the coarse
wide-context features it computed **and** features at the finer scale that
never passed through the coarsest representation.

Worth being exact about what a skip is. It is **not** the raw input pixels,
and it does not restore the detail pooling removed. It is a feature map
produced by that encoder block's two convolutions — already transformed,
just not yet downsampled. So the guarantee is access to fine-scale feature
information, not perfect preservation of anything.

They are **concatenated, not summed**, and the widths are checked in the
tests: decoder stage 1 receives 32 upsampled + 32 skip = **64 channels**,
stage 2 receives 16 + 16 = **32 channels**. Adding them instead would force
the network to treat a fine-detail feature and a wide-context feature as the
same kind of quantity and would commit to a fixed one-to-one mixing.
Concatenating hands both to the next convolution and lets it learn the
mixing, at the cost of twice as many channels to convolve over.

### The 44×44 receptive field, stated carefully

Accumulated through the deepest path — a size-preserving convolution adds
`(k−1)·jump`, a 2×2 stride-2 pool adds `jump` and doubles it, and a
non-overlapping 2×2 stride-2 transposed convolution halves the jump and adds
no extent:

| after | field | jump |
| --- | --- | --- |
| encoder 1 (conv, conv) | 5 | 1 |
| pool 1 | 6 | 2 |
| encoder 2 (conv, conv) | 14 | 2 |
| pool 2 | 16 | 4 |
| bottleneck (conv, conv) | 32 | 4 |
| up-conv 1 | 32 | 2 |
| decoder 1 (conv, conv) | 40 | 2 |
| up-conv 2 | 40 | 1 |
| decoder 2 (conv, conv) | **44** | 1 |
| 1×1 head | 44 | 1 |

Three things this number is not. It is the **maximum over paths, not the only
path**: the skip connections deliberately provide shallower routes carrying
smaller-scale local information, and an output pixel's value mixes
contributions from all of them. It is a statement about **pixels**, not
anatomical coverage or clinical context. And it is not latency — no method in
this benchmark has had its inference cost measured yet.

### Residual, and identical to the CNN where it counts

```
raw_restored = degraded + UNet(degraded)
restored     = clamp(raw_restored, 0, 1)
```

The same image-level parameterization as the residual CNN, so the two methods
differ inside the box and nowhere else. The 1×1 head is zero-initialized, so
before training the correction is identically zero and the untrained U-Net
**is** the no-restoration baseline.

Everything on the training side was copied from
[configs/cnn.yaml](configs/cnn.yaml) unchanged into
[configs/unet.yaml](configs/unet.yaml) (SHA-256 `8baba29199c516ac…`, frozen
before the first real gradient step):

| | |
| --- | --- |
| loss | full-frame **L1** on the **raw, unclamped** restoration |
| optimizer | Adam, lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| schedule | none |
| epochs | 30, all of them, no early stopping |
| batch size | 32 |
| sampling | `patient_balanced_v1`, 4160 draws per epoch, 25 patients |
| validation | all 885 slices, canonical order, exactly once |
| augmentation | none |
| seed | 2026 |
| checkpoint rule | lowest patient-weighted validation full-frame MAE, epochs 1..30, ties to the earlier epoch |

**None of it was adjusted for this architecture**, and that is the
methodological point: if the U-Net had also been given its own learning rate
or its own epoch count, a difference in the result could be any of those
rather than the model.

To be precise about what that does and does not mean: **neither model's
recipe was ever hyperparameter-tuned.** The CNN's values were one predeclared
development configuration, chosen before any result existed and never
searched over, and the U-Net inherits them unchanged. So the accurate
limitation is that **the U-Net inherits the CNN benchmark's predeclared
training recipe rather than receiving architecture-specific tuning** — it is
measured under that recipe, not at its best. A configuration predeclared for
a 28k-parameter CNN need not be a good one for a 117k-parameter U-Net.
Tuning either would be a search, and no milestone so far has run one.

> **Erratum, and why the frozen file still reads differently.** A comment
> inside [configs/unet.yaml](configs/unet.yaml) describes the inherited
> values as "a recipe tuned for a 28k-parameter CNN". That phrasing is
> wrong: no hyperparameter search was ever run, for either model. The file
> is **not** edited to fix it, because its SHA-256 is recorded inside the
> checkpoint and the run summary, and the evaluation's provenance gate
> re-hashes the file and refuses to score a checkpoint whose recorded hash
> no longer matches. Correcting a comment would therefore invalidate the
> Milestone 9 result unless the model were retrained. The statement above is
> the accurate one; the config comment is a known wording defect in a frozen
> artifact, recorded here rather than silently repaired.

### The epoch-0 identity check

| | |
| --- | --- |
| zero-initialized U-Net, patient-weighted validation full MAE | 0.016358921897 |
| committed degraded baseline, same figure | 0.016358921877 |
| absolute difference | **2.03e-11** |
| tolerance | 1e-6 |

Read from the committed
[degraded_baseline_validation_patients.csv](outputs/metrics/degraded_baseline_validation_patients.csv),
never a typed literal. The difference is **bit-identical to the CNN's**,
which is the strongest available evidence that both architectures reduce to
exactly the same identity path and are being scored by exactly the same code.

### The training run

30 epochs on one RTX 5070 Ti;
[outputs/runs/unet_seed2026/training_history.csv](outputs/runs/unet_seed2026/training_history.csv)
has every epoch. Beside the CNN's run, which used the same data in the same
order:

| epoch | CNN train L1 | CNN val MAE | U-Net train L1 | U-Net val MAE |
| --- | --- | --- | --- | --- |
| 0 | — | 0.01635892 | — | 0.01635892 |
| 1 | 0.01614127 | 0.01225671 | 0.01397254 | 0.01089852 |
| 5 | 0.01126369 | 0.01003056 | 0.01090295 | 0.00979053 |
| 10 | 0.01097686 | 0.00990126 | 0.01059083 | 0.00943554 |
| 15 | 0.01081030 | 0.00962256 | 0.01047340 | 0.00955504 |
| 20 | 0.01069439 | 0.00966141 | 0.01037877 | 0.00931955 |
| 25 | 0.01060133 | 0.00938976 | 0.01032570 | 0.00935770 |
| **29** | 0.01053796 | **0.00935309** | 0.01028418 | **0.00912664** |
| 30 | 0.01052905 | 0.00953269 | 0.01023744 | 0.00913001 |

Both runs selected **epoch 29** under the same predeclared rule, independently.
The U-Net's training loss is lower throughout, its validation curve oscillates
by a similar ±0.0003 from epoch to epoch, and it trends downwards to the end.
**No overfitting is visible in either run**, in the narrow sense that neither
validation curve turned around and rose while training loss kept falling.
That is a description of these two runs, not a general property.

### Validation result: four methods side by side

Same 6 patients, same 885 slices, same masks, same metric code, same frozen
degradation. Patient-weighted, every patient equally weighted:

| Region | Metric | No restoration | CLAHE | Residual CNN | **U-Net** |
| --- | --- | --- | --- | --- | --- |
| full | MAE | 0.016359 | 0.023570 | 0.009353 | **0.009127** |
| full | MSE | 0.00072130 | 0.00123279 | 0.00034916 | **0.00033334** |
| full | PSNR | 31.4813 | 29.1561 | 34.6583 | **34.8623** |
| full | SSIM | 0.781346 | 0.620769 | 0.953860 | **0.955142** |
| body | MAE | 0.028187 | 0.036281 | 0.019940 | **0.019410** |
| body | MSE | 0.00141527 | 0.00229926 | 0.00074864 | **0.00071440** |
| body | PSNR | 28.5620 | 26.4831 | 31.3631 | **31.5750** |
| body | SSIM | 0.810430 | 0.768085 | 0.897097 | **0.899887** |

**Did the U-Net beat no restoration? Yes, on all eight metrics** — full-frame
PSNR +3.38 dB, body PSNR +3.01 dB, and all six patients improved on every
one of the eight.

Spread across the six patients, and the secondary slice-weighted figure:

| Metric | patient std | min | max | slice-weighted |
| --- | --- | --- | --- | --- |
| full PSNR | 0.5288 | 33.9375 | 35.3368 | 34.9607 |
| body PSNR | 0.5085 | 30.9361 | 32.0636 | 31.7223 |
| full SSIM | 0.008233 | 0.945376 | 0.969305 | 0.955615 |
| body SSIM | 0.013891 | 0.882629 | 0.919956 | 0.901752 |

Descriptive acquisition-group breakdown, three patients each. With n = 3 per
group there is nothing to conclude, and no significance test is run:

| Group | full PSNR | body PSNR | full SSIM | body SSIM |
| --- | --- | --- | --- | --- |
| A | 35.1087 | 32.0088 | 0.955263 | 0.903687 |
| B | 34.6158 | 31.1412 | 0.955021 | 0.896087 |

### The architecture comparison: U-Net versus residual CNN

This is what Milestone 9 exists to measure. Both are learned residual models
trained under an identical policy, so the intended difference between them is
the model. Paired per-patient deltas, `delta = U-Net − CNN`:

| Region | Metric | mean delta | U-Net better? | patients improved |
| --- | --- | --- | --- | --- |
| full | MAE | −0.000227 | yes | 6 / 6 |
| full | MSE | −0.000016 | yes | 6 / 6 |
| full | PSNR | **+0.2039** | yes | 6 / 6 |
| full | SSIM | +0.001282 | yes | 5 / 6 |
| body | MAE | −0.000530 | yes | 6 / 6 |
| body | MSE | −0.000034 | yes | 6 / 6 |
| body | PSNR | **+0.2119** | yes | 6 / 6 |
| body | SSIM | +0.002790 | yes | 5 / 6 |

Per patient, on the two PSNR metrics:

| Subject | Group | Δ full PSNR | Δ full SSIM | Δ body PSNR | Δ body SSIM |
| --- | --- | --- | --- | --- | --- |
| 4 | B | +0.207455 | +0.001355 | +0.204618 | +0.002815 |
| 14 | B | +0.145474 | **−0.000028** | +0.140947 | **−0.000076** |
| 17 | B | +0.155706 | +0.001021 | +0.173838 | +0.002778 |
| 23 | A | +0.229841 | +0.001780 | +0.245465 | +0.003591 |
| 24 | A | +0.267384 | +0.001977 | +0.279688 | +0.004072 |
| 34 | A | +0.217609 | +0.001587 | +0.226896 | +0.003557 |
| **mean** | | **+0.203912** | **+0.001282** | **+0.211908** | **+0.002790** |

The U-Net is ahead on the mean of all eight metrics. The two 5/6 counts are
subject 14, where SSIM moved against it by 0.000028 and 0.000076 — at the
fourth and fifth decimal place, which is not a meaningful reversal in either
direction.

**How large is this, really?** Stated without implying a mechanism: **the
U-Net used 4.12× the trainable parameters** of the CNN (116,753 against
28,353) **and scored +0.20 dB higher full-frame PSNR in this single-seed
validation run.** Both models are about **+3 dB** over no restoration, so the
gap between them is roughly **one fifteenth** of the distance either travels
from the baseline. Across five predeclared seeds the paired difference
averaged **+0.228 dB**; see
[Milestone 10](#milestone-10-the-multi-seed-result).

The parameter count is reported beside the score, not as its cause. Nothing
here isolates which change produced the difference, and no latency has been
measured for either model, so the parameter ratio is not a statement about
what it costs to run.

**What this does not establish.** Under the same frozen corruption and the
same training policy, the lightweight U-Net performed better on validation,
which is consistent with the larger multi-scale context being useful. It does
**not** show that the larger receptive field *caused* the improvement. The
two architectures differ in pooling, in having a decoder at all, in
concatenative skips, in parameter count and in the entire computational
graph, and all of those changed together. Isolating any one of them would be
a separate, declared experiment.

Most importantly: **these are one seed each**, so the correct reading of this
table on its own is "this U-Net run scored slightly better than this CNN
run", not "U-Nets are better here".

When this section was written the run-to-run spread of either architecture
was unmeasured and could have been of comparable size, which is why no
architecture effect was claimed.
[Milestone 10](#milestone-10-the-multi-seed-result) has since trained five
predeclared seeds per architecture. All eight metrics favoured the U-Net at
each of the five seeds. Across them, the paired U-Net-minus-CNN full-frame
PSNR difference averaged +0.228 dB (sample SD 0.050 dB; range +0.195 to
+0.315 dB), and this run's seed-2026 difference, +0.203912 dB, was near the
lower end of the observed range. The smallest observed difference
(+0.195 dB) exceeded either architecture's observed five-seed full-PSNR
range. That is still a descriptive validation result from five seeds: it
does not bound what further seeds could produce, it remains confounded
across every architectural change listed above, and it is not a statement
that U-Nets are better in general.

### What the clamp is doing

The reported metrics come from the clamped output, so the clamp is part of
the method. As for the CNN, two quantities are kept apart: the **predicted
correction** `raw − degraded`, measured on the unclamped output, and the
**post-clamp change** `restored − degraded`, which is what survived into the
scored image.

| | U-Net | CNN, for comparison |
| --- | --- | --- |
| raw minimum, before clamping | −0.030941 | −0.025799 |
| raw maximum, before clamping | 1.057549 | 1.056146 |
| fraction of pixels raw < 0 | 0.501801 | 0.508982 |
| fraction of pixels raw > 1 | 0.005444 | 0.005719 |
| fraction changed by the clamp | 0.507245 | 0.514702 |
| **mean absolute predicted correction** | **0.012313** | 0.012068 |
| per-slice, q05 / q50 / q95 | 0.010548 / 0.012111 / 0.014543 | 0.010319 / 0.011903 / 0.014088 |
| mean absolute post-clamp change | 0.012126 | 0.011788 |
| per-slice, q05 / q50 / q95 | 0.010311 / 0.011929 / 0.014371 | 0.009980 / 0.011617 / 0.013836 |
| non-finite outputs | 0 | 0 |

The two models behave remarkably alike here. The U-Net clamps a slightly
smaller fraction of the frame (50.7% against 51.5%) and asks for a slightly
larger correction. Because the degraded image always lies inside [0, 1], the
gap between the two correction figures is exactly the overshoot the clamp
discarded: 0.012313 − 0.012126 = **0.000187** per pixel, against the CNN's
0.000280. So the U-Net overshoots less, and neither overshoots much.

These numbers are descriptive. **None is an optimization objective**, and no
model, loss or config decision was made from them.

### Reproducibility of this run

Same scope as the CNN's: **same repository, same config, same environment,
same hardware and same seed reproduce this run.** Bitwise identity across
different GPUs, drivers or PyTorch builds is **not** claimed.

Within that scope it was checked rather than asserted:

* The training run was executed **twice**, start to finish. Both runs
  produced a byte-identical `training_history.csv` (`91834d342b7785ea…`), a
  byte-identical `run_summary.json`, and a byte-identical checkpoint
  (`15e430f83ade635c…`). This is a determinism check, **not** a second seed.
* The evaluation was likewise run twice and produced byte-identical values in
  all six output files.
* `torch.use_deterministic_algorithms(True)` in strict mode — not
  `warn_only` — with `cudnn.benchmark=False`, `cudnn.deterministic=True` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Max-pooling and transposed convolution
  both have deterministic implementations available on this device, so the
  extra layers cost nothing in rigour.
* CUDA is required; the training command refuses to run on CPU.
* After selection, the saved checkpoint was reloaded in-process and compared
  against the model still in memory: **26 state-dict tensors, 0 mismatches**,
  and **0 differing pixels** on four deterministic probe images.
* **Checkpoint provenance** is verified before a single validation image is
  opened — 15 conditions, each against something computed independently of
  the checkpoint, including re-running the predeclared selection rule over
  the tracked history to re-derive epoch 29.
* **Sample alignment** is enforced before anything is written, now against
  **three** reference tables: 885 rows, 885 unique keys, 0 duplicates, and 0
  missing, 0 extra and 0 order mismatches against the degraded baseline,
  CLAHE **and** the CNN. All four methods scored the same 885 slices in the
  same order, which is what makes every delta in this section paired.

Both gates are the same shared code the CNN's evaluation clears, in
[src/ct_restoration/evaluation_integrity.py](src/ct_restoration/evaluation_integrity.py).

## Milestone 10: the multi-seed result

Milestone 9 measured one CNN run against one U-Net run and found the U-Net
ahead by +0.20 dB full-frame PSNR. With one seed each, that gap could not be
separated from ordinary run-to-run variation, because the run-to-run spread
of either architecture was unmeasured. Milestone 10 measured that spread.

The design below was written and committed **before any of the eight new runs
existed**, so none of it could have been chosen to suit a number. It lives in
[configs/multiseed/plan.yaml](configs/multiseed/plan.yaml), whose SHA-256 is
recorded in the final multi-seed summary. To be precise about the scope of
that: the aggregate `multiseed_summary.json` records the hash of the plan
file the summarizer actually read. Individual training run summaries record
their own config's hash, not the plan's.

The eight new runs were then executed in one session under a single frozen
commit, in the pre-registered order, one process at a time. The plan was not
edited, and the plan SHA-256 recorded in the aggregate summary is the one
that was committed before any of them ran. The plan file therefore still
reads `milestone_stage: 10A - design frozen, execution pending`: that line
records the moment of the freeze, and the file is byte-pinned, so it is
deliberately not edited now that the runs exist.

### The seed set

| | |
| --- | --- |
| statistical seeds | **2026, 2027, 2028, 2029, 2030** |
| architectures | residual CNN, lightweight U-Net |
| total runs | 10 |
| reused, not retrained | 2 - the Milestone 8 CNN and Milestone 9 U-Net, both at seed 2026 |
| **new runs executed** | **8, all completed** |

Seed 2026 is **reused, not retrained**. Its checkpoint, history and run
summary are the committed ones, and it keeps the checkpoint it already
selected.

The seed list is part of the experiment definition. No seed may be added
later because a result looks unusual, and none may be dropped because a
result looks poor - either would turn the reported spread into the spread of
whichever seeds survived inspection.

### What varies, and what cannot

Within each architecture, the only thing that differs across its five
training-seed runs is **training randomness**: weight initialization and
patient-balanced sampler ordering, both driven by that run's seed. Each of
the eight new runs had its own frozen config under `configs/multiseed/`,
identical to the canonical one except for a single line, and each config's
SHA-256 is pinned in the plan and was re-verified against the checkpoint
that run produced.

Between the matched CNN and U-Net runs at a given seed label, **architecture
differs by design**, while the data, degradation, sampler algorithm and
statistical-seed label are aligned.

**The degradation does not vary.** Its seed lives in
[configs/degradation.yaml](configs/degradation.yaml) and cannot reach the
training path - the Dataset constructor takes no seed argument at all - so
every seed of every architecture sees pixel-identical degraded images for the
same slice. If the corruption also varied by seed, a difference between seeds
could be either cause and the design could not tell them apart.

### The unit of analysis

Each run collapses slice to patient to equal-weight patient mean first,
exactly as Milestones 5-9 do. Only then are seeds compared.

**Six patients times five seeds is not thirty observations.** The six patient
values inside one run are not independent repetitions of training, and the
five seed values are not independent patients. The two levels stay separate.

And five seeds are not five patients: they are five repetitions of one
training procedure on **fixed data**. The spread measured is the spread of
that procedure on this dataset, and says nothing about variation across
patients, scanners or institutions.

### Two comparison quantities, kept distinct

For each seed *S* and each of the eight metrics:

| quantity | definition | sign |
| --- | --- | --- |
| **raw delta** | `U-Net(S) - CNN(S)` | metric's own sign: positive favours the U-Net for PSNR/SSIM, **negative** favours it for MAE/MSE |
| **oriented improvement** | `U-Net - CNN` for PSNR/SSIM, `CNN - U-Net` for MAE/MSE | **positive always means the U-Net did better** |

Both are reported. The raw delta is arithmetic a reader can check by
subtracting two table entries; the oriented improvement is what summary
sentences are built from, so "the mean improvement is positive" never has to
be read alongside "except for the four lower-is-better metrics, where it is
the other way round".

### What gets reported

Per architecture and metric: all five seed values, their mean, their **sample
standard deviation (ddof = 1)**, min and max. The five values are printed
beside the summary because at n=5 the list is the more honest report - a mean
and a standard deviation hide whether the spread came from one outlying run
or four evenly scattered ones.

Paired, per metric: all five raw deltas, all five oriented improvements, the
mean and standard deviation of each, and how many of the five seeds favour
each architecture, with ties counted separately.

Forbidden, and refused in code: choosing the best seed, averaging or
ensembling checkpoints, discarding outliers, weighting seeds unequally,
ranking seeds, building a composite score, or reporting only the favourable
seeds.

### The words the result is allowed to use

No p-values, no significance claims, no confidence intervals. At n=5 those
would add ceremony rather than evidence.

One metric may be called **"directionally consistent across training seeds"**
only when *both* hold: the mean oriented improvement points that way, **and**
at least **4 of 5** seeds favour that architecture. A 5/5 result may be
stated as "all five seeds favoured ...". **A 3/2 split is never called
consistent**, and there is deliberately no permitted phrase for it - the
split has to be described.

An overall statement that one architecture is directionally better requires
**all eight metrics** to pass that rule. Otherwise the pattern is reported
metric by metric. No architecture is called "generally better" on the
strength of one metric, and the four headline quality figures do not override
contradictory MAE/MSE evidence.

### Checkpoint selection stays per run

Each of the ten runs trains 30 epochs and independently selects the epoch
with the lowest patient-weighted validation full-frame MAE, ties to the
earlier epoch, epoch 0 never eligible. No common epoch across seeds, no seed
chosen by its validation score, no architecture chosen by its best seed.

### Hold-out

Validation only. Test and stress stay sealed, and the plan records
`test_allowed: false` and `stress_allowed: false` - the summarizer refuses a
plan that says otherwise.

### What a shared seed number does and does not mean

Milestone 10 trained both architectures at each of the five seeds, and the
paired comparison only means something if "the same seed" is described
accurately.

Seed *S* for the CNN and seed *S* for the U-Net **does** mean:

* the same statistical-seed label, so the two runs form a pair;
* the same patient-balanced sampler ordering, because the sampler algorithm
  and its base seed are shared and the sampler derives every stream from
  `SHA-256(algorithm, seed, epoch, stream)` independently of the model;
* a deterministic initialization within each architecture - rerunning seed
  *S* for one architecture reproduces that architecture's weights exactly.

It **does not** mean:

* identical parameter values, which is impossible: the two networks have
  different shapes and different parameter counts;
* the same random draws assigned parameter-by-parameter - the two models
  consume the RNG stream in different orders and amounts;
* matched tensors of any kind across architectures.

So a per-seed difference `U-Net(S) - CNN(S)` pairs two runs that saw **the
same data in the same order**, which is what makes pairing worth doing. It
does not pair two networks that started from "the same" weights, and no
claim of that sort should be made from it.

The per-slice degradation is unaffected by any of this. Its seed lives in
`configs/degradation.yaml` and never reaches the training path - the Dataset
constructor takes no seed argument at all - so every seed of every
architecture sees pixel-identical degraded images.

### The result

Every number below is read from [outputs/metrics/multiseed/multiseed_summary.json](outputs/metrics/multiseed/multiseed_summary.json),
which records the SHA-256 of the plan it was produced under. Per-architecture
spread across the five training seeds, patient-weighted validation:

| metric | CNN mean | CNN sd | CNN min-max | U-Net mean | U-Net sd | U-Net min-max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_mae | 0.009403 | 0.000057 | 0.009353-0.009499 | 0.009133 | 0.000022 | 0.009100-0.009155 |
| full_mse | 0.000351 | 0.000003 | 0.000349-0.000357 | 0.000333 | 0.000001 | 0.000332-0.000334 |
| full_psnr | 34.6374 | 0.0429 | 34.5617-34.6668 | 34.8652 | 0.0114 | 34.8471-34.8769 |
| full_ssim | 0.953615 | 0.000250 | 0.953266-0.953860 | 0.955134 | 0.000327 | 0.954624-0.955488 |
| body_mae | 0.020050 | 0.000128 | 0.019940-0.020264 | 0.019425 | 0.000028 | 0.019404-0.019474 |
| body_mse | 0.000753 | 0.000007 | 0.000748-0.000764 | 0.000714 | 0.000002 | 0.000712-0.000717 |
| body_psnr | 31.3419 | 0.0386 | 31.2743-31.3698 | 31.5759 | 0.0092 | 31.5603-31.5836 |
| body_ssim | 0.896522 | 0.000564 | 0.895782-0.897097 | 0.899851 | 0.000759 | 0.898665-0.900718 |

The standard deviations are **sample** standard deviations (ddof = 1) over
five values, as the plan requires.

Paired within each seed, oriented so that **positive always means the U-Net
did better**, whichever direction the metric runs:

| metric | mean improvement | sd | min | max | seeds favouring U-Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_mae | 0.000270 | 0.000074 | 0.000220 | 0.000399 | **5 / 5** |
| full_mse | 0.000018 | 0.000004 | 0.000015 | 0.000025 | **5 / 5** |
| full_psnr | 0.2279 | 0.0500 | 0.1951 | 0.3152 | **5 / 5** |
| full_ssim | 0.001519 | 0.000446 | 0.001088 | 0.002066 | **5 / 5** |
| body_mae | 0.000625 | 0.000135 | 0.000530 | 0.000860 | **5 / 5** |
| body_mse | 0.000039 | 0.000007 | 0.000034 | 0.000051 | **5 / 5** |
| body_psnr | 0.2340 | 0.0414 | 0.2095 | 0.3062 | **5 / 5** |
| body_ssim | 0.003329 | 0.000983 | 0.002387 | 0.004442 | **5 / 5** |

The five per-seed full-frame PSNR advantages, in plan order (2026 … 2030),
are printed rather than summarized because at n=5 the list is the more
honest report:

`2026: +0.2039 dB  2027: +0.2015 dB  2028: +0.3152 dB  2029: +0.2237 dB  2030: +0.1951 dB`

### What this does and does not establish

Under the rule frozen before the runs - mean direction **and** at least 4 of
5 seeds - the U-Net qualifies as **directionally consistent across training
seeds on all eight metrics**, which is the one overall statement the plan
permits. All eight metrics favoured the U-Net at each of the five
predeclared seeds. The eight metrics are not independent of one another -
MAE, MSE and PSNR are all computed from the same pixel residuals - so this is
one pattern observed at five seeds, not forty separate confirmations.

Each of the following is stated narrowly, because each is easy to
overstate.

**What the five seeds show.** Across the five predeclared training seeds,
the paired U-Net-minus-CNN full-frame PSNR difference averaged +0.228 dB
(sample SD 0.050 dB; range +0.195 to +0.315 dB), with all five seeds
favouring the U-Net. The smallest observed full-PSNR difference (+0.195 dB)
exceeded either architecture's observed five-seed full-PSNR range (CNN
0.105 dB, U-Net 0.030 dB). The seed-2026 difference behind Milestone 9's
+0.20 dB, +0.203912 dB, was near the lower end of the observed range - the
middle of the five values, 0.009 dB above the smallest and 0.111 dB below
the largest. Milestone 9 could say none of this, because with one seed per
architecture the spread was unmeasured.

**What they do not show.** Five seeds are five observations of one training
procedure. Their observed range is not a bound on what further seeds could
produce, and nothing here rules out training randomness as a contributor to
the gap, estimates how often a reversal would occur, or attributes the gap
to any one architectural difference.

**It is still a descriptive result on one dataset.** No p-value, confidence
interval or significance test is reported; the plan forbids them at n=5 and
the summarizer records `significance_tested: false`. Five seeds measure the
spread of *this training procedure on fixed data*. They say nothing about
variation across patients, scanners, institutions or dose levels, and
nothing about real low-dose CT, which this degradation does not reproduce.

The observed seed-to-seed spread also differed by metric. For PSNR the
U-Net's five-seed sample SD was smaller than the CNN's (full-frame 0.0114
vs 0.0429 dB; body 0.0092 vs 0.0386 dB); for both SSIM metrics it was
slightly larger (full 0.000327 vs 0.000250; body 0.000759 vs 0.000564). A
sample SD from five values is itself imprecise, so these describe the ten
runs; they are not a claim that either architecture trains more stably.

### How the eight runs were kept honest

| check | result |
| --- | --- |
| scientific code changed during execution | none observed - the design commit precedes the first run, and `git diff` against it was empty across `src/`, `scripts/`, `configs/` after the last; the run artifacts do not themselves record a commit ([see below](#before-milestones-11-and-12-requirements-recorded-in-advance)) |
| seed witnesses per run | 5 agree (plan key, frozen config, checkpoint payload, run summary, metric summary) |
| config SHA-256 witnesses per run | 5 agree, against the hash pinned in the plan before the run |
| checkpoints per run | exactly 1 - no run produced candidates to choose between |
| unplanned seeds on disk | 0, by the summarizer's own detector and an independent scan |
| slices scored | 885, identical keys in identical order for all 10 runs |
| degradation `global_seed` | 2026 in all 10 runs - the training seed cannot reach it |
| statistics | recomputed independently twice - 321 values from the per-seed summaries, 288 from the per-slice tables upward - 0 disagreements in either |
| repeat execution | each of the eight runs executed once; **not** independently bitwise-repeated |
| hold-out | validation only; test and stress unread |

### What was and was not looked at

Training and validation image content was read numerically. **No test or
stress image content was read** in Milestone 9 or Milestone 10 — both
training commands and both evaluation commands refuse those splits through
the same hold-out gate, and all ten learned-method validation summaries
record `test_images_read: 0` and `stress_images_read: 0`.

**No validation image was inspected visually at any point.**
`scripts/qc_unet.py` has no `--split` option: it reads **training** slices
only, and it ran after the two seed-2026 checkpoints had been selected
numerically and their canonical metrics written. Its six-panel figures —
clean, degraded, CNN restored, U-Net restored, the U-Net's correction, and
its remaining error — go to a git-ignored directory, and nothing about
either model was changed after looking at them. No figure of any kind was
rendered for seeds 2027-2030: Milestone 10 read its results as numbers only.

Nothing in the architecture, the loss, the optimizer, the learning rate, the
epoch count, the batch size or the checkpoint criterion was changed after the
first validation number existed, and nothing in the Milestone 10 plan or its
eight configs was changed after the first Milestone 10 run began.

These are **five-seed** validation development results for both learned
methods, and validation development results are all they are. The final
comparison on the held-out test split happens only once every method
decision is frozen.

## Before Milestones 11 and 12: requirements recorded in advance

Nothing in this section has been implemented or run. It records, before the
held-out test split is opened, what Milestones 11 and 12 must satisfy, so the
requirements cannot be shaped by the results they govern.

### Milestone 11: the one-time held-out test evaluation

The run artifacts of Milestones 8-10 record each run's config SHA-256 and
checkpoint SHA-256, but **not** the git commit they were produced under. The
statement that all eight Milestone 10 runs used one frozen commit rests on the
design commit preceding the first run and on `git diff` being empty across
`src/`, `scripts/` and `configs/` after the last. Milestone 11 closes that gap
before it reads a single test image. It must record, and refuse to proceed
without:

* the exact git commit of the evaluating code;
* a clean working tree - no modified or untracked file under `src/`,
  `scripts/`, `configs/` or `data/splits/`;
* the SHA-256 of every checkpoint scored, checked against the one recorded in
  its tracked run summary;
* the SHA-256 of every config those checkpoints were trained from;
* the test protocol itself - which methods, which checkpoints, which metrics,
  which aggregation and which comparisons - written down and committed before
  the test split is read;
* an explicit record that every method decision was frozen before the test
  split was opened.

The test split is evaluated **once**. A result that looks wrong is reported,
not re-run under a changed protocol.

### Milestone 12: latency, after one known fix

The U-Net's `forward()` validates its input on every call
([src/ct_restoration/models/unet.py](src/ct_restoration/models/unet.py)),
and that validation reads back a finiteness flag, a minimum and a maximum -
three device-to-host synchronizations per forward pass. The CNN's `forward()`
performs no such check. Neither affects any image-quality number, but timed
as it stands the U-Net would be charged for synchronization the CNN never
pays. Before any latency is measured:

1. the benchmark-distorting synchronization is removed from the timed path,
   for both models on equal terms;
2. the change is shown to leave every model output numerically unchanged;
3. the latency protocol is frozen;
4. only then is latency measured.

Milestone 11 is unaffected: it evaluates image quality with the frozen model
implementation and checkpoints exactly as they are.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
uv run pytest
```

PyTorch is resolved from the CUDA 13.0 wheel index declared in
`pyproject.toml`, not from PyPI, so `uv sync` preserves the GPU build.

## Repository layout

```
src/ct_restoration/   library code (importable package)
  data/               DICOM reading, HU conversion, preprocessing, CHAOS layout,
                      patient split, low-dose-like degradation, supervised
                      Dataset, patient-balanced sampler, DataLoaders
  classical/          CLAHE and its predeclared parameter search
  models/             the residual CNN, the lightweight U-Net, their shared
                      input contract, the benchmark adapter and the shared
                      raw-output diagnostics
  metrics.py          MAE / MSE / PSNR / SSIM, shared by every method
  evaluation.py       body mask, patient aggregation, paired deltas, hold-out gate
  benchmark.py        the shared run harness every method is scored through
  training.py         training loss, patient-weighted aggregation, epoch selection
  training_loop.py    the model-agnostic epoch loop, validation pass and
                      checkpoint handling both learned methods share
  evaluation_integrity.py  the hard gates every canonical evaluation clears:
                      checkpoint provenance and ordered sample alignment
  reproducibility.py  seeding and deterministic-algorithm settings
scripts/              runnable commands (cohort audit, split generation,
                      degradation audit, body-mask audit, baseline and CLAHE
                      evaluation, CLAHE tuning, Dataset/DataLoader audit,
                      CNN and U-Net training, evaluation and visual QC,
                      multi-seed aggregation)
tests/                pytest suite, fully synthetic, no downloads
configs/              YAML experiment settings
configs/multiseed/    the pre-registered multi-seed plan and one frozen
                      config per training seed, each SHA-256 pinned
data/README.md        dataset provenance
data/splits/          the frozen patient split and slice manifest (tracked)
data/raw/, processed/ image data (git-ignored)
outputs/audit/        measured dataset facts (figures there are git-ignored)
outputs/metrics/      tracked per-slice, per-patient and split-level scores
outputs/metrics/multiseed/  one directory per additional training seed, plus
                      the aggregate five-seed summary (tracked)
outputs/runs/         per-epoch training histories and run summaries for all
                      ten runs (tracked)
outputs/checkpoints/  model weights (git-ignored; only their SHA-256 is tracked)
```

Dataset provenance, licensing and the exact archive used are recorded in
[data/README.md](data/README.md).

Raw DICOM files, processed image arrays, and model checkpoints are never
committed.

## Disclaimer

This project evaluates image-restoration methods using public CT data and
controlled synthetic degradation. Image-quality metrics such as PSNR and SSIM
do not establish clinical utility.

The synthetic degradation is an image-domain approximation. It does not
reproduce the full physics of low-dose CT acquisition and reconstruction. This
is not a medical device, is not clinically validated, and must not be used for
diagnosis or treatment.
