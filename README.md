# Low-Dose CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public abdominal CT data.

> **Status: in progress (Milestone 6 of 15 - CLAHE implemented, tuned on
> validation and frozen).**
>
> Two methods are measured so far: the no-restoration degraded baseline and
> CLAHE. **CLAHE scored worse than doing nothing on every metric and every
> patient**, which is reported as it stands. No neural network exists yet:
> there is no CNN or U-Net result anywhere in this document.
>
> Every measured number here is a **validation** development result. No test
> or stress number exists, and no test or stress image content has been read
> since the split was frozen.
>
> Every number reported here is reproducible from files committed under
> `outputs/`.

## Research question

Given a clean reference CT slice and a controlled low-dose-like degradation of
it, how well do classical and deep-learning restoration methods recover image
quality, and what does each method cost in inference latency?

## Methods to be compared

| Method | Type | Status |
| --- | --- | --- |
| Degraded input (no restoration) | mandatory reference baseline | implemented; [measured on validation](#validation-degraded-baseline) |
| CLAHE | classical local contrast enhancement | implemented, validation-tuned and frozen; [worse than no restoration](#validation-result-clahe-versus-no-restoration) |
| Small residual CNN | deep learning | not implemented |
| Lightweight U-Net | deep learning | not implemented |

All four will be evaluated on identical patients, identical clean targets and
identical degraded inputs, through the same frozen metric code, using MAE, MSE,
PSNR, SSIM, and inference latency.

The final comparison will be made on the held-out **test** split, once every
method decision is frozen. Nothing has been evaluated on test or stress yet;
the baseline above is a validation figure, used to develop and sanity-check the
measurement, not a final result.

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
validation data. It is three patients. No robustness claim is made here,
because no model has been evaluated yet.

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
**before any method exists**. CLAHE, the CNN and the U-Net will all call the
same functions on the same slices with the same masks, so a later comparison
reflects the methods rather than the measurement.

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
improvement for MAE and MSE. A later milestone will look at the distribution
of those six paired deltas — how many patients improved, by how much, and
whether any got worse. No such comparison exists yet, and none is implied by
the table above.

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
The winner therefore lies at the boundary of the tested parameter space, so
the sweep does **not** establish what would happen outside that space —
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
`--overwrite`, so CLAHE cannot be quietly retuned once CNN or U-Net results
exist.

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
                      patient split, low-dose-like degradation
  metrics.py          MAE / MSE / PSNR / SSIM, shared by every method
  evaluation.py       body mask, patient aggregation, hold-out gate
scripts/              runnable commands (dataset audit, split generation,
                      degradation audit, body-mask audit, baseline evaluation)
tests/                pytest suite, fully synthetic, no downloads
configs/              YAML experiment settings
data/README.md        dataset provenance
data/splits/          the frozen patient split and slice manifest (tracked)
data/raw/, processed/ image data (git-ignored)
outputs/audit/        measured dataset facts (figures there are git-ignored)
outputs/metrics/      tracked per-slice, per-patient and split-level scores
outputs/              runs, final results
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
