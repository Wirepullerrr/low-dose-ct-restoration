# Low-Dose CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public abdominal CT data.

> **Status: in progress (Milestone 3 of 15 - frozen patient-level split).**
> No restoration experiment has been run yet. This README contains no
> restoration results, and will only report numbers that are reproducible from
> files committed in `outputs/`.

## Research question

Given a clean reference CT slice and a controlled low-dose-like degradation of
it, how well do classical and deep-learning restoration methods recover image
quality, and what does each method cost in inference latency?

## Methods to be compared

| Method | Type | Status |
| --- | --- | --- |
| Degraded input (no restoration) | mandatory reference baseline | not implemented |
| CLAHE | classical local contrast enhancement | not implemented |
| Small residual CNN | deep learning | not implemented |
| Lightweight U-Net | deep learning | not implemented |

All four will be evaluated on identical test patients, identical clean targets,
and identical degraded inputs, using MAE, MSE, PSNR, SSIM, and inference
latency.

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
* **Background is large, and the metric policy is still open.** No CHAOS CT
  slice declares `PixelPaddingValue` or `PixelPaddingRangeLimit`, so there is
  no formally declared padding to mask. Even so, about 45 percent of an average
  frame is air below -900 HU, and one subject carries an additional uniform
  -2048 HU region outside the reconstruction circle, consistent with
  out-of-field or reconstruction-background padding. All of it clips to 0.0
  after windowing. Large, simple background regions can dominate full-frame
  image-quality metrics and may make them less sensitive to restoration quality
  inside the anatomy. How the degradation and each method behave there is not
  yet known, because no degradation has been implemented. The evaluation
  milestone must therefore decide explicitly whether to report full-frame
  metrics, body-region metrics, or both, and record the reasoning. The
  degradation now implemented adds a second reason to settle this; see
  [the clipped-background caveat](#the-clipped-background-caveat).

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
scripts/              runnable commands (dataset audit, split generation,
                      degradation audit)
tests/                pytest suite, fully synthetic, no downloads
configs/              YAML experiment settings
data/README.md        dataset provenance
data/splits/          the frozen patient split and slice manifest (tracked)
data/raw/, processed/ image data (git-ignored)
outputs/audit/        measured dataset facts (figures there are git-ignored)
outputs/              metrics, runs, final results
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
