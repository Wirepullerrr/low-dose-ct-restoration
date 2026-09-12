# Low-Dose CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public abdominal CT data.

> **Status: in progress (Milestone 2 of 15 - real dataset audit).**
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
split than the 20 subjects of the training archive alone. The exact
train/validation/test assignment is **not decided yet** and will be designed in
the next milestone. It needs an explicit design decision rather than a fixed
ratio, because the four parameter groups hold 16, 21, 2 and 1 subjects, so the
two rare groups cannot be represented in every split. How to handle that, and
what it means for what the benchmark measures, is part of that decision.

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
  metrics, body-region metrics, or both, and record the reasoning.

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
  data/               DICOM reading, HU conversion, preprocessing, CHAOS layout
scripts/              runnable commands (dataset audit)
tests/                pytest suite, fully synthetic, no downloads
configs/              YAML experiment settings
data/                 README.md records provenance; image data is git-ignored
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
