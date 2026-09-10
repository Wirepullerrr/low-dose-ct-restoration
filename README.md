# Low-Dose CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public abdominal CT data.

> **Status: in progress (Milestone 1 of 15 - CT preprocessing).**
> No experiments have been run yet. This README contains no results, and will
> only report numbers that are reproducible from files committed in `outputs/`.

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
| CT geometry | 512x512 in-plane, 3 to 3.2 mm inter-slice distance |

CHAOS was assembled for organ segmentation, and the challenge is described in
Kavur et al., "CHAOS Challenge - combined (CT-MR) healthy abdominal organ
segmentation", *Medical Image Analysis* vol. 69, 2021.

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
* **Padding handling is unresolved and gates the first real-data audit.**
  `pixel_padding_mask` currently only reports declared padding; nothing is
  rewritten, because choosing a replacement value would be inventing data.
  Before any metric is reported on real CHAOS slices, that audit must check
  whether `PixelPaddingValue` and `PixelPaddingRangeLimit` occur at all, how
  large the padding and background region is, whether padding always maps
  outside the selected HU window, and whether PSNR and SSIM should exclude
  padding or use a valid-body mask. This is not cosmetic: a large region that
  is identical in the target and in every method's output adds almost no error
  to MAE and MSE, which raises PSNR, and scores near-perfectly under SSIM.
  Whole-frame metrics can therefore look strong because of empty background
  rather than restored anatomy. The policy stays undecided until real slices
  have been inspected.

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
  data/               DICOM reading, HU conversion, CT preprocessing
tests/                pytest suite, fully synthetic, no downloads
configs/              YAML experiment settings
data/                 raw / processed / splits  (contents are git-ignored)
outputs/              metrics, figures, runs, final results
```

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
