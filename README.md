# Low-Dose CT Image Restoration: Classical vs. Deep Learning Approaches

An engineering benchmark comparing classical and lightweight deep-learning
restoration methods on **synthetically degraded, low-dose-like CT images**
built from public CT data.

> **Status: in progress (Milestone 0 of 15 - repository foundation).**
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
public CT DICOM -> HU conversion -> CT windowing -> normalized [0,1] reference
      -> controlled low-dose-like degradation -> degraded input
      -> {CLAHE | CNN | U-Net} -> restored image
      -> MAE / MSE / PSNR / SSIM / latency -> engineering conclusion
```

Train, validation, and test splits are made **at the patient level**, never at
the slice level, because adjacent slices from one patient are highly correlated
and slice-level splitting would leak information into the test estimate.

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
tests/                pytest suite
configs/              YAML experiment settings  (created in later milestones)
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
