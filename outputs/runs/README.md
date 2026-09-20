# Training runs

One directory per training run, named for the method and the seed:
`cnn_seed2026` (Milestone 8) and `unet_seed2026` (Milestone 9). These are
**development** artifacts: they record what a run did, not what the benchmark
concluded. The benchmark numbers live in
[`outputs/metrics/`](../metrics/README.md).

| File | Contents |
| --- | --- |
| `<run>/training_history.csv` | one row per epoch, including epoch 0: training loss, validation checkpoint metric, learning rate, and which epochs were selection candidates |
| `<run>/run_summary.json` | the frozen config and its SHA-256, the architecture, the data counts, the epoch-0 identity check, the selected epoch, the checkpoint's SHA-256, and the environment the run happened in |

Neither file carries a timestamp, so re-running an unchanged definition on the
same machine rewrites them byte for byte.

## What these are and are not

**Development-only.** A training history is a record of an optimization, not
a result. Nothing in it is a benchmark number, and the validation column in it
exists for one purpose: choosing a checkpoint.

**Checkpoint selection used one predeclared criterion.** The lowest
patient-weighted validation full-frame MAE, computed from the clamped
prediction, over epochs 1..30, ties broken towards the earlier epoch. That
criterion was written into `configs/cnn.yaml` before the first training run
and its SHA-256 is recorded in the summary. Epoch 0 is the zero-initialized
identity check and is recorded but never eligible.

**The full metrics were computed once, afterwards.** MAE, MSE, PSNR and SSIM,
full frame and body region, were calculated only for the already-selected
checkpoint, through the same frozen benchmark harness every other method uses.
They are reported, not optimized against.

**Single seed.** These runs use one training seed. One seed shows what that
run did; it does not establish that the architecture is stable. Multi-seed
work is a later milestone.

**The two runs are deliberately comparable.** `unet_seed2026` uses the same
seed, epoch budget, batch size, loss, optimizer and hyperparameters, sampler
and checkpoint-selection rule as `cnn_seed2026`; none of them were adjusted
for the U-Net. The intended difference between the two runs is the model.
Both training commands call the same shared helpers in
`ct_restoration.training_loop`, so they cannot drift apart in how they train.

Neither model's recipe was ever hyperparameter-tuned: the CNN's values were
one predeclared development configuration. The U-Net inherits the CNN
benchmark's predeclared training recipe rather than receiving
architecture-specific tuning, so it is measured under that recipe rather than
at its best.

## The checkpoint binary

The model weights themselves (`outputs/checkpoints/*.pt`) are **git-ignored**.
They are regenerable from the tracked config, the tracked split and the
tracked seed, and a repository is not a model registry.

What is tracked is the checkpoint's **SHA-256**, in `run_summary.json`. That
is enough to confirm that a locally regenerated checkpoint is the same file
the reported numbers came from, without committing a binary.

The checkpoint holds plain tensors and scalars only - no pickled class
instance, no training data, no DICOM content - so it loads with
`weights_only=True` and restoring it never executes code from the file.

## Reproducibility

The claim is scoped: **same repository, same config, same environment, same
hardware and same seed reproduce the run.** Bitwise identity across different
GPUs, drivers or PyTorch builds is not claimed, because floating-point
reduction order in cuDNN kernels depends on the hardware and the algorithm
chosen.

Each summary records the torch version, the CUDA build, the GPU name and the
determinism settings the run used, so a mismatch is visible rather than
assumed away.

Both runs were executed twice end to end as a check, and each produced a
byte-identical history, summary and checkpoint. That is a determinism check,
**not** a second seed, and it is not stability evidence.

## Terms

These are CHAOS-derived artifacts. Use and redistribution of CHAOS data and of
artifacts derived from it remain subject to the applicable CHAOS dataset
terms, recorded with the dataset provenance in
[`data/README.md`](../../data/README.md).
