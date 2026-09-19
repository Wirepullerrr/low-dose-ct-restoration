"""Train the predeclared residual CNN and select one checkpoint on validation.

    uv run python scripts/train_cnn.py --root data/raw/chaos

One configuration, one seed
---------------------------
Everything scientific is frozen in :file:`configs/cnn.yaml` before this ever
runs, and its SHA-256 is recorded in the run summary. There is no learning
rate to sweep here, no depth to try, no loss to compare. A first CNN that
turns out mediocre is still an experiment; changing the configuration after
seeing the validation curve would turn the development estimate into a
fitted one.

This is a single-seed development result. One seed says what this run did,
not that the architecture is stable.

Development splits only
-----------------------
There is no ``--split`` option. Training reads train, validation reads
validation, and both go through the Milestone 7 helper that refuses test and
stress. Test and stress image content stays sealed.

Checkpoint selection
--------------------
One predeclared criterion: the lowest patient-weighted validation full-frame
MAE over epochs 1..30, computed from the clamped prediction, ties broken
towards the earlier epoch. Epoch 0 is evaluated as an identity sanity check -
the zero-initialized network must reproduce the frozen degraded baseline -
but it is not eligible to be chosen.

Outputs
-------
``outputs/runs/cnn_seed2026/training_history.csv``
``outputs/runs/cnn_seed2026/run_summary.json``
``outputs/checkpoints/cnn_seed2026_best.pt`` (git-ignored; its SHA-256 is
tracked in the summary)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ct_restoration.benchmark import write_csv, write_json  # noqa: E402
from ct_restoration.config import load_config  # noqa: E402
from ct_restoration.data.dataset import development_dataset  # noqa: E402
from ct_restoration.data.degradation import DegradationConfig  # noqa: E402
from ct_restoration.data.loaders import (  # noqa: E402
    make_training_loader,
    make_validation_loader,
)

# ct_restoration.reproducibility sets CUBLAS_WORKSPACE_CONFIG at import time. cuBLAS
# reads it when its handle is created, which happens at the first CUDA operation, not
# at import, so import order among these modules is not load-bearing - but no CUDA work
# may happen before main() calls into that module.
from ct_restoration.models.cnn import (  # noqa: E402
    CANONICAL_PARAMETER_COUNT,
    ResidualCnnConfig,
    build_model,
)
from ct_restoration.reproducibility import (  # noqa: E402
    describe_environment,
    enable_deterministic_algorithms,
    require_cuda,
    seed_everything,
)
from ct_restoration.training import (  # noqa: E402
    SELECTION_METRIC,
    TrainingError,
    l1_training_loss,
    patient_means,
    patient_weighted_mean,
    select_best_epoch,
    slice_absolute_errors,
    slice_weighted_mean,
)

#: Where this run's tracked artifacts go.
RUN_DIR = Path("outputs/runs/cnn_seed2026")

#: Where the checkpoint binary goes. Git-ignored; its SHA-256 is tracked.
CHECKPOINT_PATH = Path("outputs/checkpoints/cnn_seed2026_best.pt")

#: Tolerance for the epoch-0 identity check against the committed baseline.
#: The two numbers are computed by different code paths - the benchmark's
#: NumPy metric formulas against this script's torch reduction - so exact
#: float equality is not the right bar. A genuine plumbing mismatch would be
#: orders of magnitude larger than this.
IDENTITY_TOLERANCE = 1e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_patient_weighted_full_mae(patients_csv: Path) -> float:
    """The committed degraded-baseline figure the identity check compares to.

    Loaded from the tracked Milestone 5 output rather than hard-coded: a
    rounded literal in this file would be a second source of truth, and the
    check would start passing against the wrong number the moment either
    drifted.
    """
    if not patients_csv.exists():
        raise TrainingError(
            f"{patients_csv.as_posix()} is missing; the epoch-0 identity check compares "
            "against the committed degraded baseline. Run evaluate_degraded_baseline first."
        )
    frame = pd.read_csv(patients_csv, dtype={"subject_id": str})
    return float(frame["mean_full_mae"].mean())


@torch.inference_mode()
def evaluate_validation(model, loader, device) -> dict[str, Any]:
    """Patient-weighted validation full-frame MAE from the clamped prediction.

    The only quantity computed per epoch. The full eight-metric benchmark runs
    once, later, on the selected checkpoint: running SSIM 31 times to decide
    something that was predeclared as MAE would be wasted work and an
    invitation to peek at metrics the rule does not use.
    """
    model.eval()
    errors: list[float] = []
    subject_ids: list[str] = []
    for batch in loader:
        degraded = batch["degraded"].to(device=device, non_blocking=False)
        clean = batch["clean"].to(device=device, non_blocking=False)
        restored = model.restore(degraded)
        errors.extend(float(value) for value in slice_absolute_errors(restored, clean).cpu())
        subject_ids.extend(str(value) for value in batch["subject_id"])

    return {
        "slices": len(errors),
        "patient_weighted_full_mae": patient_weighted_mean(errors, subject_ids),
        "slice_weighted_full_mae": slice_weighted_mean(errors),
        "per_patient_full_mae": patient_means(errors, subject_ids),
    }


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> dict[str, Any]:
    """One patient-balanced pass. Returns the mean raw L1 over the epoch."""
    # The sampler owns ordering, and it must be told the epoch or every epoch
    # would draw the identical sequence.
    loader.sampler.set_epoch(epoch)

    model.train()
    total = 0.0
    seen = 0
    for batch in loader:
        degraded = batch["degraded"].to(device=device)
        clean = batch["clean"].to(device=device)

        optimizer.zero_grad(set_to_none=True)
        restored_raw = model(degraded)
        loss = l1_training_loss(restored_raw, clean)
        loss.backward()
        optimizer.step()

        count = degraded.shape[0]
        total += float(loss.detach()) * count
        seen += count

    if seen == 0:
        raise TrainingError(f"epoch {epoch} drew no samples at all")
    return {"train_raw_l1": total / seen, "draws": seen}


def save_checkpoint(
    model, path: Path, epoch: int, seed: int, config_sha: str, value: float
) -> Path:
    """A plain dictionary of tensors, not a pickled class instance.

    Keeps the file loadable with ``weights_only=True``, which means restoring
    it never executes arbitrary code from the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()
            },
            "epoch": int(epoch),
            "seed": int(seed),
            "config_sha256": str(config_sha),
            "selection_metric": SELECTION_METRIC,
            "selection_value": float(value),
        },
        path,
    )
    return path


@torch.inference_mode()
def checkpoint_round_trip(model, path: Path, device) -> dict[str, Any]:
    """Reload the saved checkpoint and confirm it reproduces the saved model.

    A checkpoint that cannot reproduce the model it was written from is a
    failed run, however good the validation curve looked: every number
    reported later comes from reloading this file, not from the process that
    trained it.

    Checked two ways - tensor equality of the state dict, and prediction
    equality on deterministic probes - because a missing buffer or a silently
    downcast tensor can pass one and fail the other.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    revived = build_model(model.config, device)
    revived.load_state_dict(payload["model_state_dict"])
    revived.eval()
    model.eval()

    saved = model.state_dict()
    tensor_mismatches = sum(
        0 if torch.equal(saved[name].to("cpu"), tensor.to("cpu")) else 1
        for name, tensor in revived.state_dict().items()
    )

    generator = torch.Generator().manual_seed(2026)
    probes = torch.rand(4, 1, 256, 256, generator=generator, dtype=torch.float32).to(device)
    prediction_mismatches = int((model.restore(probes) != revived.restore(probes)).sum())

    return {
        "state_dict_keys": len(saved),
        "state_dict_tensor_mismatches": tensor_mismatches,
        "probe_tensors": int(probes.shape[0]),
        "probe_prediction_mismatches": prediction_mismatches,
        "reproduces_saved_model": bool(tensor_mismatches == 0 and prediction_mismatches == 0),
        "loaded_with_weights_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chaos", help="extracted CHAOS directory")
    parser.add_argument("--manifest", default="data/splits/chaos_slice_manifest.csv")
    parser.add_argument("--cnn-config", default="cnn.yaml")
    parser.add_argument("--preprocessing-config", default="baseline.yaml")
    parser.add_argument("--degradation-config", default="degradation.yaml")
    parser.add_argument(
        "--baseline-patients",
        default="outputs/metrics/degraded_baseline_validation_patients.csv",
    )
    parser.add_argument("--run-dir", default=RUN_DIR.as_posix())
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH.as_posix())
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda for the canonical run; cpu is debugging only and non-canonical",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="debug only: override the frozen epoch budget. 0 uses the config.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    run_dir = Path(arguments.run_dir)
    config_path = Path("configs") / arguments.cnn_config
    if not config_path.exists():
        config_path = Path(arguments.cnn_config)

    document = load_config(arguments.cnn_config)
    model_config = ResidualCnnConfig.from_mapping(document)
    training = document["training"]
    data_policy = document["data"]
    selection = document["checkpoint_selection"]
    config_sha = file_sha256(config_path)

    epochs = int(arguments.epochs) if arguments.epochs else int(training["epochs"])
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])

    print("FROZEN CONFIGURATION")
    print("-" * 68)
    print(config_path.read_text(encoding="utf-8"))
    print("-" * 68)
    print(f"config sha256   : {config_sha}")
    print()

    if arguments.device == "cuda":
        device = require_cuda()
    else:
        print("WARNING: non-canonical run, --device is not cuda", file=sys.stderr)
        device = torch.device(arguments.device)

    determinism = enable_deterministic_algorithms(warn_only=False)
    seeding = seed_everything(seed)
    environment = describe_environment(device)

    print("ENVIRONMENT")
    for key, value in environment.items():
        print(f"  {key:26} {value}")
    for key, value in determinism.items():
        print(f"  {key:26} {value}")
    print()

    preprocessing = load_config(arguments.preprocessing_config)["preprocessing"]
    degradation = DegradationConfig.from_mapping(load_config(arguments.degradation_config))

    # development_dataset refuses test and stress; the split names come from
    # the frozen config, not from the command line.
    train_dataset = development_dataset(
        str(data_policy["train_split"]),
        arguments.manifest,
        arguments.root,
        preprocessing,
        degradation,
    )
    validation_dataset = development_dataset(
        str(data_policy["validation_split"]),
        arguments.manifest,
        arguments.root,
        preprocessing,
        degradation,
    )

    train_loader = make_training_loader(
        train_dataset,
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        num_workers=int(data_policy["num_workers"]),
    )
    validation_loader = make_validation_loader(
        validation_dataset, batch_size=batch_size, num_workers=int(data_policy["num_workers"])
    )

    model = build_model(model_config, device)
    parameters = model.parameter_count()

    print("MODEL AND DATA")
    print(f"  algorithm                {model_config.algorithm}")
    print(f"  trainable parameters     {parameters:,}")
    print(f"  receptive field          {model.receptive_field} x {model.receptive_field} px")
    print(
        f"  train                    {len(train_dataset.patients)} patients, "
        f"{len(train_dataset)} slices"
    )
    print(
        f"  validation               {len(validation_dataset.patients)} patients, "
        f"{len(validation_dataset)} slices"
    )
    print(
        f"  sampler                  {train_loader.sampler.algorithm}, "
        f"{train_loader.sampler.epoch_size} draws/epoch"
    )
    print(f"  batch size               {batch_size}")
    print(f"  loss                     {training['loss']} on the raw restoration")
    print(f"  optimizer                {training['optimizer']}, lr {training['learning_rate']}")
    print(f"  epochs                   {epochs}")
    print("  test and stress image content is NOT read by this command")
    print()

    if parameters != CANONICAL_PARAMETER_COUNT:
        print(
            f"note: {parameters} trainable parameters, not the canonical "
            f"{CANONICAL_PARAMETER_COUNT}",
            file=sys.stderr,
        )

    # -- epoch 0: the zero-initialized network must be the identity ---------
    epoch_zero = evaluate_validation(model, validation_loader, device)
    baseline_mae = baseline_patient_weighted_full_mae(Path(arguments.baseline_patients))
    difference = abs(epoch_zero["patient_weighted_full_mae"] - baseline_mae)
    identity_ok = difference <= IDENTITY_TOLERANCE

    print("EPOCH 0 IDENTITY CHECK")
    print(f"  zero-initialized CNN     {epoch_zero['patient_weighted_full_mae']:.12f}")
    print(f"  committed degraded base  {baseline_mae:.12f}")
    print(f"  absolute difference      {difference:.3e}  (tolerance {IDENTITY_TOLERANCE:.0e})")
    print(f"  identity reproduced      {'YES' if identity_ok else 'NO'}")
    print()
    if not identity_ok:
        print(
            "error: the zero-initialized CNN does not reproduce the frozen degraded baseline. "
            "The learned-method evaluation path is not aligned with the benchmark; stopping "
            "before training rather than reporting a number built on it.",
            file=sys.stderr,
        )
        return 3

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    learning_rate = float(training["learning_rate"])

    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_raw_l1": None,
            "validation_patient_weighted_full_mae": epoch_zero["patient_weighted_full_mae"],
            "validation_slice_weighted_full_mae": epoch_zero["slice_weighted_full_mae"],
            "learning_rate": learning_rate,
            "is_checkpoint_candidate": False,
            "is_best_so_far": False,
        }
    ]

    print(f"{'epoch':>6}{'train_raw_l1':>16}{'val_pw_full_mae':>18}{'val_sw_full_mae':>18}  best")
    print("-" * 66)
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        trained = train_one_epoch(model, train_loader, optimizer, device, epoch)
        measured = evaluate_validation(model, validation_loader, device)
        value = measured["patient_weighted_full_mae"]

        improved = value < best_value
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to("cpu").clone()
                for name, tensor in model.state_dict().items()
            }

        history.append(
            {
                "epoch": epoch,
                "train_raw_l1": trained["train_raw_l1"],
                "validation_patient_weighted_full_mae": value,
                "validation_slice_weighted_full_mae": measured["slice_weighted_full_mae"],
                "learning_rate": learning_rate,
                "is_checkpoint_candidate": True,
                "is_best_so_far": improved,
            }
        )
        print(
            f"{epoch:>6}{trained['train_raw_l1']:>16.8f}{value:>18.8f}"
            f"{measured['slice_weighted_full_mae']:>18.8f}  {'<--' if improved else ''}"
        )

    frame = pd.DataFrame(history)
    # The rule is applied to the recorded table, not to the running minimum
    # above, so the selection in the summary is the selection a reader can
    # reproduce from the tracked CSV.
    if selection["primary_metric"] != "patient_weighted_full_mae":
        raise TrainingError(
            f"configs declares checkpoint metric {selection['primary_metric']!r}, but this "
            f"script only implements 'patient_weighted_full_mae' (column {SELECTION_METRIC!r})."
        )
    selected_epoch = select_best_epoch(
        frame,
        metric=SELECTION_METRIC,
        epoch_zero_eligible=bool(selection["epoch_zero_eligible"]),
    )
    if selected_epoch != best_epoch or best_state is None:
        raise TrainingError(
            f"the running best epoch ({best_epoch}) and the selection rule applied to the "
            f"history ({selected_epoch}) disagree; refusing to save an ambiguous checkpoint."
        )
    selected_value = float(frame.loc[frame["epoch"] == selected_epoch, SELECTION_METRIC].iloc[0])

    model.load_state_dict(best_state)
    checkpoint_path = save_checkpoint(
        model, Path(arguments.checkpoint), selected_epoch, seed, config_sha, selected_value
    )

    round_trip = checkpoint_round_trip(model, checkpoint_path, device)
    if not round_trip["reproduces_saved_model"]:
        print(
            "error: the saved checkpoint does not reproduce the selected model. Refusing to "
            "report numbers that would be computed by reloading it.",
            file=sys.stderr,
        )
        return 4

    history_csv = write_csv(frame, run_dir / "training_history.csv")
    final_value = float(frame[SELECTION_METRIC].iloc[-1])

    summary = {
        "milestone": "8 - small residual CNN, single-seed development run",
        "result_class": (
            "SINGLE-SEED VALIDATION DEVELOPMENT RESULT. One seed shows what this run did; "
            "it does not establish that the architecture is stable. Multi-seed work is a "
            "later milestone, and the test split remains sealed."
        ),
        "model": {
            **model_config.as_dict(),
            "trainable_parameters": int(parameters),
            "receptive_field_pixels": int(model.receptive_field),
            "receptive_field_note": (
                "Five stacked 3x3 stride-1 convolutions: 1 + 5 * 2 = 11 pixels. A statement "
                "about pixels only, not about anatomical context."
            ),
        },
        "config": {
            "path": config_path.as_posix(),
            "sha256": config_sha,
            "frozen_before_training": True,
        },
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": "full-frame L1 on the raw (unclamped) restoration",
            "optimizer": str(training["optimizer"]),
            "learning_rate": learning_rate,
            "betas": [float(value) for value in training["betas"]],
            "eps": float(training["eps"]),
            "weight_decay": float(training["weight_decay"]),
            "scheduler": str(training["scheduler"]),
            "gradient_clipping": str(training["gradient_clipping"]),
            "mixed_precision": bool(training["mixed_precision"]),
            "augmentation": str(training["augmentation"]),
        },
        "data": {
            "sampler_algorithm": train_loader.sampler.algorithm,
            "train_patients": len(train_dataset.patients),
            "train_slices": len(train_dataset),
            "train_draws_per_epoch": train_loader.sampler.epoch_size,
            "validation_patients": len(validation_dataset.patients),
            "validation_slices": len(validation_dataset),
            "validation_sampling": "canonical_sequential, exactly once",
            "num_workers": int(data_policy["num_workers"]),
            "drop_last": False,
        },
        "epoch_zero_identity_check": {
            "description": (
                "The zero-initialized network outputs a correction of exactly 0, so its "
                "clamped restoration is the degraded image. Its validation MAE must match "
                "the committed degraded baseline, or the learned-method evaluation path "
                "disagrees with the frozen benchmark."
            ),
            "zero_initialized_cnn_patient_weighted_full_mae": epoch_zero[
                "patient_weighted_full_mae"
            ],
            "committed_degraded_baseline_patient_weighted_full_mae": baseline_mae,
            "baseline_source": Path(arguments.baseline_patients).as_posix(),
            # The tracked summary rounds floats to 10 places, which would print
            # a genuine 2e-11 difference as a flat 0.0 and read as exact
            # equality. It is not exact: the two numbers come from different
            # reduction orders in different code paths. The string keeps the
            # real magnitude visible.
            "absolute_difference": difference,
            "absolute_difference_exact": f"{difference:.6e}",
            "tolerance": IDENTITY_TOLERANCE,
            "difference_note": (
                "Not expected to be exactly zero: the benchmark computes this with NumPy "
                "metric code and this script with a torch reduction, so the two differ by "
                "floating-point summation order. A plumbing mismatch would be orders of "
                "magnitude larger."
            ),
            "identity_reproduced": bool(identity_ok),
        },
        "checkpoint_selection": {
            "split": str(selection["split"]),
            "primary_metric": SELECTION_METRIC,
            "direction": str(selection["direction"]),
            "tie_breaker": str(selection["tie_breaker"]),
            "epoch_zero_eligible": bool(selection["epoch_zero_eligible"]),
            "predeclared": True,
            "best_epoch": int(selected_epoch),
            "best_validation_patient_weighted_full_mae": selected_value,
            "final_epoch_validation_patient_weighted_full_mae": final_value,
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": file_sha256(checkpoint_path),
            "tracked_in_git": False,
            "contents": [
                "model_state_dict",
                "epoch",
                "seed",
                "config_sha256",
                "selection_metric",
                "selection_value",
            ],
            "note": "Plain tensors only; loadable with weights_only=True.",
            "round_trip": round_trip,
        },
        "environment": {**environment, **determinism, **seeding},
        "reproducibility_claim": (
            "Same repository, config, environment, hardware and seed reproduce this run. "
            "Bitwise identity across different GPUs or driver versions is NOT claimed."
        ),
        "inspection_policy": {
            "train_images_read": True,
            "validation_images_read": True,
            "test_images_read": 0,
            "stress_images_read": 0,
            "validation_images_visually_inspected": False,
        },
        "outputs": {"training_history_csv": history_csv.as_posix()},
    }
    summary_path = write_json(summary, run_dir / "run_summary.json")

    print("-" * 66)
    print(f"best epoch      : {selected_epoch}")
    print(f"best val MAE    : {selected_value:.8f}  (epoch 0 identity {baseline_mae:.8f})")
    print(f"final val MAE   : {final_value:.8f}")
    print(f"checkpoint      : {checkpoint_path}")
    print(
        f"round trip      : {round_trip['state_dict_tensor_mismatches']} tensor and "
        f"{round_trip['probe_prediction_mismatches']} prediction mismatches on reload"
    )
    print(f"history         : {history_csv}")
    print(f"summary         : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
