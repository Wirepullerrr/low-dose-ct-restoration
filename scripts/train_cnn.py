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
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ct_restoration.benchmark import write_csv, write_json  # noqa: E402
from ct_restoration.config import load_config, resolve_config_path  # noqa: E402
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
from ct_restoration.evaluation_integrity import repository_path  # noqa: E402
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
from ct_restoration.run_layout import (  # noqa: E402
    RunLayoutError,
    require_writable_run_destination,
)
from ct_restoration.training import (  # noqa: E402
    SELECTION_METRIC,
    TrainingError,
    select_best_epoch,
)
from ct_restoration.training_loop import (  # noqa: E402
    baseline_patient_weighted_full_mae,
    checkpoint_round_trip,
    evaluate_validation,
    file_sha256,
    save_checkpoint,
    train_one_epoch,
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
        "--overwrite",
        action="store_true",
        help="replace an existing run directory and checkpoint. Never used by multi-seed "
        "automation: each seed writes to its own destination, so needing this means the "
        "destination was wrong.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda for the canonical run; cpu is debugging only and non-canonical",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    run_dir = Path(arguments.run_dir)

    # Destination safety first. Before the dataset is opened, before the
    # optimizer exists, before a single gradient step and before any file is
    # written: a refused run must cost an error message, not a lost checkpoint.
    try:
        require_writable_run_destination(
            run_dir, Path(arguments.checkpoint), overwrite=arguments.overwrite
        )
    except RunLayoutError as error:
        print(f"error: {error}", file=sys.stderr)
        return 7

    config_path = resolve_config_path(arguments.cnn_config)

    document = load_config(arguments.cnn_config)
    model_config = ResidualCnnConfig.from_mapping(document)
    training = document["training"]
    data_policy = document["data"]
    selection = document["checkpoint_selection"]
    config_sha = file_sha256(config_path)

    # From the frozen config only. A CLI override would let a five-epoch run
    # carry the SHA-256 of a config that declares thirty, and the provenance
    # gate - which compares the checkpoint against that config's bytes -
    # would pass. Debug runs belong in the synthetic tests, not in a scientific
    # field behind the frozen config.
    epochs = int(training["epochs"])
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

    round_trip = checkpoint_round_trip(model, checkpoint_path, device, build_model)
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
            "path": repository_path(config_path),
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
