"""Tests for the training command's policy and its determinism.

Fully synthetic and CPU-only: the miniature on-disk cohort from
``conftest.py`` stands in for CHAOS, and the "training run" here is a handful
of steps on 16x16 images.

Two things are being protected. First, that the command cannot be pointed at a
sealed split now that a model exists to point at it. Second, that the run is
reproducible in the specific sense this project claims - same repository,
config, environment, hardware and seed give the same run - which is only
checkable by actually doing it twice.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from ct_restoration.data.dataset import development_dataset
from ct_restoration.data.loaders import make_training_loader, make_validation_loader
from ct_restoration.evaluation import EvaluationError, HeldOutSplitError
from ct_restoration.models.cnn import build_model
from ct_restoration.reproducibility import (
    ReproducibilityError,
    describe_environment,
    enable_deterministic_algorithms,
    seed_everything,
)
from ct_restoration.training import l1_training_loss

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = _load("train_cnn")


# --------------------------------------------------------------------------
# Hold-out discipline
# --------------------------------------------------------------------------


def test_the_training_command_has_no_split_option() -> None:
    """The splits come from the frozen config, not from the command line."""
    destinations = {action.dest for action in train.build_parser()._actions}

    assert "split" not in destinations


def test_the_frozen_config_names_only_development_splits() -> None:
    from ct_restoration.config import load_config

    data_policy = load_config("cnn.yaml")["data"]

    assert data_policy["train_split"] == "train"
    assert data_policy["validation_split"] == "validation"


@pytest.mark.parametrize("split", ["test", "stress"])
def test_a_sealed_split_cannot_be_turned_into_a_training_set(synthetic_cohort, split) -> None:
    """A model existing is not a reason to loosen the gate."""
    root, manifest, preprocessing = synthetic_cohort

    with pytest.raises(HeldOutSplitError, match="sealed until the final benchmark"):
        development_dataset(split, manifest, root, preprocessing)


def test_an_unknown_split_is_refused(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort

    with pytest.raises(EvaluationError, match="Unknown split"):
        development_dataset("holdout", manifest, root, preprocessing)


# --------------------------------------------------------------------------
# The epoch-0 identity baseline comes from the committed file
# --------------------------------------------------------------------------


def test_the_identity_baseline_is_read_from_the_committed_output() -> None:
    """Not a hard-coded literal: that would be a second source of truth."""
    import pandas as pd

    path = Path("outputs/metrics/degraded_baseline_validation_patients.csv")
    expected = float(pd.read_csv(path, dtype={"subject_id": str})["mean_full_mae"].mean())

    assert train.baseline_patient_weighted_full_mae(path) == pytest.approx(expected, abs=0)


def test_a_missing_baseline_file_is_refused(tmp_path) -> None:
    from ct_restoration.training import TrainingError

    with pytest.raises(TrainingError, match="epoch-0 identity check"):
        train.baseline_patient_weighted_full_mae(tmp_path / "absent.csv")


def test_the_identity_tolerance_is_tight() -> None:
    """Loose enough for two float reduction orders, far tighter than a bug."""
    assert train.IDENTITY_TOLERANCE <= 1e-6


# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------


def test_seeding_reports_what_it_seeded() -> None:
    report = seed_everything(2026)

    assert report["seed"] == 2026
    assert report["python_random"] and report["numpy_global"] and report["torch_cpu"]


def test_seeding_makes_initialization_repeatable() -> None:
    seed_everything(2026)
    first = build_model().state_dict()
    seed_everything(2026)
    second = build_model().state_dict()

    assert all(torch.equal(first[name], second[name]) for name in first)


def test_a_different_seed_initializes_differently() -> None:
    seed_everything(2026)
    first = build_model().state_dict()
    seed_everything(2027)
    second = build_model().state_dict()

    assert any(not torch.equal(first[name], second[name]) for name in first)


@pytest.mark.parametrize("seed", [1.0, "2026", True, None])
def test_a_malformed_seed_is_refused(seed) -> None:
    with pytest.raises(ReproducibilityError, match="seed must be an integer"):
        seed_everything(seed)


def test_determinism_settings_are_strict_by_default() -> None:
    report = enable_deterministic_algorithms()

    assert report["use_deterministic_algorithms"] is True
    assert report["deterministic_warn_only"] is False
    assert report["cudnn_benchmark"] is False
    assert report["cudnn_deterministic"] is True
    assert report["cublas_workspace_config"] == ":4096:8"


def test_the_environment_block_carries_no_timestamp_or_hostname() -> None:
    facts = describe_environment("cpu")

    assert "torch_version" in facts
    assert not any(
        key in facts for key in ("timestamp", "generated_utc", "hostname", "user", "date")
    )


# --------------------------------------------------------------------------
# A deterministic mini-run, twice
# --------------------------------------------------------------------------


def mini_run(synthetic_cohort, seed: int, epochs: int = 2) -> dict:
    """The real training procedure, on the miniature cohort, on CPU.

    Deliberately calls the same loaders, the same sampler, the same loss and
    the same per-epoch helpers the canonical run uses, so this checks the
    actual path rather than a simplified copy of it.
    """
    root, manifest, preprocessing = synthetic_cohort
    seed_everything(seed)

    dataset = development_dataset("train", manifest, root, preprocessing)
    validation = development_dataset("validation", manifest, root, preprocessing)
    loader = make_training_loader(dataset, batch_size=4, seed=seed, num_workers=0)
    validation_loader = make_validation_loader(validation, batch_size=4, num_workers=0)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    orders: list[list[str]] = []
    losses: list[float] = []
    validations: list[float] = []

    for epoch in range(1, epochs + 1):
        loader.sampler.set_epoch(epoch)
        orders.append([dataset.sample_keys[index] for index in loader.sampler.epoch_indices(epoch)])
        losses.append(
            train.train_one_epoch(model, loader, optimizer, device, epoch)["train_raw_l1"]
        )
        validations.append(
            train.evaluate_validation(model, validation_loader, device)["patient_weighted_full_mae"]
        )

    return {
        "initial": initial,
        "orders": orders,
        "losses": losses,
        "validations": validations,
        "final": {name: tensor.clone() for name, tensor in model.state_dict().items()},
    }


def test_the_same_seed_reproduces_the_whole_mini_run(synthetic_cohort) -> None:
    first = mini_run(synthetic_cohort, seed=2026)
    second = mini_run(synthetic_cohort, seed=2026)

    assert all(torch.equal(first["initial"][n], second["initial"][n]) for n in first["initial"])
    assert first["orders"] == second["orders"]
    assert first["losses"] == second["losses"]
    assert first["validations"] == second["validations"]
    assert all(torch.equal(first["final"][n], second["final"][n]) for n in first["final"])


def test_a_different_seed_changes_the_mini_run(synthetic_cohort) -> None:
    first = mini_run(synthetic_cohort, seed=2026)
    other = mini_run(synthetic_cohort, seed=7)

    assert any(not torch.equal(first["initial"][n], other["initial"][n]) for n in first["initial"])
    assert first["orders"] != other["orders"]
    assert any(not torch.equal(first["final"][n], other["final"][n]) for n in first["final"])


def test_the_mini_run_actually_trains(synthetic_cohort) -> None:
    run = mini_run(synthetic_cohort, seed=2026, epochs=3)

    assert all(value == value and value < float("inf") for value in run["losses"])
    assert any(not torch.equal(run["initial"][name], run["final"][name]) for name in run["initial"])


# --------------------------------------------------------------------------
# The training loop uses the Milestone 7 policy, not a convenient shortcut
# --------------------------------------------------------------------------


def test_the_epoch_is_pushed_into_the_sampler(synthetic_cohort, monkeypatch) -> None:
    """Without set_epoch every epoch would draw the identical sequence."""
    root, manifest, preprocessing = synthetic_cohort
    dataset = development_dataset("train", manifest, root, preprocessing)
    loader = make_training_loader(dataset, batch_size=4, seed=2026, num_workers=0)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    seen: list[int] = []
    original = loader.sampler.set_epoch
    monkeypatch.setattr(
        loader.sampler, "set_epoch", lambda epoch: (seen.append(epoch), original(epoch))[1]
    )

    for epoch in (1, 2, 3):
        train.train_one_epoch(model, loader, optimizer, torch.device("cpu"), epoch)

    assert seen == [1, 2, 3]


def test_the_training_loader_stays_patient_balanced(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort
    dataset = development_dataset("train", manifest, root, preprocessing)
    loader = make_training_loader(dataset, batch_size=4, seed=2026, num_workers=0)

    drawn = train.train_one_epoch(
        build_model(), loader, torch.optim.Adam(build_model().parameters()), torch.device("cpu"), 1
    )

    assert drawn["draws"] == len(dataset)
    assert loader.sampler.algorithm == "patient_balanced_v1"
    counts = {}
    for index in loader.sampler.epoch_indices(1):
        counts[dataset.subject_ids[index]] = counts.get(dataset.subject_ids[index], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_validation_visits_every_slice_exactly_once(synthetic_cohort) -> None:
    root, manifest, preprocessing = synthetic_cohort
    validation = development_dataset("validation", manifest, root, preprocessing)
    loader = make_validation_loader(validation, batch_size=3, num_workers=0)

    measured = train.evaluate_validation(build_model(), loader, torch.device("cpu"))

    assert measured["slices"] == len(validation)
    assert set(measured["per_patient_full_mae"]) == set(validation.patients)


def test_epoch_zero_validation_equals_the_degraded_error(synthetic_cohort) -> None:
    """The zero-initialized model is the identity, so its MAE is the corruption's."""
    root, manifest, preprocessing = synthetic_cohort
    validation = development_dataset("validation", manifest, root, preprocessing)
    loader = make_validation_loader(validation, batch_size=3, num_workers=0)

    measured = train.evaluate_validation(build_model(), loader, torch.device("cpu"))

    direct = []
    subjects = []
    for position in range(len(validation)):
        sample = validation[position]
        direct.append(float(torch.abs(sample["degraded"] - sample["clean"]).mean()))
        subjects.append(sample["subject_id"])
    from ct_restoration.training import patient_weighted_mean

    assert measured["patient_weighted_full_mae"] == pytest.approx(
        patient_weighted_mean(direct, subjects), abs=1e-12
    )


# --------------------------------------------------------------------------
# Checkpoint round-trip
# --------------------------------------------------------------------------


def test_a_saved_checkpoint_reproduces_its_model(tmp_path) -> None:
    """A checkpoint that cannot reproduce its own selected model is a failure."""
    seed_everything(2026)
    model = build_model()
    # Train a little so the state is not the trivial zero initialization.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(4)
    degraded = torch.rand(4, 1, 16, 16, generator=generator, dtype=torch.float32)
    clean = torch.clamp(degraded - 0.05, 0.0, 1.0)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        l1_training_loss(model(degraded), clean).backward()
        optimizer.step()

    path = train.save_checkpoint(model, tmp_path / "best.pt", 7, 2026, "abc", 0.0123)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    revived = build_model()
    revived.load_state_dict(payload["model_state_dict"])

    probe = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        assert torch.equal(model.eval().restore(probe), revived.eval().restore(probe))
    assert payload["epoch"] == 7
    assert payload["seed"] == 2026
    assert payload["config_sha256"] == "abc"
    assert payload["selection_value"] == pytest.approx(0.0123)


def test_the_checkpoint_holds_plain_tensors_only(tmp_path) -> None:
    """weights_only=True must be able to load it: no arbitrary code on restore."""
    path = train.save_checkpoint(build_model(), tmp_path / "best.pt", 1, 2026, "abc", 0.1)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {
        "model_state_dict",
        "epoch",
        "seed",
        "config_sha256",
        "selection_metric",
        "selection_value",
    }
    assert all(isinstance(tensor, torch.Tensor) for tensor in payload["model_state_dict"].values())


def test_the_config_hash_is_a_hash_of_the_committed_file() -> None:
    import hashlib

    path = Path("configs/cnn.yaml")

    assert train.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# both training commands resolve their config by the one shared rule
# --------------------------------------------------------------------------


def test_the_bare_and_prefixed_config_names_resolve_to_the_same_file():
    # train_cnn.py used to carry its own copy of this lookup rule. A second
    # copy is a second thing to keep in step with the loader, and the file
    # that gets hashed for provenance must be the file that got loaded.
    from ct_restoration.config import resolve_config_path

    bare = resolve_config_path("cnn.yaml")
    prefixed = resolve_config_path("configs/cnn.yaml")
    assert bare.resolve() == prefixed.resolve()
    assert bare.resolve().name == "cnn.yaml"
    assert bare.exists()


def test_both_training_commands_use_the_shared_resolver():
    import ast

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("train_cnn.py", "train_unet.py"):
        source = (scripts / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        code = source.replace(ast.get_docstring(tree) or "", "")
        assert "resolve_config_path(" in code, name
        # the hand-rolled copy of the rule must be gone
        assert 'Path("configs") /' not in code, name


def test_the_recorded_config_path_is_repository_relative_for_both_commands():
    # resolve_config_path returns an absolute path; writing that into a
    # tracked artifact would leak a local username and break byte-identical
    # regeneration across machines.

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("train_cnn.py", "train_unet.py"):
        code = (scripts / name).read_text(encoding="utf-8")
        assert '"path": repository_path(config_path),' in code, name
        assert '"path": config_path.as_posix(),' not in code, name
        assert '"path": config_path.name,' not in code, name


# --------------------------------------------------------------------------
# no scientific training value can be set from the command line
# --------------------------------------------------------------------------

#: Arguments a training command may legitimately take: where to read from,
#: where to write to, which device, and one explicit destructive override.
#: None of these is a term in the experiment definition.
NON_SCIENTIFIC_ARGUMENTS = {
    "--help",
    "--root",
    "--manifest",
    "--cnn-config",
    "--unet-config",
    "--preprocessing-config",
    "--degradation-config",
    "--baseline-patients",
    "--run-dir",
    "--checkpoint",
    "--overwrite",
    "--device",
}

#: Every field that defines the experiment. A CLI override for any of these
#: would let a run carry the SHA-256 of a config describing a different
#: experiment, and the provenance gate would still pass.
FROZEN_SCIENTIFIC_FIELDS = (
    "seed",
    "epochs",
    "batch-size",
    "batch_size",
    "learning-rate",
    "learning_rate",
    "optimizer",
    "loss",
    "scheduler",
    "augmentation",
    "betas",
    "eps",
    "weight-decay",
    "weight_decay",
    "gradient-clipping",
    "mixed-precision",
    "primary-metric",
    "selection-metric",
    "tie-breaker",
    "sampler",
)


def _training_parser(name: str):
    import importlib.util
    import sys

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_parser()


def _options(name: str) -> set[str]:
    return {
        option
        for action in _training_parser(name)._actions
        for option in action.option_strings
        if option.startswith("--")
    }


@pytest.mark.parametrize("script", ["train_cnn", "train_unet"])
def test_the_training_parser_does_not_accept_an_epochs_override(script):
    assert "--epochs" not in _options(script)


@pytest.mark.parametrize("script", ["train_cnn", "train_unet"])
def test_an_epochs_override_is_rejected_by_the_parser(script):
    parser = _training_parser(script)
    with pytest.raises(SystemExit):
        parser.parse_args(["--epochs", "5"])


@pytest.mark.parametrize("script", ["train_cnn", "train_unet"])
def test_no_frozen_scientific_field_is_exposed_on_the_command_line(script):
    options = _options(script)
    for field in FROZEN_SCIENTIFIC_FIELDS:
        assert f"--{field}" not in options, f"{script} exposes --{field}"


@pytest.mark.parametrize("script", ["train_cnn", "train_unet"])
def test_every_training_argument_is_a_location_or_device_not_a_hyperparameter(script):
    # An allow-list, not a deny-list: a scientific override added later fails
    # this test even if nobody thought to forbid it by name.
    unexpected = _options(script) - NON_SCIENTIFIC_ARGUMENTS
    assert not unexpected, f"{script} gained non-location arguments: {sorted(unexpected)}"


@pytest.mark.parametrize("script", ["train_cnn.py", "train_unet.py"])
def test_the_epoch_budget_is_read_only_from_the_frozen_config(script):
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text(encoding="utf-8")
    assert 'epochs = int(training["epochs"])' in source, script
    assert "arguments.epochs" not in source, script
    assert "--epochs" not in source, script
