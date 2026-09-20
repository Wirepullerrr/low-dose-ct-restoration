"""Tests for run destinations: how they are named, and when writing is refused.

The refusal is the reason this module exists. Checkpoints are git-ignored, so
overwriting one destroys it permanently, and the tracked run summary pins its
SHA-256 - a retrained replacement fails the evaluation provenance gate rather
than quietly taking its place. Multi-seed work runs the training command many
times with only a seed changing, which is exactly the situation where a
forgotten ``--run-dir`` costs a result that cannot be rebuilt.

Fully synthetic: nothing here opens a CHAOS file, a checkpoint or the GPU.
Two tests invoke the real training commands as subprocesses, deliberately
pointing them at a dataset root that does not exist, to prove the refusal
happens before any data is touched.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from ct_restoration.run_layout import (
    CANONICAL_RUN_ARTIFACTS,
    METHODS,
    METRICS_ROOT,
    MULTISEED_METRICS_ROOT,
    RunLayoutError,
    checkpoint_path,
    describe_run_destination,
    require_method,
    require_seed,
    require_writable_run_destination,
    run_directory,
    seed_metrics_directory,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


# ---------------------------------------------------------------------------
# 1. the overwrite guard
# ---------------------------------------------------------------------------


def _occupy(tmp_path, *, checkpoint: bool, run_dir: bool):
    destination = tmp_path / "run"
    ckpt = tmp_path / "best.pt"
    if run_dir:
        destination.mkdir()
        (destination / "run_summary.json").write_text("{}", encoding="utf-8")
    if checkpoint:
        ckpt.write_bytes(b"not really a checkpoint")
    return destination, ckpt


def test_an_existing_checkpoint_is_refused(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=True, run_dir=False)
    with pytest.raises(RunLayoutError, match="already exists"):
        require_writable_run_destination(run_dir, ckpt)


def test_a_non_empty_run_directory_is_refused(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=False, run_dir=True)
    with pytest.raises(RunLayoutError, match="already contains"):
        require_writable_run_destination(run_dir, ckpt)


def test_both_occupied_is_refused_and_both_are_named(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=True, run_dir=True)
    with pytest.raises(RunLayoutError) as caught:
        require_writable_run_destination(run_dir, ckpt)
    message = str(caught.value)
    assert "checkpoint" in message
    assert "run directory" in message


def test_a_run_directory_holding_an_unrecognised_file_is_still_refused(tmp_path):
    # Occupied means occupied. A directory with something unexpected in it is
    # a situation to look at, not one to write a training run into.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stray_notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RunLayoutError, match="already contains"):
        require_writable_run_destination(run_dir, tmp_path / "best.pt")


def test_a_fresh_destination_is_accepted(tmp_path):
    report = require_writable_run_destination(tmp_path / "new", tmp_path / "new.pt")
    assert report["destination_was_occupied"] is False
    assert report["overwrite_authorized"] is False
    assert report["checked_before_any_training_work"] is True


def test_an_existing_but_empty_run_directory_is_accepted(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = require_writable_run_destination(run_dir, tmp_path / "best.pt")
    assert report["destination_was_occupied"] is False


def test_overwrite_permits_replacement_only_when_supplied(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=True, run_dir=True)
    with pytest.raises(RunLayoutError):
        require_writable_run_destination(run_dir, ckpt, overwrite=False)
    report = require_writable_run_destination(run_dir, ckpt, overwrite=True)
    assert report["overwrite_authorized"] is True
    assert report["destination_was_occupied"] is True


def test_the_guard_writes_nothing_and_creates_nothing(tmp_path):
    # It must be safe to call before deciding to train: a guard that created
    # the directory it was asked about would make the second call pass.
    run_dir = tmp_path / "never_created"
    ckpt = tmp_path / "never_created.pt"
    require_writable_run_destination(run_dir, ckpt)
    assert not run_dir.exists()
    assert not ckpt.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_describe_reports_occupancy_without_judging_it(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=True, run_dir=True)
    state = describe_run_destination(run_dir, ckpt)
    assert state["checkpoint_exists"] is True
    assert state["run_dir_entries"] == 1
    assert state["run_dir_canonical_artifacts"] == ["run_summary.json"]


def test_the_refusal_explains_why_a_checkpoint_cannot_just_be_remade(tmp_path):
    run_dir, ckpt = _occupy(tmp_path, checkpoint=True, run_dir=False)
    with pytest.raises(RunLayoutError) as caught:
        require_writable_run_destination(run_dir, ckpt)
    message = str(caught.value)
    assert "not tracked in git" in message
    assert "provenance gate" in message
    assert "--overwrite" in message


# ---------------------------------------------------------------------------
# 2. the canonical seed-2026 destinations are protected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_the_canonical_destinations_are_occupied_and_therefore_refused(method):
    # This is the Milestone 8 / Milestone 9 protection, stated as a test: the
    # default destination of each training command already holds a completed
    # run, so the default command cannot start.
    run_dir = run_directory(method, 2026)
    ckpt = checkpoint_path(method, 2026)
    if not run_dir.exists():
        pytest.skip(f"{run_dir} not present in this checkout")
    with pytest.raises(RunLayoutError):
        require_writable_run_destination(run_dir, ckpt)


@pytest.mark.parametrize(("script", "method"), [("train_cnn.py", "cnn"), ("train_unet.py", "unet")])
def test_the_training_script_defaults_point_at_the_canonical_destination(script, method):
    source = (SCRIPTS / script).read_text(encoding="utf-8")
    assert f'RUN_DIR = Path("{run_directory(method, 2026).as_posix()}")' in source
    assert f'CHECKPOINT_PATH = Path("{checkpoint_path(method, 2026).as_posix()}")' in source


@pytest.mark.parametrize("script", ["train_cnn.py", "train_unet.py"])
def test_the_guard_runs_before_any_data_or_optimizer_work(script):
    # Order matters more than presence: a guard that ran after the dataset
    # was built would still refuse, but only after reading images.
    tree = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
    main = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    # ast.walk yields breadth-first, not in source order, so sort by line.
    calls = sorted(
        (
            (node.lineno, node.func.id)
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ),
    )
    names = [name for _, name in calls]
    assert "require_writable_run_destination" in names, script
    guard = names.index("require_writable_run_destination")
    for later in ("development_dataset", "make_training_loader", "build_model", "require_cuda"):
        if later in names:
            assert guard < names.index(later), f"{script}: guard runs after {later}"


@pytest.mark.parametrize("script", ["train_cnn.py", "train_unet.py"])
def test_the_real_command_refuses_without_touching_a_dataset(script):
    # Invoked with a dataset root that does not exist. Reaching the refusal
    # proves nothing opened the data: if the guard ran late, this would fail
    # with a missing-manifest error instead.
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / script),
            "--root",
            "no/such/dataset",
            "--manifest",
            "no/such/manifest.csv",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    assert completed.returncode == 7, completed.stderr[-2000:]
    assert "Refusing to start training" in completed.stderr


# ---------------------------------------------------------------------------
# 3. the future run / checkpoint path contract
# ---------------------------------------------------------------------------


def test_the_seed_2026_paths_are_exactly_what_milestone_8_and_9_used():
    assert run_directory("cnn", 2026) == Path("outputs/runs/cnn_seed2026")
    assert run_directory("unet", 2026) == Path("outputs/runs/unet_seed2026")
    assert checkpoint_path("cnn", 2026) == Path("outputs/checkpoints/cnn_seed2026_best.pt")
    assert checkpoint_path("unet", 2026) == Path("outputs/checkpoints/unet_seed2026_best.pt")


@pytest.mark.parametrize("seed", [0, 7, 2027, 123456])
@pytest.mark.parametrize("method", METHODS)
def test_a_future_seed_builds_a_distinct_destination(method, seed):
    assert run_directory(method, seed) != run_directory(method, 2026) or seed == 2026
    assert checkpoint_path(method, seed).name.endswith(f"seed{seed}_best.pt")


def test_no_two_methods_or_seeds_share_a_destination():
    destinations = [run_directory(method, seed) for method in METHODS for seed in (2026, 2027, 7)]
    assert len(set(destinations)) == len(destinations)


@pytest.mark.parametrize("bad", [True, False, 2026.0, "2026", None, [2026]])
def test_a_seed_that_is_not_a_genuine_integer_is_refused(bad):
    with pytest.raises(RunLayoutError, match="must be an integer"):
        require_seed(bad)


def test_a_negative_seed_is_refused():
    with pytest.raises(RunLayoutError, match="non-negative"):
        require_seed(-1)


@pytest.mark.parametrize("bad", ["CNN", "u-net", "clahe", "", 1, None])
def test_an_unknown_method_is_refused(bad):
    with pytest.raises(RunLayoutError, match="method must be one of"):
        require_method(bad)


def test_the_canonical_artifact_names_are_the_ones_training_writes():
    assert set(CANONICAL_RUN_ARTIFACTS) == {"run_summary.json", "training_history.csv"}


# ---------------------------------------------------------------------------
# 4. the per-seed metric directory layout
# ---------------------------------------------------------------------------


def test_a_seed_metric_directory_is_never_the_canonical_one():
    # The Milestone 8 and 9 tables live at the root of outputs/metrics. No
    # additional seed may write there.
    assert seed_metrics_directory(2027) != METRICS_ROOT
    assert seed_metrics_directory(2026) != METRICS_ROOT
    assert MULTISEED_METRICS_ROOT.parent == METRICS_ROOT


def test_seed_metric_directories_do_not_collide():
    directories = [seed_metrics_directory(seed) for seed in (2026, 2027, 7, 0)]
    assert len(set(directories)) == len(directories)


def test_the_seed_metric_directory_is_the_documented_shape():
    assert seed_metrics_directory(2027) == Path("outputs/metrics/multiseed/seed2027")


@pytest.mark.parametrize("bad", [True, "2027", 2027.0])
def test_a_seed_metric_directory_refuses_a_malformed_seed(bad):
    with pytest.raises(RunLayoutError):
        seed_metrics_directory(bad)
