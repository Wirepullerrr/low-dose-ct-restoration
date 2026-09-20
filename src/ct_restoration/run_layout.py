"""Where a training run writes, and the refusal that stops it clobbering another.

Two separate jobs, kept in one module because they are the same concern seen
from two sides: naming a run's destination, and proving that destination is
free before anything expensive or destructive happens.

Why the refusal exists
----------------------
Until this module, ``scripts/train_cnn.py`` with no arguments would silently
overwrite ``outputs/checkpoints/cnn_seed2026_best.pt``. Checkpoints are
git-ignored, so that file is not recoverable from history; and the tracked
run summary pins its SHA-256, so a retrained replacement would correctly fail
the evaluation provenance gate. One forgetful command would therefore make
the Milestone 8 and Milestone 9 results permanently unverifiable.

That risk is acceptable for a single hand-run experiment and unacceptable for
multi-seed automation, which runs the training command many times with only a
seed changing between invocations.

Why the paths are built here rather than spelled out at each call site
----------------------------------------------------------------------
``f"outputs/runs/{method}_seed{seed}"`` written out by hand in five places is
five chances to disagree about whether the U-Net's directory is ``unet_seed7``
or ``u_net_seed7``, and a disagreement between the trainer and the evaluator
is a run that cannot be scored. One function, strictly validated, means a
future seed cannot be spelled two ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The two learned methods this benchmark trains. CLAHE and the degraded
#: baseline are not trained and have no run directory.
METHODS: tuple[str, ...] = ("cnn", "unet")

#: Root for per-run training records (tracked in git).
RUNS_ROOT = Path("outputs/runs")

#: Root for trained weights (git-ignored; see the module docstring).
CHECKPOINTS_ROOT = Path("outputs/checkpoints")

#: Root for benchmark metrics.
METRICS_ROOT = Path("outputs/metrics")

#: Where additional statistical seeds write their metrics, one directory per
#: seed, so a new seed can never overwrite the canonical single-seed result.
MULTISEED_METRICS_ROOT = METRICS_ROOT / "multiseed"

#: Files that mark a directory as holding a real run rather than scratch.
CANONICAL_RUN_ARTIFACTS: tuple[str, ...] = ("run_summary.json", "training_history.csv")


class RunLayoutError(RuntimeError):
    """A run destination is invalid, or is already occupied."""


def require_method(method: Any) -> str:
    """One of the two learned methods, spelled exactly.

    Raises:
        RunLayoutError: ``method`` is not ``"cnn"`` or ``"unet"``.
    """
    if not isinstance(method, str) or method not in METHODS:
        raise RunLayoutError(
            f"method must be one of {list(METHODS)}, got {type(method).__name__} {method!r}"
        )
    return method


def require_seed(seed: Any) -> int:
    """A genuine non-negative integer seed.

    Deliberately refuses to coerce. ``int(2026.9)`` is 2026 and ``int("2026")``
    is 2026, so a coercing reader would accept a malformed seed and then build
    a directory name that disagrees with what the caller believed it asked
    for. ``True`` is an ``int`` in Python and is refused explicitly.

    Raises:
        RunLayoutError: ``seed`` is not an integer, or is negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RunLayoutError(
            f"seed must be an integer, got {type(seed).__name__} {seed!r}. A seed is part "
            "of a run's identity and is never parsed from a string or rounded from a float."
        )
    if seed < 0:
        raise RunLayoutError(f"seed must be non-negative, got {seed}")
    return seed


def run_directory(method: Any, seed: Any) -> Path:
    """``outputs/runs/<method>_seed<seed>``."""
    return RUNS_ROOT / f"{require_method(method)}_seed{require_seed(seed)}"


def checkpoint_path(method: Any, seed: Any) -> Path:
    """``outputs/checkpoints/<method>_seed<seed>_best.pt``."""
    return CHECKPOINTS_ROOT / f"{require_method(method)}_seed{require_seed(seed)}_best.pt"


def seed_metrics_directory(seed: Any) -> Path:
    """``outputs/metrics/multiseed/seed<seed>`` - one directory per extra seed.

    The canonical single-seed metrics stay where they are, at the root of
    ``outputs/metrics``. A later seed never writes there, so no automation
    mistake can overwrite the Milestone 8 or Milestone 9 numbers.
    """
    return MULTISEED_METRICS_ROOT / f"seed{require_seed(seed)}"


def describe_run_destination(run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    """What already exists at a destination, without judging it."""
    existing_artifacts = (
        sorted(name for name in CANONICAL_RUN_ARTIFACTS if (run_dir / name).exists())
        if run_dir.is_dir()
        else []
    )
    contents = sorted(entry.name for entry in run_dir.iterdir()) if run_dir.is_dir() else []
    return {
        "run_dir": run_dir.as_posix(),
        "run_dir_exists": run_dir.exists(),
        "run_dir_entries": len(contents),
        "run_dir_canonical_artifacts": existing_artifacts,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_exists": checkpoint.exists(),
    }


def require_writable_run_destination(
    run_dir: Path,
    checkpoint: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Refuse to start a training run that would destroy an existing one.

    Called before the dataset is opened, before the optimizer exists and
    before a single gradient step, so a refused run costs an error message
    rather than an hour of GPU time and a lost checkpoint.

    A run directory counts as occupied if it holds *anything*, not merely a
    recognised artifact: a directory with an unexpected file in it is a
    situation to look at, not to write into.

    Args:
        run_dir: where the training history and run summary would be written.
        checkpoint: where the selected weights would be written.
        overwrite: caller explicitly authorises replacing both. Intended for
            deliberate re-runs of a seed whose result is *not* relied upon.
            Multi-seed automation must not pass this: each seed gets its own
            destination, so needing it means the destination was wrong.

    Returns:
        A record of what was checked, for the run summary.

    Raises:
        RunLayoutError: the destination is occupied and ``overwrite`` is not set.
    """
    state = describe_run_destination(run_dir, checkpoint)
    occupied: list[str] = []
    if state["checkpoint_exists"]:
        occupied.append(f"checkpoint {checkpoint.as_posix()} already exists")
    if state["run_dir_entries"]:
        artifacts = state["run_dir_canonical_artifacts"]
        detail = f" (including {', '.join(artifacts)})" if artifacts else ""
        occupied.append(
            f"run directory {run_dir.as_posix()} already contains "
            f"{state['run_dir_entries']} entries{detail}"
        )

    if occupied and not overwrite:
        raise RunLayoutError(
            "Refusing to start training: "
            + "; and ".join(occupied)
            + ". Training would replace a completed run. Checkpoints are not tracked in "
            "git, so an overwritten one cannot be recovered, and the tracked run summary "
            "pins its SHA-256 - a retrained replacement would fail the evaluation "
            "provenance gate rather than quietly substitute itself. Write this run "
            "somewhere else with --run-dir and --checkpoint, or pass --overwrite if you "
            "genuinely intend to discard what is there."
        )

    return {
        **state,
        "overwrite_authorized": bool(overwrite),
        "destination_was_occupied": bool(occupied),
        "checked_before_any_training_work": True,
    }
