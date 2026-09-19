"""Making a training run repeatable, and being precise about what that means.

The claim this module supports
------------------------------
**Same repository, same config, same environment, same hardware, same seed
produces the same run.** That is the claim, and it is the useful one: it means
a reported number can be re-derived and a change in a result can be attributed
to the change that was made rather than to the draw.

It is explicitly **not** a claim of bitwise identity across different GPUs,
driver versions, CUDA versions or PyTorch builds. Floating-point reduction
order in cuDNN kernels depends on the hardware and the chosen algorithm, so
the same seed on a different card can and does produce slightly different
numbers. Anyone claiming otherwise has not checked.

What gets seeded
----------------
Python's :mod:`random`, NumPy's global RNG, the PyTorch CPU RNG and all CUDA
device RNGs. Note that the two things this benchmark cares most about are
already independent of all of them: the per-slice degradation derives its own
seed from the sample key, and the patient-balanced sampler derives its own
from ``(algorithm, seed, epoch)``. Seeding the global generators here is for
what is left - weight initialization, and anything a library reaches for
implicitly.

Determinism settings
--------------------
``torch.use_deterministic_algorithms(True)`` is set strictly rather than with
``warn_only``: if this training path ever reaches a nondeterministic kernel,
the run should fail loudly rather than quietly become irreproducible.
``cudnn.benchmark`` is disabled because algorithm auto-tuning picks kernels
based on timing, which is not reproducible; ``cudnn.deterministic`` is
enabled.

cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG`` set before its handle is created, so
this module sets it at import time. Import it before touching CUDA.
"""

from __future__ import annotations

import os
import random
from typing import Any

# Must be set before the first cuBLAS handle is created, which happens at the
# first CUDA matmul-like call. Setting it at import time is the only reliable
# place; setdefault so an explicit outer choice still wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: The workspace configuration cuBLAS needs for deterministic reductions.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class ReproducibilityError(RuntimeError):
    """The environment cannot support the determinism this run requires."""


def _require_integer(name: str, value: Any) -> int:
    """Accept only a genuine integer seed; ``bool`` would read as 0 or 1."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ReproducibilityError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}"
        )
    return int(value)


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device).

    Returns:
        What was seeded, for the run summary.
    """
    value = _require_integer("seed", seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return {
        "seed": value,
        "python_random": True,
        "numpy_global": True,
        "torch_cpu": True,
        "torch_cuda": bool(torch.cuda.is_available()),
    }


def enable_deterministic_algorithms(warn_only: bool = False) -> dict[str, Any]:
    """Turn on deterministic kernels and disable cuDNN autotuning.

    Args:
        warn_only: leave ``False`` for a canonical run. ``True`` downgrades a
            nondeterministic operation from an error to a warning, which is
            useful when diagnosing which operation is the problem and is not
            an acceptable setting for a reported result.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return {
        "use_deterministic_algorithms": True,
        "deterministic_warn_only": bool(warn_only),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def require_cuda() -> torch.device:
    """Return the CUDA device, or refuse.

    A canonical training run must not silently fall back to CPU: the result
    would be a different run from the one the summary describes, and the
    environment block would be quietly wrong.

    Raises:
        ReproducibilityError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise ReproducibilityError(
            "CUDA is not available. The canonical training run must not silently fall back "
            "to CPU: re-check the environment, or run with an explicit --device cpu for "
            "debugging only, which produces a non-canonical result."
        )
    return torch.device("cuda")


def describe_environment(device: torch.device | str | None = None) -> dict[str, Any]:
    """The environment facts a reported run has to carry with it.

    No timestamp and no hostname: a tracked summary should regenerate byte for
    byte, and a hostname is neither reproducible nor anyone's business.
    """
    facts: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device) if device is not None else None,
    }
    if torch.cuda.is_available():
        index = torch.device(device).index if device is not None else 0
        properties = torch.cuda.get_device_properties(index or 0)
        facts.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return facts


def dataloader_worker_seed(worker_id: int) -> None:  # pragma: no cover - runs in a subprocess
    """Give each DataLoader worker a distinct, run-derived seed.

    Not strictly needed here - the Dataset is a pure function of the slice and
    uses no randomness at all - but a worker that inherits an unseeded RNG is
    a trap waiting for the first person who adds augmentation.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
