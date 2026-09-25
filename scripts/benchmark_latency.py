"""Measure the frozen Milestone 12 inference-latency protocol, once.

    uv run python scripts/benchmark_latency.py --preflight-only
    uv run python scripts/benchmark_latency.py

Reads :file:`configs/latency/benchmark_plan.yaml`, proves every frozen input,
and only then times CLAHE on the CPU and the two frozen checkpoints - CNN
seed 2026 and U-Net seed 2026 - on the GPU, on one deterministic synthetic
input. No CT image of any split is read: the imaging root is sealed for the
whole run.

No timing option exists
-----------------------
There is no flag for the warm-up or measured iteration count, the batch size,
the input shape, the device, a method subset or a checkpoint. Every one is
fixed in the committed plan, and the plan is refused unless it agrees with the
frozen constants in :mod:`ct_restoration.latency`. The only flag is
``--preflight-only``.

``--preflight-only``
--------------------
Runs every check - plan, git, the held-out plan and quality summary, config
and checkpoint bytes, the GPU and its backend settings, the input digests,
parameter counts, and one untimed call per model under PyTorch's sync debug
mode - and stops. It never reaches a timer, creates no CUDA event, prints no
latency and writes nothing.

The real run
------------
The same checks, which must all pass; then ``outputs/latency.incomplete/`` is
created exclusively, every method is timed in the frozen order, the samples,
summary and receipt are written, and the directory moves to
``outputs/latency/``. No latency value is printed.

Refused, never repeated
-----------------------
There is no ``--overwrite`` and no retry. If either destination exists the run
refuses. A run that fails after measurement starts leaves its staging
directory and a failure record exactly where they are, and a completed run is
never repeated to obtain a different number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.config import PROJECT_ROOT  # noqa: E402
from ct_restoration.holdout import DicomOpenMonitor  # noqa: E402
from ct_restoration.latency import (  # noqa: E402
    FROZEN,
    PLAN_PATH,
    LatencyExecutionError,
    LatencyProtocolError,
    cuda_runtime,
    execute_protocol,
    load_frozen_models,
    load_plan,
    read_git_state,
    run_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run every check and stop; times nothing and writes nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if Path.cwd().resolve() != PROJECT_ROOT:
        print(f"error: run from the repository root, {PROJECT_ROOT}", file=sys.stderr)
        return 5

    root = Path(".")
    try:
        plan = load_plan(PLAN_PATH, FROZEN)
    except LatencyProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    runtime = cuda_runtime(plan["benchmark_device"])
    git = read_git_state(root, FROZEN)

    if arguments.preflight_only:
        with DicomOpenMonitor(root / FROZEN.data_root) as sealed:
            result = run_preflight(plan, root, git, runtime, load_frozen_models, FROZEN)
        for name, passed, detail in result.report.checks:
            print(
                f"  {'PASS' if passed else 'FAIL'}  {name}"
                + (f"  - {detail}" if not passed else "")
            )
        reads = sealed.record()
        opened = sum(reads["file_opens_by_split"].values()) + sum(
            reads["refused_opens_by_split"].values()
        )
        print(f"files opened or attempted under {FROZEN.data_root}: {opened}")
        failed = len(result.report.failed)
        print(
            f"preflight: {len(result.report.checks) - failed} passed, {failed} failed; "
            "nothing timed, nothing written"
        )
        return 0 if failed == 0 and opened == 0 else 3

    try:
        final = execute_protocol(plan, root, git, runtime, load_frozen_models, FROZEN)
    except LatencyProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except LatencyExecutionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    print(f"latency measured once; record written to {final.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
