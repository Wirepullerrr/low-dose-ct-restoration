"""Execute the frozen Milestone 11 held-out test protocol, once.

    uv run python scripts/run_holdout_test.py --preflight-only
    uv run python scripts/run_holdout_test.py

The one authorized route to held-out test image content. It reads
:file:`configs/holdout/test_plan.yaml`, proves every frozen input, and only
then opens the six test patients and scores all four methods - the degraded
baseline, CLAHE, and the five frozen checkpoints of each learned architecture
- through the Milestone 5-10 benchmark code, on one shared degraded input per
slice.

No scientific option exists
---------------------------
There is no flag for a seed, a method, a metric, a split, a CLAHE parameter,
the degradation, a checkpoint, a device, a batch size or an output location.
Every one of those is fixed in the committed plan, and the plan is refused
unless it agrees with the frozen constants in
:mod:`ct_restoration.holdout`. The only flag is ``--preflight-only``.

``--preflight-only``
--------------------
Runs every check - plan, git, split and manifest hashes, the cohort from
metadata, every config, every tracked run record, every checkpoint's bytes
and provenance, absent output destinations - and stops. It cannot read a test
image: it never builds the test rows, never reaches the scoring loop, and runs
inside an audit-hook monitor that refuses to open any file under the imaging
root at all. It writes nothing.

The real run
------------
The same checks, which must all pass; then the staging directory
``outputs/metrics/holdout/test.incomplete/`` is created exclusively and the
opening recorded before the first test file is opened. When every table is
written and checked, the execution receipt is written last and the directory
moves to ``outputs/metrics/holdout/test/``. No metric value is printed.

Refused, never resumed
----------------------
There is no ``--overwrite`` and no retry. If either destination exists the
run refuses. If execution fails after opening, the staging directory, its log
and a failure record stay exactly where they are; nothing is deleted and
nothing is retried, and whether any re-run is permitted is a decision for an
integrity review. Stress is never opened.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.config import PROJECT_ROOT  # noqa: E402
from ct_restoration.holdout import (  # noqa: E402
    FROZEN,
    PLAN_PATH,
    DicomOpenMonitor,
    HoldoutExecutionError,
    HoldoutProtocolError,
    environment_facts,
    execute_protocol,
    load_learned_restorers,
    load_test_plan,
    read_git_state,
    run_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run every check and stop; reads no test image and writes nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    # Every path in the plan and in the tracked artifacts is repository
    # relative, and every artifact this writes must stay that way.
    if Path.cwd().resolve() != PROJECT_ROOT:
        print(f"error: run from the repository root, {PROJECT_ROOT}", file=sys.stderr)
        return 5

    root = Path(".")
    try:
        plan = load_test_plan(PLAN_PATH, FROZEN)
    except HoldoutProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    import torch

    cuda = torch.cuda.is_available()
    loader = functools.partial(load_learned_restorers, device=plan["device"] if cuda else "cpu")
    git = read_git_state(root, FROZEN)

    if arguments.preflight_only:
        with DicomOpenMonitor(root / plan["data_root"]) as sealed:
            result = run_preflight(plan, root, git, loader, FROZEN)
        result.report.add("CUDA available for the frozen device", cuda, plan["device"])
        for name, passed, detail in result.report.checks:
            print(
                f"  {'PASS' if passed else 'FAIL'}  {name}"
                + (f"  - {detail}" if not passed else "")
            )
        reads = sealed.record()
        opened = sum(sealed.opened.values()) + sum(sealed.refused.values())
        print(f"files opened or attempted under {plan['data_root']}: {opened}")
        print(
            f"test files opened: {reads['file_opens_by_split']['test']}; "
            f"stress files opened: {reads['file_opens_by_split']['stress']}; nothing written"
        )
        failed = len(result.report.failed)
        print(f"preflight: {len(result.report.checks) - failed} passed, {failed} failed")
        return 0 if failed == 0 and opened == 0 else 3

    if not cuda:
        print(
            f"error: the frozen device is {plan['device']!r} and CUDA is not available. "
            "The protocol does not fall back to CPU.",
            file=sys.stderr,
        )
        return 5
    try:
        execute_protocol(plan, root, git, loader, environment_facts(plan["device"]), FROZEN)
    except HoldoutProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        print("no test image was read and nothing was written.", file=sys.stderr)
        return 3
    except HoldoutExecutionError as error:
        print(f"error: {error}", file=sys.stderr)
        print(
            f"execution stopped after the test split was opened. The record is preserved "
            f"in {plan['outputs']['staging']}; do not read it for results and do not re-run "
            "without an integrity review.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
