"""Aggregate the ten Milestone 10 runs into the seed-level comparison.

    uv run python scripts/summarize_multiseed.py

Reads the frozen plan at :file:`configs/multiseed/plan.yaml` and requires
**every** run it declares. It will not summarise four seeds out of five.

Why it refuses so much
----------------------
A multi-seed result is a claim about stability, and the ways it can be
quietly wrong all look like ordinary output: a seed that failed to train and
was left out, two runs that both say "seed 2028" because a directory was
copied, a config edited between runs so two seeds are not actually the same
experiment, a NaN averaged into a mean. None of those announce themselves in
a table of numbers, so each is a refusal here rather than a footnote later.

The directory name is never the evidence. ``seed2028/`` is a label someone
typed; the seed is proved by agreement between the plan, the config's
``training.seed``, the checkpoint's recorded seed and the run summary's, all
of which are pinned by a hash the evaluation gate re-checks.

Milestone 10A note
------------------
This command exists before the runs it reads. With only seed 2026 trained it
refuses, correctly, and that refusal is the behaviour under test - the
synthetic fixtures in ``tests/test_multiseed.py`` exercise every path without
a CHAOS image.

Outputs
-------
``outputs/metrics/multiseed/seed_level_metrics.csv``
``outputs/metrics/multiseed/paired_seed_deltas.csv``
``outputs/metrics/multiseed/multiseed_summary.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_restoration.multiseed import (  # noqa: E402
    METHODS,
    METRICS,
    STATISTICAL_SEEDS,
    MultiseedError,
    overall_directional_claim,
    paired_metric_summary,
)
from ct_restoration.run_layout import (  # noqa: E402
    MULTISEED_METRICS_ROOT,
    run_directory,
    seed_metrics_directory,
)

#: The frozen experiment definition.
PLAN_PATH = Path("configs/multiseed/plan.yaml")

#: Canonical metrics root, where seed 2026's tables live.
CANONICAL_METRICS = Path("outputs/metrics")

#: The validation split has six patients. A summary reporting a different
#: count was computed over different data and is not comparable.
EXPECTED_VALIDATION_PATIENTS = 6

#: The only split any of this may read.
EXPECTED_SPLIT = "validation"

#: Splits that must never appear in a multi-seed artifact.
SEALED_SPLITS = ("test", "stress")

#: File names that mark a directory as holding a learned result for a seed.
#: A directory under the multiseed root containing any of these is a run;
#: a README or a stray note is not, and is left alone.
LEARNED_ARTIFACTS: tuple[str, ...] = tuple(
    f"{method}_{EXPECTED_SPLIT}_{suffix}"
    for method in ("cnn", "unet")
    for suffix in ("summary.json", "patients.csv", "slices.csv")
)

#: Reserved key: load_plan records which file it read and that file's hash,
#: so the summary can never quote one plan's SHA beside another plan's rules.
PLAN_SOURCE_KEY = "_source"


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    """The frozen plan, with its own internal consistency checked."""
    if not path.exists():
        raise MultiseedError(f"{path.as_posix()} is missing; the experiment has no definition.")
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))

    if PLAN_SOURCE_KEY in plan:
        raise MultiseedError(f"a plan file may not declare the reserved key {PLAN_SOURCE_KEY!r}")

    seeds = plan.get("statistical_seeds")
    if not isinstance(seeds, list) or len(set(seeds)) != len(seeds):
        raise MultiseedError(f"plan statistical_seeds must be a list of distinct seeds: {seeds!r}")
    # Fail closed against the pre-registered constant, not merely against the
    # plan's own internal consistency: a plan that quietly grew a sixth seed
    # or lost one is self-consistent and is still not this experiment.
    if sorted(seeds) != sorted(STATISTICAL_SEEDS):
        raise MultiseedError(
            f"plan statistical_seeds is {sorted(seeds)}, but the pre-registered Milestone 10 "
            f"seed set is {sorted(STATISTICAL_SEEDS)}. The seed list is part of the "
            "experiment definition; it cannot be added to or reduced after the fact."
        )
    if sorted(plan.get("methods", [])) != sorted(METHODS):
        raise MultiseedError(f"plan methods must be {list(METHODS)}, got {plan.get('methods')!r}")
    if plan.get("development_split") != EXPECTED_SPLIT:
        raise MultiseedError("plan development_split must be validation")
    if plan.get("test_allowed") or plan.get("stress_allowed"):
        raise MultiseedError("plan allows a sealed split; refusing to run")

    for method in METHODS:
        declared = sorted(plan["runs"][method])
        if declared != sorted(seeds):
            raise MultiseedError(
                f"plan runs.{method} declares seeds {declared}, but statistical_seeds is "
                f"{sorted(seeds)}. Every method runs at every seed."
            )

    # The hash travels with the plan it came from, so a summary cannot quote
    # configs/multiseed/plan.yaml's SHA while having been governed by another.
    plan[PLAN_SOURCE_KEY] = {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return plan


def metrics_summary_path(method: str, seed: int, canonical_seed: int) -> Path:
    """Where this run's validation summary lives."""
    directory = CANONICAL_METRICS if seed == canonical_seed else seed_metrics_directory(seed)
    return directory / f"{method}_validation_summary.json"


def _require_seed_value(label: str, value: Any, expected: int) -> int:
    """A seed witness: present, strictly typed, and equal to the plan's.

    Refuses to coerce. ``"2027"`` and ``2027.0`` and ``True`` are all rejected
    rather than read as 2027, because a witness of the wrong type means the
    artifact was not written by the canonical command, and a witness that
    merely looks right after coercion is not evidence of anything.
    """
    if value is None:
        raise MultiseedError(
            f"{label} is absent. Every seed witness is required: a run whose seed cannot "
            "be read is a run whose identity rests on its directory name."
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiseedError(f"{label} must be an integer, got {type(value).__name__} {value!r}")
    if value < 0:
        raise MultiseedError(f"{label} must be non-negative, got {value}")
    if value != expected:
        raise MultiseedError(
            f"{label} is {value}, but the plan places this run at seed {expected}. A "
            "directory name is not evidence of which seed produced its contents."
        )
    return value


def verify_seed_identity(
    method: str, seed: int, plan: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Prove the run's seed from four independent witnesses, all required.

    The plan, the config file's bytes and parsed ``training.seed``, the
    provenance the evaluator recorded for the checkpoint, and the training
    run summary must every one exist and agree. None is optional: an absent
    witness used to be treated as "nothing to contradict", which made the
    weakest artifact the easiest to pass.

    The config is the anchor. Its bytes are hashed into the checkpoint and
    re-hashed by the evaluation gate, so editing it to match a run breaks the
    chain rather than repairing it.

    Raises:
        MultiseedError: any witness is missing, malformed, or disagrees.
    """
    entry = plan["runs"].get(method, {}).get(seed)
    if not isinstance(entry, dict) or "config" not in entry or "sha256" not in entry:
        raise MultiseedError(
            f"the plan has no complete entry for {method} seed {seed}; a run with no "
            "pre-registered config cannot be part of this experiment."
        )

    # --- B. the config file itself -----------------------------------------
    config_path = Path(entry["config"])
    if not config_path.exists():
        raise MultiseedError(
            f"{method} seed {seed}: the plan pins {config_path.as_posix()}, which does not "
            "exist. The config defines the run; without it the run cannot be verified."
        )
    actual_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if actual_sha != entry["sha256"]:
        raise MultiseedError(
            f"{config_path.as_posix()} hashes to {actual_sha}, but the plan pins "
            f"{entry['sha256']}. A frozen config changed after the plan was written."
        )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = document.get("training") if isinstance(document, dict) else None
    if not isinstance(training, dict):
        raise MultiseedError(f"{config_path.as_posix()} declares no training section")
    config_seed = _require_seed_value(
        f"{config_path.as_posix()} training.seed", training.get("seed"), seed
    )

    # --- C. provenance recorded in the metric summary ----------------------
    checkpoint = summary.get("checkpoint") or {}
    provenance = checkpoint.get("provenance") or {}
    recorded_config_sha = checkpoint.get("config_sha256", provenance.get("config_sha256"))
    if recorded_config_sha is None:
        raise MultiseedError(
            f"{method} seed {seed}: the metric summary records no checkpoint config SHA, so "
            "there is no evidence which config the scored checkpoint came from."
        )
    if recorded_config_sha != entry["sha256"]:
        raise MultiseedError(
            f"{method} seed {seed}: the scored checkpoint was produced from a config whose "
            f"SHA-256 is {recorded_config_sha!r}, but the plan pins {entry['sha256']!r} for "
            f"{entry['config']}. This run came from a different experiment definition."
        )
    recorded_seed = _require_seed_value(
        f"{method} seed {seed}: the metric summary's recorded seed",
        provenance.get("seed", checkpoint.get("seed")),
        seed,
    )

    # --- D. the training run summary ---------------------------------------
    run_summary_path = run_directory(method, seed) / "run_summary.json"
    if not run_summary_path.exists():
        raise MultiseedError(
            f"{method} seed {seed}: {run_summary_path.as_posix()} is missing. The tracked "
            "record of the training run is required evidence of its seed."
        )
    run_document = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_seed = _require_seed_value(
        f"{run_summary_path.as_posix()} training.seed",
        (run_document.get("training") or {}).get("seed"),
        seed,
    )
    run_config_sha = (run_document.get("config") or {}).get("sha256")
    if run_config_sha is not None and run_config_sha != entry["sha256"]:
        raise MultiseedError(
            f"{run_summary_path.as_posix()} records config SHA {run_config_sha!r}, but the "
            f"plan pins {entry['sha256']!r}. The training run and the plan disagree about "
            "which config defined this run."
        )

    # --- E. agreement -------------------------------------------------------
    return {
        "plan_seed": seed,
        "config_path": config_path.as_posix(),
        "config_sha256": entry["sha256"],
        "config_training_seed": config_seed,
        "metric_summary_recorded_seed": recorded_seed,
        "metric_summary_recorded_config_sha256": recorded_config_sha,
        "run_summary_seed": run_seed,
        "run_summary_config_sha256": run_config_sha,
        "witnesses_checked": 4,
        "agree": True,
    }


def validate_run_summary(method: str, seed: int, summary: dict[str, Any]) -> dict[str, float]:
    """Check one run's summary and return its eight patient-weighted means."""
    if summary.get("split") != EXPECTED_SPLIT:
        raise MultiseedError(
            f"{method} seed {seed}: split is {summary.get('split')!r}, expected "
            f"{EXPECTED_SPLIT!r}. Multi-seed analysis reads validation only."
        )
    for sealed in SEALED_SPLITS:
        if sealed in json.dumps(summary.get("outputs", {})):
            raise MultiseedError(
                f"{method} seed {seed}: the artifact references the sealed {sealed!r} split."
            )
    patients = summary.get("patients")
    if patients != EXPECTED_VALIDATION_PATIENTS:
        raise MultiseedError(
            f"{method} seed {seed}: summary reports {patients!r} patients, expected "
            f"{EXPECTED_VALIDATION_PATIENTS}. A different patient count is a different "
            "measurement and is not comparable across seeds."
        )

    primary = summary.get("primary_result", {})
    missing = [metric for metric in METRICS if metric not in primary]
    if missing:
        raise MultiseedError(
            f"{method} seed {seed}: missing metrics {missing}. All {len(METRICS)} are "
            "mandatory; a partial metric set cannot be compared against a full one."
        )
    values: dict[str, float] = {}
    for metric in METRICS:
        block = primary[metric]
        if not isinstance(block, dict) or "mean" not in block:
            raise MultiseedError(f"{method} seed {seed}: {metric} has no patient-weighted mean")
        values[metric] = block["mean"]
    # Caught on the run that produced it, named, rather than inside a mean
    # where a single NaN silently turns all eight summary figures into NaN.
    for metric, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MultiseedError(
                f"{method} seed {seed}: {metric} is {type(value).__name__} {value!r}"
            )
        if not math.isfinite(float(value)):
            raise MultiseedError(
                f"{method} seed {seed}: {metric} is {value!r}. A non-finite metric means "
                "that run is broken; it is never averaged in or silently skipped."
            )
    return values


def discover_unplanned_seeds(plan: dict[str, Any], root: Path | None = None) -> list[int]:
    """Seeds that have learned results on disk but are not in the plan.

    Looking for what should NOT be there, which iterating the plan can never
    do: a loop over planned seeds finds a sixth seed's directory exactly as
    often as it finds nothing, which is never. The earlier version of this
    check subtracted the planned seeds from a dict built out of the planned
    seeds and was therefore always empty.

    Only learned metric artifacts count. A README under the multiseed root is
    documentation, not a run, and is ignored.
    """
    directory = MULTISEED_METRICS_ROOT if root is None else root
    if not directory.is_dir():
        return []
    planned = set(plan["statistical_seeds"])
    found: set[int] = set()
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("seed"):
            continue
        label = entry.name[len("seed") :]
        if not label.isdigit():
            continue
        if any((entry / name).exists() for name in LEARNED_ARTIFACTS):
            seed = int(label)
            if seed not in planned:
                found.add(seed)
    return sorted(found)


def collect(plan: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    """Every planned run's eight metrics, or a refusal naming what is missing."""
    canonical_seed = plan["canonical_existing_seed"]
    seeds = sorted(plan["statistical_seeds"])

    # Before anything is read or written: an unplanned result is as much a
    # deviation from the pre-registered design as a missing one, and silently
    # leaving it out would make the report a summary of the seeds that suited
    # it rather than of the experiment that was declared.
    unplanned = discover_unplanned_seeds(plan)
    if unplanned:
        raise MultiseedError(
            f"learned results exist for unplanned seed(s) {unplanned}, which are not in the "
            f"frozen plan's seed set {seeds}. Adding a seed after results exist breaks the "
            "pre-registration: the reported spread would become the spread of whichever "
            "seeds were kept. Either this result belongs to a different experiment and must "
            "be moved out of the multiseed metrics root, or the plan was not the plan."
        )
    collected: dict[str, dict[int, dict[str, float]]] = {method: {} for method in METHODS}
    absent: list[str] = []

    for method in METHODS:
        for seed in seeds:
            path = metrics_summary_path(method, seed, canonical_seed)
            if not path.exists():
                absent.append(f"{method} seed {seed} ({path.as_posix()})")
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            verify_seed_identity(method, seed, plan, summary)
            collected[method][seed] = validate_run_summary(method, seed, summary)

    if absent:
        raise MultiseedError(
            "the experiment is incomplete; "
            f"{len(absent)} of {len(seeds) * len(METHODS)} planned runs are missing:\n  "
            + "\n  ".join(absent)
            + "\nA multi-seed summary over the runs that happen to exist is a summary of "
            "which runs finished, not of the experiment that was pre-registered."
        )

    # Both dictionaries were filled from the same planned seed list and the
    # absent check above would have fired otherwise, so they necessarily
    # match here; the real unplanned-seed check is the disk scan above.
    assert sorted(collected["cnn"]) == sorted(collected["unet"]) == seeds
    return collected


def build_summary(
    plan: dict[str, Any], collected: dict[str, dict[int, dict[str, float]]]
) -> dict[str, Any]:
    """The predeclared seed-level summary."""
    per_metric = {
        metric: paired_metric_summary(
            metric,
            {seed: values[metric] for seed, values in collected["cnn"].items()},
            {seed: values[metric] for seed, values in collected["unet"].items()},
        )
        for metric in METRICS
    }
    return {
        "milestone": 10,
        "result_class": (
            "MULTI-SEED VALIDATION DEVELOPMENT RESULT. Five training seeds per architecture "
            "on fixed data. This measures the spread of the training procedure, not "
            "variation across patients, scanners or institutions. The test split remains "
            "sealed."
        ),
        "plan_path": plan[PLAN_SOURCE_KEY]["path"],
        "plan_sha256": plan[PLAN_SOURCE_KEY]["sha256"],
        "seeds": sorted(plan["statistical_seeds"]),
        "split": EXPECTED_SPLIT,
        "patients": EXPECTED_VALIDATION_PATIENTS,
        "unit_of_analysis": "statistical training seed, after per-run patient aggregation",
        "pooled_patients_and_seeds": False,
        "significance_testing": False,
        "per_metric": per_metric,
        "overall": overall_directional_claim(per_metric),
    }


def build_tables(collected: dict[str, dict[int, dict[str, float]]], summary: dict[str, Any]):
    """The two flat tables that accompany the JSON."""
    seed_rows = [
        {"method": method, "seed": seed, **{metric: values[metric] for metric in METRICS}}
        for method in METHODS
        for seed, values in sorted(collected[method].items())
    ]
    delta_rows = [
        {
            "seed": seed,
            "metric": metric,
            "cnn": collected["cnn"][seed][metric],
            "unet": collected["unet"][seed][metric],
            "raw_delta_unet_minus_cnn": summary["per_metric"][metric]["raw_delta_by_seed"][seed],
            "oriented_improvement_positive_is_unet": summary["per_metric"][metric][
                "oriented_improvement_by_seed"
            ][seed],
        }
        for metric in METRICS
        for seed in summary["seeds"]
    ]
    return pd.DataFrame(seed_rows), pd.DataFrame(delta_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN_PATH.as_posix())
    parser.add_argument("--output-dir", default=MULTISEED_METRICS_ROOT.as_posix())
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        plan = load_plan(Path(arguments.plan))
        collected = collect(plan)
    except MultiseedError as error:
        print(f"error: {error}", file=sys.stderr)
        print("no multi-seed artifact was written.", file=sys.stderr)
        return 9

    summary = build_summary(plan, collected)
    seed_table, delta_table = build_tables(collected, summary)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(output_dir / "seed_level_metrics.csv", index=False, lineterminator="\n")
    delta_table.to_csv(output_dir / "paired_seed_deltas.csv", index=False, lineterminator="\n")
    (output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"seeds           : {summary['seeds']}")
    for metric in METRICS:
        block = summary["per_metric"][metric]
        print(
            f"  {metric:<10} U-Net better on {block['seeds_favouring_unet']}/"
            f"{len(block['seeds'])} seeds   mean oriented improvement "
            f"{block['oriented_improvement']['mean']:+.6f} "
            f"(sd {block['oriented_improvement']['std']:.6f})"
        )
    print(f"overall         : {summary['overall']['statement']}")
    print(f"summary         : {(output_dir / 'multiseed_summary.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
