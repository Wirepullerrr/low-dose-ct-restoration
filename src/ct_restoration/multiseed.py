"""The Milestone 10 multi-seed analysis, defined before any result exists.

Everything here was written while the only trained runs were seed 2026's, so
none of these rules could have been chosen to suit a number. That is the point
of the module: the direction that counts as "better", the statistic that gets
reported, and the words allowed in the conclusion are all fixed in advance.

What a seed is, and is not
--------------------------
One seed is one repetition of the same training procedure under different
training randomness - weight initialization and patient-balanced sampler
ordering. Five seeds are five such repetitions on **fixed data**: the same 25
training patients, the same 6 validation patients, the same frozen per-slice
degradation. They are not five patients, not a sample from a population, and
not an independent replication of the data.

So the spread this module measures is the spread of the training procedure on
this dataset. It says nothing about variation across patients, scanners or
institutions.

The unit of analysis
--------------------
Each run collapses slice -> patient -> equal-weight patient mean first,
exactly as Milestones 5-9 do. Only then are seeds compared. The six patient
values inside one run are not independent repetitions of training, and the
five seed values are not independent patients, so the two levels are never
pooled into thirty observations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: The eight reported metrics, in report order. All eight are mandatory.
METRICS: tuple[str, ...] = (
    "full_mae",
    "full_mse",
    "full_psnr",
    "full_ssim",
    "body_mae",
    "body_mse",
    "body_psnr",
    "body_ssim",
)

#: Reported first because the project has emphasised image quality since
#: Milestone 5. They do NOT override contradictory MAE/MSE evidence, and there
#: is deliberately no composite score that would let them do so.
HEADLINE_METRICS: tuple[str, ...] = ("full_psnr", "full_ssim", "body_psnr", "body_ssim")

#: Metrics where a smaller value is a better restoration.
LOWER_IS_BETTER: frozenset[str] = frozenset({"full_mae", "full_mse", "body_mae", "body_mse"})

#: Metrics where a larger value is a better restoration.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"full_psnr", "full_ssim", "body_psnr", "body_ssim"})

#: The frozen seed set. Adding a seed because a result looks unusual, or
#: dropping one because it looks poor, would turn the reported spread into the
#: spread of whichever seeds survived inspection.
STATISTICAL_SEEDS: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)

#: Both learned architectures run at every seed.
METHODS: tuple[str, ...] = ("cnn", "unet")

#: Sample standard deviation. Five runs are a sample of the training
#: procedure's behaviour, not the entire population of runs that could exist.
STD_DDOF = 1

#: A metric may be called directionally consistent only at this many seeds or
#: more, out of five. Four of five leaves one disagreeing run visible in the
#: report; three of five is a split and is never called consistent.
MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY = 4

#: The phrase permitted when the rule above is satisfied.
DIRECTIONAL_PHRASE = "directionally consistent across training seeds"


class MultiseedError(ValueError):
    """A multi-seed analysis input is malformed or incomplete."""


def require_metric(metric: str) -> str:
    """One of the eight frozen metrics.

    Raises:
        MultiseedError: ``metric`` is not one of them.
    """
    if metric not in METRICS:
        raise MultiseedError(f"unknown metric {metric!r}; expected one of {list(METRICS)}")
    return metric


def _require_finite(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MultiseedError(f"{label} must be a real number, got {type(value).__name__} {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise MultiseedError(
            f"{label} is {value!r}. A NaN or infinite metric means the run that produced "
            "it is broken; it is never averaged into a summary or silently skipped."
        )
    return number


def raw_delta(metric: str, cnn_value: float, unet_value: float) -> float:
    """``U-Net - CNN``, in the metric's own units and sign.

    Deliberately NOT reoriented. This is the quantity a reader can check by
    subtracting two numbers in the tables, so it keeps the sign the metric
    naturally has: for PSNR and SSIM a positive delta favours the U-Net, and
    for MAE and MSE a negative delta favours the U-Net.
    """
    require_metric(metric)
    return _require_finite("unet value", unet_value) - _require_finite("cnn value", cnn_value)


def oriented_improvement(metric: str, cnn_value: float, unet_value: float) -> float:
    """The same comparison with a single meaning: positive is always U-Net better.

    Existing beside :func:`raw_delta` rather than replacing it because the two
    answer different questions. The raw delta is arithmetic a reader can
    verify; the oriented improvement is what a summary sentence should be
    built from, so that "mean improvement is positive" never has to be read
    together with "except for MAE and MSE, where it is the other way round".
    Four of the eight metrics are lower-is-better, so that caveat is exactly
    the kind a report gets wrong once and then repeats.
    """
    require_metric(metric)
    cnn = _require_finite("cnn value", cnn_value)
    unet = _require_finite("unet value", unet_value)
    return unet - cnn if metric in HIGHER_IS_BETTER else cnn - unet


def favours_unet(metric: str, cnn_value: float, unet_value: float) -> bool:
    """Whether this seed's pair favours the U-Net at all.

    An exact tie is not a win for either side; :func:`favours_cnn` is
    independently false there too, so ties are counted separately rather than
    assigned to whichever architecture the report happens to be about.
    """
    return oriented_improvement(metric, cnn_value, unet_value) > 0.0


def favours_cnn(metric: str, cnn_value: float, unet_value: float) -> bool:
    """Whether this seed's pair favours the CNN at all."""
    return oriented_improvement(metric, cnn_value, unet_value) < 0.0


def describe_values(values: Sequence[float]) -> dict[str, Any]:
    """Mean, sample standard deviation (ddof=1), min and max across seeds.

    The individual values are returned alongside the summary because with five
    observations the list itself is the more honest report: a mean and a
    standard deviation over n=5 hide whether the spread came from one outlying
    run or from four evenly scattered ones.

    Raises:
        MultiseedError: fewer than two values, or a non-finite one.
    """
    numbers = [_require_finite(f"value[{index}]", value) for index, value in enumerate(values)]
    if len(numbers) < 2:
        raise MultiseedError(
            f"a sample standard deviation with ddof={STD_DDOF} needs at least two values, "
            f"got {len(numbers)}"
        )
    mean = sum(numbers) / len(numbers)
    variance = sum((number - mean) ** 2 for number in numbers) / (len(numbers) - STD_DDOF)
    return {
        "values": numbers,
        "n": len(numbers),
        "mean": mean,
        "std": math.sqrt(variance),
        "std_ddof": STD_DDOF,
        "min": min(numbers),
        "max": max(numbers),
    }


def paired_metric_summary(
    metric: str,
    cnn_by_seed: Mapping[int, float],
    unet_by_seed: Mapping[int, float],
) -> dict[str, Any]:
    """The predeclared per-metric comparison across seeds.

    Pairs by seed rather than comparing two independent means. Pairing by seed
    aligns the data and the patient-balanced sampler ordering across
    architectures. The within-seed delta therefore gives a controlled
    descriptive comparison under matched experimental seed labels, but it is
    not a pure causal estimate of architecture alone: the two architectures
    have different parameterizations and consume initialization randomness
    differently.

    Raises:
        MultiseedError: the two seed sets differ, or any value is non-finite.
    """
    require_metric(metric)
    if set(cnn_by_seed) != set(unet_by_seed):
        raise MultiseedError(
            f"{metric}: the CNN and U-Net seed sets differ - CNN has "
            f"{sorted(cnn_by_seed)}, U-Net has {sorted(unet_by_seed)}. A paired comparison "
            "needs both architectures at every seed."
        )

    seeds = sorted(cnn_by_seed)
    raw = {seed: raw_delta(metric, cnn_by_seed[seed], unet_by_seed[seed]) for seed in seeds}
    oriented = {
        seed: oriented_improvement(metric, cnn_by_seed[seed], unet_by_seed[seed]) for seed in seeds
    }
    unet_wins = sum(1 for value in oriented.values() if value > 0.0)
    cnn_wins = sum(1 for value in oriented.values() if value < 0.0)
    ties = sum(1 for value in oriented.values() if value == 0.0)

    oriented_stats = describe_values([oriented[seed] for seed in seeds])
    return {
        "metric": metric,
        "direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
        "seeds": seeds,
        "cnn": describe_values([cnn_by_seed[seed] for seed in seeds]),
        "unet": describe_values([unet_by_seed[seed] for seed in seeds]),
        "raw_delta_definition": "unet_minus_cnn",
        "raw_delta_by_seed": raw,
        "raw_delta": describe_values([raw[seed] for seed in seeds]),
        "oriented_improvement_definition": (
            "unet - cnn for higher-is-better metrics, cnn - unet for lower-is-better; "
            "positive always means the U-Net did better"
        ),
        "oriented_improvement_by_seed": oriented,
        "oriented_improvement": oriented_stats,
        "seeds_favouring_unet": unet_wins,
        "seeds_favouring_cnn": cnn_wins,
        "seeds_tied": ties,
        "directionally_consistent_for_unet": _directional(
            oriented_stats["mean"], unet_wins, len(seeds)
        ),
        "directionally_consistent_for_cnn": _directional(
            -oriented_stats["mean"], cnn_wins, len(seeds)
        ),
    }


def _directional(mean_oriented: float, wins: int, n_seeds: int) -> bool:
    """Both halves of the predeclared rule: mean direction AND seed count."""
    return mean_oriented > 0.0 and wins >= MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY


def directional_phrase(summary: Mapping[str, Any]) -> str | None:
    """The permitted sentence for one metric, or ``None`` if none applies.

    Returns the exact wording rather than a flag so a report cannot reach for
    "consistent" at 3/5 by accident: at a split there is simply no phrase to
    use, and the caller has to describe the split.
    """
    seeds = len(summary["seeds"])
    if summary["seeds_favouring_unet"] == seeds:
        return f"all {seeds} seeds favoured the U-Net"
    if summary["seeds_favouring_cnn"] == seeds:
        return f"all {seeds} seeds favoured the CNN"
    if summary["directionally_consistent_for_unet"]:
        return f"the U-Net was {DIRECTIONAL_PHRASE}"
    if summary["directionally_consistent_for_cnn"]:
        return f"the CNN was {DIRECTIONAL_PHRASE}"
    return None


def overall_directional_claim(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Whether an architecture may be described as directionally better overall.

    Requires **all eight** metrics to satisfy the per-metric rule. One metric
    is never enough: four of the eight measure error and four measure
    similarity, and a method can improve one family while worsening the other.
    When the requirement is not met the answer is not "no difference" - it is
    that the pattern has to be reported metric by metric.
    """
    missing = [metric for metric in METRICS if metric not in summaries]
    if missing:
        raise MultiseedError(
            f"an overall claim needs all {len(METRICS)} metrics; missing {missing}"
        )
    unet = [metric for metric in METRICS if summaries[metric]["directionally_consistent_for_unet"]]
    cnn = [metric for metric in METRICS if summaries[metric]["directionally_consistent_for_cnn"]]
    if len(unet) == len(METRICS):
        architecture: str | None = "unet"
    elif len(cnn) == len(METRICS):
        architecture = "cnn"
    else:
        architecture = None
    return {
        "architecture": architecture,
        "metrics_directional_for_unet": unet,
        "metrics_directional_for_cnn": cnn,
        "requires_all_metrics": len(METRICS),
        "requires_minimum_seeds_favouring": MIN_SEEDS_FOR_DIRECTIONAL_CONSISTENCY,
        "statement": (
            f"The {architecture} was {DIRECTIONAL_PHRASE} on all {len(METRICS)} metrics."
            if architecture
            else "No overall directional claim is permitted; report the pattern metric by metric."
        ),
        "significance_tested": False,
        "note": (
            "Descriptive only. Five seeds measure the spread of this training procedure on "
            "fixed data; no p-value, significance claim or confidence interval is reported, "
            "and none is implied."
        ),
    }
