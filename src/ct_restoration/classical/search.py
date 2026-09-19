"""The predeclared CLAHE search space and its deterministic selection rule.

Both are fixed in :file:`configs/clahe_search.yaml` before any candidate is
scored. Keeping them here, in tested library code rather than inside a script,
means the rule that picks a winner can be exercised on synthetic tables where
the right answer is known by construction.

Why predeclare at all
---------------------
A validation split is only an honest development estimate while the number of
things tried against it stays bounded and the criterion stays fixed. Widening
the grid because the first sweep disappoints, or switching the primary metric
to whichever one the method happened to win on, both quietly convert
validation into training data for the search itself.

The rule
--------
Rank by patient-weighted mean **body SSIM**, higher first. CLAHE is a local
contrast and structure method, so a local structural measure inside the
anatomy is the metric most aligned with what it attempts. Ties break on body
PSNR, then full SSIM, then full PSNR, then the lower clip limit, then the
smaller tile grid - a total order, so there is always exactly one winner and
it never depends on row order.

Choosing the primary metric in advance does not make it the only one that
matters. The winner may still be worse than no restoration on that metric or
any other; :mod:`scripts.tune_clahe` reports all eight either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from ct_restoration.classical.clahe import ALGORITHM_VERSION, QUANTIZATION_BITS, ClaheConfig

#: Prefix of the aggregated metric columns the selection rule reads.
PATIENT_WEIGHTED_PREFIX = "patient_weighted_"

#: The predeclared primary selection metric, higher is better.
PRIMARY_METRIC = "body_ssim"

#: Metrics consulted in order, all higher-is-better, before the structural
#: tie-breakers on the parameters themselves.
METRIC_TIE_BREAKERS: tuple[str, ...] = ("body_psnr", "full_ssim", "full_psnr")


class SearchError(ValueError):
    """The search space or a candidate table is not usable as given."""


def _require_positive_real(name: str, value: Any) -> float:
    """Accept only a genuine finite positive real number.

    Deliberately refuses to coerce. ``float("0.5")`` succeeds, so a coercing
    reader would accept a quoted YAML value and silently define a different
    frozen experiment from the one the file appears to declare. ``bool`` is
    refused too: it is an ``int`` subclass, so ``True`` would otherwise read
    as a clip limit of 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise SearchError(
            f"{name} must be a real number, got {type(value).__name__} {value!r}. "
            "It is not parsed from a string: a frozen experiment definition does not "
            "silently repair malformed YAML."
        )
    if not np.isfinite(value) or float(value) <= 0:
        raise SearchError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def _require_positive_integer(name: str, value: Any) -> int:
    """Accept only a genuine positive integer.

    ``int(4.8)`` is 4 and ``int("4")`` is 4, so a coercing reader would turn a
    typo into a different but plausible-looking tile grid and change what the
    sweep measured without anything reporting a problem. A float is refused
    even when integral: ``4.0`` in the YAML means the file was not written the
    way this experiment declares it.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise SearchError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}. "
            "It is not rounded, truncated or parsed from a string."
        )
    if int(value) < 1:
        raise SearchError(f"{name} must be >= 1, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class ClaheSearchSpace:
    """The frozen set of CLAHE candidates."""

    implementation: str = ALGORITHM_VERSION
    input_quantization_bits: int = QUANTIZATION_BITS
    clip_limit: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    tile_grid_size: tuple[tuple[int, int], ...] = ((4, 4), (8, 8), (16, 16))

    def __post_init__(self) -> None:
        if self.implementation != ALGORITHM_VERSION:
            raise SearchError(
                f"Unsupported search implementation {self.implementation!r}; this module "
                f"searches {ALGORITHM_VERSION!r} only."
            )
        if (
            isinstance(self.input_quantization_bits, bool)
            or not isinstance(self.input_quantization_bits, int | np.integer)
            or int(self.input_quantization_bits) != QUANTIZATION_BITS
        ):
            raise SearchError(
                f"input_quantization_bits must be the integer {QUANTIZATION_BITS}, got "
                f"{type(self.input_quantization_bits).__name__} "
                f"{self.input_quantization_bits!r}"
            )
        if isinstance(self.clip_limit, str) or not hasattr(self.clip_limit, "__len__"):
            raise SearchError(f"clip_limit must be a sequence of values, got {self.clip_limit!r}")
        if not self.clip_limit:
            raise SearchError("clip_limit must list at least one value")
        if isinstance(self.tile_grid_size, str) or not hasattr(self.tile_grid_size, "__len__"):
            raise SearchError(
                f"tile_grid_size must be a sequence of grids, got {self.tile_grid_size!r}"
            )
        if not self.tile_grid_size:
            raise SearchError("tile_grid_size must list at least one grid")

        # Types first: the duplicate checks below hash these values, and a
        # malformed entry must be reported as malformed rather than as an
        # unhashable-type crash.
        for value in self.clip_limit:
            _require_positive_real("clip_limit entries", value)
        for grid in self.tile_grid_size:
            if isinstance(grid, str) or not hasattr(grid, "__len__") or len(grid) != 2:
                raise SearchError(
                    f"tile_grid_size must be a list of [rows, columns] pairs, got {grid!r}"
                )
            for value in grid:
                _require_positive_integer("tile_grid_size entries", value)

        if len(set(self.clip_limit)) != len(self.clip_limit):
            raise SearchError(f"clip_limit repeats a value: {list(self.clip_limit)}")
        grids = [tuple(grid) for grid in self.tile_grid_size]
        if len(set(grids)) != len(grids):
            raise SearchError(f"tile_grid_size repeats a grid: {[list(g) for g in grids]}")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ClaheSearchSpace:
        """Build the space from the tracked YAML, refusing surprises.

        Values are passed through without coercion so that validation, not a
        silent repair, decides whether the file declares a usable experiment.
        A quoted ``"0.5"`` or a float ``4.0`` tile dimension is refused rather
        than parsed into the value it resembles.

        Raises:
            SearchError: not a mapping, a missing or unrecognised key, or a
                value that is not of the declared type.
        """
        if not isinstance(mapping, dict):
            raise SearchError(f"Search config must be a mapping, got {type(mapping).__name__}")
        section = mapping.get("search", mapping)
        if not isinstance(section, dict):
            raise SearchError(f"'search' section must be a mapping, got {type(section).__name__}")

        fields = ("implementation", "input_quantization_bits", "clip_limit", "tile_grid_size")
        missing = sorted(set(fields) - set(section))
        if missing:
            raise SearchError(f"Search config is missing key(s): {missing}")
        unknown = sorted(set(section) - set(fields))
        if unknown:
            raise SearchError(
                f"Search config has unrecognised key(s): {unknown}. "
                f"Known keys are {sorted(fields)}."
            )

        clips = section["clip_limit"]
        if isinstance(clips, str) or not isinstance(clips, list | tuple):
            raise SearchError(f"clip_limit must be a list of values, got {clips!r}")

        grids = section["tile_grid_size"]
        if isinstance(grids, str) or not isinstance(grids, list | tuple):
            raise SearchError(f"tile_grid_size must be a list of [rows, columns], got {grids!r}")
        for grid in grids:
            if isinstance(grid, str) or not isinstance(grid, list | tuple) or len(grid) != 2:
                raise SearchError(
                    f"tile_grid_size must be a list of [rows, columns] pairs, got {grid!r}"
                )

        # tuple() here is structural only - YAML gives lists, the dataclass
        # holds tuples. The values themselves are never converted; __post_init__
        # rejects anything that is not already the declared type.
        return cls(
            implementation=section["implementation"],
            input_quantization_bits=section["input_quantization_bits"],
            clip_limit=tuple(clips),
            tile_grid_size=tuple(tuple(grid) for grid in grids),
        )

    def candidates(self) -> list[ClaheConfig]:
        """Every candidate, in a canonical order independent of file order.

        Sorted by clip limit, then tile rows, then tile columns, so the list is
        the same whatever order the YAML happened to list values in.
        """
        configs = [
            ClaheConfig(
                algorithm=self.implementation,
                input_quantization_bits=self.input_quantization_bits,
                clip_limit=float(clip),
                tile_grid_size=(int(grid[0]), int(grid[1])),
            )
            for clip, grid in product(self.clip_limit, self.tile_grid_size)
        ]
        return sorted(configs, key=lambda c: (c.clip_limit, c.tile_rows, c.tile_columns))

    def as_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "input_quantization_bits": int(self.input_quantization_bits),
            "clip_limit": [float(value) for value in self.clip_limit],
            "tile_grid_size": [[int(rows), int(cols)] for rows, cols in self.tile_grid_size],
        }


def selection_key(row: Any) -> tuple[float, ...]:
    """The lexicographic sort key implementing the predeclared rule.

    Negated for the higher-is-better metrics so that a plain ascending sort
    puts the winner first. The final two entries fall back to the gentler
    setting - lower clip limit, then smaller grid - which is a deliberate
    preference for the more conservative of two configurations that measured
    identically.

    Raises:
        SearchError: the row lacks a column the rule needs.
    """
    needed = [
        f"{PATIENT_WEIGHTED_PREFIX}{PRIMARY_METRIC}",
        *(f"{PATIENT_WEIGHTED_PREFIX}{name}" for name in METRIC_TIE_BREAKERS),
        "clip_limit",
        "tile_grid_rows",
        "tile_grid_cols",
    ]
    for column in needed:
        if column not in row:
            raise SearchError(f"candidate row is missing {column!r}, which the rule needs")

    metrics = needed[: 1 + len(METRIC_TIE_BREAKERS)]
    return (
        *(-float(row[column]) for column in metrics),
        float(row["clip_limit"]),
        int(row["tile_grid_rows"]),
        int(row["tile_grid_cols"]),
    )


def rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the selection rule, adding ``selection_rank`` and ``selected``.

    The result is sorted best-first and is invariant to the order the rows
    arrived in: ranking reads only the values, never the position.

    Raises:
        SearchError: the table is empty, or two rows describe the same
            candidate, which would make "the winner" ambiguous.
    """
    if candidates.empty:
        raise SearchError("cannot rank an empty candidate table")

    identity = ["clip_limit", "tile_grid_rows", "tile_grid_cols"]
    missing = [column for column in identity if column not in candidates]
    if missing:
        raise SearchError(f"candidate table is missing column(s): {missing}")
    duplicated = candidates.duplicated(subset=identity)
    if duplicated.any():
        repeats = candidates.loc[duplicated, identity].to_dict("records")
        raise SearchError(f"candidate table repeats parameter set(s): {repeats}")

    ranked = candidates.copy()
    ranked["_key"] = [selection_key(row) for _, row in ranked.iterrows()]
    ranked = ranked.sort_values("_key").drop(columns="_key").reset_index(drop=True)
    ranked["selection_rank"] = range(1, len(ranked) + 1)
    ranked["selected"] = ranked["selection_rank"] == 1
    return ranked


def selected_config(ranked: pd.DataFrame) -> ClaheConfig:
    """The winning configuration from a ranked table.

    Raises:
        SearchError: no row, or more than one row, is marked selected.
    """
    if "selected" not in ranked:
        raise SearchError("ranked table has no 'selected' column; call rank_candidates first")
    winners = ranked[ranked["selected"]]
    if len(winners) != 1:
        raise SearchError(f"expected exactly one selected candidate, found {len(winners)}")
    row = winners.iloc[0]
    return ClaheConfig(
        clip_limit=float(row["clip_limit"]),
        tile_grid_size=(int(row["tile_grid_rows"]), int(row["tile_grid_cols"])),
    )
