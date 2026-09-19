"""Tests for the predeclared CLAHE search space and selection rule.

Fully synthetic. The candidate tables here are hand-built so the correct
winner is known by construction, which is the only way to check that the rule
implemented is the rule that was declared.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ct_restoration.classical.clahe import ClaheConfig
from ct_restoration.classical.search import (
    METRIC_TIE_BREAKERS,
    PATIENT_WEIGHTED_PREFIX,
    PRIMARY_METRIC,
    ClaheSearchSpace,
    SearchError,
    rank_candidates,
    selected_config,
    selection_key,
)
from ct_restoration.config import load_config


def candidate_row(
    clip: float,
    grid: tuple[int, int],
    body_ssim: float,
    body_psnr: float = 25.0,
    full_ssim: float = 0.5,
    full_psnr: float = 28.0,
) -> dict:
    """One candidate row with the four metrics the rule consults."""
    return {
        "clip_limit": clip,
        "tile_grid_rows": grid[0],
        "tile_grid_cols": grid[1],
        f"{PATIENT_WEIGHTED_PREFIX}body_ssim": body_ssim,
        f"{PATIENT_WEIGHTED_PREFIX}body_psnr": body_psnr,
        f"{PATIENT_WEIGHTED_PREFIX}full_ssim": full_ssim,
        f"{PATIENT_WEIGHTED_PREFIX}full_psnr": full_psnr,
    }


# --------------------------------------------------------------------------
# The search space
# --------------------------------------------------------------------------


def test_committed_search_config_matches_the_declared_space() -> None:
    space = ClaheSearchSpace.from_mapping(load_config("clahe_search.yaml"))

    assert space.implementation == "opencv_clahe_v1"
    assert space.input_quantization_bits == 8
    assert list(space.clip_limit) == [0.5, 1.0, 2.0, 4.0]
    assert [list(grid) for grid in space.tile_grid_size] == [[4, 4], [8, 8], [16, 16]]


def test_the_space_holds_exactly_twelve_unique_candidates() -> None:
    candidates = ClaheSearchSpace.from_mapping(load_config("clahe_search.yaml")).candidates()

    assert len(candidates) == 12
    tuples = {(c.clip_limit, c.tile_rows, c.tile_columns) for c in candidates}
    assert len(tuples) == 12


def test_every_candidate_is_a_valid_config() -> None:
    for candidate in ClaheSearchSpace().candidates():
        assert isinstance(candidate, ClaheConfig)
        assert candidate.clip_limit > 0
        assert candidate.tile_rows >= 1


def test_candidate_generation_is_independent_of_file_ordering() -> None:
    """The YAML could list values in any order; the sweep must not care."""
    forward = ClaheSearchSpace(
        clip_limit=(0.5, 1.0, 2.0, 4.0), tile_grid_size=((4, 4), (8, 8), (16, 16))
    ).candidates()
    shuffled = ClaheSearchSpace(
        clip_limit=(4.0, 0.5, 2.0, 1.0), tile_grid_size=((16, 16), (4, 4), (8, 8))
    ).candidates()

    assert [c.label for c in forward] == [c.label for c in shuffled]


def test_candidates_are_in_a_canonical_sorted_order() -> None:
    labels = [(c.clip_limit, c.tile_rows, c.tile_columns) for c in ClaheSearchSpace().candidates()]

    assert labels == sorted(labels)


def test_a_repeated_clip_limit_is_refused() -> None:
    with pytest.raises(SearchError, match="repeats a value"):
        ClaheSearchSpace(clip_limit=(0.5, 0.5, 1.0))


def test_a_repeated_grid_is_refused() -> None:
    with pytest.raises(SearchError, match="repeats a grid"):
        ClaheSearchSpace(tile_grid_size=((4, 4), (4, 4)))


def test_search_config_refuses_unknown_keys() -> None:
    document = load_config("clahe_search.yaml")["search"]

    with pytest.raises(SearchError, match="unrecognised key"):
        ClaheSearchSpace.from_mapping({**document, "gamma": [1.0]})


def test_search_config_refuses_missing_keys() -> None:
    with pytest.raises(SearchError, match="missing key"):
        ClaheSearchSpace.from_mapping({"implementation": "opencv_clahe_v1"})


# --------------------------------------------------------------------------
# Malformed search YAML is refused, never silently repaired
# --------------------------------------------------------------------------
#
# A frozen experiment definition that quietly coerces its own values is worse
# than one that crashes: float("0.5") and int(4.8) both succeed, so a typo
# would define a different sweep from the one the tracked file appears to
# declare, and nothing in the run would report it.


def declared() -> dict:
    """The committed search section, to be corrupted one key at a time."""
    return dict(load_config("clahe_search.yaml")["search"])


@pytest.mark.parametrize(
    "clips",
    [
        ["0.5", 1.0, 2.0, 4.0],  # a quoted number is not a number
        [0.5, "1.0"],
        [True, 1.0],  # bool is an int subclass; True would read as 1.0
        [False, 1.0],
        [None, 1.0],
        [[0.5], 1.0],
    ],
)
def test_a_non_numeric_clip_limit_is_refused(clips) -> None:
    with pytest.raises(SearchError, match="clip_limit entries must be a real number"):
        ClaheSearchSpace.from_mapping({**declared(), "clip_limit": clips})


@pytest.mark.parametrize("clips", [[0.0, 1.0], [-0.5, 1.0], [float("nan")], [float("inf")]])
def test_a_non_positive_or_non_finite_clip_limit_is_refused(clips) -> None:
    with pytest.raises(SearchError, match="clip_limit entries must be finite and positive"):
        ClaheSearchSpace.from_mapping({**declared(), "clip_limit": clips})


def test_a_clip_limit_string_is_not_parsed_into_the_value_it_resembles() -> None:
    """float("0.5") == 0.5, which is exactly the silent repair being refused."""
    with pytest.raises(SearchError):
        ClaheSearchSpace.from_mapping({**declared(), "clip_limit": ["0.5"]})


@pytest.mark.parametrize("clips", ["0.5", 0.5, None])
def test_clip_limit_must_be_a_list(clips) -> None:
    with pytest.raises(SearchError, match="clip_limit must be a list"):
        ClaheSearchSpace.from_mapping({**declared(), "clip_limit": clips})


@pytest.mark.parametrize(
    "grids",
    [
        [[4.0, 4]],  # integral float is still a float
        [[4, 4.0]],
        [[4.8, 4]],  # int(4.8) == 4 is the truncation being refused
        [["4", 4]],
        [[True, 4]],
        [[4, None]],
    ],
)
def test_a_non_integer_grid_dimension_is_refused(grids) -> None:
    with pytest.raises(SearchError, match="tile_grid_size entries must be an integer"):
        ClaheSearchSpace.from_mapping({**declared(), "tile_grid_size": grids})


@pytest.mark.parametrize("grids", [[[0, 4]], [[4, 0]], [[-4, 4]]])
def test_a_non_positive_grid_dimension_is_refused(grids) -> None:
    with pytest.raises(SearchError, match="tile_grid_size entries must be >= 1"):
        ClaheSearchSpace.from_mapping({**declared(), "tile_grid_size": grids})


@pytest.mark.parametrize("grids", [[[4]], [[4, 4, 4]], [4], ["44"]])
def test_a_malformed_grid_pair_is_refused(grids) -> None:
    with pytest.raises(SearchError, match="tile_grid_size must be a list"):
        ClaheSearchSpace.from_mapping({**declared(), "tile_grid_size": grids})


@pytest.mark.parametrize("bits", [8.0, "8", True, 16])
def test_a_wrong_quantization_depth_is_refused(bits) -> None:
    with pytest.raises(SearchError, match="input_quantization_bits"):
        ClaheSearchSpace.from_mapping({**declared(), "input_quantization_bits": bits})


def test_the_committed_search_yaml_still_parses_unchanged() -> None:
    """Strictness must not have narrowed the declared space itself."""
    space = ClaheSearchSpace.from_mapping(load_config("clahe_search.yaml"))

    assert [(c.clip_limit, c.tile_rows, c.tile_columns) for c in space.candidates()] == [
        (0.5, 4, 4),
        (0.5, 8, 8),
        (0.5, 16, 16),
        (1.0, 4, 4),
        (1.0, 8, 8),
        (1.0, 16, 16),
        (2.0, 4, 4),
        (2.0, 8, 8),
        (2.0, 16, 16),
        (4.0, 4, 4),
        (4.0, 8, 8),
        (4.0, 16, 16),
    ]


# --------------------------------------------------------------------------
# The selection rule
# --------------------------------------------------------------------------


def test_body_ssim_is_the_primary_metric() -> None:
    assert PRIMARY_METRIC == "body_ssim"
    assert METRIC_TIE_BREAKERS == ("body_psnr", "full_ssim", "full_psnr")


def test_the_highest_body_ssim_wins() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(0.5, (4, 4), body_ssim=0.70, body_psnr=99.0, full_psnr=99.0),
            candidate_row(1.0, (8, 8), body_ssim=0.80, body_psnr=1.0, full_psnr=1.0),
            candidate_row(2.0, (16, 16), body_ssim=0.75),
        ]
    )

    ranked = rank_candidates(frame)

    # Body SSIM decides, even though the loser leads on every tie-breaker.
    assert ranked.iloc[0]["clip_limit"] == 1.0
    assert bool(ranked.iloc[0]["selected"])
    assert list(ranked["selection_rank"]) == [1, 2, 3]


def test_body_psnr_breaks_a_body_ssim_tie() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(0.5, (4, 4), body_ssim=0.80, body_psnr=20.0),
            candidate_row(1.0, (8, 8), body_ssim=0.80, body_psnr=30.0),
        ]
    )

    assert rank_candidates(frame).iloc[0]["clip_limit"] == 1.0


def test_full_ssim_breaks_a_tie_after_body_psnr() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(0.5, (4, 4), body_ssim=0.8, body_psnr=25.0, full_ssim=0.4),
            candidate_row(1.0, (8, 8), body_ssim=0.8, body_psnr=25.0, full_ssim=0.6),
        ]
    )

    assert rank_candidates(frame).iloc[0]["clip_limit"] == 1.0


def test_full_psnr_breaks_a_tie_after_full_ssim() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(
                0.5, (4, 4), body_ssim=0.8, body_psnr=25.0, full_ssim=0.5, full_psnr=20.0
            ),
            candidate_row(
                1.0, (8, 8), body_ssim=0.8, body_psnr=25.0, full_ssim=0.5, full_psnr=30.0
            ),
        ]
    )

    assert rank_candidates(frame).iloc[0]["clip_limit"] == 1.0


def test_the_lower_clip_limit_wins_a_full_metric_tie() -> None:
    """Exactly the case the real sweep hits: at a 16x16 grid on 256x256
    images, OpenCV maps clip limits 0.5 and 1.0 to the same internal integer
    threshold, so the two candidates measure identically."""
    frame = pd.DataFrame(
        [
            candidate_row(1.0, (16, 16), body_ssim=0.8),
            candidate_row(0.5, (16, 16), body_ssim=0.8),
        ]
    )

    ranked = rank_candidates(frame)

    assert ranked.iloc[0]["clip_limit"] == 0.5
    assert ranked.iloc[1]["clip_limit"] == 1.0


def test_the_smaller_grid_wins_when_even_the_clip_limit_ties() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(0.5, (16, 16), body_ssim=0.8),
            candidate_row(0.5, (4, 4), body_ssim=0.8),
        ]
    )

    assert rank_candidates(frame).iloc[0]["tile_grid_rows"] == 4


@pytest.mark.parametrize("permutation", ["reversed", "shuffled", "rotated"])
def test_row_order_cannot_change_the_winner(permutation) -> None:
    import random

    rows = [
        candidate_row(0.5, (4, 4), body_ssim=0.70),
        candidate_row(1.0, (8, 8), body_ssim=0.82),
        candidate_row(2.0, (16, 16), body_ssim=0.75),
        candidate_row(4.0, (4, 4), body_ssim=0.60),
    ]
    if permutation == "reversed":
        permuted = list(reversed(rows))
    elif permutation == "rotated":
        permuted = rows[2:] + rows[:2]
    else:
        permuted = list(rows)
        random.Random(11).shuffle(permuted)

    original = rank_candidates(pd.DataFrame(rows))
    other = rank_candidates(pd.DataFrame(permuted))

    assert list(original["clip_limit"]) == list(other["clip_limit"])
    assert list(original["selection_rank"]) == list(other["selection_rank"])
    assert selected_config(original) == selected_config(other)


def test_exactly_one_candidate_is_selected() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(clip, (4, 4), body_ssim=0.5 + index * 0.01)
            for index, clip in enumerate([0.5, 1.0, 2.0, 4.0])
        ]
    )

    ranked = rank_candidates(frame)

    assert int(ranked["selected"].sum()) == 1
    assert bool(ranked.iloc[0]["selected"])


def test_selected_config_returns_the_winning_parameters() -> None:
    frame = pd.DataFrame(
        [
            candidate_row(0.5, (4, 4), body_ssim=0.70),
            candidate_row(2.0, (16, 16), body_ssim=0.90),
        ]
    )

    config = selected_config(rank_candidates(frame))

    assert config.clip_limit == 2.0
    assert config.tile_grid_size == (16, 16)


def test_ranking_an_empty_table_is_refused() -> None:
    with pytest.raises(SearchError, match="empty candidate table"):
        rank_candidates(pd.DataFrame())


def test_a_duplicated_parameter_set_is_refused() -> None:
    """Two rows for one candidate would make 'the winner' ambiguous."""
    frame = pd.DataFrame(
        [candidate_row(0.5, (4, 4), body_ssim=0.7), candidate_row(0.5, (4, 4), body_ssim=0.9)]
    )

    with pytest.raises(SearchError, match="repeats parameter set"):
        rank_candidates(frame)


def test_a_missing_metric_column_is_refused() -> None:
    frame = pd.DataFrame([candidate_row(0.5, (4, 4), body_ssim=0.7)]).drop(
        columns=[f"{PATIENT_WEIGHTED_PREFIX}full_psnr"]
    )

    with pytest.raises(SearchError, match="which the rule needs"):
        rank_candidates(frame)


def test_the_selection_key_orders_higher_metrics_first() -> None:
    better = selection_key(candidate_row(0.5, (4, 4), body_ssim=0.9))
    worse = selection_key(candidate_row(0.5, (4, 4), body_ssim=0.8))

    assert better < worse
