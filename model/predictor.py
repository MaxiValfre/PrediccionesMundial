"""
World Cup 2026 Prediction Engine
Dixon-Coles model with FIFA World Rankings, contextual adjustments,
head-to-head history, probability timelines, and Monte Carlo simulation.
"""

import copy
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMULATIONS = 50_000
MAX_GOALS_MATRIX = 8          # 0..7 goals per side for probability matrix
BASE_GOALS = 1.35             # average expected goals baseline
ELO_DIVISOR = 400.0
HOME_ADVANTAGE_GOALS = 0.15   # extra xG for host nations
SEED = 2026
FULL_MATCH_MINUTES = 90

HOST_COUNTRIES = {"USA", "Mexico", "Canada"}

# How many 3rd-placed teams qualify
BEST_THIRD_QUALIFY = 8

# Dixon-Coles correlation parameter (empirically fitted to international data)
DIXON_COLES_RHO = -0.13

# Group stage draw bias: increase draw probability by ~8%
GROUP_DRAW_BIAS = 0.08

# Model metadata
MODEL_NAME = "Dixon-Coles + FIFA Rankings"
MODEL_VERSION = "3.1.0"


# ---------------------------------------------------------------------------
# FIFA World Rankings (June 11, 2026) — official ranking points
# ---------------------------------------------------------------------------

FIFA_RANKINGS: dict[str, float] = {
    "Argentina": 1877.27,
    "Spain": 1874.71,
    "France": 1870.70,
    "England": 1828.02,
    "Portugal": 1767.85,
    "Brazil": 1765.86,
    "Morocco": 1755.10,
    "Netherlands": 1753.57,
    "Belgium": 1742.24,
    "Germany": 1735.77,
    "Colombia": 1698.35,
    "Mexico": 1687.48,
    "Senegal": 1684.07,
    "Uruguay": 1673.07,
    "USA": 1671.23,
    "Japan": 1661.58,
    "Switzerland": 1650.06,
    "Iran": 1619.58,
    # Estimated from FIFA ranking positions
    "South Korea": 1610.0,
    "Australia": 1600.0,
    "DR Congo": 1460.0,
    "Croatia": 1770.0,
    "Ghana": 1440.0,
    "Panama": 1520.0,
    "Ecuador": 1580.0,
    "Turkey": 1570.0,
    "Sweden": 1560.0,
    "Scotland": 1530.0,
    "Egypt": 1520.0,
    "Ivory Coast": 1515.0,
    "Canada": 1510.0,
    "Saudi Arabia": 1500.0,
    "Norway": 1490.0,
    "Tunisia": 1480.0,
    "Algeria": 1480.0,
    "Austria": 1470.0,
    "Paraguay": 1460.0,
    "Czech Republic": 1450.0,

    "Iraq": 1420.0,
    "Qatar": 1410.0,
    "New Zealand": 1400.0,
    "Bosnia": 1380.0,
    "Uzbekistan": 1370.0,
    "Jordan": 1360.0,
    "Cape Verde": 1350.0,
    "Haiti": 1340.0,
    "South Africa": 1400.0,
    "Curaçao": 1280.0,
}

DEFAULT_RATING = 1350.0  # fallback for unknown teams


# ---------------------------------------------------------------------------
# Dynamic ratings loader — prefer updated Elo from ratings_history.json
# ---------------------------------------------------------------------------

def _load_dynamic_ratings() -> dict[str, float]:
    """Load the latest ratings from ratings_history.json if available.

    Falls back to the hardcoded FIFA_RANKINGS if no history file exists.
    This allows the predictor to use Bayesian-updated Elo ratings after
    results have been applied by the updater module.
    """
    ratings_path = Path(__file__).resolve().parent.parent / "data" / "ratings_history.json"
    if ratings_path.exists():
        try:
            with open(ratings_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                # Find the latest snapshot (order: initial < after_md1 < after_md2 < after_md3)
                key_order = ["initial", "after_md1", "after_md2", "after_md3"]
                latest = None
                for k in key_order:
                    if k in history:
                        latest = history[k]
                for k in sorted(history.keys()):
                    if k not in key_order:
                        latest = history[k]
                if latest:
                    # Merge with FIFA_RANKINGS to ensure all teams have a rating
                    merged = dict(FIFA_RANKINGS)
                    merged.update(latest)
                    return merged
        except (json.JSONDecodeError, IOError):
            pass
    return dict(FIFA_RANKINGS)


def get_active_ratings() -> dict[str, float]:
    """Public accessor for the currently active ratings.

    Used by other modules to get the ratings the predictor will use.
    """
    return _load_dynamic_ratings()


# ---------------------------------------------------------------------------
# World Cup Performance Factor — xG boost for historically strong WC teams
# ---------------------------------------------------------------------------

WC_PERFORMANCE_BOOST: dict[str, float] = {
    "Argentina": 0.15,    # defending champion
    "France": 0.08,       # 2018 winner, 2022 finalist
    "Germany": 0.08,      # 2014 winner, historically dominant
    "Brazil": 0.08,       # most titles, 5x champion
    "Spain": 0.08,        # 2010 winner
}


# ---------------------------------------------------------------------------
# Confederation Strength — xG adjustment based on recent confederation form
# ---------------------------------------------------------------------------

CONFEDERATION: dict[str, str] = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "France": "UEFA", "Spain": "UEFA", "England": "UEFA", "Germany": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Switzerland": "UEFA", "Croatia": "UEFA", "Sweden": "UEFA",
    "Scotland": "UEFA", "Austria": "UEFA",
    "Czech Republic": "UEFA", "Norway": "UEFA", "Turkey": "UEFA",
    "Bosnia": "UEFA",
    "Morocco": "CAF", "Senegal": "CAF", "DR Congo": "CAF", "Egypt": "CAF",
    "Ivory Coast": "CAF", "Tunisia": "CAF", "Ghana": "CAF",
    "Algeria": "CAF", "South Africa": "CAF", "Cape Verde": "CAF",
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Haiti": "CONCACAF", "Panama": "CONCACAF", "Curaçao": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC",
    "Qatar": "AFC", "Uzbekistan": "AFC", "Jordan": "AFC",
    "New Zealand": "OFC",
}

CONFEDERATION_BOOST: dict[str, float] = {
    "UEFA": 0.05,
    "CONMEBOL": 0.03,
    "CAF": 0.0,
    "CONCACAF": 0.0,
    "AFC": 0.0,
    "OFC": 0.0,
}


# ---------------------------------------------------------------------------
# Head-to-Head Historical Adjustments
# ---------------------------------------------------------------------------

H2H_ADJUSTMENTS: dict[tuple[str, str], float] = {
    # (team_a, team_b): xG adjustment for team_a
    ("Argentina", "Algeria"): 0.05,
    ("Argentina", "Mexico"): 0.06,
    ("Brazil", "Morocco"): 0.0,
    ("Brazil", "Scotland"): 0.05,
    ("France", "Senegal"): -0.05,   # Senegal upset France in 2002 WC
    ("Germany", "Ivory Coast"): 0.03,
    ("Germany", "Japan"): -0.03,    # Japan upset Germany in 2022 WC
    ("Spain", "Saudi Arabia"): -0.03, # Saudi upset Spain in 2022 WC
    ("Spain", "Morocco"): -0.02,    # Morocco eliminated Spain in 2022
    ("USA", "Australia"): 0.02,
    ("USA", "England"): -0.02,
    ("Netherlands", "Japan"): 0.02,
    ("England", "USA"): 0.02,
    ("France", "Belgium"): 0.03,
    ("Portugal", "Morocco"): -0.03, # Morocco eliminated Portugal in 2022
    ("Morocco", "Belgium"): 0.04,   # Morocco beat Belgium in 2022 WC
    ("Japan", "Germany"): 0.04,     # Japan beat Germany in 2022 WC
    ("Saudi Arabia", "Argentina"): 0.05, # Saudi upset Argentina in 2022 WC
}


# ---------------------------------------------------------------------------
# Fatigue/Travel adjustment (time zone difference from base)
# ---------------------------------------------------------------------------

# Base time zones (UTC offset) for confederations
_BASE_TZ: dict[str, int] = {
    "CONMEBOL": -3,
    "UEFA": 1,
    "CAF": 1,
    "CONCACAF": -5,  # mix, but center
    "AFC": 8,
    "OFC": 12,
}

# Matches played in North America (~UTC-5 to UTC-7)
_MATCH_TZ = -5


def _travel_adjustment(team: str) -> float:
    """Small xG penalty for teams playing far from home time zone."""
    conf = CONFEDERATION.get(team, "")
    base_tz = _BASE_TZ.get(conf, 0)
    diff = abs(base_tz - _MATCH_TZ)
    if diff > 12:
        diff = 24 - diff
    # Max penalty ~0.04 for teams 12 hours away (AFC, OFC)
    if diff <= 3:
        return 0.0
    return -0.02 * (diff / 12)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tournament_data() -> dict:
    data_path = Path(__file__).resolve().parent.parent / "data" / "world_cup_2026.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Form adjustments from matchday results (momentum-based)
# ---------------------------------------------------------------------------

def compute_form_adjustments(data: dict) -> dict[str, float]:
    """Return per-team xG adjustment based on matchday results.

    Momentum system:
      Won by 2+: +0.08
      Won by 1:  +0.04
      Drew:       0
      Lost by 1: -0.04
      Lost by 2+: -0.08
    """
    adj: dict[str, float] = {}
    for group_name, group in data["groups"].items():
        for md_key in ["matchday1", "matchday2", "matchday3"]:
            for match in group["matches"].get(md_key, []):
                if match["status"] != "played":
                    continue
                sa, sb = match["score_a"], match["score_b"]
                ta, tb = match["team_a"], match["team_b"]
                margin = sa - sb
                if margin >= 2:
                    adj[ta] = adj.get(ta, 0) + 0.08
                    adj[tb] = adj.get(tb, 0) - 0.08
                elif margin == 1:
                    adj[ta] = adj.get(ta, 0) + 0.04
                    adj[tb] = adj.get(tb, 0) - 0.04
                elif margin == 0:
                    pass  # draw: no change
                elif margin == -1:
                    adj[ta] = adj.get(ta, 0) - 0.04
                    adj[tb] = adj.get(tb, 0) + 0.04
                else:  # margin <= -2
                    adj[ta] = adj.get(ta, 0) - 0.08
                    adj[tb] = adj.get(tb, 0) + 0.08
    return adj


# ---------------------------------------------------------------------------
# Dixon-Coles correction factor
# ---------------------------------------------------------------------------

def dixon_coles_tau(x: int, y: int, lambda_val: float,
                    mu_val: float, rho: float) -> float:
    """Dixon-Coles correction for low-scoring outcomes.

    Adjusts the independent Poisson probabilities for scores (0,0), (0,1),
    (1,0), and (1,1) which are systematically mis-predicted by independent
    Poisson models.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_val * mu_val * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_val * rho
    elif x == 1 and y == 0:
        return 1.0 + mu_val * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Expected goals & match outcome probabilities
# ---------------------------------------------------------------------------

def _get_h2h_adjustment(team_a: str, team_b: str) -> tuple[float, float]:
    """Get head-to-head xG adjustments for both teams."""
    adj_a = H2H_ADJUSTMENTS.get((team_a, team_b), 0.0)
    adj_b = H2H_ADJUSTMENTS.get((team_b, team_a), 0.0)
    # Also check reverse for team_b
    if adj_a == 0.0:
        reverse = H2H_ADJUSTMENTS.get((team_b, team_a), 0.0)
        if reverse != 0.0:
            adj_a = -reverse
    if adj_b == 0.0:
        reverse = H2H_ADJUSTMENTS.get((team_a, team_b), 0.0)
        if reverse != 0.0:
            adj_b = -reverse
    return adj_a, adj_b


def expected_goals(rating_a: float, rating_b: float,
                   team_a: str = "", team_b: str = "",
                   home_a: bool = False, home_b: bool = False,
                   form_a: float = 0.0, form_b: float = 0.0,
                   knockout: bool = False) -> tuple[float, float]:
    """Compute expected goals for each side using FIFA rankings + contextual factors."""
    diff = (rating_a - rating_b) / ELO_DIVISOR
    xg_a = BASE_GOALS * (10 ** (diff / 2))
    xg_b = BASE_GOALS * (10 ** (-diff / 2))

    # Home advantage
    if home_a:
        xg_a += HOME_ADVANTAGE_GOALS
    if home_b:
        xg_b += HOME_ADVANTAGE_GOALS

    # Form / momentum
    xg_a += form_a
    xg_b += form_b

    # World Cup performance boost
    xg_a += WC_PERFORMANCE_BOOST.get(team_a, 0.0)
    xg_b += WC_PERFORMANCE_BOOST.get(team_b, 0.0)

    # Confederation strength
    conf_a = CONFEDERATION.get(team_a, "")
    conf_b = CONFEDERATION.get(team_b, "")
    xg_a += CONFEDERATION_BOOST.get(conf_a, 0.0)
    xg_b += CONFEDERATION_BOOST.get(conf_b, 0.0)

    # Head-to-head
    h2h_a, h2h_b = _get_h2h_adjustment(team_a, team_b)
    xg_a += h2h_a
    xg_b += h2h_b

    # Travel/fatigue
    xg_a += _travel_adjustment(team_a)
    xg_b += _travel_adjustment(team_b)

    # Cap
    xg_a = max(0.25, min(xg_a, 4.0))
    xg_b = max(0.25, min(xg_b, 4.0))

    # Knockout matches: slightly more attacking → boost both slightly
    if knockout:
        xg_a *= 1.05
        xg_b *= 1.05
        xg_a = min(xg_a, 4.0)
        xg_b = min(xg_b, 4.0)

    return xg_a, xg_b


def score_probabilities(xg_a: float, xg_b: float,
                        rho: float = DIXON_COLES_RHO) -> np.ndarray:
    """Return an (N x N) matrix where cell [i][j] = P(team_a=i, team_b=j).

    Uses the Dixon-Coles model which corrects independent Poisson for low
    scores (0-0, 1-0, 0-1, 1-1).
    """
    n = MAX_GOALS_MATRIX
    pa = np.array([poisson.pmf(k, xg_a) for k in range(n)])
    pb = np.array([poisson.pmf(k, xg_b) for k in range(n)])
    # Normalise tails
    pa[-1] = 1.0 - pa[:-1].sum()
    pb[-1] = 1.0 - pb[:-1].sum()

    # Independent Poisson outer product
    matrix = np.outer(pa, pb)

    # Apply Dixon-Coles correction for low scores
    for i in range(min(2, n)):
        for j in range(min(2, n)):
            tau = dixon_coles_tau(i, j, xg_a, xg_b, rho)
            matrix[i][j] *= tau

    # Re-normalize so matrix sums to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def _round_cache_value(value: float) -> float:
    return round(float(value), 6)


def _apply_group_draw_bias_to_matrix(matrix: np.ndarray) -> np.ndarray:
    """Scale win/draw/loss cells so group-stage sampling matches displayed odds."""
    raw = match_outcome_probs(matrix, group_stage=False)
    adjusted = match_outcome_probs(matrix, group_stage=True)
    scaled = matrix.copy()

    win_factor = adjusted["win"] / raw["win"] if raw["win"] > 0 else 1.0
    draw_factor = adjusted["draw"] / raw["draw"] if raw["draw"] > 0 else 1.0
    loss_factor = adjusted["loss"] / raw["loss"] if raw["loss"] > 0 else 1.0

    rows, cols = scaled.shape
    for i in range(rows):
        for j in range(cols):
            if i > j:
                scaled[i][j] *= win_factor
            elif i == j:
                scaled[i][j] *= draw_factor
            else:
                scaled[i][j] *= loss_factor

    total = scaled.sum()
    if total > 0:
        scaled /= total
    return scaled


@lru_cache(maxsize=8192)
def _cached_sampling_distribution(
    xg_a: float,
    xg_b: float,
    group_stage: bool,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = score_probabilities(xg_a, xg_b, rho)
    if group_stage:
        matrix = _apply_group_draw_bias_to_matrix(matrix)
    cdf = np.cumsum(matrix.reshape(-1))
    if len(cdf) > 0:
        cdf[-1] = 1.0
    return matrix, cdf


def get_sampling_distribution(
    xg_a: float,
    xg_b: float,
    group_stage: bool = False,
    rho: float = DIXON_COLES_RHO,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached sampling structures for repeated match simulations."""
    return _cached_sampling_distribution(
        _round_cache_value(xg_a),
        _round_cache_value(xg_b),
        group_stage,
        _round_cache_value(rho),
    )


def clamp_match_minute(minute: Optional[int | float]) -> int:
    """Clamp live match minute to a sensible football range."""
    if minute is None:
        return 0
    return max(0, min(int(minute), 130))


def remaining_match_fraction(minute: Optional[int | float]) -> float:
    """Fraction of regular time still to play for live conditional forecasts."""
    clamped = clamp_match_minute(minute)
    return max(0.0, min(1.0, (FULL_MATCH_MINUTES - min(clamped, FULL_MATCH_MINUTES)) / FULL_MATCH_MINUTES))


def remaining_expected_goals(xg_a: float, xg_b: float,
                             minute: Optional[int | float]) -> tuple[float, float, float]:
    """Scale full-match xG to the remaining regular-time share."""
    remaining_fraction = remaining_match_fraction(minute)
    return xg_a * remaining_fraction, xg_b * remaining_fraction, remaining_fraction


def build_display_score_matrix(matrix: np.ndarray, size: int = 6) -> list[list[float]]:
    """Convert a score matrix into a small normalized heatmap slice."""
    score_matrix_raw = matrix[:size, :size]
    sm_total = score_matrix_raw.sum()
    score_matrix_norm = score_matrix_raw / sm_total if sm_total > 0 else score_matrix_raw
    return [
        [round(float(score_matrix_norm[i][j]), 5) for j in range(size)]
        for i in range(score_matrix_norm.shape[0])
    ]


def build_live_score_matrix(current_score_a: int, current_score_b: int,
                            additional_matrix: np.ndarray) -> np.ndarray:
    """Shift additional-goals probabilities into final-score space."""
    rows, cols = additional_matrix.shape
    total = np.zeros((current_score_a + rows, current_score_b + cols))
    for i in range(rows):
        for j in range(cols):
            total[current_score_a + i][current_score_b + j] += additional_matrix[i][j]
    return total


def live_match_state(current_score_a: int, current_score_b: int,
                     minute: Optional[int | float],
                     xg_a: float, xg_b: float,
                     group_stage: bool = False) -> dict:
    """Project a live match from the current score and remaining time."""
    rem_xg_a, rem_xg_b, remaining_fraction = remaining_expected_goals(xg_a, xg_b, minute)
    additional_matrix, _ = get_sampling_distribution(
        rem_xg_a,
        rem_xg_b,
        group_stage=group_stage,
    )
    final_matrix = build_live_score_matrix(current_score_a, current_score_b, additional_matrix)
    outcome = match_outcome_probs(final_matrix, group_stage=False)
    pred_score = most_likely_score(final_matrix)
    return {
        "matrix": final_matrix,
        "outcome": outcome,
        "predicted_score": pred_score,
        "remaining_xg_a": rem_xg_a,
        "remaining_xg_b": rem_xg_b,
        "remaining_fraction": remaining_fraction,
        "minute": clamp_match_minute(minute),
    }


def match_outcome_probs(matrix: np.ndarray,
                        group_stage: bool = False) -> dict[str, float]:
    """Win / Draw / Loss probabilities from score matrix (team_a perspective).

    If group_stage=True, applies draw bias (groups have more draws historically).
    """
    rows, cols = matrix.shape
    win = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i > j)
    draw = sum(matrix[i][i] for i in range(min(rows, cols)))
    loss = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i < j)
    total = win + draw + loss

    win /= total
    draw /= total
    loss /= total

    # Group stage draw bias: redistribute from extreme outcomes toward draws
    if group_stage:
        shift = GROUP_DRAW_BIAS * (win + loss)
        # Take proportionally from win and loss
        win_share = win / (win + loss) if (win + loss) > 0 else 0.5
        loss_share = loss / (win + loss) if (win + loss) > 0 else 0.5
        win -= shift * win_share
        loss -= shift * loss_share
        draw += shift
        # Ensure non-negative
        win = max(win, 0.001)
        loss = max(loss, 0.001)
        draw = max(draw, 0.001)
        # Re-normalize
        total = win + draw + loss
        win /= total
        draw /= total
        loss /= total

    return {"win": win, "draw": draw, "loss": loss}


def most_likely_score(matrix: np.ndarray) -> tuple[int, int]:
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(idx[0]), int(idx[1])


# ---------------------------------------------------------------------------
# Betting odds conversion
# ---------------------------------------------------------------------------

def prob_to_decimal_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    return round(1.0 / max(prob, 0.001), 2)


def _apply_table_result(table: dict, team_a: str, team_b: str,
                        score_a: int, score_b: int) -> None:
    """Apply a single match result to a compact standings table."""
    table[team_a]["gf"] += score_a
    table[team_a]["ga"] += score_b
    table[team_a]["gd"] += score_a - score_b
    table[team_a]["played"] += 1
    table[team_b]["gf"] += score_b
    table[team_b]["ga"] += score_a
    table[team_b]["gd"] += score_b - score_a
    table[team_b]["played"] += 1
    if score_a > score_b:
        table[team_a]["pts"] += 3
    elif score_a < score_b:
        table[team_b]["pts"] += 3
    else:
        table[team_a]["pts"] += 1
        table[team_b]["pts"] += 1


# ---------------------------------------------------------------------------
# Simulate a single match (returns goals_a, goals_b)
# ---------------------------------------------------------------------------

def simulate_match(rng: np.random.Generator,
                   xg_a: float, xg_b: float,
                   group_stage: bool = False) -> tuple[int, int]:
    """Sample a match result from the Dixon-Coles score probability matrix."""
    matrix, cdf = get_sampling_distribution(xg_a, xg_b, group_stage=group_stage)
    idx = int(np.searchsorted(cdf, rng.random(), side="right"))
    cols = matrix.shape[1]
    return idx // cols, idx % cols


def simulate_knockout_match(rng: np.random.Generator,
                            xg_a: float, xg_b: float) -> tuple[int, int, str]:
    """Simulate knockout match with Dixon-Coles sampling.
    If draw after 90 min -> extra time (DC with 1/3 xG) -> penalties."""
    ga, gb = simulate_match(rng, xg_a, xg_b, group_stage=False)
    if ga != gb:
        return ga, gb, "90min"
    # Extra time: ~30 min -> scale xG by 1/3
    et_a, et_b = simulate_match(rng, xg_a / 3, xg_b / 3, group_stage=False)
    ga += et_a
    gb += et_b
    if ga != gb:
        return ga, gb, "aet"

    if rng.random() < penalty_win_probability(xg_a, xg_b):
        return ga, gb, "pen_a"
    else:
        return ga, gb, "pen_b"


# ---------------------------------------------------------------------------
# Group simulation
# ---------------------------------------------------------------------------

def simulate_group(rng: np.random.Generator, group: dict,
                   ratings: dict[str, float], form: dict[str, float],
                   played_results: list[dict]) -> list[dict]:
    """Simulate a group from the current state, including live matches."""
    teams = group["teams"]
    table = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0} for t in teams}

    # Resolve the current group state: completed matches count fully, live matches
    # continue from the current scoreline, scheduled matches start from 0-0.
    for md_key in ["matchday1", "matchday2", "matchday3"]:
        for m in group["matches"].get(md_key, []):
            ta, tb = m["team_a"], m["team_b"]
            status = m.get("status", "scheduled")
            if status == "played":
                sa = m.get("score_a")
                sb = m.get("score_b")
                if sa is not None and sb is not None:
                    _apply_table_result(table, ta, tb, sa, sb)
                continue

            ha = ta in HOST_COUNTRIES
            hb = tb in HOST_COUNTRIES
            xg_a, xg_b = expected_goals(
                ratings.get(ta, DEFAULT_RATING),
                ratings.get(tb, DEFAULT_RATING),
                team_a=ta, team_b=tb,
                home_a=ha, home_b=hb,
                form_a=form.get(ta, 0), form_b=form.get(tb, 0),
            )
            if status == "in_progress" and m.get("score_a") is not None and m.get("score_b") is not None:
                rem_xg_a, rem_xg_b, _ = remaining_expected_goals(xg_a, xg_b, m.get("minute"))
                add_a, add_b = simulate_match(rng, rem_xg_a, rem_xg_b, group_stage=True)
                ga = int(m.get("score_a", 0)) + add_a
                gb = int(m.get("score_b", 0)) + add_b
            else:
                ga, gb = simulate_match(rng, xg_a, xg_b, group_stage=True)
            _apply_table_result(table, ta, tb, ga, gb)

    # Sort: pts -> gd -> gf -> rating tiebreak
    standings = sorted(
        table.items(),
        key=lambda x: (
            x[1]["pts"], x[1]["gd"], x[1]["gf"],
            ratings.get(x[0], DEFAULT_RATING),
        ),
        reverse=True,
    )
    return [{"team": t, **s, "rank": i + 1} for i, (t, s) in enumerate(standings)]


# ---------------------------------------------------------------------------
# Best third-placed teams selection
# ---------------------------------------------------------------------------

def select_best_thirds(third_placed: list[dict],
                       ratings: Optional[dict[str, float]] = None) -> list[dict]:
    """Pick the best 8 third-placed teams."""
    if ratings is None:
        ratings = _load_dynamic_ratings()
    sorted_thirds = sorted(
        third_placed,
        key=lambda x: (
            x["pts"], x["gd"], x["gf"],
            ratings.get(x["team"], DEFAULT_RATING),
        ),
        reverse=True,
    )
    return sorted_thirds[:BEST_THIRD_QUALIFY]


# ---------------------------------------------------------------------------
# Knockout bracket structure
# ---------------------------------------------------------------------------

BRACKET_R32 = [
    # Official FIFA 2026 Round of 32 structure ordered to match the
    # sequential R16_PAIRINGS below.
    ("2A", "2B"),        # Match 73
    ("1F", "2C"),        # Match 75
    ("1E", "3ABCDF"),    # Match 74
    ("1I", "3CDFGH"),    # Match 77
    ("1C", "2F"),        # Match 76
    ("2E", "2I"),        # Match 78
    ("1A", "3CEFHI"),    # Match 79
    ("1L", "3EHIJK"),    # Match 80
    ("2K", "2L"),        # Match 83
    ("1H", "2J"),        # Match 84
    ("1D", "3BEFIJ"),    # Match 81
    ("1G", "3AEHIJ"),    # Match 82
    ("1J", "2H"),        # Match 86
    ("2D", "2G"),        # Match 88
    ("1B", "3EFGIJ"),    # Match 85
    ("1K", "3DEIJL"),    # Match 87
]

WILDCARD_SLOTS = tuple(
    slot
    for match in BRACKET_R32
    for slot in match
    if slot.startswith("3") and len(slot) > 2
)

R16_PAIRINGS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
QF_PAIRINGS = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_PAIRINGS = [(0, 1), (2, 3)]


@lru_cache(maxsize=512)
def _assign_wildcard_groups(selected_groups: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Assign qualified third-place groups to the official wildcard slots.

    Each wildcard slot has a FIFA-defined pool of allowed third-place groups.
    A deterministic backtracking assignment keeps the bracket valid for any
    qualifying combination without inventing impossible matchups.
    """
    candidates = {
        slot: [group for group in selected_groups if group in slot[1:]]
        for slot in WILDCARD_SLOTS
    }
    ordered_slots = sorted(WILDCARD_SLOTS, key=lambda slot: (len(candidates[slot]), slot))
    assignment: dict[str, str] = {}
    used_groups: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(ordered_slots):
            return True

        slot = ordered_slots[index]
        for group in candidates[slot]:
            if group in used_groups:
                continue
            assignment[slot] = group
            used_groups.add(group)
            if backtrack(index + 1):
                return True
            used_groups.remove(group)
            assignment.pop(slot, None)

        return False

    if not backtrack(0):
        raise ValueError(f"Could not assign wildcard slots for groups: {selected_groups}")

    return tuple((slot, assignment[slot]) for slot in WILDCARD_SLOTS)


def assign_wildcard_slots(best_thirds: list[dict]) -> dict[str, str]:
    """Map official wildcard slot tokens (e.g. 3CEFHI) to actual team names."""
    best_by_group = {entry["group"]: entry["team"] for entry in best_thirds}
    selected_groups = tuple(sorted(best_by_group))
    slot_groups = dict(_assign_wildcard_groups(selected_groups))
    return {slot: best_by_group[group] for slot, group in slot_groups.items()}


def resolve_bracket_slot(slot: str, group_standings: dict,
                         best_thirds_map: dict) -> Optional[str]:
    """Resolve seeded bracket slots to concrete team names."""
    if slot.startswith("3") and len(slot) > 2:
        return best_thirds_map.get(slot)

    pos = int(slot[0])  # 1, 2, or 3
    group = slot[1]
    standings = group_standings.get(group, [])
    if pos <= 2:
        if len(standings) >= pos:
            return standings[pos - 1]["team"]
    elif pos == 3:
        return best_thirds_map.get(slot) or best_thirds_map.get(group)
    return None


def simulate_group_stage(rng: np.random.Generator, data: dict,
                         ratings: dict[str, float],
                         form: dict[str, float]) -> tuple[dict, list[dict], dict[str, str]]:
    """Simulate the full group stage and resolve wildcard third-place slots."""
    group_results = {}
    all_thirds = []

    for gname, gdata in data["groups"].items():
        played = []
        for md_key in ["matchday1", "matchday2", "matchday3"]:
            for match in gdata["matches"].get(md_key, []):
                if match["status"] == "played":
                    played.append(match)

        standings = simulate_group(rng, gdata, ratings, form, played)
        group_results[gname] = standings
        if len(standings) >= 3:
            third = dict(standings[2])
            third["group"] = gname
            all_thirds.append(third)

    best_thirds = select_best_thirds(all_thirds, ratings=ratings)
    wildcard_slots = assign_wildcard_slots(best_thirds)
    return group_results, best_thirds, wildcard_slots


def penalty_win_probability(xg_a: float, xg_b: float) -> float:
    """Approximate penalty-shootout edge from the underlying xG advantage."""
    stronger_prob = 0.5 + 0.1 * (xg_a - xg_b) / max(xg_a + xg_b, 0.5)
    return max(0.40, min(0.60, stronger_prob))


def project_knockout_match(ta: str, tb: str,
                           ratings: dict[str, float],
                           form: dict[str, float]) -> dict[str, float | str]:
    """Project a knockout matchup using 90m + ET + penalties logic."""
    ha = ta in HOST_COUNTRIES
    hb = tb in HOST_COUNTRIES
    xg_a, xg_b = expected_goals(
        ratings.get(ta, DEFAULT_RATING),
        ratings.get(tb, DEFAULT_RATING),
        team_a=ta, team_b=tb,
        home_a=ha, home_b=hb,
        form_a=form.get(ta, 0), form_b=form.get(tb, 0),
        knockout=True,
    )
    outcome_90 = match_outcome_probs(score_probabilities(xg_a, xg_b))
    outcome_et = match_outcome_probs(score_probabilities(xg_a / 3, xg_b / 3))
    pen_a = penalty_win_probability(xg_a, xg_b)
    prob_a = outcome_90["win"] + outcome_90["draw"] * (
        outcome_et["win"] + outcome_et["draw"] * pen_a
    )
    prob_b = 1.0 - prob_a
    winner = ta if prob_a >= prob_b else tb

    return {
        "team_a": ta,
        "team_b": tb,
        "winner": winner,
        "prob_a": float(round(prob_a, 3)),
        "prob_b": float(round(prob_b, 3)),
    }


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(rng: np.random.Generator, data: dict,
                        ratings: dict, form: dict) -> dict:
    """Simulate entire tournament once. Return team results + final pair."""
    group_results, best_thirds, best_thirds_map = simulate_group_stage(
        rng, data, ratings, form,
    )
    best_third_teams = {t["team"] for t in best_thirds}

    # Track how far each team goes
    results = {}
    for gname, st in group_results.items():
        for entry in st:
            team = entry["team"]
            rank = entry["rank"]
            if rank <= 2:
                results[team] = "R32"
            elif team in best_third_teams:
                results[team] = "R32"
            else:
                results[team] = "Group"

    def _run_knockout_round(matchups, teams_in):
        """Generic knockout round runner."""
        winners = []
        for i, j in matchups:
            ta, tb = teams_in[i], teams_in[j]
            ha = ta in HOST_COUNTRIES
            hb = tb in HOST_COUNTRIES
            xg_a, xg_b = expected_goals(
                ratings.get(ta, DEFAULT_RATING),
                ratings.get(tb, DEFAULT_RATING),
                team_a=ta, team_b=tb,
                home_a=ha, home_b=hb,
                form_a=form.get(ta, 0), form_b=form.get(tb, 0),
                knockout=True,
            )
            ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
            if how == "pen_b":
                winner = tb
            elif how == "pen_a":
                winner = ta
            else:
                winner = ta if ga > gb else tb
            winners.append(winner)
        return winners

    # --- Round of 32 ---
    r32_winners = []
    for slot_a, slot_b in BRACKET_R32:
        ta = resolve_bracket_slot(slot_a, group_results, best_thirds_map)
        tb = resolve_bracket_slot(slot_b, group_results, best_thirds_map)
        if ta is None or tb is None:
            r32_winners.append(ta or tb or "Unknown")
            continue
        ha = ta in HOST_COUNTRIES
        hb = tb in HOST_COUNTRIES
        xg_a, xg_b = expected_goals(
            ratings.get(ta, DEFAULT_RATING),
            ratings.get(tb, DEFAULT_RATING),
            team_a=ta, team_b=tb,
            home_a=ha, home_b=hb,
            form_a=form.get(ta, 0), form_b=form.get(tb, 0),
            knockout=True,
        )
        ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
        if how == "pen_b":
            winner = tb
        elif how == "pen_a":
            winner = ta
        else:
            winner = ta if ga > gb else tb
        r32_winners.append(winner)
        results[winner] = "R16"

    # --- Round of 16 ---
    r16_winners = []
    for i, j in R16_PAIRINGS:
        ta, tb = r32_winners[i], r32_winners[j]
        ha = ta in HOST_COUNTRIES
        hb = tb in HOST_COUNTRIES
        xg_a, xg_b = expected_goals(
            ratings.get(ta, DEFAULT_RATING),
            ratings.get(tb, DEFAULT_RATING),
            team_a=ta, team_b=tb,
            home_a=ha, home_b=hb,
            form_a=form.get(ta, 0), form_b=form.get(tb, 0),
            knockout=True,
        )
        ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
        if how == "pen_b":
            winner = tb
        elif how == "pen_a":
            winner = ta
        else:
            winner = ta if ga > gb else tb
        r16_winners.append(winner)
        results[winner] = "QF"

    # --- Quarterfinals ---
    qf_winners = []
    for i, j in QF_PAIRINGS:
        ta, tb = r16_winners[i], r16_winners[j]
        ha = ta in HOST_COUNTRIES
        hb = tb in HOST_COUNTRIES
        xg_a, xg_b = expected_goals(
            ratings.get(ta, DEFAULT_RATING),
            ratings.get(tb, DEFAULT_RATING),
            team_a=ta, team_b=tb,
            home_a=ha, home_b=hb,
            form_a=form.get(ta, 0), form_b=form.get(tb, 0),
            knockout=True,
        )
        ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
        if how == "pen_b":
            winner = tb
        elif how == "pen_a":
            winner = ta
        else:
            winner = ta if ga > gb else tb
        qf_winners.append(winner)
        results[winner] = "SF"

    # --- Semifinals ---
    sf_winners = []
    sf_losers = []
    for i, j in SF_PAIRINGS:
        ta, tb = qf_winners[i], qf_winners[j]
        ha = ta in HOST_COUNTRIES
        hb = tb in HOST_COUNTRIES
        xg_a, xg_b = expected_goals(
            ratings.get(ta, DEFAULT_RATING),
            ratings.get(tb, DEFAULT_RATING),
            team_a=ta, team_b=tb,
            home_a=ha, home_b=hb,
            form_a=form.get(ta, 0), form_b=form.get(tb, 0),
            knockout=True,
        )
        ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
        if how == "pen_b":
            winner, loser = tb, ta
        elif how == "pen_a":
            winner, loser = ta, tb
        else:
            winner = ta if ga > gb else tb
            loser = tb if ga > gb else ta
        sf_winners.append(winner)
        sf_losers.append(loser)
        results[winner] = "Final"

    # --- Final ---
    ta, tb = sf_winners[0], sf_winners[1]
    ha = ta in HOST_COUNTRIES
    hb = tb in HOST_COUNTRIES
    xg_a, xg_b = expected_goals(
        ratings.get(ta, DEFAULT_RATING),
        ratings.get(tb, DEFAULT_RATING),
        team_a=ta, team_b=tb,
        home_a=ha, home_b=hb,
        form_a=form.get(ta, 0), form_b=form.get(tb, 0),
        knockout=True,
    )
    ga, gb, how = simulate_knockout_match(rng, xg_a, xg_b)
    if how == "pen_b":
        champion, runner_up = tb, ta
    elif how == "pen_a":
        champion, runner_up = ta, tb
    else:
        champion = ta if ga > gb else tb
        runner_up = tb if ga > gb else ta
    results[champion] = "Winner"
    results[runner_up] = "Final"

    # Return results plus the final pairing for tracking
    final_pair = tuple(sorted([ta, tb]))
    return results, champion, final_pair


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]


def run_monte_carlo(data: dict, n_sims: int = SIMULATIONS) -> tuple[dict, dict]:
    """Run full MC simulation and aggregate results.

    Returns (team_probs, extra_stats).
    extra_stats includes most_common_final, most_common_winner, and
    per-team win counts for confidence interval calculation.
    """
    ratings = _load_dynamic_ratings()
    form = compute_form_adjustments(data)

    all_teams = set()
    for g in data["groups"].values():
        all_teams.update(g["teams"])

    # Counters
    reach_round = {t: defaultdict(int) for t in all_teams}
    win_counts = {t: 0 for t in all_teams}
    final_pairs = Counter()
    winner_counts = Counter()

    rng = np.random.default_rng(SEED)

    for sim in range(n_sims):
        results, champion, final_pair = simulate_tournament(
            rng, data, ratings, form,
        )
        winner_counts[champion] += 1
        final_pairs[final_pair] += 1
        for team in all_teams:
            best = results.get(team, "Group")
            idx = ROUND_ORDER.index(best)
            for r in ROUND_ORDER[:idx + 1]:
                reach_round[team][r] += 1
            if best == "Winner":
                win_counts[team] += 1

    # Convert to probabilities
    probs = {}
    for team in sorted(all_teams):
        win_p = reach_round[team].get("Winner", 0) / n_sims
        probs[team] = {
            "elo": ratings.get(team, DEFAULT_RATING),
            "form": form.get(team, 0),
            "group_stage": reach_round[team].get("R32", 0) / n_sims,
            "r32": reach_round[team].get("R32", 0) / n_sims,
            "r16": reach_round[team].get("R16", 0) / n_sims,
            "qf": reach_round[team].get("QF", 0) / n_sims,
            "sf": reach_round[team].get("SF", 0) / n_sims,
            "final": reach_round[team].get("Final", 0) / n_sims,
            "winner": win_p,
        }

    # Confidence intervals (±1 std dev for binomial)
    confidence_intervals = {}
    for team in all_teams:
        p = win_counts[team] / n_sims
        # Standard error of binomial proportion
        se = math.sqrt(p * (1 - p) / n_sims) if n_sims > 0 else 0
        confidence_intervals[team] = round(se, 6)

    # Most common final and winner
    most_common_final_pair = final_pairs.most_common(1)[0] if final_pairs else (("?", "?"), 0)
    most_common_winner_team = winner_counts.most_common(1)[0] if winner_counts else ("?", 0)

    extra_stats = {
        "most_common_final": list(most_common_final_pair[0]),
        "most_common_final_count": most_common_final_pair[1],
        "most_common_winner": most_common_winner_team[0],
        "most_common_winner_count": most_common_winner_team[1],
        "confidence_intervals": confidence_intervals,
    }

    return probs, extra_stats


# ---------------------------------------------------------------------------
# Match predictions for specific dates
# ---------------------------------------------------------------------------

def predict_matches(data: dict) -> list[dict]:
    """Generate predictions for scheduled, live, and completed matches.

    For played matches, uses archived pre-match predictions from prediction_log
    when available, preserving forecast honesty.
    """
    ratings = _load_dynamic_ratings()
    form = compute_form_adjustments(data)

    # Load archived predictions for honesty
    log_path = Path(__file__).resolve().parent.parent / "data" / "prediction_log.json"
    archived: dict[tuple[str, str], dict] = {}
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f)
            for entry in log:
                key = (entry.get("team_a", ""), entry.get("team_b", ""))
                archived[key] = entry
        except (json.JSONDecodeError, IOError):
            pass

    predictions = []

    for gname, gdata in data["groups"].items():
        for md_key in ["matchday1", "matchday2", "matchday3"]:
            for m in gdata["matches"].get(md_key, []):
                ta, tb = m["team_a"], m["team_b"]

                # Check for archived prediction (honest pre-match forecast)
                arch = archived.get((ta, tb)) or archived.get((tb, ta))

                if m["status"] == "played" and arch:
                    # Use archived pre-match prediction
                    ha = ta in HOST_COUNTRIES
                    hb = tb in HOST_COUNTRIES
                    xg_a_current, xg_b_current = expected_goals(
                        ratings.get(ta, DEFAULT_RATING),
                        ratings.get(tb, DEFAULT_RATING),
                        team_a=ta, team_b=tb,
                        home_a=ha, home_b=hb,
                        form_a=form.get(ta, 0), form_b=form.get(tb, 0),
                    )

                    # Extract archived prediction fields using the real log schema.
                    prob_win_a = arch.get("prob_win_a")
                    prob_draw = arch.get("prob_draw")
                    prob_win_b = arch.get("prob_win_b")
                    if prob_win_a is None or prob_draw is None or prob_win_b is None:
                        arch_probs = arch.get("predicted_probs", {})
                        prob_win_a = arch_probs.get("win", 0.33)
                        prob_draw = arch_probs.get("draw", 0.33)
                        prob_win_b = arch_probs.get("loss", 0.33)

                    arch_xg_a = arch.get("xg_a", xg_a_current)
                    arch_xg_b = arch.get("xg_b", xg_b_current)
                    pred_score_a = arch.get("predicted_score_a", 0)
                    pred_score_b = arch.get("predicted_score_b", 0)

                    # If the archive has team_b as first team, flip
                    if arch.get("team_a") == tb:
                        prob_win_a, prob_win_b = prob_win_b, prob_win_a
                        arch_xg_a, arch_xg_b = arch_xg_b, arch_xg_a
                        pred_score_a, pred_score_b = pred_score_b, pred_score_a

                    matrix = score_probabilities(arch_xg_a, arch_xg_b)

                    score_matrix_list = build_display_score_matrix(matrix)

                    odds_a = prob_to_decimal_odds(prob_win_a)
                    odds_draw = prob_to_decimal_odds(prob_draw)
                    odds_b = prob_to_decimal_odds(prob_win_b)

                    pred = {
                        "date": m["date"],
                        "group": gname,
                        "team_a": ta,
                        "team_b": tb,
                        "status": m["status"],
                        "actual_score_a": m.get("score_a"),
                        "actual_score_b": m.get("score_b"),
                        "xg_a": round(arch_xg_a, 2),
                        "xg_b": round(arch_xg_b, 2),
                        "predicted_score_a": pred_score_a,
                        "predicted_score_b": pred_score_b,
                        "prob_win_a": round(prob_win_a, 4),
                        "prob_draw": round(prob_draw, 4),
                        "prob_win_b": round(prob_win_b, 4),
                        "confidence": round(max(prob_win_a, prob_draw, prob_win_b), 4),
                        "score_matrix": score_matrix_list,
                        "odds_a": odds_a,
                        "odds_draw": odds_draw,
                        "odds_b": odds_b,
                        "prediction_source": "archived",
                    }
                    predictions.append(pred)
                elif m["status"] == "in_progress":
                    ha = ta in HOST_COUNTRIES
                    hb = tb in HOST_COUNTRIES
                    xg_a, xg_b = expected_goals(
                        ratings.get(ta, DEFAULT_RATING),
                        ratings.get(tb, DEFAULT_RATING),
                        team_a=ta, team_b=tb,
                        home_a=ha, home_b=hb,
                        form_a=form.get(ta, 0), form_b=form.get(tb, 0),
                    )
                    current_score_a = int(m.get("score_a") or 0)
                    current_score_b = int(m.get("score_b") or 0)
                    live_state = live_match_state(
                        current_score_a,
                        current_score_b,
                        m.get("minute"),
                        xg_a,
                        xg_b,
                        group_stage=True,
                    )
                    outcome = live_state["outcome"]
                    pred_score = live_state["predicted_score"]
                    matrix = live_state["matrix"]
                    score_matrix_list = build_display_score_matrix(matrix)

                    odds_a = prob_to_decimal_odds(outcome["win"])
                    odds_draw = prob_to_decimal_odds(outcome["draw"])
                    odds_b = prob_to_decimal_odds(outcome["loss"])

                    pred = {
                        "date": m["date"],
                        "group": gname,
                        "team_a": ta,
                        "team_b": tb,
                        "status": m["status"],
                        "actual_score_a": current_score_a,
                        "actual_score_b": current_score_b,
                        "live_minute": live_state["minute"],
                        "remaining_minutes": round(live_state["remaining_fraction"] * FULL_MATCH_MINUTES, 1),
                        "xg_a": round(xg_a, 2),
                        "xg_b": round(xg_b, 2),
                        "remaining_xg_a": round(live_state["remaining_xg_a"], 2),
                        "remaining_xg_b": round(live_state["remaining_xg_b"], 2),
                        "predicted_score_a": pred_score[0],
                        "predicted_score_b": pred_score[1],
                        "prob_win_a": round(outcome["win"], 4),
                        "prob_draw": round(outcome["draw"], 4),
                        "prob_win_b": round(outcome["loss"], 4),
                        "confidence": round(max(outcome.values()), 4),
                        "score_matrix": score_matrix_list,
                        "odds_a": odds_a,
                        "odds_draw": odds_draw,
                        "odds_b": odds_b,
                        "prediction_source": "live_conditional",
                    }
                    predictions.append(pred)
                else:
                    # Fresh prediction (not yet played or no archive)
                    ha = ta in HOST_COUNTRIES
                    hb = tb in HOST_COUNTRIES
                    xg_a, xg_b = expected_goals(
                        ratings.get(ta, DEFAULT_RATING),
                        ratings.get(tb, DEFAULT_RATING),
                        team_a=ta, team_b=tb,
                        home_a=ha, home_b=hb,
                        form_a=form.get(ta, 0), form_b=form.get(tb, 0),
                    )
                    matrix = score_probabilities(xg_a, xg_b)
                    outcome = match_outcome_probs(matrix, group_stage=True)
                    pred_score = most_likely_score(matrix)

                    score_matrix_list = build_display_score_matrix(matrix)

                    odds_a = prob_to_decimal_odds(outcome["win"])
                    odds_draw = prob_to_decimal_odds(outcome["draw"])
                    odds_b = prob_to_decimal_odds(outcome["loss"])

                    pred = {
                        "date": m["date"],
                        "group": gname,
                        "team_a": ta,
                        "team_b": tb,
                        "status": m["status"],
                        "actual_score_a": m.get("score_a"),
                        "actual_score_b": m.get("score_b"),
                        "xg_a": round(xg_a, 2),
                        "xg_b": round(xg_b, 2),
                        "predicted_score_a": pred_score[0],
                        "predicted_score_b": pred_score[1],
                        "prob_win_a": round(outcome["win"], 4),
                        "prob_draw": round(outcome["draw"], 4),
                        "prob_win_b": round(outcome["loss"], 4),
                        "confidence": round(max(outcome.values()), 4),
                        "score_matrix": score_matrix_list,
                        "odds_a": odds_a,
                        "odds_draw": odds_draw,
                        "odds_b": odds_b,
                        "prediction_source": "scheduled",
                    }
                    predictions.append(pred)

    predictions.sort(key=lambda x: x["date"])
    return predictions


# ---------------------------------------------------------------------------
# Group projected standings
# ---------------------------------------------------------------------------

def project_group_standings(data: dict, n_sims: int = 5000) -> dict:
    """For each group, simulate n_sims times and get projected final standings.

    Includes best-third qualification for 2026 format (top 2 + best 8 thirds).
    """
    ratings = _load_dynamic_ratings()
    form = compute_form_adjustments(data)
    rng = np.random.default_rng(SEED + 1)

    group_names = sorted(data["groups"].keys())

    # Initialize counters
    team_counts: dict[str, dict[str, dict[str, int]]] = {}
    for gname in group_names:
        team_counts[gname] = {
            t: {"1st": 0, "2nd": 0, "3rd": 0, "4th": 0, "qualify": 0}
            for t in data["groups"][gname]["teams"]
        }

    for _ in range(n_sims):
        all_standings: dict[str, list[dict]] = {}
        all_thirds: list[dict] = []

        # Simulate all groups together
        for gname in group_names:
            gdata = data["groups"][gname]
            played = []
            for md_key in ["matchday1", "matchday2", "matchday3"]:
                for m in gdata["matches"].get(md_key, []):
                    if m["status"] == "played":
                        played.append(m)

            st = simulate_group(rng, gdata, ratings, form, played)
            all_standings[gname] = st

            # Track positions
            for entry in st:
                rank = entry["rank"]
                team = entry["team"]
                pos_key = f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}"
                team_counts[gname][team][pos_key] += 1
                if rank <= 2:
                    team_counts[gname][team]["qualify"] += 1

            # Collect third-placed teams
            if len(st) >= 3:
                third = dict(st[2])  # copy
                third["group"] = gname
                all_thirds.append(third)

        # Select best 8 thirds across all groups
        best_thirds = select_best_thirds(all_thirds, ratings=ratings)
        best_third_teams = {t["team"] for t in best_thirds}

        # Credit qualifying thirds
        for gname in group_names:
            st = all_standings[gname]
            if len(st) >= 3:
                third_team = st[2]["team"]
                if third_team in best_third_teams:
                    team_counts[gname][third_team]["qualify"] += 1

    # Convert to probabilities
    projections = {}
    for gname in group_names:
        proj = {}
        for team, counts in team_counts[gname].items():
            proj[team] = {k: round(v / n_sims, 4) for k, v in counts.items()}
        projections[gname] = proj

    return projections


# ---------------------------------------------------------------------------
# Knockout bracket prediction
# ---------------------------------------------------------------------------

def predict_knockout_bracket(data: dict) -> dict:
    """Predict a valid knockout bracket using official 2026 slot constraints."""
    ratings = _load_dynamic_ratings()
    form = compute_form_adjustments(data)

    rng = np.random.default_rng(SEED + 2)
    n = 5000

    # Count full valid Round-of-32 lineups so the displayed bracket remains
    # globally consistent and cannot duplicate a team across slots.
    lineup_counts = Counter()
    for _ in range(n):
        group_results, _, wildcard_slots = simulate_group_stage(rng, data, ratings, form)
        lineup = []
        for slot_a, slot_b in BRACKET_R32:
            ta = resolve_bracket_slot(slot_a, group_results, wildcard_slots)
            tb = resolve_bracket_slot(slot_b, group_results, wildcard_slots)
            if ta and tb:
                lineup.append((ta, tb))
        if len(lineup) == len(BRACKET_R32):
            lineup_counts[tuple(lineup)] += 1

    # Build bracket
    bracket = {"r32": [], "r16": [], "qf": [], "sf": [], "final": {}}

    # R32
    r32_teams = []
    selected_lineup = lineup_counts.most_common(1)[0][0] if lineup_counts else ()
    for index, (slot_a, slot_b) in enumerate(BRACKET_R32):
        if index < len(selected_lineup):
            ta, tb = selected_lineup[index]
        else:
            ta = slot_a
            tb = slot_b
        matchup = project_knockout_match(ta, tb, ratings, form)
        bracket["r32"].append(matchup)
        r32_teams.append(matchup["winner"])

    def _bracket_round(pairings, teams_in):
        """Predict a bracket round."""
        round_matches = []
        winners = []
        for i, j in pairings:
            ta, tb = teams_in[i], teams_in[j]
            matchup = project_knockout_match(ta, tb, ratings, form)
            round_matches.append(matchup)
            winners.append(matchup["winner"])
        return round_matches, winners

    # R16
    r16_matches, r16_teams = _bracket_round(R16_PAIRINGS, r32_teams)
    bracket["r16"] = r16_matches

    # QF
    qf_matches, qf_teams = _bracket_round(QF_PAIRINGS, r16_teams)
    bracket["qf"] = qf_matches

    # SF
    sf_matches, sf_teams = _bracket_round(SF_PAIRINGS, qf_teams)
    bracket["sf"] = sf_matches

    # Final
    ta, tb = sf_teams[0], sf_teams[1]
    bracket["final"] = project_knockout_match(ta, tb, ratings, form)

    return bracket


# ---------------------------------------------------------------------------
# Probability timeline (for dashboard "stock charts")
# ---------------------------------------------------------------------------

def build_probability_timeline(data: dict, n_sims: int = 10_000) -> dict:
    """Generate probability snapshots at each matchday for the trading charts.

    Returns dict: team -> list of {matchday, win_pct, qualify_pct}.
    Simulates as if we're at different points in the tournament.
    """
    ratings = _load_dynamic_ratings()
    all_teams = set()
    for g in data["groups"].values():
        all_teams.update(g["teams"])

    timeline: dict[str, list[dict]] = {t: [] for t in all_teams}

    # Pre-tournament snapshot (no form, no results)
    rng_pre = np.random.default_rng(SEED + 100)
    empty_form: dict[str, float] = {}
    pre_reach = {t: defaultdict(int) for t in all_teams}

    # Build clean pre-tournament data (reset all matches to unplayed)
    pre_data = copy.deepcopy(data)
    for _gname, _gdata in pre_data["groups"].items():
        for _md_key in ["matchday1", "matchday2", "matchday3"]:
            for _m in _gdata["matches"].get(_md_key, []):
                _m["status"] = "scheduled"
                _m.pop("score_a", None)
                _m.pop("score_b", None)

    for _ in range(n_sims):
        results, _, _ = simulate_tournament(rng_pre, pre_data, ratings, empty_form)
        for team in all_teams:
            best = results.get(team, "Group")
            idx = ROUND_ORDER.index(best)
            for r in ROUND_ORDER[:idx + 1]:
                pre_reach[team][r] += 1

    for team in all_teams:
        timeline[team].append({
            "matchday": "pre",
            "win_pct": round(pre_reach[team].get("Winner", 0) / n_sims, 4),
            "qualify_pct": round(pre_reach[team].get("R32", 0) / n_sims, 4),
        })

    # Post-MD1 snapshot (with actual form adjustments from MD1 results)
    form_md1 = compute_form_adjustments(data)
    rng_md1 = np.random.default_rng(SEED + 101)
    md1_reach = {t: defaultdict(int) for t in all_teams}

    for _ in range(n_sims):
        results, _, _ = simulate_tournament(rng_md1, data, ratings, form_md1)
        for team in all_teams:
            best = results.get(team, "Group")
            idx = ROUND_ORDER.index(best)
            for r in ROUND_ORDER[:idx + 1]:
                md1_reach[team][r] += 1

    for team in all_teams:
        timeline[team].append({
            "matchday": "md1",
            "win_pct": round(md1_reach[team].get("Winner", 0) / n_sims, 4),
            "qualify_pct": round(md1_reach[team].get("R32", 0) / n_sims, 4),
        })

    return timeline


# ---------------------------------------------------------------------------
# Generate full predictions output
# ---------------------------------------------------------------------------

def generate_predictions() -> dict:
    """Main entry: generate all predictions and return as dict."""
    data = load_tournament_data()

    print("Running Monte Carlo simulations ({:,} iterations)...".format(SIMULATIONS))
    tournament_probs, mc_extra = run_monte_carlo(data, SIMULATIONS)

    print("Predicting individual matches...")
    match_predictions = predict_matches(data)

    print("Projecting group standings...")
    group_projections = project_group_standings(data)

    print("Building knockout bracket...")
    bracket = predict_knockout_bracket(data)

    print("Building probability timeline...")
    probability_timeline = build_probability_timeline(data)

    # Power rankings by tournament win probability
    ci = mc_extra["confidence_intervals"]
    power_rankings = sorted(
        [
            {
                "team": t,
                **p,
                "confidence_interval": round(ci.get(t, 0), 6),
            }
            for t, p in tournament_probs.items()
        ],
        key=lambda x: x["winner"],
        reverse=True,
    )

    # Compute stats
    played_matches = [m for m in match_predictions if m["status"] == "played"]
    total_goals = sum(
        m["actual_score_a"] + m["actual_score_b"]
        for m in played_matches
        if m["actual_score_a"] is not None
    )
    num_played = len(played_matches)
    upsets = 0
    for m in played_matches:
        if m["actual_score_a"] is not None and m["actual_score_b"] is not None:
            if m["actual_score_a"] > m["actual_score_b"] and m["prob_win_a"] < 0.35:
                upsets += 1
            elif m["actual_score_b"] > m["actual_score_a"] and m["prob_win_b"] < 0.35:
                upsets += 1

    stats = {
        "matches_played": num_played,
        "matches_live": sum(1 for m in match_predictions if m["status"] == "in_progress"),
        "total_goals": total_goals,
        "avg_goals_per_match": round(total_goals / max(num_played, 1), 2),
        "upsets": upsets,
        "simulations": SIMULATIONS,
        # New stats
        "most_common_final": mc_extra["most_common_final"],
        "most_common_final_pct": round(
            mc_extra["most_common_final_count"] / SIMULATIONS, 4
        ),
        "most_common_winner": mc_extra["most_common_winner"],
        "most_common_winner_pct": round(
            mc_extra["most_common_winner_count"] / SIMULATIONS, 4
        ),
    }

    # Model info
    model_info = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "parameters": {
            "simulations": SIMULATIONS,
            "base_goals": BASE_GOALS,
            "elo_divisor": ELO_DIVISOR,
            "home_advantage": HOME_ADVANTAGE_GOALS,
            "dixon_coles_rho": DIXON_COLES_RHO,
            "group_draw_bias": GROUP_DRAW_BIAS,
        },
        "features": [
            "FIFA World Rankings (June 2026)",
            "Dixon-Coles low-score correction",
            "World Cup performance boost",
            "Confederation strength adjustment",
            "Head-to-head historical data",
            "Matchday momentum system",
            "Home advantage for host nations",
            "Travel/fatigue penalty",
            "Group stage draw bias",
            "Conditional live match forecasting",
            "Probability timeline tracking",
            "Betting odds conversion",
            "Bayesian Elo rating updates",
            "TheSportsDB live/final event feed",
            "Automatic result fetching",
            "Brier score calibration tracking",
        ],
        "confidence_metric": "binomial_std_error",
    }

    # Check if we're using dynamic ratings
    ratings_path = Path(__file__).resolve().parent.parent / "data" / "ratings_history.json"
    if ratings_path.exists():
        model_info["ratings_source"] = "dynamic (Bayesian-updated Elo)"
    else:
        model_info["ratings_source"] = "static (FIFA Rankings June 2026)"

    # Today's matches
    today = date.today().isoformat()
    todays_matches = [m for m in match_predictions if m["date"] == today]

    output = {
        "generated_date": today,
        "today": today,
        "todays_matches": todays_matches,
        "match_predictions": match_predictions,
        "tournament_probabilities": tournament_probs,
        "power_rankings": power_rankings,
        "group_projections": group_projections,
        "knockout_bracket": bracket,
        "stats": stats,
        "groups": data["groups"],
        "flags": data.get("flags", {}),
        # New fields
        "probability_timeline": probability_timeline,
        "model_info": model_info,
    }

    return output


if __name__ == "__main__":
    output = generate_predictions()
    out_path = Path(__file__).resolve().parent.parent / "dashboard" / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {out_path}")

    # Print today's matches
    print(f"\n{'='*60}")
    print(f"  TODAY'S PREDICTIONS — {output['today']}")
    print(f"{'='*60}")
    for m in output["todays_matches"]:
        flag_a = output["flags"].get(m["team_a"], "")
        flag_b = output["flags"].get(m["team_b"], "")
        print(f"\n  {flag_a} {m['team_a']} vs {m['team_b']} {flag_b}")
        print(f"  Group {m['group']}")
        print(f"  Predicted: {m['predicted_score_a']}-{m['predicted_score_b']}")
        print(f"  Win: {m['prob_win_a']*100:.1f}% | Draw: {m['prob_draw']*100:.1f}% | Loss: {m['prob_win_b']*100:.1f}%")
        print(f"  xG: {m['xg_a']} - {m['xg_b']}")
        print(f"  Odds: {m['odds_a']} / {m['odds_draw']} / {m['odds_b']}")

    # Print model info
    print(f"\n{'='*60}")
    print(f"  MODEL: {output['model_info']['name']} v{output['model_info']['version']}")
    print(f"{'='*60}")
    print(f"  Simulations: {output['stats']['simulations']:,}")
    print(f"  Most likely final: {' vs '.join(output['stats']['most_common_final'])}")
    print(f"    ({output['stats']['most_common_final_pct']*100:.1f}% of simulations)")
    print(f"  Most likely winner: {output['stats']['most_common_winner']}")
    print(f"    ({output['stats']['most_common_winner_pct']*100:.1f}% of simulations)")
