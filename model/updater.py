"""
Data updater + Bayesian rating update for World Cup 2026 predictions.

Handles:
- Updating world_cup_2026.json with new match results
- FIFA Elo rating updates after each result
- Storing pre-match predictions for accuracy tracking
- Recalculating group standings
"""

import json
import logging
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

from model.calibrator import actual_result_category, brier_score, classify_result

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOURNAMENT_FILE = DATA_DIR / "world_cup_2026.json"
RATINGS_HISTORY_FILE = DATA_DIR / "ratings_history.json"
PREDICTION_LOG_FILE = DATA_DIR / "prediction_log.json"
RESULTS_UPDATE_FILE = DATA_DIR / "results_update.json"

# FIFA Elo K-factors
K_GROUP = 50
K_KNOCKOUT = 60


# ---------------------------------------------------------------------------
# Elo rating update
# ---------------------------------------------------------------------------

def elo_expected(rating_a: float, rating_b: float) -> float:
    """Compute expected result for team A (FIFA Elo formula)."""
    delta = rating_a - rating_b
    return 1.0 / (10.0 ** (-delta / 600.0) + 1.0)


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: int,
    score_b: int,
    k: float = K_GROUP,
) -> tuple[float, float]:
    """Update Elo ratings for both teams after a match.

    Args:
        rating_a: Current rating of team A.
        rating_b: Current rating of team B.
        score_a: Goals scored by team A.
        score_b: Goals scored by team B.
        k: K-factor (50 for group stage, 60 for knockout).

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    w_expected_a = elo_expected(rating_a, rating_b)
    w_expected_b = 1.0 - w_expected_a

    # Actual result
    if score_a > score_b:
        w_actual_a = 1.0
        w_actual_b = 0.0
    elif score_a < score_b:
        w_actual_a = 0.0
        w_actual_b = 1.0
    else:
        w_actual_a = 0.5
        w_actual_b = 0.5

    new_a = rating_a + k * (w_actual_a - w_expected_a)
    new_b = rating_b + k * (w_actual_b - w_expected_b)

    return round(new_a, 2), round(new_b, 2)


# ---------------------------------------------------------------------------
# Load/save helpers
# ---------------------------------------------------------------------------

def load_tournament_data() -> dict:
    """Load tournament data from JSON."""
    with open(TOURNAMENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tournament_data(data: dict) -> None:
    """Save tournament data back to JSON."""
    with open(TOURNAMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Tournament data saved to %s", TOURNAMENT_FILE)


def load_ratings_history() -> dict:
    """Load ratings history. Returns empty structure if file doesn't exist."""
    if RATINGS_HISTORY_FILE.exists():
        with open(RATINGS_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ratings_history(history: dict) -> None:
    """Save ratings history to JSON."""
    with open(RATINGS_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info("Ratings history saved to %s", RATINGS_HISTORY_FILE)


def load_prediction_log() -> list[dict]:
    """Load existing prediction log."""
    if PREDICTION_LOG_FILE.exists():
        with open(PREDICTION_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_prediction_log(log: list[dict]) -> None:
    """Save prediction log to JSON."""
    with open(PREDICTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    logger.info("Prediction log saved to %s", PREDICTION_LOG_FILE)


def normalize_update_status(entry: dict) -> str:
    """Normalize inbound update status while keeping old files compatible."""
    status = str(entry.get("status", "played")).strip().lower()
    if status in {"live", "in_progress", "in-progress", "inprogress"}:
        return "in_progress"
    return "played"


def get_latest_ratings(ratings_history: dict) -> Optional[dict[str, float]]:
    """Extract the latest ratings from the history dict.

    The history has keys like 'initial', 'after_md1', 'after_md2', etc.
    Returns the values from the last chronological key, or None if empty.
    """
    if not ratings_history:
        return None

    # Order: initial < after_md1 < after_md2 < after_md3
    key_order = ["initial", "after_md1", "after_md2", "after_md3"]
    latest = None
    for k in key_order:
        if k in ratings_history:
            latest = ratings_history[k]
    # Also check any other keys
    for k in sorted(ratings_history.keys()):
        if k not in key_order:
            latest = ratings_history[k]
    return latest


# ---------------------------------------------------------------------------
# Match finding & updating
# ---------------------------------------------------------------------------

def find_match_in_data(
    data: dict,
    team_a: str,
    team_b: str,
    allowed_statuses: Optional[set[str]] = None,
) -> Optional[tuple[str, str, int, dict]]:
    """Find a match in the tournament data across allowed statuses.

    Returns (group_name, matchday_key, match_index, match_dict) or None.
    Searches both orderings (team_a/team_b and team_b/team_a).
    """
    if allowed_statuses is None:
        allowed_statuses = {"scheduled"}

    for gname, gdata in data["groups"].items():
        for md_key in ["matchday1", "matchday2", "matchday3"]:
            for i, match in enumerate(gdata["matches"].get(md_key, [])):
                if match.get("status") not in allowed_statuses:
                    continue
                if (match["team_a"] == team_a and match["team_b"] == team_b) or \
                   (match["team_a"] == team_b and match["team_b"] == team_a):
                    return (gname, md_key, i, match)
    return None


def recalculate_standings(group_data: dict) -> list[dict]:
    """Recalculate group standings from played and live matches.

    Returns sorted standings list.
    """
    teams = group_data["teams"]
    table = {
        t: {"team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0}
        for t in teams
    }

    for md_key in ["matchday1", "matchday2", "matchday3"]:
        for match in group_data["matches"].get(md_key, []):
            if match.get("status") not in {"played", "in_progress"}:
                continue
            ta = match["team_a"]
            tb = match["team_b"]
            sa = match["score_a"]
            sb = match["score_b"]
            if sa is None or sb is None:
                continue

            table[ta]["played"] += 1
            table[tb]["played"] += 1
            table[ta]["gf"] += sa
            table[ta]["ga"] += sb
            table[tb]["gf"] += sb
            table[tb]["ga"] += sa
            table[ta]["gd"] += sa - sb
            table[tb]["gd"] += sb - sa

            if sa > sb:
                table[ta]["won"] += 1
                table[ta]["points"] += 3
                table[tb]["lost"] += 1
            elif sa < sb:
                table[tb]["won"] += 1
                table[tb]["points"] += 3
                table[ta]["lost"] += 1
            else:
                table[ta]["drawn"] += 1
                table[tb]["drawn"] += 1
                table[ta]["points"] += 1
                table[tb]["points"] += 1

    # Sort: points > gd > gf > alphabetical
    standings = sorted(
        table.values(),
        key=lambda x: (x["points"], x["gd"], x["gf"], x["team"]),
        reverse=True,
    )
    return standings


def build_pre_match_snapshot(
    group_name: str,
    match: dict,
    data: dict,
    current_ratings: dict[str, float],
) -> dict:
    """Capture the pre-match model state before a match goes live or final."""
    from model.predictor import (
        DEFAULT_RATING,
        HOST_COUNTRIES,
        compute_form_adjustments,
        expected_goals,
        match_outcome_probs,
        most_likely_score,
        score_probabilities,
    )

    team_a = match["team_a"]
    team_b = match["team_b"]
    form = compute_form_adjustments(data)
    home_a = team_a in HOST_COUNTRIES
    home_b = team_b in HOST_COUNTRIES
    xg_a, xg_b = expected_goals(
        current_ratings.get(team_a, DEFAULT_RATING),
        current_ratings.get(team_b, DEFAULT_RATING),
        team_a=team_a,
        team_b=team_b,
        home_a=home_a,
        home_b=home_b,
        form_a=form.get(team_a, 0),
        form_b=form.get(team_b, 0),
    )
    matrix = score_probabilities(xg_a, xg_b)
    outcome = match_outcome_probs(matrix, group_stage=True)
    pred_score = most_likely_score(matrix)

    return {
        "date": match.get("date", ""),
        "group": group_name,
        "team_a": team_a,
        "team_b": team_b,
        "predicted_score_a": pred_score[0],
        "predicted_score_b": pred_score[1],
        "prob_win_a": round(outcome["win"], 4),
        "prob_draw": round(outcome["draw"], 4),
        "prob_win_b": round(outcome["loss"], 4),
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "captured_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def build_log_entry_from_snapshot(snapshot: dict,
                                  actual_score_a: int,
                                  actual_score_b: int) -> dict:
    """Turn a stored pre-match snapshot into a final accuracy log entry."""
    actual = actual_result_category(actual_score_a, actual_score_b)
    bs = brier_score(
        snapshot.get("prob_win_a", 0.33),
        snapshot.get("prob_draw", 0.33),
        snapshot.get("prob_win_b", 0.33),
        actual,
    )
    result_cat = classify_result(
        snapshot.get("predicted_score_a", 0),
        snapshot.get("predicted_score_b", 0),
        actual_score_a,
        actual_score_b,
    )

    return {
        "date": snapshot.get("date", ""),
        "group": snapshot.get("group", ""),
        "team_a": snapshot.get("team_a", ""),
        "team_b": snapshot.get("team_b", ""),
        "predicted_score_a": snapshot.get("predicted_score_a", 0),
        "predicted_score_b": snapshot.get("predicted_score_b", 0),
        "prob_win_a": snapshot.get("prob_win_a", 0.33),
        "prob_draw": snapshot.get("prob_draw", 0.33),
        "prob_win_b": snapshot.get("prob_win_b", 0.33),
        "xg_a": snapshot.get("xg_a", 0.0),
        "xg_b": snapshot.get("xg_b", 0.0),
        "actual_score_a": actual_score_a,
        "actual_score_b": actual_score_b,
        "result_category": result_cat,
        "brier_score": round(bs, 4),
    }


def determine_matchday_label(data: dict) -> str:
    """Determine which matchday we're updating (for ratings history key).

    Counts played matches per group to determine the current matchday.
    """
    max_played = 0
    for gdata in data["groups"].values():
        played_count = 0
        for md_key in ["matchday1", "matchday2", "matchday3"]:
            for m in gdata["matches"].get(md_key, []):
                if m["status"] == "played":
                    played_count += 1
        max_played = max(max_played, played_count)

    if max_played <= 2:
        return "after_md1"
    elif max_played <= 4:
        return "after_md2"
    else:
        return "after_md3"


# ---------------------------------------------------------------------------
# Main update pipeline
# ---------------------------------------------------------------------------

def log_predictions_for_matches(
    new_results: list[dict],
    data: dict,
    current_ratings: dict[str, float],
) -> list[dict]:
    """Generate pre-match prediction log entries for matches about to be updated.

    This captures what the model predicted BEFORE seeing the actual result,
    which is essential for honest accuracy tracking.
    """
    log_entries = []

    for result in new_results:
        if normalize_update_status(result) != "played":
            continue

        ta = result["team_a"]
        tb = result["team_b"]
        sa = result["score_a"]
        sb = result["score_b"]

        found = find_match_in_data(data, ta, tb, allowed_statuses={"scheduled", "in_progress"})
        if found is None:
            continue
        gname, md_key, idx, match = found

        # Use the match's canonical ordering
        canon_ta = match["team_a"]
        canon_tb = match["team_b"]

        # Flip scores if result teams are reversed from canonical order
        if canon_ta == tb and canon_tb == ta:
            actual_sa, actual_sb = sb, sa
        else:
            actual_sa, actual_sb = sa, sb

        snapshot = match.get("pre_match_prediction")
        if snapshot is None:
            snapshot = build_pre_match_snapshot(gname, match, data, current_ratings)

        entry = build_log_entry_from_snapshot(snapshot, actual_sa, actual_sb)
        log_entries.append(entry)

    return log_entries


def apply_live_updates(
    live_updates: list[dict],
    data: dict,
    current_ratings: dict[str, float],
) -> tuple[dict, int]:
    """Apply in-progress match snapshots without touching Elo ratings."""
    applied = 0
    groups_to_update = set()

    for update in live_updates:
        ta = update["team_a"]
        tb = update["team_b"]
        sa = update["score_a"]
        sb = update["score_b"]
        minute = int(update.get("minute", 0))

        found = find_match_in_data(data, ta, tb, allowed_statuses={"scheduled", "in_progress"})
        if found is None:
            logger.debug("Live match %s vs %s not found as scheduled/live — skipping.", ta, tb)
            continue

        gname, md_key, idx, match = found
        canon_ta = match["team_a"]
        canon_tb = match["team_b"]

        if match.get("pre_match_prediction") is None:
            match["pre_match_prediction"] = build_pre_match_snapshot(gname, match, data, current_ratings)

        if canon_ta == tb and canon_tb == ta:
            match["score_a"] = sb
            match["score_b"] = sa
        else:
            match["score_a"] = sa
            match["score_b"] = sb

        match["status"] = "in_progress"
        match["minute"] = minute
        data["groups"][gname]["matches"][md_key][idx] = match
        groups_to_update.add(gname)
        applied += 1

    for gname in groups_to_update:
        data["groups"][gname]["standings"] = recalculate_standings(data["groups"][gname])
        logger.info("Recalculated live standings for Group %s", gname)

    return data, applied


def apply_results(
    new_results: list[dict],
    data: dict,
    ratings: dict[str, float],
) -> tuple[dict, dict[str, float], int]:
    """Apply new results to tournament data and update Elo ratings.

    Args:
        new_results: List of result dicts with team_a, team_b, score_a, score_b.
        data: Tournament data dict (modified in-place).
        ratings: Current ratings dict (modified in-place).

    Returns:
        Tuple of (updated_data, updated_ratings, num_applied).
    """
    applied = 0
    groups_to_update = set()

    for result in new_results:
        ta = result["team_a"]
        tb = result["team_b"]
        sa = result["score_a"]
        sb = result["score_b"]

        found = find_match_in_data(data, ta, tb, allowed_statuses={"scheduled", "in_progress"})
        if found is None:
            logger.debug("Match %s vs %s not found as scheduled/live — skipping.", ta, tb)
            continue

        gname, md_key, idx, match = found

        # Apply the result using the canonical team ordering
        canon_ta = match["team_a"]
        canon_tb = match["team_b"]

        if canon_ta == tb and canon_tb == ta:
            # Flip scores to match canonical order
            match["score_a"] = sb
            match["score_b"] = sa
        else:
            match["score_a"] = sa
            match["score_b"] = sb

        match["status"] = "played"
        match.pop("minute", None)
        match.pop("pre_match_prediction", None)
        data["groups"][gname]["matches"][md_key][idx] = match

        # Elo update (use canonical order scores)
        actual_sa = match["score_a"]
        actual_sb = match["score_b"]
        rating_a = ratings.get(canon_ta, 1400.0)
        rating_b = ratings.get(canon_tb, 1400.0)
        new_a, new_b = elo_update(rating_a, rating_b, actual_sa, actual_sb, K_GROUP)
        ratings[canon_ta] = new_a
        ratings[canon_tb] = new_b
        logger.info(
            "Applied: %s %d-%d %s | Elo: %s %.0f->%.0f, %s %.0f->%.0f",
            canon_ta, actual_sa, actual_sb, canon_tb,
            canon_ta, rating_a, new_a,
            canon_tb, rating_b, new_b,
        )

        groups_to_update.add(gname)
        applied += 1

    # Recalculate standings for affected groups
    for gname in groups_to_update:
        data["groups"][gname]["standings"] = recalculate_standings(
            data["groups"][gname]
        )
        logger.info("Recalculated standings for Group %s", gname)

    return data, ratings, applied


def run_update(new_results: list[dict]) -> dict:
    """Execute the full update pipeline.

    1. Load current data and ratings
    2. Log pre-match predictions
    3. Apply results to tournament data
    4. Update Elo ratings
    5. Save everything

    Args:
        new_results: List of result dicts.

    Returns:
        Summary dict with update statistics.
    """
    if not new_results:
        return {
            "results_found": 0,
            "results_applied": 0,
            "live_updates_applied": 0,
            "ratings_updated": False,
            "prediction_entries_added": 0,
        }

    # Load current state
    data = load_tournament_data()
    ratings_history = load_ratings_history()
    prediction_log = load_prediction_log()

    # Get current ratings: use latest from history, or fall back to
    # the elo_ratings in tournament data, or FIFA_RANKINGS from predictor
    current_ratings = get_latest_ratings(ratings_history)
    if current_ratings is None:
        # Use the elo_ratings from tournament data as initial
        current_ratings = dict(data.get("elo_ratings", {}))

    # Save initial ratings if not already saved
    if "initial" not in ratings_history:
        ratings_history["initial"] = dict(current_ratings)

    live_updates = []
    final_results = []
    for result in new_results:
        status = normalize_update_status(result)
        allowed_statuses = {"scheduled", "in_progress"}
        found = find_match_in_data(data, result["team_a"], result["team_b"], allowed_statuses=allowed_statuses)
        if found is None:
            continue
        if status == "in_progress":
            live_updates.append(result)
        else:
            final_results.append(result)

    if not live_updates and not final_results:
        return {
            "results_found": len(new_results),
            "results_applied": 0,
            "live_updates_applied": 0,
            "ratings_updated": False,
            "prediction_entries_added": 0,
        }

    live_applied = 0
    if live_updates:
        data, live_applied = apply_live_updates(live_updates, data, current_ratings)

    unique_entries = []
    num_applied = 0
    md_label = ""
    updated_ratings = dict(current_ratings)
    if final_results:
        # Step 1: Log predictions BEFORE applying final results.
        new_log_entries = log_predictions_for_matches(final_results, data, current_ratings)

        existing_keys = {
            (e["team_a"], e["team_b"], e.get("date", ""))
            for e in prediction_log
        }
        unique_entries = [
            e for e in new_log_entries
            if (e["team_a"], e["team_b"], e.get("date", "")) not in existing_keys
        ]
        prediction_log.extend(unique_entries)

        # Step 2: Apply final results and update Elo.
        ratings_copy = dict(current_ratings)
        data, updated_ratings, num_applied = apply_results(final_results, data, ratings_copy)

        # Step 3: Save updated ratings to history.
        md_label = determine_matchday_label(data)
        ratings_history[md_label] = dict(updated_ratings)

    # Step 4: Save everything
    save_tournament_data(data)
    if num_applied > 0:
        save_ratings_history(ratings_history)
    if unique_entries:
        save_prediction_log(prediction_log)

    return {
        "results_found": len(new_results),
        "results_applied": num_applied,
        "live_updates_applied": live_applied,
        "ratings_updated": num_applied > 0,
        "prediction_entries_added": len(unique_entries),
        "matchday_label": md_label,
    }


def get_rating_changes(ratings_history: dict) -> list[dict]:
    """Compute rating changes between the initial and latest ratings.

    Returns a list sorted by biggest positive change first.
    """
    if len(ratings_history) < 2:
        return []

    initial = ratings_history.get("initial", {})
    latest = get_latest_ratings(ratings_history)
    if latest is None or not initial:
        return []

    changes = []
    all_teams = set(initial.keys()) | set(latest.keys())
    for team in all_teams:
        old = initial.get(team, 0)
        new = latest.get(team, old)
        diff = new - old
        if abs(diff) > 0.01:
            changes.append({
                "team": team,
                "initial": old,
                "current": new,
                "change": round(diff, 2),
            })

    changes.sort(key=lambda x: x["change"], reverse=True)
    return changes
