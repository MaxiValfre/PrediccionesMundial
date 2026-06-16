#!/usr/bin/env python3
"""
World Cup 2026 Prediction Runner
Loads data, runs the model, generates predictions JSON, and prints today's picks.

Usage:
    python run_predictions.py              # Predict only (no data changes)
    python run_predictions.py --update     # Fetch results, update data, then predict
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Fix Windows console encoding for emoji flags
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.predictor import generate_predictions


def run_update_cycle() -> dict:
    """Execute the full update cycle: fetch results, update data, update ratings.

    Returns summary dict with update statistics.
    """
    from model.fetcher import (
        fetch_latest_results,
        fetch_match_updates_from_thesportsdb,
        fetch_results_from_file,
    )
    from model.updater import (
        RESULTS_UPDATE_FILE,
        get_rating_changes,
        load_prediction_log,
        load_ratings_history,
        run_update,
    )

    print("\n--- Auto-Update Cycle ---\n")

    # Step 1: Try fetching live/final updates from TheSportsDB
    print("Fetching live/final updates from TheSportsDB...")
    sportsdb_updates = fetch_match_updates_from_thesportsdb()
    if sportsdb_updates:
        live_count = sum(1 for r in sportsdb_updates if r.get("status") == "in_progress")
        final_count = len(sportsdb_updates) - live_count
        print(f"  Found {len(sportsdb_updates)} updates from TheSportsDB ({live_count} live, {final_count} final).")
    else:
        print("  No updates from TheSportsDB.")

    # Step 2: Try fetching finished results from Wikipedia
    print("Fetching latest results from Wikipedia...")
    web_results = fetch_latest_results()
    if web_results:
        print(f"  Found {len(web_results)} results from Wikipedia.")
    else:
        print("  No results from Wikipedia (offline or page not parseable).")

    # Step 3: Check local file
    print(f"Checking local results file: {RESULTS_UPDATE_FILE}")
    file_results = fetch_results_from_file(str(RESULTS_UPDATE_FILE))
    if file_results:
        print(f"  Found {len(file_results)} results from local file.")
    else:
        print("  No local results file or empty.")

    def update_priority(entry: dict) -> int:
        status = str(entry.get("status", "played")).strip().lower()
        if status in {"live", "in_progress", "in-progress", "inprogress"}:
            return 1
        return 2

    # Merge updates by team pair.
    # Final results outrank live snapshots so a stale local live file cannot
    # override a completed Wikipedia result.
    all_results = {}
    for r in sportsdb_updates:
        key = tuple(sorted([r["team_a"], r["team_b"]]))
        all_results[key] = r
    for r in web_results:
        key = tuple(sorted([r["team_a"], r["team_b"]]))
        existing = all_results.get(key)
        if existing is None or update_priority(r) >= update_priority(existing):
            all_results[key] = r
    for r in file_results:
        key = tuple(sorted([r["team_a"], r["team_b"]]))
        existing = all_results.get(key)
        if existing is None or update_priority(r) >= update_priority(existing):
            all_results[key] = r

    merged = list(all_results.values())
    print(f"\n  Total unique results to process: {len(merged)}")

    if not merged:
        print("  No new results found. Continuing with existing data.\n")
        return {"results_applied": 0}

    # Step 3: Run the update pipeline
    summary = run_update(merged)

    print(f"\n  Results applied: {summary['results_applied']}")
    print(f"  Live updates applied: {summary.get('live_updates_applied', 0)}")
    print(f"  Prediction log entries added: {summary['prediction_entries_added']}")
    if summary.get("ratings_updated"):
        print(f"  Ratings updated (saved as '{summary.get('matchday_label', 'unknown')}')")

        # Show top rating changes
        history = load_ratings_history()
        changes = get_rating_changes(history)
        if changes:
            print("\n  Top Rating Changes:")
            risers = [c for c in changes if c["change"] > 0][:5]
            fallers = [c for c in changes if c["change"] < 0][-5:]
            for c in risers:
                print(f"    {c['team']:<20s} {c['initial']:.0f} -> {c['current']:.0f}  (+{c['change']:.1f})")
            if fallers:
                for c in reversed(fallers):
                    print(f"    {c['team']:<20s} {c['initial']:.0f} -> {c['current']:.0f}  ({c['change']:.1f})")

    print("\n--- Update Complete ---\n")
    return summary


def print_accuracy_report():
    """Load prediction log and print accuracy metrics."""
    from model.calibrator import compute_accuracy, format_accuracy_report
    from model.updater import load_prediction_log

    log = load_prediction_log()
    if not log:
        return

    metrics = compute_accuracy(log)
    if metrics["total_predictions"] == 0:
        return

    print(f"\n{'='*60}")
    print("  MODEL ACCURACY REPORT")
    print(f"{'='*60}")
    print(format_accuracy_report(metrics))


def main():
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Prediction Engine"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch latest results and update data/ratings before predicting.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  FIFA World Cup 2026 — Prediction Engine")
    print("=" * 60)
    print()

    # Step 1: Update cycle (optional)
    update_summary = None
    if args.update:
        update_summary = run_update_cycle()

    # Step 2: Run predictions
    output = generate_predictions()

    # Step 3: Add accuracy metrics to output if prediction log exists
    try:
        from model.calibrator import compute_accuracy
        from model.updater import get_rating_changes, load_prediction_log, load_ratings_history

        log = load_prediction_log()
        if log:
            output["accuracy_metrics"] = compute_accuracy(log)
            output["prediction_log_summary"] = {
                "total_entries": len(log),
                "latest_date": max(e.get("date", "") for e in log) if log else "",
            }

        history = load_ratings_history()
        if history:
            changes = get_rating_changes(history)
            if changes:
                output["rating_updates"] = changes
    except Exception as e:
        logging.getLogger(__name__).warning("Could not load accuracy data: %s", e)

    # Step 4: Save predictions JSON for dashboard
    dashboard_dir = Path(__file__).resolve().parent / "dashboard"
    out_path = dashboard_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to: {out_path}")

    # Embed predictions directly into index.html (fixes CORS on file://)
    html_path = dashboard_dir / "index.html"
    html = html_path.read_text(encoding="utf-8")
    json_inline = json.dumps(output, ensure_ascii=False)
    # Reset any previous injection so re-runs work
    import re
    html = re.sub(
        r'var EMBEDDED_PREDICTIONS = .+?;',
        f'var EMBEDDED_PREDICTIONS = {json_inline};',
        html,
        count=1,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"Predictions embedded into: {html_path}")

    # Print today's matches
    print(f"\n{'='*60}")
    print(f"  TODAY'S PREDICTIONS — {output['today']}")
    print(f"{'='*60}")

    if not output["todays_matches"]:
        print("  No matches scheduled for today.")
    else:
        for m in output["todays_matches"]:
            flag_a = output["flags"].get(m["team_a"], "")
            flag_b = output["flags"].get(m["team_b"], "")
            print(f"\n  {flag_a} {m['team_a']} vs {m['team_b']} {flag_b}")
            print(f"  Group {m['group']}")
            print(f"  Predicted Score: {m['predicted_score_a']}-{m['predicted_score_b']}")
            w = m['prob_win_a'] * 100
            d = m['prob_draw'] * 100
            l = m['prob_win_b'] * 100
            print(f"  {m['team_a']} Win: {w:.1f}% | Draw: {d:.1f}% | {m['team_b']} Win: {l:.1f}%")
            print(f"  Expected Goals: {m['xg_a']} vs {m['xg_b']}")

    # Top 10 tournament winners
    print(f"\n{'='*60}")
    print("  TOP 10 — Tournament Win Probability")
    print(f"{'='*60}")
    for i, team in enumerate(output["power_rankings"][:10]):
        flag = output["flags"].get(team["team"], "")
        pct = team["winner"] * 100
        bar = "█" * int(pct * 2)
        print(f"  {i+1:2d}. {flag} {team['team']:<16s} {pct:5.1f}% {bar}")

    # Stats
    print(f"\n{'='*60}")
    print("  TOURNAMENT STATS")
    print(f"{'='*60}")
    s = output["stats"]
    print(f"  Matches played:      {s['matches_played']}")
    print(f"  Total goals:         {s['total_goals']}")
    print(f"  Avg goals/match:     {s['avg_goals_per_match']}")
    print(f"  Upsets:              {s['upsets']}")
    print(f"  MC Simulations:      {s['simulations']:,}")

    # Accuracy report (if update was run or log exists)
    if args.update:
        print_accuracy_report()

    print()


if __name__ == "__main__":
    main()
