"""
Model calibration and accuracy tracking for World Cup 2026 predictions.

Provides Brier score calculation, calibration analysis, and accuracy summaries
to measure how well the Dixon-Coles model performs against actual results.
"""

import math
from collections import defaultdict


def brier_score(
    prob_win_a: float,
    prob_draw: float,
    prob_win_b: float,
    actual_result: str,
) -> float:
    """Compute the Brier score for a single match prediction.

    Lower is better: 0 = perfect, ~0.667 = worst (confident and wrong).

    Args:
        prob_win_a: Predicted probability of team A winning.
        prob_draw: Predicted probability of a draw.
        prob_win_b: Predicted probability of team B winning.
        actual_result: One of 'home_win', 'draw', 'away_win'.

    Returns:
        Brier score (float). Lower = better calibration.
    """
    actual = [0.0, 0.0, 0.0]
    if actual_result == "home_win":
        actual[0] = 1.0
    elif actual_result == "draw":
        actual[1] = 1.0
    elif actual_result == "away_win":
        actual[2] = 1.0
    else:
        raise ValueError(f"Unknown result: {actual_result!r}")

    predicted = [prob_win_a, prob_draw, prob_win_b]
    return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / 3.0


def classify_result(
    predicted_score_a: int,
    predicted_score_b: int,
    actual_score_a: int,
    actual_score_b: int,
) -> str:
    """Classify a prediction result.

    Returns:
        'exact': Predicted the exact score.
        'correct_direction': Predicted the right winner/draw.
        'wrong': Predicted wrong outcome.
    """
    if predicted_score_a == actual_score_a and predicted_score_b == actual_score_b:
        return "exact"

    # Determine predicted and actual outcomes
    if predicted_score_a > predicted_score_b:
        pred_outcome = "home_win"
    elif predicted_score_a < predicted_score_b:
        pred_outcome = "away_win"
    else:
        pred_outcome = "draw"

    if actual_score_a > actual_score_b:
        actual_outcome = "home_win"
    elif actual_score_a < actual_score_b:
        actual_outcome = "away_win"
    else:
        actual_outcome = "draw"

    if pred_outcome == actual_outcome:
        return "correct_direction"
    return "wrong"


def actual_result_category(score_a: int, score_b: int) -> str:
    """Convert a score to a result category string."""
    if score_a > score_b:
        return "home_win"
    elif score_a < score_b:
        return "away_win"
    return "draw"


def compute_calibration(prediction_log: list[dict]) -> dict:
    """Analyze calibration across probability buckets.

    For binned probability ranges, check what fraction of predictions
    in that confidence bucket actually occurred.

    Returns dict with:
        - buckets: list of {range, predicted_avg, actual_freq, count}
        - calibration_error: mean absolute calibration error
    """
    if not prediction_log:
        return {"buckets": [], "calibration_error": 0.0}

    # Collect (predicted_prob, actual_occurred) for the predicted-winner outcome
    observations: list[tuple[float, int]] = []
    for entry in prediction_log:
        prob_a = entry.get("prob_win_a", 0)
        prob_d = entry.get("prob_draw", 0)
        prob_b = entry.get("prob_win_b", 0)

        actual_a = entry.get("actual_score_a")
        actual_b = entry.get("actual_score_b")
        if actual_a is None or actual_b is None:
            continue

        actual = actual_result_category(actual_a, actual_b)

        # Record the highest predicted probability and whether it hit
        probs = {"home_win": prob_a, "draw": prob_d, "away_win": prob_b}
        predicted_outcome = max(probs, key=probs.get)
        predicted_prob = probs[predicted_outcome]
        hit = 1 if predicted_outcome == actual else 0
        observations.append((predicted_prob, hit))

    if not observations:
        return {"buckets": [], "calibration_error": 0.0}

    # Bin into decile-ish buckets
    bucket_edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    buckets = []
    total_ce = 0.0
    bucket_count = 0

    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        in_bucket = [(p, h) for p, h in observations if lo <= p < hi]
        if not in_bucket:
            continue
        avg_pred = sum(p for p, _ in in_bucket) / len(in_bucket)
        actual_freq = sum(h for _, h in in_bucket) / len(in_bucket)
        ce = abs(avg_pred - actual_freq)
        total_ce += ce
        bucket_count += 1
        buckets.append({
            "range": f"{lo:.0%}-{hi:.0%}",
            "predicted_avg": round(avg_pred, 4),
            "actual_freq": round(actual_freq, 4),
            "count": len(in_bucket),
            "calibration_error": round(ce, 4),
        })

    mean_ce = total_ce / bucket_count if bucket_count > 0 else 0.0

    return {
        "buckets": buckets,
        "calibration_error": round(mean_ce, 4),
    }


def compute_accuracy(prediction_log: list[dict]) -> dict:
    """Compute overall accuracy metrics from the prediction log.

    Args:
        prediction_log: List of prediction entries with actual results.

    Returns:
        Dict with accuracy summary metrics.
    """
    if not prediction_log:
        return {
            "total_predictions": 0,
            "exact_scores": 0,
            "correct_direction": 0,
            "wrong_direction": 0,
            "accuracy_pct": 0.0,
            "avg_brier_score": 0.0,
            "brier_rating": "N/A",
            "calibration": {"buckets": [], "calibration_error": 0.0},
        }

    # Filter to entries that have actual results
    evaluated = [
        e for e in prediction_log
        if e.get("actual_score_a") is not None and e.get("actual_score_b") is not None
    ]

    if not evaluated:
        return {
            "total_predictions": 0,
            "exact_scores": 0,
            "correct_direction": 0,
            "wrong_direction": 0,
            "accuracy_pct": 0.0,
            "avg_brier_score": 0.0,
            "brier_rating": "N/A",
            "calibration": {"buckets": [], "calibration_error": 0.0},
        }

    exact = 0
    correct = 0
    wrong = 0
    brier_total = 0.0

    for entry in evaluated:
        cat = entry.get("result_category")
        if cat is None:
            cat = classify_result(
                entry.get("predicted_score_a", 0),
                entry.get("predicted_score_b", 0),
                entry["actual_score_a"],
                entry["actual_score_b"],
            )

        if cat == "exact":
            exact += 1
        elif cat == "correct_direction":
            correct += 1
        else:
            wrong += 1

        bs = entry.get("brier_score")
        if bs is None:
            actual = actual_result_category(
                entry["actual_score_a"], entry["actual_score_b"]
            )
            bs = brier_score(
                entry.get("prob_win_a", 0.33),
                entry.get("prob_draw", 0.33),
                entry.get("prob_win_b", 0.33),
                actual,
            )
        brier_total += bs

    n = len(evaluated)
    avg_brier = brier_total / n

    # Rate the Brier score
    if avg_brier < 0.10:
        rating = "Excellent"
    elif avg_brier < 0.18:
        rating = "Good"
    elif avg_brier < 0.25:
        rating = "Fair"
    else:
        rating = "Poor"

    accuracy_pct = round((exact + correct) / n * 100, 1)

    calibration = compute_calibration(evaluated)

    return {
        "total_predictions": n,
        "exact_scores": exact,
        "correct_direction": correct,
        "wrong_direction": wrong,
        "accuracy_pct": accuracy_pct,
        "avg_brier_score": round(avg_brier, 4),
        "brier_rating": rating,
        "calibration": calibration,
    }


def format_accuracy_report(metrics: dict) -> str:
    """Format accuracy metrics as a human-readable report string."""
    if metrics["total_predictions"] == 0:
        return "  No predictions with actual results to evaluate yet.\n"

    lines = [
        f"  Predictions evaluated: {metrics['total_predictions']}",
        f"  Exact scores:          {metrics['exact_scores']}",
        f"  Correct direction:     {metrics['correct_direction']}",
        f"  Wrong direction:       {metrics['wrong_direction']}",
        f"  Accuracy:              {metrics['accuracy_pct']}%",
        f"  Avg Brier Score:       {metrics['avg_brier_score']:.4f} ({metrics['brier_rating']})",
    ]

    cal = metrics.get("calibration", {})
    if cal.get("buckets"):
        lines.append(f"  Calibration Error:     {cal['calibration_error']:.4f}")
        lines.append("")
        lines.append("  Calibration Buckets:")
        lines.append(f"  {'Range':<12} {'Predicted':>10} {'Actual':>10} {'Count':>6}")
        for b in cal["buckets"]:
            lines.append(
                f"  {b['range']:<12} {b['predicted_avg']:>10.1%} "
                f"{b['actual_freq']:>10.1%} {b['count']:>6}"
            )

    return "\n".join(lines) + "\n"
