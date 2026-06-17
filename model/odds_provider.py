"""
Market odds provider for World Cup 2026.

Fetches real bookmaker odds from The Odds API and caches them locally
to minimize API usage (free tier: 500 requests/month).
"""

import json
import logging
import os
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ODDS_CACHE_FILE = DATA_DIR / "market_odds_cache.json"
TOURNAMENT_FILE = DATA_DIR / "world_cup_2026.json"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
# The Odds API sport keys for FIFA World Cup
SPORT_KEYS = ["soccer_fifa_world_cup", "soccer_fifa_world_cup_winner"]

# Team name mapping from bookmaker names to our internal names
BOOKMAKER_TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia",
    "Bosnia Herzegovina": "Bosnia",
    "Czechia": "Czech Republic",
    "Czech Rep": "Czech Republic",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Dem. Rep. Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "New Zealand FC": "New Zealand",
    "Aotearoa New Zealand": "New Zealand",
}


def _odds_refresh_minutes() -> int:
    """Return the minimum age before hitting The Odds API again."""
    raw = os.getenv("ODDS_REFRESH_MINUTES", "360").strip()
    try:
        value = int(raw)
    except ValueError:
        return 360
    return max(15, value)


def _get_api_key() -> Optional[str]:
    """Get The Odds API key from environment."""
    key = os.getenv("ODDS_API_KEY", "").strip()
    return key if key else None


def _normalize_team(name: str) -> str:
    """Normalize bookmaker team name to our internal format."""
    name = name.strip()
    return BOOKMAKER_TEAM_ALIASES.get(name, name)


def _load_cache() -> dict:
    """Load cached odds from disk."""
    if ODDS_CACHE_FILE.exists():
        try:
            with open(ODDS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"matches": {}, "outright": {}, "last_fetch": None}


def _cache_is_fresh(cache: dict) -> bool:
    """Return True when the odds cache is still inside its refresh window."""
    last_fetch = cache.get("last_fetch")
    if not last_fetch:
        return False

    try:
        fetched_at = datetime.fromisoformat(str(last_fetch))
    except ValueError:
        return False

    age = datetime.utcnow() - fetched_at
    return age < timedelta(minutes=_odds_refresh_minutes())


def _has_near_term_matches(window_days: int = 1) -> bool:
    """Return True when there are upcoming/live matches in the next window."""
    try:
        with open(TOURNAMENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return True

    target_days = {
        (date.today() + timedelta(days=offset)).isoformat()
        for offset in range(window_days + 1)
    }
    for group in data.get("groups", {}).values():
        for md_key in ("matchday1", "matchday2", "matchday3"):
            for match in group.get("matches", {}).get(md_key, []):
                if match.get("date") not in target_days:
                    continue
                if match.get("status") in {"scheduled", "in_progress"}:
                    return True
    return False


def _save_cache(cache: dict) -> None:
    """Persist odds cache to disk."""
    with open(ODDS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _fetch_odds_api(sport_key: str, markets: str = "h2h",
                    regions: str = "eu,uk,us") -> Optional[list]:
    """Call The Odds API for a specific sport and market."""
    api_key = _get_api_key()
    if not api_key:
        return None

    url = (
        f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        f"?apiKey={api_key}&markets={markets}&regions={regions}"
        f"&oddsFormat=decimal"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info("Odds API: %s requests remaining", remaining)
            return json.load(resp)
    except Exception as e:
        logger.warning("Failed to fetch odds from The Odds API: %s", e)
        return None


def _compute_consensus_odds(bookmakers: list) -> Optional[dict]:
    """Average h2h odds across bookmakers into consensus probabilities."""
    all_home = []
    all_draw = []
    all_away = []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if len(outcomes) == 3:
                names = list(outcomes.keys())
                # Find Draw
                draw_odds = outcomes.get("Draw")
                if draw_odds is None:
                    continue
                home_name = [n for n in names if n != "Draw"][0]
                away_name = [n for n in names if n != "Draw" and n != home_name][0]
                all_home.append(outcomes[home_name])
                all_draw.append(draw_odds)
                all_away.append(outcomes[away_name])

    if not all_home:
        return None

    avg_home = sum(all_home) / len(all_home)
    avg_draw = sum(all_draw) / len(all_draw)
    avg_away = sum(all_away) / len(all_away)

    # Convert decimal odds to probabilities (with overround removal)
    raw_prob_home = 1.0 / avg_home
    raw_prob_draw = 1.0 / avg_draw
    raw_prob_away = 1.0 / avg_away
    total = raw_prob_home + raw_prob_draw + raw_prob_away

    return {
        "prob_home": round(raw_prob_home / total, 4),
        "prob_draw": round(raw_prob_draw / total, 4),
        "prob_away": round(raw_prob_away / total, 4),
        "odds_home": round(avg_home, 2),
        "odds_draw": round(avg_draw, 2),
        "odds_away": round(avg_away, 2),
        "bookmaker_count": len(all_home),
    }


def fetch_market_odds() -> dict:
    """Fetch and cache market odds for World Cup matches.

    Returns dict with:
        matches: {(team_a, team_b): {prob_home, prob_draw, prob_away, ...}}
        outright: {team: probability} (if available)
        last_fetch: ISO timestamp
    """
    cache = _load_cache()
    api_key = _get_api_key()

    if not api_key:
        logger.info("No ODDS_API_KEY configured — using cached odds only.")
        return cache

    if _cache_is_fresh(cache):
        logger.info(
            "Odds cache still fresh (< %d min) — skipping API refresh.",
            _odds_refresh_minutes(),
        )
        return cache

    if not _has_near_term_matches(window_days=1):
        logger.info("No near-term matches — skipping odds refresh.")
        return cache

    # Try to fetch match odds
    events = _fetch_odds_api("soccer_fifa_world_cup", markets="h2h")
    if events:
        for event in events:
            home = _normalize_team(event.get("home_team", ""))
            away = _normalize_team(event.get("away_team", ""))
            if not home or not away:
                continue

            consensus = _compute_consensus_odds(event.get("bookmakers", []))
            if consensus:
                # Store with both orderings for easy lookup
                key = f"{home}|{away}"
                consensus["home_team"] = home
                consensus["away_team"] = away
                consensus["commence_time"] = event.get("commence_time", "")
                cache["matches"][key] = consensus

        cache["last_fetch"] = datetime.utcnow().isoformat(timespec="seconds")
        _save_cache(cache)
        logger.info("Updated market odds cache: %d matches", len(cache["matches"]))

    return cache


def get_market_odds_for_match(team_a: str, team_b: str,
                              cache: Optional[dict] = None) -> Optional[dict]:
    """Look up cached market odds for a specific match."""
    if cache is None:
        cache = _load_cache()

    matches = cache.get("matches", {})
    # Try both orderings
    key1 = f"{team_a}|{team_b}"
    key2 = f"{team_b}|{team_a}"

    entry = matches.get(key1) or matches.get(key2)
    if not entry:
        return None

    # Normalize to team_a perspective
    if entry.get("home_team") == team_a:
        return {
            "market_prob_a": entry["prob_home"],
            "market_prob_draw": entry["prob_draw"],
            "market_prob_b": entry["prob_away"],
            "market_odds_a": entry["odds_home"],
            "market_odds_draw": entry["odds_draw"],
            "market_odds_b": entry["odds_away"],
            "bookmaker_count": entry.get("bookmaker_count", 0),
        }
    else:
        return {
            "market_prob_a": entry["prob_away"],
            "market_prob_draw": entry["prob_draw"],
            "market_prob_b": entry["prob_home"],
            "market_odds_a": entry["odds_away"],
            "market_odds_draw": entry["odds_draw"],
            "market_odds_b": entry["odds_home"],
            "bookmaker_count": entry.get("bookmaker_count", 0),
        }
