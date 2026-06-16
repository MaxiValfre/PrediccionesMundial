"""
Automatic result fetcher for World Cup 2026.

Scrapes completed match results from Wikipedia and provides
a fallback for manual JSON file updates, including live snapshots.
"""

import json
import logging
import os
import re
import urllib.request
from datetime import date, timedelta
from urllib.parse import urlencode
from typing import Optional

logger = logging.getLogger(__name__)

# Wikipedia URL for the 2026 FIFA World Cup
WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
THESPORTSDB_V1_BASE = "https://www.thesportsdb.com/api/v1/json"
THESPORTSDB_V2_BASE = "https://www.thesportsdb.com/api/v2/json"
THESPORTSDB_FREE_KEY = "123"
THESPORTSDB_WORLD_CUP_LEAGUE_ID = "4429"

# Known team name aliases used on Wikipedia vs. our data
TEAM_ALIASES: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Côte d’Ivoire": "Ivory Coast",
    "Bosnia and Herzegovina": "Bosnia",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Aotearoa New Zealand": "New Zealand",
    "Bosnia & Herzegovina": "Bosnia",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
}

# All 48 teams in our dataset for validation
VALID_TEAMS = {
    "Argentina", "France", "Brazil", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Colombia", "Uruguay", "Japan", "USA", "Mexico",
    "Sweden", "Morocco", "South Korea", "Switzerland", "Australia", "Denmark",
    "Iran", "Ecuador", "Turkey", "Serbia", "Senegal", "Nigeria", "Tunisia",
    "Scotland", "Egypt", "Ivory Coast", "Canada", "Saudi Arabia", "Norway",
    "Algeria", "Austria", "Paraguay", "Czech Republic", "Iraq", "New Zealand",
    "Qatar", "Cameroon", "Haiti", "Bosnia", "Jordan", "Cape Verde",
    "Uzbekistan", "Curaçao", "South Africa",
}


def _normalize_team_name(name: str) -> str:
    """Normalize a team name from Wikipedia to our internal format."""
    name = name.strip()
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    return name


def _fetch_wiki_html(url: str = WIKI_URL, timeout: int = 15) -> Optional[str]:
    """Fetch the raw HTML from Wikipedia. Returns None on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "WorldCup2026Predictor/1.0 (educational project)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to fetch Wikipedia page: %s", e)
        return None


def _fetch_json_url(url: str, headers: Optional[dict[str, str]] = None,
                    timeout: int = 15) -> Optional[dict]:
    """Fetch a JSON payload and decode it safely."""
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception as e:
        logger.warning("Failed to fetch JSON from %s: %s", url, e)
        return None


def _thesportsdb_api_key() -> str:
    """Return the configured TheSportsDB API key or the free key."""
    return os.getenv("THESPORTSDB_API_KEY", THESPORTSDB_FREE_KEY).strip() or THESPORTSDB_FREE_KEY


def _thesportsdb_league_id() -> str:
    """Return the configured league id for the World Cup."""
    return os.getenv("THESPORTSDB_LEAGUE_ID", THESPORTSDB_WORLD_CUP_LEAGUE_ID).strip() or THESPORTSDB_WORLD_CUP_LEAGUE_ID


def _thesportsdb_reference_dates(window_days: int = 1) -> list[str]:
    """Fetch a compact date window to absorb timezone differences."""
    override = os.getenv("THESPORTSDB_REFERENCE_DATE", "").strip()
    if override:
        try:
            base = date.fromisoformat(override)
        except ValueError:
            logger.warning("Invalid THESPORTSDB_REFERENCE_DATE=%s", override)
            base = date.today()
    else:
        base = date.today()
    return [
        (base + timedelta(days=offset)).isoformat()
        for offset in range(-window_days, window_days + 1)
    ]


def _parse_live_minute(status: str) -> Optional[int]:
    """Extract a representative minute from a provider status string."""
    status = status.strip().upper()
    if not status:
        return None
    if status == "HT":
        return 45
    if status in {"ET", "AET"}:
        return 105
    if status in {"PEN", "FT", "NS", "POSTP", "CANC", "ABD"}:
        return None

    match = re.search(r"(\d{1,3})", status)
    if match:
        minute = int(match.group(1))
        if "+" in status:
            extra = re.search(r"\+(\d{1,2})", status)
            if extra:
                minute += int(extra.group(1))
        return minute
    return None


def _event_to_update(event: dict) -> Optional[dict]:
    """Map a TheSportsDB event payload to our updater format."""
    team_a = _normalize_team_name(event.get("strHomeTeam", ""))
    team_b = _normalize_team_name(event.get("strAwayTeam", ""))
    if team_a not in VALID_TEAMS or team_b not in VALID_TEAMS:
        return None

    status = str(event.get("strStatus", "")).strip().upper()
    score_a = event.get("intHomeScore")
    score_b = event.get("intAwayScore")
    if score_a is None or score_b is None:
        return None

    try:
        score_a = int(score_a)
        score_b = int(score_b)
    except (TypeError, ValueError):
        return None

    base = {
        "date": event.get("dateEvent") or event.get("dateEventLocal") or "",
        "group": event.get("strGroup", ""),
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
    }

    if status in {"FT", "AET", "PEN"}:
        base["status"] = "played"
        return base

    if status in {"NS", "TBD", "POSTP", "CANC"}:
        return None

    minute = _parse_live_minute(status)
    if minute is None:
        return None

    base["status"] = "in_progress"
    base["minute"] = minute
    return base


def _fetch_thesportsdb_schedule_day(day: str, league_id: str) -> list[dict]:
    """Fetch a single day of World Cup events from TheSportsDB free API."""
    params = urlencode({"d": day, "l": league_id})
    url = f"{THESPORTSDB_V1_BASE}/{THESPORTSDB_FREE_KEY}/eventsday.php?{params}"
    payload = _fetch_json_url(url)
    if not payload:
        return []
    return payload.get("events") or []


def _fetch_thesportsdb_premium_livescore(league_id: str) -> list[dict]:
    """Fetch live scores from TheSportsDB v2 if a premium key is configured."""
    api_key = _thesportsdb_api_key()
    if api_key == THESPORTSDB_FREE_KEY:
        return []

    url = f"{THESPORTSDB_V2_BASE}/livescore/{league_id}"
    payload = _fetch_json_url(url, headers={"X-API-KEY": api_key})
    if not payload:
        return []

    return payload.get("events") or payload.get("event") or []


def fetch_match_updates_from_thesportsdb() -> list[dict]:
    """Fetch live and finished World Cup updates from TheSportsDB.

    Strategy:
    - Premium key configured: try v2 livescore first for fresher live state.
    - Always fall back to free v1 day schedule around today's date, which gives
      us NS / live / FT statuses for the World Cup league.
    """
    league_id = _thesportsdb_league_id()
    raw_events = _fetch_thesportsdb_premium_livescore(league_id)
    if not raw_events:
        for day in _thesportsdb_reference_dates(window_days=1):
            raw_events.extend(_fetch_thesportsdb_schedule_day(day, league_id))

    updates = []
    seen = set()
    for event in raw_events:
        mapped = _event_to_update(event)
        if not mapped:
            continue
        key = (mapped["team_a"], mapped["team_b"], mapped.get("date", ""), mapped["status"])
        if key in seen:
            continue
        seen.add(key)
        updates.append(mapped)

    logger.info("Fetched %d live/final updates from TheSportsDB.", len(updates))
    return updates


def _parse_results_from_html(html: str) -> list[dict]:
    """Parse match results from Wikipedia HTML.

    Looks for patterns in group stage match tables.
    Wikipedia formats scores as: Team A  v  Team B (scheduled)
    or Team A  3–1  Team B (played).
    The exact HTML structure varies, so we use multiple regex strategies.
    """
    results = []

    # Strategy 1: Look for match summary rows in wikitables
    # Pattern: team_a ... score ... team_b with date context
    # Wikipedia typically has score in format "X–Y" (en dash) or "X-Y"
    #
    # Match row pattern in group stage tables:
    # <td>...team_a...</td>...<td>X–Y</td>...<td>...team_b...</td>
    score_pattern = re.compile(
        r'(?:title="([^"]+)"[^>]*>[^<]*</a>\s*</td>'
        r'[^<]*<td[^>]*>\s*(\d+)\s*[–\-]\s*(\d+)\s*</td>'
        r'[^<]*<td[^>]*>[^<]*<a[^>]*title="([^"]+)")',
        re.DOTALL,
    )

    for match in score_pattern.finditer(html):
        team_a_raw = match.group(1)
        score_a = int(match.group(2))
        score_b = int(match.group(3))
        team_b_raw = match.group(4)

        # Normalize names — Wikipedia links to "X national football team"
        team_a_raw = re.sub(
            r"\s*national\s+(association\s+)?football\s+team\s*$", "", team_a_raw, flags=re.IGNORECASE
        )
        team_b_raw = re.sub(
            r"\s*national\s+(association\s+)?football\s+team\s*$", "", team_b_raw, flags=re.IGNORECASE
        )

        team_a = _normalize_team_name(team_a_raw)
        team_b = _normalize_team_name(team_b_raw)

        if team_a in VALID_TEAMS and team_b in VALID_TEAMS:
            results.append({
                "team_a": team_a,
                "team_b": team_b,
                "score_a": score_a,
                "score_b": score_b,
            })

    # Strategy 2: Broader pattern for infobox/match-report style
    # "Team A 3–1 Team B" plain text patterns
    text_pattern = re.compile(
        r'title="([^"]*?(?:national[^"]*?team|football[^"]*?))"[^>]*>'
        r'([^<]+)</a>\s*'
        r'(\d+)\s*[–\-]\s*(\d+)\s*'
        r'<a[^>]*title="([^"]*?(?:national[^"]*?team|football[^"]*?))"[^>]*>'
        r'([^<]+)</a>',
        re.DOTALL,
    )

    seen = set()
    for match in text_pattern.finditer(html):
        team_a_title = match.group(1)
        score_a = int(match.group(3))
        score_b = int(match.group(4))
        team_b_title = match.group(5)

        team_a_raw = re.sub(
            r"\s*national\s+(association\s+)?football\s+team\s*$", "", team_a_title, flags=re.IGNORECASE
        )
        team_b_raw = re.sub(
            r"\s*national\s+(association\s+)?football\s+team\s*$", "", team_b_title, flags=re.IGNORECASE
        )

        team_a = _normalize_team_name(team_a_raw)
        team_b = _normalize_team_name(team_b_raw)

        if team_a in VALID_TEAMS and team_b in VALID_TEAMS:
            key = (team_a, team_b)
            if key not in seen:
                seen.add(key)
                results.append({
                    "team_a": team_a,
                    "team_b": team_b,
                    "score_a": score_a,
                    "score_b": score_b,
                })

    # Deduplicate: keep first occurrence for each team pair
    unique = {}
    for r in results:
        key = tuple(sorted([r["team_a"], r["team_b"]]))
        if key not in unique:
            unique[key] = r

    return list(unique.values())


def fetch_latest_results() -> list[dict]:
    """Fetch completed World Cup 2026 match results from Wikipedia.

    Returns list of dicts with keys:
        team_a, team_b, score_a, score_b

    Returns empty list on failure (graceful degradation).
    """
    html = _fetch_wiki_html()
    if html is None:
        logger.info("No Wikipedia data available — using existing data only.")
        return []

    results = _parse_results_from_html(html)
    logger.info("Fetched %d results from Wikipedia.", len(results))
    return results


def fetch_results_from_file(filepath: str) -> list[dict]:
    """Load results from a local JSON file for manual updates.

    Expected format:
    {
        "last_updated": "2026-06-16",
        "results": [
            {"date": "...", "group": "...", "team_a": "...", "team_b": "...",
             "score_a": N, "score_b": N, "status": "played"},
            {"date": "...", "group": "...", "team_a": "...", "team_b": "...",
             "score_a": N, "score_b": N, "status": "in_progress", "minute": 63}
        ]
    }

    Returns empty list if file doesn't exist or is invalid.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Validate structure
        valid = []
        for r in results:
            status = str(r.get("status", "played")).strip().lower()
            if status in {"live", "in_progress", "in-progress", "inprogress"}:
                status = "in_progress"
            else:
                status = "played"

            if all(k in r for k in ("team_a", "team_b", "score_a", "score_b")):
                if (r["score_a"] is not None and r["score_b"] is not None
                        and isinstance(r["score_a"], int)
                        and isinstance(r["score_b"], int)):
                    if status == "in_progress":
                        minute = r.get("minute")
                        if minute is None or not isinstance(minute, int):
                            logger.warning("Skipping live update without integer minute: %s", r)
                            continue
                        valid.append({**r, "status": status, "minute": minute})
                    else:
                        valid.append({**r, "status": status})
                else:
                    logger.warning("Skipping result with invalid scores: %s", r)
            else:
                logger.warning("Skipping malformed result entry: %s", r)
        logger.info("Loaded %d results from %s", len(valid), filepath)
        return valid
    except FileNotFoundError:
        logger.info("Manual results file not found: %s", filepath)
        return []
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse results file %s: %s", filepath, e)
        return []
