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
from datetime import date, datetime, timedelta
from pathlib import Path
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
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Panamá": "Panama",
}

# All 48 teams in our dataset for validation
VALID_TEAMS = {
    "Argentina", "France", "Brazil", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Colombia", "Uruguay", "Japan", "USA", "Mexico",
    "Sweden", "Morocco", "South Korea", "Switzerland", "Australia", "DR Congo",
    "Iran", "Ecuador", "Turkey", "Croatia", "Senegal", "Ghana", "Tunisia",
    "Scotland", "Egypt", "Ivory Coast", "Canada", "Saudi Arabia", "Norway",
    "Algeria", "Austria", "Paraguay", "Czech Republic", "Iraq", "New Zealand",
    "Qatar", "Panama", "Haiti", "Bosnia", "Jordan", "Cape Verde",
    "Uzbekistan", "Curaçao", "South Africa",
}

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "world_cup_2026.json"


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


def _fetch_wiki_wikitext(page_title: str, timeout: int = 15) -> Optional[str]:
    """Fetch parsed MediaWiki wikitext for a specific page title."""
    params = urlencode({
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "format": "json",
    })
    payload = _fetch_json_url(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "WorldCup2026Predictor/1.0 (educational project)"},
        timeout=timeout,
    )
    if not payload:
        return None

    try:
        return str(payload["parse"]["wikitext"]["*"])
    except (KeyError, TypeError):
        logger.warning("Wikipedia wikitext payload missing for page %s", page_title)
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
    if status == "1H":
        # TheSportsDB sometimes only exposes the current phase, not the exact
        # minute. Use a first-half midpoint instead of misreading `1H` as 1'.
        return 23
    if status == "HT":
        return 45
    if status == "2H":
        # Same idea for second-half phase-only updates.
        return 68
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


def _resolve_event_date(event: dict) -> str:
    """Resolve the local calendar date for a provider event.

    TheSportsDB often gives a UTC `dateEvent` plus separate UTC/local kickoff
    times. We normalize to the venue-local day so the stored schedule stays
    aligned with the tournament dataset.
    """
    event_date = str(event.get("dateEvent") or "").strip()
    if not event_date:
        return str(event.get("dateEventLocal") or "").strip()

    utc_time = str(event.get("strTime") or "").strip()
    local_time = str(event.get("strTimeLocal") or "").strip()
    if not utc_time or not local_time:
        return str(event.get("dateEventLocal") or event_date)

    try:
        resolved_date = datetime.strptime(event_date, "%Y-%m-%d").date()
        utc_clock = datetime.strptime(utc_time, "%H:%M:%S").time()
        local_clock = datetime.strptime(local_time, "%H:%M:%S").time()
    except ValueError:
        return str(event.get("dateEventLocal") or event_date)

    utc_minutes = utc_clock.hour * 60 + utc_clock.minute
    local_minutes = local_clock.hour * 60 + local_clock.minute
    delta_minutes = local_minutes - utc_minutes

    if delta_minutes > 12 * 60:
        resolved_date -= timedelta(days=1)
    elif delta_minutes < -12 * 60:
        resolved_date += timedelta(days=1)

    return resolved_date.isoformat()


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
        "date": _resolve_event_date(event),
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


def _parse_results_from_group_wikitext(wikitext: str) -> list[dict]:
    """Parse completed match results from structured World Cup group wikitext.

    Group subpages expose each match as a `football box` block with explicit
    date/score fields, which is more reliable than scraping the main page HTML.
    """
    results = []
    match_pattern = re.compile(
        r"===(?P<team_a>[^=]+?)\s+vs\s+(?P<team_b>[^=]+?)===.*?"
        r"<section begin=.*?/>\{\{#invoke:football box\|main(?P<body>.*?)\}\}<section end=.*?/>",
        re.DOTALL,
    )
    score_pattern = re.compile(r"\|score=\{\{score link\|[^|]*\|([^}]+)\}\}")
    date_pattern = re.compile(r"\|date=\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")
    attendance_pattern = re.compile(r"\|attendance=([0-9][0-9,]*)")

    for match in match_pattern.finditer(wikitext):
        team_a = _normalize_team_name(match.group("team_a").strip())
        team_b = _normalize_team_name(match.group("team_b").strip())
        if team_a not in VALID_TEAMS or team_b not in VALID_TEAMS:
            continue

        body = match.group("body")
        score_match = score_pattern.search(body)
        date_match = date_pattern.search(body)
        if not score_match or not date_match:
            continue

        score_value = score_match.group(1).strip()
        played_match = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", score_value)
        if not played_match:
            continue

        match_date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        attendance_match = attendance_pattern.search(body)

        # Same-day Wikipedia edits can briefly show a live score. Requiring an
        # attendance figure for current-day matches reduces false FT positives
        # while still letting older finished matches pass through.
        if match_date == date.today().isoformat() and attendance_match is None:
            continue

        results.append({
            "date": match_date,
            "team_a": team_a,
            "team_b": team_b,
            "score_a": int(played_match.group(1)),
            "score_b": int(played_match.group(2)),
            "status": "played",
        })

    return results


def _fetch_group_page_results() -> list[dict]:
    """Fetch completed results from the dedicated World Cup group pages."""
    page_titles = []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        recent_days = _thesportsdb_reference_dates(window_days=1)[:2]
        recent_groups = set()
        for group_name, group_data in data.get("groups", {}).items():
            for matchday_key in ("matchday1", "matchday2", "matchday3"):
                for match in group_data.get("matches", {}).get(matchday_key, []):
                    if match.get("date") in recent_days:
                        recent_groups.add(group_name)
                        break
                if group_name in recent_groups:
                    break

        page_titles = [
            f"2026 FIFA World Cup Group {group_name}"
            for group_name in sorted(recent_groups)
        ]
    except (OSError, json.JSONDecodeError, TypeError):
        page_titles = []

    if not page_titles:
        page_titles = [f"2026 FIFA World Cup Group {group_name}" for group_name in "ABCDEFGHIJKL"]

    results = []
    for page_title in page_titles:
        wikitext = _fetch_wiki_wikitext(page_title)
        if not wikitext:
            continue
        results.extend(_parse_results_from_group_wikitext(wikitext))

    unique = {}
    for result in results:
        key = tuple(sorted([result["team_a"], result["team_b"]]))
        unique[key] = result
    return list(unique.values())


def fetch_latest_results() -> list[dict]:
    """Fetch completed World Cup 2026 match results from Wikipedia.

    Returns list of dicts with keys:
        team_a, team_b, score_a, score_b

    Returns empty list on failure (graceful degradation).
    """
    results = _fetch_group_page_results()
    if results:
        logger.info("Fetched %d results from Wikipedia group pages.", len(results))
        return results

    html = _fetch_wiki_html()
    if html is None:
        logger.info("No Wikipedia data available — using existing data only.")
        return []

    results = _parse_results_from_html(html)
    logger.info("Fetched %d results from Wikipedia main page.", len(results))
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
