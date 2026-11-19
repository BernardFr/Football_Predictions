"""
ESPN's public (undocumented) soccer feed: schedules, results and lineups. No key needed.

    scoreboard: {BASE}/{slug}/scoreboard?dates=YYYYMMDD   -> events of that day (+ season calendar)
    summary:    {BASE}/{slug}/summary?event=ID            -> rosters with starters, subs, player stats

Everything is cached under data/espn/. Finished matches are never fetched twice.
The feed is unofficial and can change; parsing is deliberately defensive.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
DEFAULT_UA = "football-predictions/1.0 (+personal research; Python requests)"
DEFAULT_DELAY_SECONDS = 0.4          # pause before each live request, to be polite to the feed
DEFAULT_TIMEOUT_SECONDS = 30
HTTP_FORBIDDEN = 403
CACHE_SUBDIR = "espn"
DAY_FORMAT = "%Y%m%d"
SEASON_START_MMDD = "07-01"          # seasons run 1 July .. 30 June
SEASON_END_MMDD = "06-30"
CALENDAR_REFRESH_DAYS = 3            # re-fetch the season calendar when older than this, in season
RECENT_REFRESH_DAYS = 3              # re-fetch scoreboards this recent: results/lineups may still change
SYNC_LOOKAHEAD_DAYS = 7              # sync scoreboards up to this far ahead of today
UPCOMING_GRACE_HOURS = 2             # keep "pre" games that were due to start up to this long ago
SCOREBOARD_LOG_EVERY = 25
SUMMARY_LOG_EVERY = 50
STATE_PRE = "pre"                    # ESPN event states
STATE_POST = "post"
STATUS_TEXT_KEYS = ("name", "description", "detail", "shortDetail")
EXTRA_TIME_MARKERS = ("aet", "extra", "pen")
MATCH_MINUTES = 90
STARTING_XI = 11
DEFAULT_SUB_MINUTE = 60              # substitution flagged but no minute given
GOALKEEPER_CODES = ("G", "GK")
GOALKEEPER = "G"
DEFAULT_POSITION = "M"


class EspnError(RuntimeError):
    pass


class EspnClient:
    def __init__(self, cache_dir: Path, user_agent: str = DEFAULT_UA, delay: float = DEFAULT_DELAY_SECONDS,
                 timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 now: pd.Timestamp | None = None):
        self.cache = Path(cache_dir) / CACHE_SUBDIR
        self.now = now  # UTC, naive; None = the real clock (set for replays)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.delay, self.timeout = delay, timeout
        self.requests_made = 0

    def _now(self) -> pd.Timestamp:
        return self.now if self.now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _get(self, url: str, path: Path, refresh: bool) -> dict:
        if path.exists() and not refresh:
            return json.loads(path.read_text())
        time.sleep(self.delay)
        response = self.session.get(url, timeout=self.timeout)
        self.requests_made += 1
        if response.status_code == HTTP_FORBIDDEN:
            raise EspnError(f"ESPN refused the request (403) for {url}. Try another user_agent in src/football_predictions.toml [lineups].")
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text)
        return response.json()

    def scoreboard(self, slug: str, day: str, refresh: bool = False) -> dict:
        return self._get(f"{BASE}/{slug}/scoreboard?dates={day}", self.cache / slug / "scoreboard" / f"{day}.json", refresh)

    def summary(self, slug: str, event_id: str, refresh: bool = False) -> dict:
        return self._get(f"{BASE}/{slug}/summary?event={event_id}", self.cache / slug / "summary" / f"{event_id}.json", refresh)

    # ------------------------------------------------------------------ season sync
    def _calendar(self, slug: str, season: str) -> list[pd.Timestamp]:
        """
        Match days ESPN attaches to any scoreboard of the season. Re-fetched every few days during
        the season, because postponed games get new dates.
        """
        start = int(season[:4])
        day = f"{start}{SEASON_START_MMDD.replace('-', '')}"   # same trick soccerdata uses to get the calendar
        path = self.cache / slug / "scoreboard" / f"{day}.json"
        now = self._now()
        in_season = pd.Timestamp(f"{start}-{SEASON_START_MMDD}") <= now < pd.Timestamp(f"{start + 1}-{SEASON_START_MMDD}")
        stale = path.exists() and (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")) > pd.Timedelta(days=CALENDAR_REFRESH_DAYS)
        data = self.scoreboard(slug, day, refresh=in_season and stale and self.now is None)
        cal = []
        for league_entry in data.get("leagues", []):
            for entry in league_entry.get("calendar", []):
                value = entry if isinstance(entry, str) else (entry.get("startDate") or entry.get("value") if isinstance(entry, dict) else None)
                if value:
                    try:
                        cal.append(pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo is None
                                   else pd.Timestamp(value).tz_convert("UTC").tz_localize(None))
                    except ValueError:
                        continue
        lo, hi = pd.Timestamp(f"{start}-{SEASON_START_MMDD}"), pd.Timestamp(f"{start + 1}-{SEASON_END_MMDD}")
        return [match_day for match_day in cal if lo <= match_day <= hi]

    def season_days(self, slug: str, season: str, every_day_fallback: bool = True) -> list[str]:
        """Match days of a season (YYYYMMDD), from ESPN's calendar."""
        start = int(season[:4])
        today = self._now().normalize()
        lo, hi = pd.Timestamp(f"{start}-{SEASON_START_MMDD}"), pd.Timestamp(f"{start + 1}-{SEASON_END_MMDD}")
        days = sorted({match_day.strftime(DAY_FORMAT) for match_day in self._calendar(slug, season)})
        if not days and every_day_fallback:   # no calendar: fall back to every day of the season up to today
            days = [match_day.strftime(DAY_FORMAT) for match_day in pd.date_range(lo, min(hi, today))]
        return days

    def upcoming(self, slug: str, season: str, n_days: int) -> pd.DataFrame:
        """Scheduled (not yet started) games from today through today + days. A handful of requests."""
        today = self._now().normalize()
        end = today + pd.Timedelta(days=n_days)
        cal = self._calendar(slug, season)
        if cal:          # only the match days in the window; none means no games (e.g. international break)
            window = sorted({match_day.strftime(DAY_FORMAT) for match_day in cal if today <= match_day.normalize() <= end})
        else:            # no calendar at all: ask for each day in the window
            window = [match_day.strftime(DAY_FORMAT) for match_day in pd.date_range(today, end)]
        events = []
        for day in window:
            events.extend(parse_scoreboard(self.scoreboard(slug, day, refresh=self.now is None)))
        ev = pd.DataFrame(events)
        if ev.empty:
            return ev
        ev = ev.drop_duplicates("event_id")
        ev = ev[(ev.state == STATE_PRE) & (ev.date >= self._now() - pd.Timedelta(hours=UPCOMING_GRACE_HOURS))].copy()
        ev["season"] = season
        return ev.sort_values("date").reset_index(drop=True)

    def sync_season(self, slug: str, season: str, log=print, summaries: bool = True,
                    every_day_fallback: bool = True) -> pd.DataFrame:
        """Fetch (or read from cache) every event of the season, plus summaries of finished events."""
        today = self._now().normalize()
        events = []
        day_lst = self.season_days(slug, season, every_day_fallback)
        for i, day in enumerate(day_lst):
            day_ts = pd.Timestamp(day)
            if day_ts > today + pd.Timedelta(days=SYNC_LOOKAHEAD_DAYS):
                continue
            recent = day_ts >= today - pd.Timedelta(days=RECENT_REFRESH_DAYS)       # results/lineups may still be changing
            sb = self.scoreboard(slug, day, refresh=recent)
            events.extend(parse_scoreboard(sb))
            if log and i % SCOREBOARD_LOG_EVERY == 0:
                log(f"  {slug} {season}: scoreboard {i + 1}/{len(day_lst)} ({self.requests_made} requests so far)")
        ev = pd.DataFrame(events).drop_duplicates("event_id") if events else pd.DataFrame()
        if ev.empty:
            return ev
        start = int(season[:4])
        ev = ev[(ev.date >= pd.Timestamp(f"{start}-{SEASON_START_MMDD}"))
                & (ev.date < pd.Timestamp(f"{start + 1}-{SEASON_START_MMDD}"))]
        ev["season"] = season
        if not summaries:
            return ev.reset_index(drop=True)
        done = ev[ev.state == STATE_POST]
        for j, eid in enumerate(done.event_id):
            self.summary(slug, eid)
            if log and j % SUMMARY_LOG_EVERY == 0 and self.requests_made:
                log(f"  {slug} {season}: summaries {j + 1}/{len(done)} ({self.requests_made} requests so far)")
        return ev.reset_index(drop=True)


# ---------------------------------------------------------------------- parsing
def _ts(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_convert("UTC").tz_localize(None) if timestamp.tzinfo else timestamp


def parse_scoreboard(data: dict) -> list[dict]:
    out = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        sides = {competitor.get("homeAway"): competitor for competitor in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        st = (comp.get("status") or event.get("status") or {}).get("type", {})
        state = st.get("state") or (STATE_POST if st.get("completed") else STATE_PRE)

        def score(competitor):
            try:
                return int(competitor.get("score"))
            except (TypeError, ValueError):
                return None

        def first_two_periods(competitor):
            vals = [period.get("value") for period in (competitor.get("linescores") or []) if isinstance(period, dict)]
            if len(vals) < 2:
                return None
            first, second = vals[0], vals[1]
            if first is None or second is None:
                return None
            try:
                return int(first) + int(second)
            except (TypeError, ValueError):
                return None

        hs, as_ = score(sides["home"]), score(sides["away"])
        text = " ".join(str(st.get(key, "")) for key in STATUS_TEXT_KEYS).lower()
        extra = state == STATE_POST and any(key in text for key in EXTRA_TIME_MARKERS)
        h90, a90 = hs, as_
        if extra and hs is not None and as_ is not None:
            # level after 90 by definition; use the half scores if ESPN gives them, else the lower total
            h90, a90 = first_two_periods(sides["home"]), first_two_periods(sides["away"])
            if h90 is None or a90 is None or h90 != a90:
                h90 = a90 = min(hs, as_)
        out.append({
            "event_id": str(event["id"]), "date": _ts(event.get("date") or comp.get("date")), "state": state,
            "home_id": str(sides["home"]["team"]["id"]), "away_id": str(sides["away"]["team"]["id"]),
            "home_name": sides["home"]["team"].get("displayName") or sides["home"]["team"].get("name"),
            "away_name": sides["away"]["team"].get("displayName") or sides["away"]["team"].get("name"),
            "home_score": hs if state == STATE_POST else None,
            "away_score": as_ if state == STATE_POST else None,
            "home_90": h90 if state == STATE_POST else None,
            "away_90": a90 if state == STATE_POST else None,
            "extra_time": bool(extra),
            "neutral": bool(comp.get("neutralSite", False)),
        })
    return out


_POS = [("G", "G"), ("GK", "G"), ("CF", "F"), ("ST", "F"), ("F", "F"), ("LW", "F"), ("RW", "F"),
        ("W", "F"), ("AM", "M"), ("DM", "M"), ("M", "M"), ("CD", "D"), ("CB", "D"), ("LB", "D"),
        ("RB", "D"), ("WB", "D"), ("D", "D"), ("SW", "D")]


def position_group(abbr: str | None) -> str:
    code = (abbr or "").upper()
    if code in GOALKEEPER_CODES:
        return GOALKEEPER
    for prefix, grp in _POS:
        if code.startswith(prefix) or code.endswith(prefix):
            return grp
    return DEFAULT_POSITION


def _clock(value) -> int | None:
    text = value.get("displayValue") if isinstance(value, dict) else value
    nums = [int(digits) for digits in re.findall(r"\d{1,3}", str(text or ""))]
    return sum(nums) if nums else None


def _sub_minute(player: dict, key: str) -> int | None:
    value = player.get(key)
    if isinstance(value, dict):
        return _clock(value.get("clock")) if value.get("didSub") else None
    if value:   # boolean form: the minute lives in plays[]
        subs = [pl for pl in player.get("plays", []) if pl.get("substitution")]
        if subs:
            idx = 0 if key == "subbedIn" or len(subs) == 1 else 1
            return _clock(subs[idx].get("clock"))
        return DEFAULT_SUB_MINUTE
    return None


def _stat(player: dict, *names) -> float:
    stats = {entry.get("name"): entry.get("value") for entry in player.get("stats", []) if isinstance(entry, dict)}
    for stat_name in names:
        value = stats.get(stat_name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def parse_lineups(summary: dict, event_id: str) -> pd.DataFrame:
    """One row per player in the match-day squad: starter flag, minutes, goals + assists."""
    rows = []
    for team_roster in summary.get("rosters", []):
        team = team_roster.get("team", {})
        for player in team_roster.get("roster", []) or []:
            ath = player.get("athlete", {})
            starter = bool(player.get("starter"))
            sub_in = _sub_minute(player, "subbedIn")
            sub_out = _sub_minute(player, "subbedOut")
            if starter:
                start, end = 0, (sub_out if sub_out is not None else MATCH_MINUTES)
            elif sub_in is not None:
                start, end = sub_in, (sub_out if sub_out is not None else MATCH_MINUTES)
            else:
                start = end = 0
            rows.append({
                "event_id": str(event_id), "team_id": str(team.get("id")),
                "athlete_id": str(ath.get("id") or ath.get("displayName")),
                "player": ath.get("displayName") or ath.get("fullName"),
                "pos": position_group((player.get("position") or {}).get("abbreviation")),
                "starter": starter, "minutes": max(min(end, MATCH_MINUTES) - min(start, MATCH_MINUTES), 0),
                "ga": _stat(player, "totalGoals") + _stat(player, "goalAssists"),
            })
    return pd.DataFrame(rows)


def has_lineups(lineups: pd.DataFrame) -> bool:
    if lineups.empty:
        return False
    counts = lineups[lineups.starter].groupby("team_id").size()
    return len(counts) == 2 and bool((counts >= STARTING_XI).all())
