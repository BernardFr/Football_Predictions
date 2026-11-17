"""ESPN's public (undocumented) soccer feed: schedules, results and lineups. No key needed.

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


class EspnError(RuntimeError):
    pass


class EspnClient:
    def __init__(self, cache_dir: Path, user_agent: str = DEFAULT_UA, delay: float = 0.4, timeout: int = 30,
                 now: pd.Timestamp | None = None):
        self.cache = Path(cache_dir) / "espn"
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
        r = self.session.get(url, timeout=self.timeout)
        self.requests_made += 1
        if r.status_code == 403:
            raise EspnError(f"ESPN refused the request (403) for {url}. Try another user_agent in src/football_predictions.toml [lineups].")
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return r.json()

    def scoreboard(self, slug: str, day: str, refresh: bool = False) -> dict:
        return self._get(f"{BASE}/{slug}/scoreboard?dates={day}", self.cache / slug / "scoreboard" / f"{day}.json", refresh)

    def summary(self, slug: str, event_id: str, refresh: bool = False) -> dict:
        return self._get(f"{BASE}/{slug}/summary?event={event_id}", self.cache / slug / "summary" / f"{event_id}.json", refresh)

    # ------------------------------------------------------------------ season sync
    def _calendar(self, slug: str, season: str) -> list[pd.Timestamp]:
        """Match days ESPN attaches to any scoreboard of the season. Re-fetched every few days during
        the season, because postponed games get new dates."""
        start = int(season[:4])
        day = f"{start}0701"                         # same trick soccerdata uses to get the calendar
        path = self.cache / slug / "scoreboard" / f"{day}.json"
        now = self._now()
        in_season = pd.Timestamp(f"{start}-07-01") <= now < pd.Timestamp(f"{start + 1}-07-01")
        stale = path.exists() and (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")) > pd.Timedelta(days=3)
        data = self.scoreboard(slug, day, refresh=in_season and stale and self.now is None)
        cal = []
        for lg in data.get("leagues", []):
            for c in lg.get("calendar", []):
                v = c if isinstance(c, str) else (c.get("startDate") or c.get("value") if isinstance(c, dict) else None)
                if v:
                    try:
                        cal.append(pd.Timestamp(v).tz_localize(None) if pd.Timestamp(v).tzinfo is None
                                   else pd.Timestamp(v).tz_convert("UTC").tz_localize(None))
                    except ValueError:
                        continue
        lo, hi = pd.Timestamp(f"{start}-07-01"), pd.Timestamp(f"{start + 1}-06-30")
        return [d for d in cal if lo <= d <= hi]

    def season_days(self, slug: str, season: str, every_day_fallback: bool = True) -> list[str]:
        """Match days of a season (YYYYMMDD), from ESPN's calendar."""
        start = int(season[:4])
        today = self._now().normalize()
        lo, hi = pd.Timestamp(f"{start}-07-01"), pd.Timestamp(f"{start + 1}-06-30")
        days = sorted({d.strftime("%Y%m%d") for d in self._calendar(slug, season)})
        if not days and every_day_fallback:   # no calendar: fall back to every day of the season up to today
            days = [d.strftime("%Y%m%d") for d in pd.date_range(lo, min(hi, today))]
        return days

    def upcoming(self, slug: str, season: str, days: int) -> pd.DataFrame:
        """Scheduled (not yet started) games from today through today + days. A handful of requests."""
        today = self._now().normalize()
        end = today + pd.Timedelta(days=days)
        cal = self._calendar(slug, season)
        if cal:          # only the match days in the window; none means no games (e.g. international break)
            window = sorted({d.strftime("%Y%m%d") for d in cal if today <= d.normalize() <= end})
        else:            # no calendar at all: ask for each day in the window
            window = [d.strftime("%Y%m%d") for d in pd.date_range(today, end)]
        events = []
        for day in window:
            events.extend(parse_scoreboard(self.scoreboard(slug, day, refresh=self.now is None)))
        ev = pd.DataFrame(events)
        if ev.empty:
            return ev
        ev = ev.drop_duplicates("event_id")
        ev = ev[(ev.state == "pre") & (ev.date >= self._now() - pd.Timedelta(hours=2))].copy()
        ev["season"] = season
        return ev.sort_values("date").reset_index(drop=True)

    def sync_season(self, slug: str, season: str, log=print, summaries: bool = True,
                    every_day_fallback: bool = True) -> pd.DataFrame:
        """Fetch (or read from cache) every event of the season, plus summaries of finished events."""
        today = self._now().normalize()
        events = []
        days = self.season_days(slug, season, every_day_fallback)
        for i, day in enumerate(days):
            d = pd.Timestamp(day)
            if d > today + pd.Timedelta(days=7):
                continue
            recent = d >= today - pd.Timedelta(days=3)       # results/lineups may still be changing
            sb = self.scoreboard(slug, day, refresh=recent)
            events.extend(parse_scoreboard(sb))
            if log and i % 25 == 0:
                log(f"  {slug} {season}: scoreboard {i + 1}/{len(days)} ({self.requests_made} requests so far)")
        ev = pd.DataFrame(events).drop_duplicates("event_id") if events else pd.DataFrame()
        if ev.empty:
            return ev
        ev = ev[(ev.date >= pd.Timestamp(f"{season[:4]}-07-01")) & (ev.date < pd.Timestamp(f"{int(season[:4]) + 1}-07-01"))]
        ev["season"] = season
        if not summaries:
            return ev.reset_index(drop=True)
        done = ev[ev.state == "post"]
        for j, eid in enumerate(done.event_id):
            self.summary(slug, eid)
            if log and j % 50 == 0 and self.requests_made:
                log(f"  {slug} {season}: summaries {j + 1}/{len(done)} ({self.requests_made} requests so far)")
        return ev.reset_index(drop=True)


# ---------------------------------------------------------------------- parsing
def _ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def parse_scoreboard(data: dict) -> list[dict]:
    out = []
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in sides or "away" not in sides:
            continue
        st = (comp.get("status") or e.get("status") or {}).get("type", {})
        state = st.get("state") or ("post" if st.get("completed") else "pre")

        def score(c):
            try:
                return int(c.get("score"))
            except (TypeError, ValueError):
                return None

        def first_two_periods(c):
            vals = [x.get("value") for x in (c.get("linescores") or []) if isinstance(x, dict)]
            try:
                return int(vals[0]) + int(vals[1]) if len(vals) >= 2 else None
            except (TypeError, ValueError):
                return None

        hs, as_ = score(sides["home"]), score(sides["away"])
        text = " ".join(str(st.get(k, "")) for k in ("name", "description", "detail", "shortDetail")).lower()
        extra = state == "post" and any(k in text for k in ("aet", "extra", "pen"))
        h90, a90 = hs, as_
        if extra and hs is not None and as_ is not None:
            # level after 90 by definition; use the half scores if ESPN gives them, else the lower total
            h90, a90 = first_two_periods(sides["home"]), first_two_periods(sides["away"])
            if h90 is None or a90 is None or h90 != a90:
                h90 = a90 = min(hs, as_)
        out.append({
            "event_id": str(e["id"]), "date": _ts(e.get("date") or comp.get("date")), "state": state,
            "home_id": str(sides["home"]["team"]["id"]), "away_id": str(sides["away"]["team"]["id"]),
            "home_name": sides["home"]["team"].get("displayName") or sides["home"]["team"].get("name"),
            "away_name": sides["away"]["team"].get("displayName") or sides["away"]["team"].get("name"),
            "home_score": hs if state == "post" else None,
            "away_score": as_ if state == "post" else None,
            "home_90": h90 if state == "post" else None,
            "away_90": a90 if state == "post" else None,
            "extra_time": bool(extra),
            "neutral": bool(comp.get("neutralSite", False)),
        })
    return out


_POS = [("G", "G"), ("GK", "G"), ("CF", "F"), ("ST", "F"), ("F", "F"), ("LW", "F"), ("RW", "F"),
        ("W", "F"), ("AM", "M"), ("DM", "M"), ("M", "M"), ("CD", "D"), ("CB", "D"), ("LB", "D"),
        ("RB", "D"), ("WB", "D"), ("D", "D"), ("SW", "D")]


def position_group(abbr: str | None) -> str:
    a = (abbr or "").upper()
    if a in ("G", "GK"):
        return "G"
    for prefix, grp in _POS:
        if a.startswith(prefix) or a.endswith(prefix):
            return grp
    return "M"


def _clock(v) -> int | None:
    s = v.get("displayValue") if isinstance(v, dict) else v
    nums = [int(x) for x in re.findall(r"\d{1,3}", str(s or ""))]
    return sum(nums) if nums else None


def _sub_minute(p: dict, key: str) -> int | None:
    v = p.get(key)
    if isinstance(v, dict):
        return _clock(v.get("clock")) if v.get("didSub") else None
    if v:   # boolean form: the minute lives in plays[]
        subs = [pl for pl in p.get("plays", []) if pl.get("substitution")]
        if subs:
            idx = 0 if key == "subbedIn" or len(subs) == 1 else 1
            return _clock(subs[idx].get("clock"))
        return 60   # flagged but no minute available
    return None


def _stat(p: dict, *names) -> float:
    stats = {s.get("name"): s.get("value") for s in p.get("stats", []) if isinstance(s, dict)}
    for n in names:
        if stats.get(n) is not None:
            try:
                return float(stats[n])
            except (TypeError, ValueError):
                pass
    return 0.0


def parse_lineups(summary: dict, event_id: str) -> pd.DataFrame:
    """One row per player in the match-day squad: starter flag, minutes, goals + assists."""
    rows = []
    for r in summary.get("rosters", []):
        team = r.get("team", {})
        for p in r.get("roster", []) or []:
            ath = p.get("athlete", {})
            starter = bool(p.get("starter"))
            sub_in = _sub_minute(p, "subbedIn")
            sub_out = _sub_minute(p, "subbedOut")
            if starter:
                start, end = 0, (sub_out if sub_out is not None else 90)
            elif sub_in is not None:
                start, end = sub_in, (sub_out if sub_out is not None else 90)
            else:
                start = end = 0
            rows.append({
                "event_id": str(event_id), "team_id": str(team.get("id")),
                "athlete_id": str(ath.get("id") or ath.get("displayName")),
                "player": ath.get("displayName") or ath.get("fullName"),
                "pos": position_group((p.get("position") or {}).get("abbreviation")),
                "starter": starter, "minutes": max(min(end, 90) - min(start, 90), 0),
                "ga": _stat(p, "totalGoals") + _stat(p, "goalAssists"),
            })
    return pd.DataFrame(rows)


def has_lineups(lineups: pd.DataFrame) -> bool:
    if lineups.empty:
        return False
    counts = lineups[lineups.starter].groupby("team_id").size()
    return len(counts) == 2 and (counts >= 11).all()
