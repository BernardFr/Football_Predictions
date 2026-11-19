"""
Domestic cup results (e.g. FA Cup, EFL Cup, Community Shield) from ESPN, as extra fitting data.

Cup games add information the league can't: a Premier League side against a Championship side is a
direct measurement of the gap between the divisions. They are also noisier (rotated line-ups, extra
time, neutral venues), so they are handled carefully:

  * Scored at 90 minutes. Games that went to extra time or penalties count as draws at the 90-minute
    score (from ESPN's half scores when available, otherwise the lower final total).
  * Neutral venues (Wembley semi-finals and finals, the Community Shield) get no home advantage.
  * Only games between teams the model already rates (the configured divisions) are used. A League
    Two or non-league side with one cup tie would be a team with almost no data.
  * They form their own pseudo-division "CUP", with its own scoring level and home advantage, and a
    weight (`cup_weight`) that `tune` picks from the backtest. Weight 0 switches cups off.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import teammap
from .espn import STATE_POST
from .sources import COLUMNS, ODDS_COLS

CUP_DIV = "CUP"
EXTRA_COLUMNS = ["neutral", "competition"]   # cup rows carry these on top of the standard COLUMNS
ERROR_SNIPPET_CHARS = 100                    # length of an error message quoted in a warning


def load(client, slug_lst: list[str], season_lst: list[str], known_teams, team_map_path: Path | None = None,
         log=print) -> pd.DataFrame:
    log = log or (lambda *_: None)
    tmap = {}
    if team_map_path is not None and Path(team_map_path).exists():
        map_df = pd.read_csv(team_map_path, dtype=str)
        tmap = dict(zip(map_df.espn_id, map_df.fd_name))
    known = sorted(set(known_teams))
    cache: dict[tuple[str, str], str | None] = {}

    def fd_name(espn_id, name):
        if espn_id in tmap and tmap[espn_id] in known:
            return tmap[espn_id]
        key = (espn_id, name)
        if key not in cache:
            cache[key] = teammap.match_name(name, known)
        return cache[key]

    frames = []
    for slug in slug_lst:
        for season in season_lst:
            try:
                ev = client.sync_season(slug, season, log=None, summaries=False, every_day_fallback=False)
            except Exception as exc:  # noqa: BLE001 - cups are optional extra data
                log(f"  warning: could not load {slug} {season} from ESPN ({type(exc).__name__}: {str(exc)[:ERROR_SNIPPET_CHARS]}); "
                    f"continuing without it")
                continue
            if ev.empty:
                continue
            ev = ev[(ev.state == STATE_POST) & ev.home_90.notna() & ev.away_90.notna()]
            if ev.empty:
                continue
            home = [fd_name(team_id, team_name) for team_id, team_name in zip(ev.home_id, ev.home_name)]
            away = [fd_name(team_id, team_name) for team_id, team_name in zip(ev.away_id, ev.away_name)]
            df = pd.DataFrame({
                "date": ev.date.dt.normalize().to_numpy(), "season": season, "div": CUP_DIV,
                "home": home, "away": away, "hg": ev.home_90.to_numpy(), "ag": ev.away_90.to_numpy(),
                **{col: np.nan for col in ODDS_COLS},
                "neutral": ev.neutral.to_numpy(bool), "competition": slug,
            }).dropna(subset=["home", "away"])
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS + EXTRA_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["hg"], out["ag"] = out.hg.astype(int), out.ag.astype(int)
    return out.drop_duplicates(["date", "home", "away"]).sort_values("date").reset_index(drop=True)
