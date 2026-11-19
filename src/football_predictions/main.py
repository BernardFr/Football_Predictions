#! /Users/bfraenkel/Documents/Code/Football_Predictions/.venv/bin/python3

"""
football-predictions <command> [--league EPL | --all] [--source openfootball] [--today YYYY-MM-DD]

Commands
  download   fetch results for the current and past seasons (cached; current season refreshed)
  tune       grid-search decay/ridge/feeder weight on the previous full season
  backtest   walk-forward backtest of a season (default: current season so far)
  predict    fit on everything played so far and predict upcoming fixtures (logged to the ledger)
  update     the feedback loop: refresh data -> score logged predictions -> backtest -> predict -> report
  report     rebuild the Excel report from saved files
  lineups-backtest   download ESPN lineups, test whether adjusting for the starting XI improves forecasts
  lineups            lineup-adjusted predictions for games in the next 36 hours whose XIs are announced
                     (run about an hour before kickoff; logged to the ledger as variant "lineup")
"""

import argparse
import json
import sys
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from custom_logger import custom_logger as logger
from custom_logger import setup_logger

from . import backtest as bt
from . import cups, ledger, lineups, report, teammap
from .config import DEFAULT_CONFIG_PATH, Config, read_toml_config, shift_season
from .csvio import read_csv, write_csv
from .espn import DEFAULT_DELAY_SECONDS, DEFAULT_UA, STATE_POST, STATE_PRE, EspnClient, has_lineups, parse_lineups
from .model import DixonColes, Hyper
from .report import DRAW_LABEL
from .sources import load_matches, make_source

PROG_NAME = "football-predictions"
COMMANDS = ["download", "tune", "backtest", "predict", "update", "report", "lineups-backtest", "lineups"]
DEFAULT_LEAGUE = "EPL"
REPLAY_HOUR_UTC = 10   # with --today, pretend the run happens at 10:00 UTC that day

# files under state/<league>/
HYPER_FILE = "hyper.json" # stores hyperparameters for the model
TUNING_FILE = "tuning.csv" # stores tuning results for the model
BACKTEST_FILE = "backtest_{season}.csv" # stores backtest results for a given season
ADJUSTMENTS_FILE = "adjustments.csv" # stores manual adjustments to model predictions
RATINGS_FILE = "ratings.csv" # stores team ratings
UPCOMING_FILE = "upcoming.csv" # stores upcoming fixtures
LEDGER_FILE = "ledger.csv" # stores logged predictions and results
TEAM_MAP_FILE = "team_map.csv" # stores mapping of team names to canonical names
TEAM_MAP_OVERRIDES_FILE = "team_map_overrides.csv" # stores manual overrides for team mapping
LINEUP_COEFS_FILE = "lineup_coefs.json" # stores lineup adjustment coefficients
LINEUP_BACKTEST_FILE = "lineup_backtest.csv" # stores lineup backtest results
LINEUP_COMPARISON_FILE = "lineup_comparison.csv" # stores comparison of base vs lineup predictions
LINEUP_UPCOMING_FILE = "lineup_upcoming.csv"
REPORT_FILE = "{league}_report.xlsx"   # under reports/
MIN_CSV_BYTES = 2                      # smaller files hold no data (an empty frame writes a bare newline)
JSON_INDENT = 2

NO_ADJUSTMENT = 1.0                    # multiplier when adjustments.csv leaves a column out
ESPN_SCHEDULE = "espn"                 # `schedule` value for games found only on ESPN
DEFAULT_LINEUP_SEASONS = 2
LINEUP_VARIANT = "lineup"
LINEUP_WINDOW_HOURS = 36               # `lineups` covers games kicking off within this window
LINEUP_GRACE_HOURS = 1                 # ... or that were due to start up to this long ago
LINEUP_LEDGER_DROP_PREFIXES = ("base_", "h_", "a_")   # diagnostic columns kept out of the ledger
PREDICT_DISPLAY_COLS = ["date", "home", "away", "pick", "confidence", "band", "p_home", "p_draw", "p_away",
                        "exp_gd", "gd_range_80", "likely_score"]
PERCENT_DISPLAY_COLS = ("confidence", "p_home", "p_draw", "p_away")
LINEUP_DISPLAY_COLS = ["home", "away", "base", "lineup", "pick_team", "confidence", "exp_gd",
                       "h_miss", "a_miss", "h_gk", "a_gk"]


class Run:
    def __init__(self, cfg, league, source_name):
        """Initialize a Run instance for a given league and data source."""
        self.cfg, self.league = cfg, league
        self.source = make_source(source_name, cfg.data_dir)
        self.divs = league.divs(self.source.name)
        self.top = self.divs[0]
        self.season = league.current_season
        self.state = cfg.state_dir / league.key
        self.state.mkdir(parents=True, exist_ok=True)

    def seasons(self, last: str) -> list[str]:
        return [shift_season(last, -k) for k in range(self.cfg.history_seasons, -1, -1)]

    def matches(self, last: str, refresh: bool = False) -> pd.DataFrame:
        seasons = self.seasons(last)
        match_df = load_matches(self.source, seasons, self.divs, refresh_season=self.season if refresh else None)
        match_df = match_df.assign(neutral=False)
        cup = self.cup_matches(seasons, set(match_df.home) | set(match_df.away))
        if not cup.empty:
            match_df = pd.concat([match_df, cup[match_df.columns]], ignore_index=True).sort_values(["date", "div", "home"]).reset_index(drop=True)
        return match_df[match_df.date < pd.Timestamp(self.cfg.today)]

    def cup_matches(self, seasons: list[str], known_teams) -> pd.DataFrame:
        """Cup games between rated teams, cached for this run (they are only fetched from ESPN once)."""
        if not self.league.cups or not self.league.espn:
            return pd.DataFrame()
        key = tuple(seasons)
        if not hasattr(self, "_cups"):
            self._cups = {}
        if key not in self._cups:
            self._cups[key] = cups.load(self._espn(), self.league.cups, seasons, known_teams,
                                        self.state / TEAM_MAP_FILE)
        return self._cups[key]

    def hyper(self) -> Hyper:
        params = Hyper.from_dict(self.cfg.model)
        path = self.state / HYPER_FILE
        if path.exists():
            params = Hyper.from_dict({**params.to_dict(), **json.loads(path.read_text())})
        return params

    # ------------------------------------------------------------- commands
    def download(self):
        match_df = load_matches(self.source, self.seasons(self.season) + [shift_season(self.season, -self.cfg.history_seasons - 1)],
                         self.divs, refresh_season=self.season)
        logger.info("matches per season and division:\n" + match_df.groupby(["season", "div"]).size().unstack(fill_value=0).to_string())
        cup = self.cup_matches(self.seasons(self.season), set(match_df.home) | set(match_df.away))
        if not cup.empty:
            logger.info("cup games between rated teams (from ESPN):\n"
                        + cup.groupby(["season", "competition"]).size().unstack(fill_value=0).to_string())

    def tune(self):
        season = shift_season(self.season, -1)
        logger.info(f"tuning on {season} (walk-forward, weekly refits)")
        best, table = bt.tune(self.matches(season), season, self.top, Hyper.from_dict(self.cfg.model), self.cfg.tune_grid)
        (self.state / HYPER_FILE).write_text(json.dumps(best.to_dict(), indent=JSON_INDENT))
        table.to_csv(self.state / TUNING_FILE, index=False)
        logger.info(f"best: {best.to_dict()} -> {self.state / HYPER_FILE}")

    def backtest(self, season: str | None = None, refresh: bool = False) -> pd.DataFrame:
        test_season = season or self.season
        df = bt.walk_forward(self.matches(test_season, refresh), test_season, self.top, self.hyper())
        if df.empty:
            logger.info(f"no {self.top} matches played in {test_season} yet")
            return df
        write_csv(df, self.state / BACKTEST_FILE.format(season=test_season))
        from .metrics import confidence_bands, summary
        conf = self.cfg.confidence
        logger.info(f"{self.league.name} {test_season}: walk-forward backtest on {len(df)} matches\n"
                    + summary(df).to_string(index=False, float_format=lambda value: f"{value:.3f}") + "\n"
                    + confidence_bands(df, conf["high"], conf["medium"]).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return df

    def _adjustments(self, fixture_df: pd.DataFrame) -> list[dict | None]:
        path = self.state / ADJUSTMENTS_FILE
        if not path.exists():
            return [None] * len(fixture_df)
        adj = pd.read_csv(path)
        if "until" in adj:
            adj["until"] = pd.to_datetime(adj["until"], errors="coerce")
        out = []
        for fixture in fixture_df.itertuples(index=False):
            adjust = {}
            for side, team in (("home", fixture.home), ("away", fixture.away)):
                rows = adj[adj.team == team]
                if "until" in adj:
                    rows = rows[rows.until.isna() | (rows.until >= fixture.date)]
                for entry in rows.itertuples(index=False):
                    adjust[f"{side}_attack"] = adjust.get(f"{side}_attack", NO_ADJUSTMENT) * float(getattr(entry, "attack", NO_ADJUSTMENT))
                    adjust[f"{side}_defence"] = adjust.get(f"{side}_defence", NO_ADJUSTMENT) * float(getattr(entry, "defence", NO_ADJUSTMENT))
            out.append(adjust or None)
        return out

    def predict(self, refresh: bool = False) -> pd.DataFrame:
        match_df = self.matches(self.season, refresh)
        model = DixonColes(self.hyper()).fit(match_df, pd.Timestamp(self.cfg.today), self.top)
        ratings = model.ratings()
        ratings.to_csv(self.state / RATINGS_FILE, index=False)
        fx = self.source.fixtures(self.season, self.top, self.cfg.today, self.cfg.upcoming_days)
        fx = fx.assign(schedule=self.source.name)
        fx = self._espn_fixtures(fx, model)
        if fx.empty:
            logger.info(f"no {self.league.name} games scheduled in the next {self.cfg.upcoming_days} days "
                        f"({self.source.name}{' or ESPN' if self.league.espn else ''}); nothing to predict "
                        f"(international break or end of season?)")
            pd.DataFrame().to_csv(self.state / UPCOMING_FILE, index=False)
            return fx
        rows = []
        for fixture, adj in zip(fx.itertuples(index=False), self._adjustments(fx)):
            pred = model.predict(str(fixture.home), str(fixture.away), adjust=adj)
            unseen = [team for team in (fixture.home, fixture.away) if team not in model.team_index]
            if unseen:
                logger.warning(f"no history for {unseen}; using a typical promoted-team rating")
            rows.append({"date": fixture.date, "season": fixture.season, **pred, "odds_h": fixture.odds_h, "odds_d": fixture.odds_d,
                         "odds_a": fixture.odds_a, "adjusted": bool(adj), "schedule": fixture.schedule})
        up = pd.DataFrame(rows)
        write_csv(up, self.state / UPCOMING_FILE)
        ledger.record(self.state / LEDGER_FILE, up)
        conf = self.cfg.confidence
        show = report.display_predictions(up, conf["high"], conf["medium"])
        logger.info(f"{self.league.name}: {len(up)} upcoming match(es), model fitted on {model.n_matches} matches")
        n_espn = int((up.schedule == ESPN_SCHEDULE).sum())
        if n_espn:
            logger.info(f"{n_espn} game(s) from ESPN's schedule (no bookmaker odds until {self.source.name} lists them)")
        logger.info("predictions:\n" + show[PREDICT_DISPLAY_COLS].to_string(
            index=False, formatters={col: "{:.0%}".format for col in PERCENT_DISPLAY_COLS},
            float_format=lambda value: f"{value:+.2f}"))
        return up

    def _espn_fixtures(self, fx: pd.DataFrame, model) -> pd.DataFrame:
        """Add games from ESPN's schedule that the main source hasn't listed yet."""
        if not self.league.espn or self.cfg.lineups.get("espn_schedule", True) is False:
            return fx
        try:
            ev = self._espn().upcoming(self.league.espn, self.season, self.cfg.upcoming_days)
        except Exception as exc:  # noqa: BLE001 - ESPN is a fallback; never block predictions
            logger.warning(f"ESPN schedule unavailable ({exc}); using {self.source.name} fixtures only")
            return fx
        if ev.empty:
            return fx
        path = self.state / TEAM_MAP_FILE
        tmap = {}
        if path.exists():
            map_df = pd.read_csv(path, dtype=str)
            tmap = dict(zip(map_df.espn_id, map_df.fd_name))
        known = list(model.team_index)

        def fd_name(espn_id, espn_name):
            return tmap.get(espn_id) or teammap.match_name(espn_name, known)

        have = set(zip(fx.home, fx.away))
        rows, unmapped = [], set()
        for event in ev.itertuples(index=False):
            home, away = fd_name(event.home_id, event.home_name), fd_name(event.away_id, event.away_name)
            if home is None or away is None:
                unmapped |= {name for name, mapped in ((event.home_name, home), (event.away_name, away)) if mapped is None}
                continue
            if (home, away) in have:
                continue
            rows.append({"date": cast(pd.Timestamp, event.date).normalize(), "season": self.season, "div": self.top, "home": home, "away": away,
                         "hg": np.nan, "ag": np.nan, "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
                         "schedule": ESPN_SCHEDULE})
        if unmapped:
            logger.warning(f"skipped ESPN games for teams with no football-data name: {sorted(unmapped)}; "
                           f"add them to {self.state / TEAM_MAP_OVERRIDES_FILE} and run lineups-backtest, or wait for "
                           f"{self.source.name} to list the games")
        if not rows:
            return fx
        return pd.concat([fx, pd.DataFrame(rows)], ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)

    def settle(self, refresh: bool = True) -> int:
        res = load_matches(self.source, [self.season], [self.top], refresh_season=self.season if refresh else None)
        n_settled = ledger.settle(self.state / LEDGER_FILE, res)
        logger.info(f"scored {n_settled} logged prediction(s) against new results")
        return n_settled

    def report(self):
        def read(name, dates=("date",)):
            csv_path = self.state / name
            if not csv_path.exists() or csv_path.stat().st_size < MIN_CSV_BYTES:
                return None
            return read_csv(csv_path, parse_dates=[date_col for date_col in dates if date_col in pd.read_csv(csv_path, nrows=0).columns])
        path = self.cfg.reports_dir / REPORT_FILE.format(league=self.league.key)
        conf = self.cfg.confidence
        coefs = self.state / LINEUP_COEFS_FILE
        report.write(path, self.league.name, read(UPCOMING_FILE), read(BACKTEST_FILE.format(season=self.season)),
                     self.season, ledger.load(self.state / LEDGER_FILE), read(RATINGS_FILE, ()),
                     read(TUNING_FILE, ()), conf["high"], conf["medium"],
                     lineup_comparison=read(LINEUP_COMPARISON_FILE, ()),
                     lineup_coefs=json.loads(coefs.read_text()) if coefs.exists() else None,
                     lineup_upcoming=read(LINEUP_UPCOMING_FILE))
        logger.info(f"report -> {path}")

    def update(self):
        self.settle(refresh=True)
        self.backtest(refresh=False)   # settle() already refreshed the current season
        self.predict(refresh=False)
        self.report()

    # ------------------------------------------------------------- lineups (ESPN)
    def _espn(self) -> EspnClient:
        if not self.league.espn:
            raise SystemExit(f"no espn slug configured for {self.league.key} in src/football_predictions.toml")
        lc = self.cfg.lineups
        return EspnClient(self.cfg.data_dir, lc.get("user_agent", DEFAULT_UA), float(lc.get("delay_seconds", DEFAULT_DELAY_SECONDS)),
                          now=self.cfg.now)

    def _lineup_model(self, client) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Sync ESPN, map teams, build features, walk-forward the adjustment. Returns what predict needs."""
        lc = self.cfg.lineups
        seasons = [shift_season(self.season, -k) for k in range(int(lc.get("seasons", DEFAULT_LINEUP_SEASONS)) - 1, -1, -1)]
        logger.info(f"syncing ESPN {self.league.espn} for {seasons} (cached after the first run)")
        events = pd.concat([client.sync_season(self.league.espn, season_name) for season_name in seasons], ignore_index=True)
        logger.info(f"{len(events)} events, {client.requests_made} ESPN requests this run")
        load_matches(self.source, [self.season], [self.top], refresh_season=self.season)
        fd = load_matches(self.source, seasons, [self.top])
        tm = teammap.build(events[events.state == STATE_POST], fd, self.state / TEAM_MAP_OVERRIDES_FILE)
        tm.to_csv(self.state / TEAM_MAP_FILE, index=False)
        names = dict(zip(events.home_id, events.home_name)) | dict(zip(events.away_id, events.away_name))
        missing = sorted(names[espn_id] for espn_id in set(names) - set(tm.espn_id))
        if missing:
            logger.warning(f"no football-data match for ESPN teams {missing}; "
                           f"add them to {self.state / TEAM_MAP_OVERRIDES_FILE} (espn_name,fd_name)")
        players = lineups.player_table(client, self.league.espn, events, tm)
        base = pd.concat([bt.walk_forward(self.matches(season_name), season_name, self.top, self.hyper()) for season_name in seasons],
                         ignore_index=True)
        feat = lineups.match_features(players, events[events.state == STATE_POST], tm)
        logger.info(f"lineup features for {len(feat)} of {len(fd)} {self.top} matches")
        ev, coefs = lineups.walk_forward(feat, base, float(lc.get("ridge", lineups.DEFAULT_RIDGE)),
                                        int(lc.get("min_train_matches", lineups.DEFAULT_MIN_TRAIN)))
        lineups.save_coefs(self.state / LINEUP_COEFS_FILE, coefs, len(feat))
        if not ev.empty:
            write_csv(ev, self.state / LINEUP_BACKTEST_FILE)
            comp = lineups.compare(ev)
            comp.to_csv(self.state / LINEUP_COMPARISON_FILE, index=False)
            logger.info("lineup adjustment, walk-forward (same matches for both):\n"
                        + comp.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        else:
            logger.info("not enough lineup history yet for a walk-forward test")
        logger.info(f"coefficients: {dict(zip(lineups.COEFS, np.round(coefs, 3)))}")
        return events, tm, players, coefs

    def lineups_backtest(self):
        self._lineup_model(self._espn())
        self.report()

    def lineups_predict(self, window_hours: int = LINEUP_WINDOW_HOURS):
        client = self._espn()
        event_df, tm, players, coefs = self._lineup_model(client)
        model = DixonColes(self.hyper()).fit(self.matches(self.season), pd.Timestamp(self.cfg.today), self.top)
        now = self.cfg.now if self.cfg.now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
        up = event_df[(event_df.state == STATE_PRE) & (event_df.date >= now - pd.Timedelta(hours=LINEUP_GRACE_HOURS))
                    & (event_df.date <= now + pd.Timedelta(hours=window_hours))]
        if up.empty:
            logger.info(f"no {self.league.espn} games in the next {window_hours} hours")
            return
        got, waiting = {}, []
        for event in up.itertuples(index=False):
            event_id = str(event.event_id)
            lu = parse_lineups(client.summary(self.league.espn, event_id, refresh=True), event_id)
            if has_lineups(lu):
                got[event_id] = lu
            else:
                waiting.append(f"{event.home_name} v {event.away_name} ({event.date:%a %H:%M} UTC)")
        if waiting:
            logger.info("lineups not published yet: " + "; ".join(waiting))
        if not got:
            return
        feat = lineups.match_features(players, up[up.event_id.isin(got)], tm, lineups=got)
        row_lst = []
        for feat_row in feat.itertuples(index=False):
            base = model.predict(str(feat_row.home), str(feat_row.away))
            row = pd.Series({**dict(zip(feat.columns, feat_row)), "xg_home": base["xg_home"], "xg_away": base["xg_away"], "rho": base["rho"]})
            adjusted = lineups.enhance(row, coefs, self.hyper().max_goals)
            row_lst.append({"date": feat_row.date, "season": feat_row.season, "home": feat_row.home, "away": feat_row.away, **adjusted,
                         "base_p_home": base["p_home"], "base_p_draw": base["p_draw"], "base_p_away": base["p_away"],
                         **{feature: getattr(feat_row, feature) for feature in lineups.FEATURE_COLS}})
        out = pd.DataFrame(row_lst)
        write_csv(out, self.state / LINEUP_UPCOMING_FILE)
        ledger.record(self.state / LEDGER_FILE,
                      out.drop(columns=[col for col in out if str(col).startswith(LINEUP_LEDGER_DROP_PREFIXES)]),
                      variant=LINEUP_VARIANT)
        show = out.assign(
            base=lambda frame: frame.apply(lambda pred_row: f"{pred_row.base_p_home:.0%}/{pred_row.base_p_draw:.0%}/{pred_row.base_p_away:.0%}", axis=1),
            lineup=lambda frame: frame.apply(lambda pred_row: f"{pred_row.p_home:.0%}/{pred_row.p_draw:.0%}/{pred_row.p_away:.0%}", axis=1),
            pick_team=lambda frame: frame.apply(lambda pred_row: pred_row.home if pred_row.pick == "H" else (pred_row.away if pred_row.pick == "A" else DRAW_LABEL), axis=1))
        logger.info(f"lineup-adjusted predictions ({len(out)}):\n"
                    + show[LINEUP_DISPLAY_COLS].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
        self.report()


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Parse the command line, then load the TOML config it points to (with --today applied); returns (config, args)."""
    parser = argparse.ArgumentParser(prog=PROG_NAME, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    league_group = parser.add_mutually_exclusive_group()
    league_group.add_argument("--league", default=DEFAULT_LEAGUE)
    league_group.add_argument("--all", action="store_true", help="every enabled league")
    parser.add_argument("--season", help="backtest: season to test, e.g. 2025-26")
    parser.add_argument("--source", help="override the data source in the config file")
    parser.add_argument("--today", help="pretend today is this date (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    if not args.today:
        return read_toml_config(args.config), args
    today = date.fromisoformat(args.today)
    now = pd.Timestamp(today) + pd.Timedelta(hours=REPLAY_HOUR_UTC)
    return read_toml_config(args.config, today, now), args


def main(argv=None):
    setup_logger(DEFAULT_CONFIG_PATH.stem)   # football_predictions.toml -> football_predictions.log
    cfg, args = load_config(argv)
    league_lst = [league for league in cfg.leagues if league.enabled] if args.all else [cfg.league(args.league)]
    for lg in league_lst:
        run = Run(cfg, lg, args.source or cfg.source)
        logger.info(f"== {lg.name} ({lg.key}) season {lg.current_season}, source {run.source.name}, divisions {run.divs}")
        match args.command:
            case "download": run.download()
            case "tune": run.tune()
            case "backtest": run.backtest(args.season, refresh=True)
            case "predict": run.predict(refresh=True)
            case "update": run.update()
            case "report": run.report()
            case "lineups-backtest": run.lineups_backtest()
            case "lineups": run.lineups_predict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
