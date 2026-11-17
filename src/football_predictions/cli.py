"""football-predictions <command> [--league EPL | --all] [--source openfootball] [--today YYYY-MM-DD]

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
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import cups, ledger, lineups, report, teammap
from .config import load_config, shift_season
from .espn import DEFAULT_UA, EspnClient, has_lineups, parse_lineups
from .model import DixonColes, Hyper
from .sources import load_matches, make_source


class Run:
    def __init__(self, cfg, league, source_name, today, now=None):
        self.cfg, self.league, self.today, self.now = cfg, league, today, now
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
        m = load_matches(self.source, seasons, self.divs, refresh_season=self.season if refresh else None)
        m = m.assign(neutral=False)
        cup = self.cup_matches(seasons, set(m.home) | set(m.away))
        if not cup.empty:
            m = pd.concat([m, cup[m.columns]], ignore_index=True).sort_values(["date", "div", "home"]).reset_index(drop=True)
        return m[m.date < pd.Timestamp(self.today)]

    def cup_matches(self, seasons: list[str], known_teams) -> pd.DataFrame:
        """Cup games between rated teams, cached for this run (they are only fetched from ESPN once)."""
        if not self.league.cups or not self.league.espn:
            return pd.DataFrame()
        key = tuple(seasons)
        if not hasattr(self, "_cups"):
            self._cups = {}
        if key not in self._cups:
            self._cups[key] = cups.load(self._espn(), self.league.cups, seasons, known_teams,
                                        self.state / "team_map.csv")
        return self._cups[key]

    def hyper(self) -> Hyper:
        h = Hyper.from_dict(self.cfg.model)
        p = self.state / "hyper.json"
        if p.exists():
            h = Hyper.from_dict({**h.to_dict(), **json.loads(p.read_text())})
        return h

    # ------------------------------------------------------------- commands
    def download(self):
        m = load_matches(self.source, self.seasons(self.season) + [shift_season(self.season, -self.cfg.history_seasons - 1)],
                         self.divs, refresh_season=self.season)
        print(m.groupby(["season", "div"]).size().unstack(fill_value=0).to_string())
        cup = self.cup_matches(self.seasons(self.season), set(m.home) | set(m.away))
        if not cup.empty:
            print("\ncup games between rated teams (from ESPN):")
            print(cup.groupby(["season", "competition"]).size().unstack(fill_value=0).to_string())

    def tune(self):
        s = shift_season(self.season, -1)
        print(f"tuning on {s} (walk-forward, weekly refits)")
        best, table = bt.tune(self.matches(s), s, self.top, Hyper.from_dict(self.cfg.model), self.cfg.tune_grid)
        (self.state / "hyper.json").write_text(json.dumps(best.to_dict(), indent=2))
        table.to_csv(self.state / "tuning.csv", index=False)
        print(f"best: {best.to_dict()} -> {self.state / 'hyper.json'}")

    def backtest(self, season: str | None = None, refresh: bool = False) -> pd.DataFrame:
        s = season or self.season
        df = bt.walk_forward(self.matches(s, refresh), s, self.top, self.hyper())
        if df.empty:
            print(f"no {self.top} matches played in {s} yet")
            return df
        df.to_csv(self.state / f"backtest_{s}.csv", index=False)
        from .metrics import confidence_bands, summary
        print(f"\n{self.league.name} {s}: walk-forward backtest on {len(df)} matches")
        print(summary(df).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        c = self.cfg.confidence
        print(confidence_bands(df, c["high"], c["medium"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return df

    def _adjustments(self, fixtures: pd.DataFrame) -> list[dict | None]:
        path = self.state / "adjustments.csv"
        if not path.exists():
            return [None] * len(fixtures)
        adj = pd.read_csv(path)
        if "until" in adj:
            adj["until"] = pd.to_datetime(adj["until"], errors="coerce")
        out = []
        for r in fixtures.itertuples(index=False):
            a = {}
            for side, team in (("home", r.home), ("away", r.away)):
                rows = adj[adj.team == team]
                if "until" in adj:
                    rows = rows[rows.until.isna() | (rows.until >= r.date)]
                for x in rows.itertuples(index=False):
                    a[f"{side}_attack"] = a.get(f"{side}_attack", 1.0) * float(getattr(x, "attack", 1.0))
                    a[f"{side}_defence"] = a.get(f"{side}_defence", 1.0) * float(getattr(x, "defence", 1.0))
            out.append(a or None)
        return out

    def predict(self, refresh: bool = False) -> pd.DataFrame:
        m = self.matches(self.season, refresh)
        model = DixonColes(self.hyper()).fit(m, pd.Timestamp(self.today), self.top)
        ratings = model.ratings()
        ratings.to_csv(self.state / "ratings.csv", index=False)
        fx = self.source.fixtures(self.season, self.top, self.today, self.cfg.upcoming_days)
        fx = fx.assign(schedule=self.source.name)
        fx = self._espn_fixtures(fx, model)
        if fx.empty:
            print(f"no {self.league.name} games scheduled in the next {self.cfg.upcoming_days} days "
                  f"({self.source.name}{' or ESPN' if self.league.espn else ''}); nothing to predict "
                  f"(international break or end of season?)")
            pd.DataFrame().to_csv(self.state / "upcoming.csv", index=False)
            return fx
        rows = []
        for r, adj in zip(fx.itertuples(index=False), self._adjustments(fx)):
            p = model.predict(r.home, r.away, adjust=adj)
            unseen = [t for t in (r.home, r.away) if t not in model.team_index]
            if unseen:
                print(f"  warning: no history for {unseen}; using a typical promoted-team rating")
            rows.append({"date": r.date, "season": r.season, **p, "odds_h": r.odds_h, "odds_d": r.odds_d,
                         "odds_a": r.odds_a, "adjusted": bool(adj), "schedule": r.schedule})
        up = pd.DataFrame(rows)
        up.to_csv(self.state / "upcoming.csv", index=False)
        ledger.record(self.state / "ledger.csv", up)
        c = self.cfg.confidence
        show = report.display_predictions(up, c["high"], c["medium"])
        print(f"\n{self.league.name}: {len(up)} upcoming match(es), model fitted on {model.n_matches} matches")
        n_espn = int((up.schedule == "espn").sum())
        if n_espn:
            print(f"  {n_espn} game(s) from ESPN's schedule (no bookmaker odds until {self.source.name} lists them)")
        print(show[["date", "home", "away", "pick", "confidence", "band", "p_home", "p_draw", "p_away",
                    "exp_gd", "gd_range_80", "likely_score"]].to_string(
            index=False, formatters={k: "{:.0%}".format for k in ("confidence", "p_home", "p_draw", "p_away")},
            float_format=lambda v: f"{v:+.2f}"))
        return up

    def _espn_fixtures(self, fx: pd.DataFrame, model) -> pd.DataFrame:
        """Add games from ESPN's schedule that the main source hasn't listed yet."""
        if not self.league.espn or self.cfg.lineups.get("espn_schedule", True) is False:
            return fx
        try:
            ev = self._espn().upcoming(self.league.espn, self.season, self.cfg.upcoming_days)
        except Exception as e:  # noqa: BLE001 - ESPN is a fallback; never block predictions
            print(f"  ESPN schedule unavailable ({e}); using {self.source.name} fixtures only")
            return fx
        if ev.empty:
            return fx
        path = self.state / "team_map.csv"
        tmap = {}
        if path.exists():
            t = pd.read_csv(path, dtype=str)
            tmap = dict(zip(t.espn_id, t.fd_name))
        known = list(model.team_index)

        def fd_name(espn_id, espn_name):
            return tmap.get(espn_id) or teammap.match_name(espn_name, known)

        have = set(zip(fx.home, fx.away))
        rows, unmapped = [], set()
        for e in ev.itertuples(index=False):
            h, a = fd_name(e.home_id, e.home_name), fd_name(e.away_id, e.away_name)
            if h is None or a is None:
                unmapped |= {n for n, m in ((e.home_name, h), (e.away_name, a)) if m is None}
                continue
            if (h, a) in have:
                continue
            rows.append({"date": e.date.normalize(), "season": self.season, "div": self.top, "home": h, "away": a,
                         "hg": np.nan, "ag": np.nan, "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
                         "schedule": "espn"})
        if unmapped:
            print(f"  warning: skipped ESPN games for teams with no football-data name: {sorted(unmapped)}; "
                  f"add them to {self.state / 'team_map_overrides.csv'} and run lineups-backtest, or wait for "
                  f"{self.source.name} to list the games")
        if not rows:
            return fx
        return pd.concat([fx, pd.DataFrame(rows)], ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)

    def settle(self, refresh: bool = True) -> int:
        res = load_matches(self.source, [self.season], [self.top], refresh_season=self.season if refresh else None)
        n = ledger.settle(self.state / "ledger.csv", res)
        print(f"scored {n} logged prediction(s) against new results")
        return n

    def report(self):
        def read(name, dates=("date",)):
            p = self.state / name
            if not p.exists() or p.stat().st_size < 2:
                return None
            return pd.read_csv(p, parse_dates=[d for d in dates if d in pd.read_csv(p, nrows=0).columns])
        path = self.cfg.reports_dir / f"{self.league.key}_report.xlsx"
        c = self.cfg.confidence
        coefs = self.state / "lineup_coefs.json"
        report.write(path, self.league.name, read("upcoming.csv"), read(f"backtest_{self.season}.csv"),
                     self.season, ledger.load(self.state / "ledger.csv"), read("ratings.csv", ()),
                     read("tuning.csv", ()), c["high"], c["medium"],
                     lineup_comparison=read("lineup_comparison.csv", ()),
                     lineup_coefs=json.loads(coefs.read_text()) if coefs.exists() else None,
                     lineup_upcoming=read("lineup_upcoming.csv"))
        print(f"report -> {path}")

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
        return EspnClient(self.cfg.data_dir, lc.get("user_agent", DEFAULT_UA), float(lc.get("delay_seconds", 0.4)),
                          now=self.now)

    def _lineup_model(self, client):
        """Sync ESPN, map teams, build features, walk-forward the adjustment. Returns what predict needs."""
        lc = self.cfg.lineups
        seasons = [shift_season(self.season, -k) for k in range(int(lc.get("seasons", 2)) - 1, -1, -1)]
        print(f"syncing ESPN {self.league.espn} for {seasons} (cached after the first run)")
        events = pd.concat([client.sync_season(self.league.espn, s) for s in seasons], ignore_index=True)
        print(f"  {len(events)} events, {client.requests_made} ESPN requests this run")
        load_matches(self.source, [self.season], [self.top], refresh_season=self.season)
        fd = load_matches(self.source, seasons, [self.top])
        tm = teammap.build(events[events.state == "post"], fd, self.state / "team_map_overrides.csv")
        tm.to_csv(self.state / "team_map.csv", index=False)
        names = dict(zip(events.home_id, events.home_name)) | dict(zip(events.away_id, events.away_name))
        missing = sorted(names[i] for i in set(names) - set(tm.espn_id))
        if missing:
            print(f"  warning: no football-data match for ESPN teams {missing}; "
                  f"add them to {self.state / 'team_map_overrides.csv'} (espn_name,fd_name)")
        players = lineups.player_table(client, self.league.espn, events, tm)
        base = pd.concat([bt.walk_forward(self.matches(s), s, self.top, self.hyper()) for s in seasons],
                         ignore_index=True)
        feat = lineups.match_features(players, events[events.state == "post"], tm)
        print(f"  lineup features for {len(feat)} of {len(fd)} {self.top} matches")
        ev, b = lineups.walk_forward(feat, base, float(lc.get("ridge", 5.0)), int(lc.get("min_train_matches", 150)))
        lineups.save_coefs(self.state / "lineup_coefs.json", b, len(feat))
        if not ev.empty:
            ev.to_csv(self.state / "lineup_backtest.csv", index=False)
            comp = lineups.compare(ev)
            comp.to_csv(self.state / "lineup_comparison.csv", index=False)
            print("\nLineup adjustment, walk-forward (same matches for both):")
            print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        else:
            print("  not enough lineup history yet for a walk-forward test")
        print("coefficients:", dict(zip(lineups.COEFS, np.round(b, 3))))
        return events, tm, players, b

    def lineups_backtest(self):
        self._lineup_model(self._espn())
        self.report()

    def lineups_predict(self, window_hours: int = 36):
        client = self._espn()
        events, tm, players, b = self._lineup_model(client)
        model = DixonColes(self.hyper()).fit(self.matches(self.season), pd.Timestamp(self.today), self.top)
        now = self.now if self.now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
        up = events[(events.state == "pre") & (events.date >= now - pd.Timedelta(hours=1))
                    & (events.date <= now + pd.Timedelta(hours=window_hours))]
        if up.empty:
            print(f"no {self.league.espn} games in the next {window_hours} hours")
            return
        got, waiting = {}, []
        for e in up.itertuples(index=False):
            lu = parse_lineups(client.summary(self.league.espn, e.event_id, refresh=True), e.event_id)
            if has_lineups(lu):
                got[e.event_id] = lu
            else:
                waiting.append(f"{e.home_name} v {e.away_name} ({e.date:%a %H:%M} UTC)")
        if waiting:
            print("lineups not published yet: " + "; ".join(waiting))
        if not got:
            return
        feat = lineups.match_features(players, up[up.event_id.isin(got)], tm, lineups=got)
        rows = []
        for r in feat.itertuples(index=False):
            base = model.predict(r.home, r.away)
            row = pd.Series({**r._asdict(), "xg_home": base["xg_home"], "xg_away": base["xg_away"], "rho": base["rho"]})
            e = lineups.enhance(row, b, self.hyper().max_goals)
            rows.append({"date": r.date, "season": r.season, "home": r.home, "away": r.away, **e,
                         "base_p_home": base["p_home"], "base_p_draw": base["p_draw"], "base_p_away": base["p_away"],
                         **{k: getattr(r, k) for k in ("h_miss", "h_att", "h_gk", "a_miss", "a_att", "a_gk")}})
        out = pd.DataFrame(rows)
        out.to_csv(self.state / "lineup_upcoming.csv", index=False)
        ledger.record(self.state / "ledger.csv", out.drop(columns=[c for c in out if c.startswith(("base_", "h_", "a_"))]),
                      variant="lineup")
        print(f"\nLineup-adjusted predictions ({len(out)}):")
        show = out.assign(
            base=lambda d: d.apply(lambda x: f"{x.base_p_home:.0%}/{x.base_p_draw:.0%}/{x.base_p_away:.0%}", axis=1),
            lineup=lambda d: d.apply(lambda x: f"{x.p_home:.0%}/{x.p_draw:.0%}/{x.p_away:.0%}", axis=1),
            pick_team=lambda d: d.apply(lambda x: x.home if x.pick == "H" else (x.away if x.pick == "A" else "Draw"), axis=1))
        print(show[["home", "away", "base", "lineup", "pick_team", "confidence", "exp_gd", "h_miss", "a_miss", "h_gk", "a_gk"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        self.report()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="football-predictions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "tune", "backtest", "predict", "update", "report",
                                        "lineups-backtest", "lineups"])
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "football_predictions.toml"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--league", default="EPL")
    g.add_argument("--all", action="store_true", help="every enabled league")
    ap.add_argument("--season", help="backtest: season to test, e.g. 2025-26")
    ap.add_argument("--source", help="override the data source in the config file")
    ap.add_argument("--today", help="pretend today is this date (YYYY-MM-DD)")
    a = ap.parse_args(argv)

    today = date.fromisoformat(a.today) if a.today else date.today()
    cfg = load_config(a.config, today)
    leagues = [l for l in cfg.leagues if l.enabled] if a.all else [cfg.league(a.league)]
    for lg in leagues:
        now = pd.Timestamp(a.today) + pd.Timedelta(hours=10) if a.today else None   # replay: 10:00 UTC that day
        run = Run(cfg, lg, a.source or cfg.source, today, now)
        print(f"== {lg.name} ({lg.key}) season {lg.current_season}, source {run.source.name}, divisions {run.divs}")
        match a.command:
            case "download": run.download()
            case "tune": run.tune()
            case "backtest": run.backtest(a.season, refresh=True)
            case "predict": run.predict(refresh=True)
            case "update": run.update()
            case "report": run.report()
            case "lineups-backtest": run.lineups_backtest()
            case "lineups": run.lineups_predict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
