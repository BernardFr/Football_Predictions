"""Walk-forward backtest and hyperparameter tuning.

Each refit point uses only matches played strictly before it, so nothing from the future leaks in.
This is the same refit-after-every-matchday loop the live `update` command runs.
"""

import itertools

import numpy as np
import pandas as pd

from .metrics import outcome, rps, onehot
from .model import DixonColes, Hyper


def base_rates(matches: pd.DataFrame, asof, top_div: str, seasons: int = 2) -> tuple[float, float, float]:
    m = matches[(matches.date < asof) & (matches["div"] == top_div)]
    m = m[m.date >= pd.Timestamp(asof) - pd.DateOffset(years=seasons)]
    r = outcome(m.hg, m.ag)
    return tuple(float((r == o).mean()) for o in ("H", "D", "A")) if len(r) else (0.45, 0.25, 0.30)


def walk_forward(matches: pd.DataFrame, season: str, top_div: str, hyper: Hyper,
                 refit: str = "date", log=None) -> pd.DataFrame:
    test = matches[(matches.season == season) & (matches["div"] == top_div)].sort_values("date")
    if test.empty:
        return pd.DataFrame()
    key = test.date if refit == "date" else test.date.dt.to_period("W-SUN").dt.start_time
    rows, model = [], None
    for _, g in test.groupby(key, sort=True):
        asof = g.date.min()
        try:
            model = DixonColes(hyper).fit(matches, asof, top_div, warm=model)
        except ValueError:
            continue
        b = base_rates(matches, asof, top_div)
        for r in g.itertuples(index=False):
            p = model.predict(r.home, r.away)
            rows.append({"date": r.date, "season": r.season, **p, "hg": r.hg, "ag": r.ag,
                         "odds_h": r.odds_h, "odds_d": r.odds_d, "odds_a": r.odds_a,
                         "p_base_h": b[0], "p_base_d": b[1], "p_base_a": b[2],
                         "home_seen": r.home in model.team_index, "away_seen": r.away in model.team_index})
        if log:
            log(f"  {asof:%Y-%m-%d}: {len(g)} match(es), fitted on {model.n_matches}")
    df = pd.DataFrame(rows)
    df["result"] = outcome(df.hg, df.ag)
    return df


def tune(matches: pd.DataFrame, season: str, top_div: str, base: Hyper, grid: dict, log=print) -> tuple[Hyper, pd.DataFrame]:
    keys = [k for k in ("half_life_days", "ridge", "feeder_weight", "cup_weight") if grid.get(k)]
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        h = Hyper(**{**base.to_dict(), **dict(zip(keys, combo))})
        bt = walk_forward(matches, season, top_div, h, refit="week")
        P = bt[["p_home", "p_draw", "p_away"]].to_numpy()
        score = float(rps(P, onehot(bt.result)).mean())
        results.append({**dict(zip(keys, combo)), "n": len(bt), "rps": score})
        log(f"  {dict(zip(keys, combo))}: RPS {score:.4f}")
    table = pd.DataFrame(results).sort_values("rps").reset_index(drop=True)
    best = table.iloc[0]
    return Hyper(**{**base.to_dict(), **{k: type(getattr(base, k))(best[k]) for k in keys}}), table
