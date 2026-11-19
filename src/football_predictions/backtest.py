"""
Walk-forward backtest and hyperparameter tuning.

Each refit point uses only matches played strictly before it, so nothing from the future leaks in.
This is the same refit-after-every-matchday loop the live `update` command runs.
"""

import itertools

import pandas as pd

from .metrics import OUTCOMES, PROB_COLS, outcome, rps, onehot
from .model import DixonColes, Hyper

BASE_RATE_SEASONS = 2                      # years of top-division results behind the base rates
FALLBACK_BASE_RATES = (0.45, 0.25, 0.30)   # home/draw/away when there is no history at all
REFIT_DAILY = "date"
REFIT_WEEKLY = "week"
WEEK_PERIOD = "W-SUN"                      # weekly refits group Monday..Sunday
TUNABLE_PARAMS = ("half_life_days", "ridge", "feeder_weight", "cup_weight")

def base_rates(matches: pd.DataFrame, asof, top_div: str, seasons: int = BASE_RATE_SEASONS) -> tuple[float, float, float]:
    """Compute base rates for home win, draw, and away win."""
    recent = matches[(matches.date < asof) & (matches["div"] == top_div)]
    recent = recent[recent.date >= pd.Timestamp(asof) - pd.DateOffset(years=seasons)]
    results = outcome(recent.hg, recent.ag)
    if not len(results):
        return FALLBACK_BASE_RATES
    p_h, p_d, p_a = (float((results == code).mean()) for code in OUTCOMES)
    return p_h, p_d, p_a


def walk_forward(matches: pd.DataFrame, season: str, top_div: str, hyper: Hyper,
                 refit: str = REFIT_DAILY, log=None) -> pd.DataFrame:
    """Perform a walk-forward backtest for a given season and division."""
    test = matches[(matches.season == season) & (matches["div"] == top_div)].sort_values("date")
    if test.empty:
        return pd.DataFrame()
    key = test.date if refit == REFIT_DAILY else test.date.dt.to_period(WEEK_PERIOD).dt.start_time
    row_lst, model = [], None
    for _, day_matches in test.groupby(key, sort=True):
        asof = day_matches.date.min()
        try:
            model = DixonColes(hyper).fit(matches, asof, top_div, warm=model)
        except ValueError:
            continue
        base = base_rates(matches, asof, top_div)
        for row in day_matches.itertuples(index=False):
            pred = model.predict(str(row.home), str(row.away))
            row_lst.append({"date": row.date, "season": row.season, **pred, "hg": row.hg, "ag": row.ag,
                         "odds_h": row.odds_h, "odds_d": row.odds_d, "odds_a": row.odds_a,
                         "p_base_h": base[0], "p_base_d": base[1], "p_base_a": base[2],
                         "home_seen": row.home in model.team_index, "away_seen": row.away in model.team_index})
        if log:
            log(f"  {asof:%Y-%m-%d}: {len(day_matches)} match(es), fitted on {model.n_matches}")
    df = pd.DataFrame(row_lst)
    df["result"] = outcome(df.hg, df.ag)
    return df


def tune(matches: pd.DataFrame, season: str, top_div: str, base: Hyper, grid: dict, log=print) -> tuple[Hyper, pd.DataFrame]:
    keys = [param for param in TUNABLE_PARAMS if grid.get(param)]
    results = []
    for combo in itertools.product(*(grid[param] for param in keys)):
        hyper = Hyper(**{**base.to_dict(), **dict(zip(keys, combo))})
        bt = walk_forward(matches, season, top_div, hyper, refit=REFIT_WEEKLY)
        P = bt[PROB_COLS].to_numpy()
        score = float(rps(P, onehot(bt.result)).mean())
        results.append({**dict(zip(keys, combo)), "n": len(bt), "rps": score})
        log(f"  {dict(zip(keys, combo))}: RPS {score:.4f}")
    table = pd.DataFrame(results).sort_values("rps").reset_index(drop=True)
    best = table.iloc[0]
    return Hyper(**{**base.to_dict(), **{param: type(getattr(base, param))(best[param]) for param in keys}}), table
