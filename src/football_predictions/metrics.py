"""Scoring rules and summary tables for W/D/L probabilities and goal-difference forecasts."""

import numpy as np
import pandas as pd

from .sources import ODDS_COLS

OUTCOMES = ("H", "D", "A")
PROB_COLS = ["p_home", "p_draw", "p_away"]
BASE_RATE_COLS = ["p_base_h", "p_base_d", "p_base_a"]
PROB_FLOOR = 1e-12                       # avoids log(0) in the log loss
BANDS = ("High", "Medium", "Low")        # confidence bands, most confident first
DEFAULT_CALIBRATION_BINS = 10


def outcome(hg, ag) -> np.ndarray:
    hg, ag = np.asarray(hg), np.asarray(ag)
    home, draw, away = OUTCOMES
    return np.where(hg > ag, home, np.where(hg == ag, draw, away))


def onehot(res) -> np.ndarray:
    res = np.asarray(res)
    return np.stack([(res == code).astype(float) for code in OUTCOMES], axis=1)


def rps(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ranked probability score (lower is better); respects H > D > A ordering."""
    cum_probs, cum_outcomes = np.cumsum(P, axis=1)[:, :2], np.cumsum(Y, axis=1)[:, :2]
    return 0.5 * ((cum_probs - cum_outcomes) ** 2).sum(axis=1)


def log_loss(P, Y) -> np.ndarray:
    return -np.log(np.clip((P * Y).sum(axis=1), PROB_FLOOR, 1))


def brier(P, Y) -> np.ndarray:
    return ((P - Y) ** 2).sum(axis=1)


def implied_probs(odds_h, odds_d, odds_a) -> np.ndarray:
    """Bookmaker probabilities with the margin removed by normalisation."""
    inv = 1 / np.column_stack([odds_h, odds_d, odds_a]).astype(float)
    return inv / inv.sum(axis=1, keepdims=True)


def _score(P, Y) -> dict:
    return {"n": len(P), "rps": rps(P, Y).mean(), "log_loss": log_loss(P, Y).mean(),
            "brier": brier(P, Y).mean(), "accuracy": (P.argmax(1) == Y.argmax(1)).mean()}


def _gd(df: pd.DataFrame, mask) -> dict:
    subset = df[mask]
    gd = (subset.hg - subset.ag).to_numpy()
    return {"gd_mae": float(np.abs(subset.exp_gd.to_numpy() - gd).mean()),
            "gd_80_coverage": float(((gd >= subset.gd_lo80) & (gd <= subset.gd_hi80)).mean())}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model vs. bookmakers vs. base rate. Bookmaker comparison uses only matches with odds.

    gd_mae: mean absolute error of the expected goal difference.
    gd_80_coverage: share of results inside the model's 80% goal-difference interval.
    """
    Y = onehot(df.result)
    P = df[PROB_COLS].to_numpy()
    allm = np.ones(len(df), bool)
    rows = [{"forecaster": "model", "subset": "all matches", **_score(P, Y), **_gd(df, allm)}]
    if set(BASE_RATE_COLS) <= set(df):
        base_probs = df[BASE_RATE_COLS].to_numpy()
        rows.append({"forecaster": "base rate (H/D/A frequencies)", "subset": "all matches", **_score(base_probs, Y)})
    has = (df[ODDS_COLS].notna().all(axis=1).to_numpy()
           if "odds_h" in df else np.zeros(len(df), bool))
    if has.any():
        bookie_probs = implied_probs(df.odds_h[has], df.odds_d[has], df.odds_a[has])
        rows.append({"forecaster": "model", "subset": "matches with odds", **_score(P[has], Y[has]), **_gd(df, has)})
        rows.append({"forecaster": "bookmakers", "subset": "matches with odds", **_score(bookie_probs, Y[has])})
    return pd.DataFrame(rows)


def confidence_bands(df: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    banded = df.assign(band=band(df.confidence, high, medium), hit=(df.pick == df.result))
    order = {band_name: rank for rank, band_name in enumerate(BANDS)}
    table = (banded.groupby("band").agg(n=("hit", "size"), mean_confidence=("confidence", "mean"), hit_rate=("hit", "mean"))
         .reset_index())
    return table.sort_values("band", key=lambda col_values: col_values.map(order)).reset_index(drop=True)


def band(conf, high: float, medium: float) -> np.ndarray:
    conf = np.asarray(conf, float)
    hi_band, mid_band, lo_band = BANDS
    return np.where(conf >= high, hi_band, np.where(conf >= medium, mid_band, lo_band))


def calibration(df: pd.DataFrame, bins: int = DEFAULT_CALIBRATION_BINS) -> pd.DataFrame:
    """Pooled reliability table over all three outcome probabilities."""
    Y = onehot(df.result)
    frames = [("model", df[PROB_COLS].to_numpy(), np.ones(len(df), bool))]
    if "odds_h" in df:
        has = df[ODDS_COLS].notna().all(axis=1).to_numpy()
        if has.any():
            bookie_probs = np.full((len(df), len(OUTCOMES)), np.nan)
            bookie_probs[has] = implied_probs(df.odds_h[has], df.odds_d[has], df.odds_a[has])
            frames.append(("bookmakers", bookie_probs, has))
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for name, P, mask in frames:
        forecast, observed = P[mask].ravel(), Y[mask].ravel()
        idx = np.clip(np.digitize(forecast, edges) - 1, 0, bins - 1)
        for bin_idx in range(bins):
            sel = idx == bin_idx
            if sel.any():
                rows.append({"forecaster": name, "bin": f"{edges[bin_idx]:.1f}-{edges[bin_idx + 1]:.1f}",
                             "n": int(sel.sum()), "mean_forecast": forecast[sel].mean(), "observed_freq": observed[sel].mean()})
    return pd.DataFrame(rows)


def team_vs_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per team: actual wins, losses and goal difference vs. what the model expected before each game.
    Expected wins/losses are the summed pre-match probabilities; expected GD is the summed exp_gd.
    """
    home = pd.DataFrame({"team": df.home, "win": df.hg > df.ag, "loss": df.hg < df.ag,
                         "p_win": df.p_home, "p_loss": df.p_away, "gd": df.hg - df.ag, "exp_gd": df.exp_gd})
    away = pd.DataFrame({"team": df.away, "win": df.ag > df.hg, "loss": df.ag < df.hg,
                         "p_win": df.p_away, "p_loss": df.p_home, "gd": df.ag - df.hg, "exp_gd": -df.exp_gd})
    games = pd.concat([home, away], ignore_index=True)
    table = games.groupby("team").agg(games=("win", "size"), wins=("win", "sum"), exp_wins=("p_win", "sum"),
                                      losses=("loss", "sum"), exp_losses=("p_loss", "sum"),
                                      gd=("gd", "sum"), exp_gd=("exp_gd", "sum")).reset_index()
    return table.assign(wins_vs_pred=table.wins - table.exp_wins,
                        losses_vs_pred=table.losses - table.exp_losses,
                        gd_vs_pred=table.gd - table.exp_gd)
