"""Scoring rules and summary tables for W/D/L probabilities and goal-difference forecasts."""

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


def outcome(hg, ag) -> np.ndarray:
    hg, ag = np.asarray(hg), np.asarray(ag)
    return np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))


def onehot(res) -> np.ndarray:
    res = np.asarray(res)
    return np.stack([(res == o).astype(float) for o in OUTCOMES], axis=1)


def rps(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ranked probability score (lower is better); respects H > D > A ordering."""
    cp, cy = np.cumsum(P, axis=1)[:, :2], np.cumsum(Y, axis=1)[:, :2]
    return 0.5 * ((cp - cy) ** 2).sum(axis=1)


def log_loss(P, Y) -> np.ndarray:
    return -np.log(np.clip((P * Y).sum(axis=1), 1e-12, 1))


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
    d = df[mask]
    gd = (d.hg - d.ag).to_numpy()
    return {"gd_mae": float(np.abs(d.exp_gd.to_numpy() - gd).mean()),
            "gd_80_coverage": float(((gd >= d.gd_lo80) & (gd <= d.gd_hi80)).mean())}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Model vs. bookmakers vs. base rate. Bookmaker comparison uses only matches with odds.

    gd_mae: mean absolute error of the expected goal difference.
    gd_80_coverage: share of results inside the model's 80% goal-difference interval.
    """
    Y = onehot(df.result)
    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    allm = np.ones(len(df), bool)
    rows = [{"forecaster": "model", "subset": "all matches", **_score(P, Y), **_gd(df, allm)}]
    if {"p_base_h", "p_base_d", "p_base_a"} <= set(df):
        B = df[["p_base_h", "p_base_d", "p_base_a"]].to_numpy()
        rows.append({"forecaster": "base rate (H/D/A frequencies)", "subset": "all matches", **_score(B, Y)})
    has = (df[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
           if "odds_h" in df else np.zeros(len(df), bool))
    if has.any():
        K = implied_probs(df.odds_h[has], df.odds_d[has], df.odds_a[has])
        rows.append({"forecaster": "model", "subset": "matches with odds", **_score(P[has], Y[has]), **_gd(df, has)})
        rows.append({"forecaster": "bookmakers", "subset": "matches with odds", **_score(K, Y[has])})
    return pd.DataFrame(rows)


def confidence_bands(df: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    d = df.assign(band=band(df.confidence, high, medium), hit=(df.pick == df.result))
    order = {"High": 0, "Medium": 1, "Low": 2}
    t = (d.groupby("band").agg(n=("hit", "size"), mean_confidence=("confidence", "mean"), hit_rate=("hit", "mean"))
         .reset_index())
    return t.sort_values("band", key=lambda s: s.map(order)).reset_index(drop=True)


def band(conf, high: float, medium: float) -> np.ndarray:
    conf = np.asarray(conf, float)
    return np.where(conf >= high, "High", np.where(conf >= medium, "Medium", "Low"))


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Pooled reliability table over all three outcome probabilities."""
    Y = onehot(df.result)
    frames = [("model", df[["p_home", "p_draw", "p_away"]].to_numpy(), np.ones(len(df), bool))]
    if "odds_h" in df:
        has = df[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
        if has.any():
            K = np.full((len(df), 3), np.nan)
            K[has] = implied_probs(df.odds_h[has], df.odds_d[has], df.odds_a[has])
            frames.append(("bookmakers", K, has))
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for name, P, mask in frames:
        p, y = P[mask].ravel(), Y[mask].ravel()
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for b in range(bins):
            sel = idx == b
            if sel.any():
                rows.append({"forecaster": name, "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                             "n": int(sel.sum()), "mean_forecast": p[sel].mean(), "observed_freq": y[sel].mean()})
    return pd.DataFrame(rows)
