"""Lineup-enhanced predictions.

For each team in a match, compare the announced starting XI with the team's recent usage, using
only matches played before kickoff:

  miss  share of the team's usual playing time missing from the XI (0 = full strength)
  att   log(attacking output of the XI / output of its usual XIs); goals + assists per 90,
        shrunk toward a position prior so one lucky goal doesn't make a player a star
  gk    1 if the regular goalkeeper (>= 60% of GK minutes) is not starting

The base model's expected goals are then scaled:
  goals_for ~ Poisson(xg_base * exp(b1*miss_own + b2*att_own + b3*miss_opp + b4*gk_opp))
The four b's are fitted (ridge-shrunk toward 0, i.e. toward "no effect") on past matches, using
the base model's walk-forward expected goals as the offset. The evaluation is walk-forward too.
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import metrics
from .espn import has_lineups, parse_lineups
from .model import summarize

FEATURES = ["miss", "att", "gk"]
COEFS = ["miss_own", "att_own", "miss_opp", "gk_opp"]
PRIOR_GA90 = {"G": 0.0, "D": 0.08, "M": 0.22, "F": 0.45}


# ------------------------------------------------------------------ data assembly
def player_table(client, slug: str, events: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Players in every finished event, with the football-data team name attached."""
    ids = dict(zip(team_map.espn_id, team_map.fd_name))
    frames = []
    for e in events[events.state == "post"].itertuples(index=False):
        lu = parse_lineups(client.summary(slug, e.event_id), e.event_id)
        if not has_lineups(lu):
            continue
        frames.append(lu.assign(date=e.date, season=e.season))
    if not frames:
        return pd.DataFrame()
    p = pd.concat(frames, ignore_index=True)
    p["team"] = p.team_id.map(ids)
    return p.dropna(subset=["team"])


def team_features(hist: pd.DataFrame, xi: pd.DataFrame, n_hist: int = 19, decay: float = 0.85,
                  shrink_90s: float = 4.0, min_hist: int = 4) -> dict | None:
    """hist: this team's player rows from earlier matches; xi: the 11 announced starters."""
    order = hist.drop_duplicates("event_id").sort_values("date", ascending=False).event_id.head(n_hist).tolist()
    if len(order) < min_hist:
        return None
    w = {eid: decay ** k for k, eid in enumerate(order)}
    h = hist[hist.event_id.isin(order)].assign(w=lambda d: d.event_id.map(w))
    h = h.assign(wmin=h.w * h.minutes, wga=h.w * h.ga)
    agg = h.groupby("athlete_id").agg(wmin=("wmin", "sum"), wga=("wga", "sum"),
                                      pos=("pos", lambda s: s.mode().iat[0]))
    team_wmin = sum(w.values()) * 990.0
    share = agg.wmin / team_wmin
    top11 = share.nlargest(11).sum()
    g = (agg.wga + shrink_90s * agg.pos.map(PRIOR_GA90)) / (agg.wmin / 90 + shrink_90s)

    xi_ids = xi.athlete_id.tolist()
    miss = float(np.clip(1 - share.reindex(xi_ids).fillna(0).sum() / top11, 0, 1)) if top11 > 0 else 0.0
    xi_g = g.reindex(xi_ids)
    xi_g = xi_g.fillna(xi.set_index("athlete_id").pos.map(PRIOR_GA90))
    xi_att = float(xi_g.sum())
    starts = h[h.starter].assign(g=lambda d: d.athlete_id.map(g))
    usual = (starts.groupby("event_id").g.sum() * pd.Series(w)).dropna().sum() / sum(w.values())
    att = float(np.log(max(xi_att, 0.05) / max(usual, 0.05)))

    gk_min = h[h.pos == "G"].groupby("athlete_id").wmin.sum()
    gk = 0.0
    if gk_min.sum() > 0:
        regular = gk_min.idxmax()
        if gk_min.max() / gk_min.sum() >= 0.6 and regular not in set(xi[xi.pos == "G"].athlete_id):
            gk = 1.0
    return {"miss": miss, "att": att, "gk": gk, "n_hist": len(order)}


def match_features(players: pd.DataFrame, events: pd.DataFrame, team_map: pd.DataFrame,
                   lineups: dict | None = None) -> pd.DataFrame:
    """Features for each event. `lineups` maps event_id -> parsed lineup (for upcoming games);
    finished games use their own starters from `players`."""
    ids = dict(zip(team_map.espn_id, team_map.fd_name))
    by_team = {t: g for t, g in players.groupby("team")} if not players.empty else {}
    by_event = {k: g for k, g in players.groupby("event_id")} if not players.empty else {}
    rows = []
    for e in events.itertuples(index=False):
        home, away = ids.get(e.home_id), ids.get(e.away_id)
        if home is None or away is None:
            continue
        lu = lineups.get(e.event_id) if lineups is not None else by_event.get(e.event_id)
        if lu is None or lu.empty or not has_lineups(lu):
            continue
        f = {}
        for side, team, tid in (("h", home, e.home_id), ("a", away, e.away_id)):
            hist = by_team.get(team, pd.DataFrame())
            hist = hist[hist.date < e.date] if not hist.empty else hist
            xi = lu[(lu.team_id == tid) & lu.starter]
            r = team_features(hist, xi) if not hist.empty else None
            if r is None:
                break
            f.update({f"{side}_{k}": v for k, v in r.items()})
        else:
            rows.append({"event_id": e.event_id, "date": e.date, "season": e.season,
                         "home": home, "away": away, **f})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ adjustment model
def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xh = df[["h_miss", "h_att", "a_miss", "a_gk"]].to_numpy(float)
    Xa = df[["a_miss", "a_att", "h_miss", "h_gk"]].to_numpy(float)
    return Xh, Xa, np.log(df.xg_home.to_numpy(float)), np.log(df.xg_away.to_numpy(float))


def fit_coefs(train: pd.DataFrame, ridge: float = 5.0) -> np.ndarray:
    Xh, Xa, oh, oa = design(train)
    X = np.vstack([Xh, Xa])
    off = np.concatenate([oh, oa])
    y = np.concatenate([train.hg.to_numpy(float), train.ag.to_numpy(float)])

    def f(b):
        eta = off + X @ b
        mu = np.exp(eta)
        return np.sum(mu - y * eta) + 0.5 * ridge * b @ b, X.T @ (mu - y) + ridge * b

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def multipliers(row, b: np.ndarray) -> tuple[float, float]:
    mh = np.exp(b @ np.array([row.h_miss, row.h_att, row.a_miss, row.a_gk], float))
    ma = np.exp(b @ np.array([row.a_miss, row.a_att, row.h_miss, row.h_gk], float))
    return float(mh), float(ma)


def enhance(row, b: np.ndarray, max_goals: int = 10) -> dict:
    mh, ma = multipliers(row, b)
    return summarize(row.xg_home * mh, row.xg_away * ma, row.rho, max_goals)


def walk_forward(feat: pd.DataFrame, base: pd.DataFrame, ridge: float = 5.0, min_train: int = 150) -> tuple[pd.DataFrame, np.ndarray]:
    """Refit coefficients before each match date on everything earlier; predict that date's games."""
    keep = ["season", "home", "away", "xg_home", "xg_away", "rho", "p_home", "p_draw", "p_away", "pick",
            "confidence", "exp_gd", "gd_lo80", "gd_hi80", "hg", "ag", "result", "odds_h", "odds_d", "odds_a"]
    df = feat.merge(base[[c for c in keep if c in base]], on=["season", "home", "away"], how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    out, b = [], np.zeros(4)
    for d, g in df.groupby(df.date.dt.normalize()):
        train = df[df.date < d]
        if len(train) < min_train:
            continue
        b = fit_coefs(train, ridge)
        for r in g.itertuples(index=False):
            e = enhance(r, b)
            out.append({"event_id": r.event_id, **{f"lu_{k}": v for k, v in e.items()}})
    ev = df.merge(pd.DataFrame(out), on="event_id", how="inner") if out else pd.DataFrame()
    final = fit_coefs(df, ridge) if len(df) >= min_train else b
    return ev, final


def compare(ev: pd.DataFrame) -> pd.DataFrame:
    """Base vs lineup-enhanced on the same matches, overall and by season.

    goal_loglik is the average Poisson log-likelihood of both teams' goals (higher is better).
    It reacts to lineup information sooner than the W/D/L scores, which need many matches.
    """
    from scipy.stats import poisson
    rows = []
    for scope, g in [("all", ev), *[(s, x) for s, x in ev.groupby("season")]]:
        Y = metrics.onehot(g.result)
        for name, cols, xg in (("base model", ["p_home", "p_draw", "p_away"], ["xg_home", "xg_away"]),
                               ("with lineups", ["lu_p_home", "lu_p_draw", "lu_p_away"], ["lu_xg_home", "lu_xg_away"])):
            P = g[cols].to_numpy()
            ll = (poisson.logpmf(g.hg, g[xg[0]]) + poisson.logpmf(g.ag, g[xg[1]])).mean()
            rows.append({"season": scope, "forecaster": name, "n": len(g),
                         "rps": metrics.rps(P, Y).mean(), "log_loss": metrics.log_loss(P, Y).mean(),
                         "accuracy": (P.argmax(1) == Y.argmax(1)).mean(), "goal_loglik": ll})
    return pd.DataFrame(rows)


def save_coefs(path, b: np.ndarray, n: int):
    path.write_text(json.dumps({"coefficients": dict(zip(COEFS, map(float, b))), "n_matches": n}, indent=2))


def load_coefs(path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array([json.loads(path.read_text())["coefficients"][k] for k in COEFS])
