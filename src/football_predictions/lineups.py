"""
Lineup-enhanced predictions.

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
from .espn import GOALKEEPER, MATCH_MINUTES, STARTING_XI, STATE_POST, has_lineups, parse_lineups
from .model import DEFAULT_MAX_GOALS, summarize

FEATURES = ["miss", "att", "gk"]
COEFS = ["miss_own", "att_own", "miss_opp", "gk_opp"]
PRIOR_GA90 = {"G": 0.0, "D": 0.08, "M": 0.22, "F": 0.45}
FEATURE_COLS = ["h_miss", "h_att", "h_gk", "a_miss", "a_att", "a_gk"]
HOME_DESIGN_COLS = ["h_miss", "h_att", "a_miss", "a_gk"]   # drives home goals, in COEFS order
AWAY_DESIGN_COLS = ["a_miss", "a_att", "h_miss", "h_gk"]   # drives away goals, in COEFS order
BASE_COLS = ["season", "home", "away", "xg_home", "xg_away", "rho", "p_home", "p_draw", "p_away", "pick",
             "confidence", "exp_gd", "gd_lo80", "gd_hi80", "hg", "ag", "result", "odds_h", "odds_d", "odds_a"]
HISTORY_MATCHES = 19            # recent matches that define a team's usual XI
HISTORY_DECAY = 0.85            # weight of the k-th most recent match = HISTORY_DECAY ** k
SHRINK_90S = 4.0                # prior strength, in 90-minute appearances
MIN_HISTORY_MATCHES = 4
MIN_ATTACK_OUTPUT = 0.05        # floor on goals + assists per 90, so the log ratio stays finite
REGULAR_GK_SHARE = 0.6          # a keeper with this share of GK minutes is the regular
DEFAULT_RIDGE = 5.0
DEFAULT_MIN_TRAIN = 150         # matches needed before the coefficients are fitted
JSON_INDENT = 2


# ------------------------------------------------------------------ data assembly
def player_table(client, slug: str, events: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Players in every finished event, with the football-data team name attached."""
    ids = dict(zip(team_map.espn_id, team_map.fd_name))
    frames = []
    for event_row in events[events.state == STATE_POST].itertuples(index=False):
        lu = parse_lineups(client.summary(slug, event_row.event_id), str(event_row.event_id))
        if not has_lineups(lu):
            continue
        frames.append(lu.assign(date=event_row.date, season=event_row.season))
    if not frames:
        return pd.DataFrame()
    player_df = pd.concat(frames, ignore_index=True)
    player_df["team"] = player_df.team_id.map(ids)
    return player_df.dropna(subset=["team"])


def team_features(hist: pd.DataFrame, xi: pd.DataFrame, n_hist: int = HISTORY_MATCHES,
                  decay: float = HISTORY_DECAY, shrink_90s: float = SHRINK_90S,
                  min_hist: int = MIN_HISTORY_MATCHES) -> dict | None:
    """hist: this team's player rows from earlier matches; xi: the 11 announced starters."""
    order = hist.drop_duplicates("event_id").sort_values("date", ascending=False).event_id.head(n_hist).tolist()
    if len(order) < min_hist:
        return None
    weights = {eid: decay ** k for k, eid in enumerate(order)}
    recent = hist[hist.event_id.isin(order)].assign(w=lambda frame: frame.event_id.map(weights))
    recent = recent.assign(wmin=recent.w * recent.minutes, wga=recent.w * recent.ga)
    agg = recent.groupby("athlete_id").agg(wmin=("wmin", "sum"), wga=("wga", "sum"),
                                      pos=("pos", lambda positions: positions.mode().iat[0]))
    team_wmin = sum(weights.values()) * STARTING_XI * MATCH_MINUTES
    share = agg.wmin / team_wmin
    top11 = share.nlargest(STARTING_XI).sum()
    ga90 = (agg.wga + shrink_90s * agg.pos.map(PRIOR_GA90)) / (agg.wmin / MATCH_MINUTES + shrink_90s)

    xi_ids = xi.athlete_id.tolist()
    miss = float(np.clip(1 - share.reindex(xi_ids).fillna(0).sum() / top11, 0, 1)) if top11 > 0 else 0.0
    xi_g = ga90.reindex(xi_ids)
    xi_g = xi_g.fillna(xi.set_index("athlete_id").pos.map(PRIOR_GA90))
    xi_att = float(xi_g.sum())
    starts = recent[recent.starter].assign(g=lambda frame: frame.athlete_id.map(ga90))
    usual = (starts.groupby("event_id").g.sum() * pd.Series(weights)).dropna().sum() / sum(weights.values())
    att = float(np.log(max(xi_att, MIN_ATTACK_OUTPUT) / max(usual, MIN_ATTACK_OUTPUT)))

    gk_min = recent[recent.pos == GOALKEEPER].groupby("athlete_id").wmin.sum()
    gk = 0.0
    if gk_min.sum() > 0:
        regular = gk_min.idxmax()
        if gk_min.max() / gk_min.sum() >= REGULAR_GK_SHARE and regular not in set(xi[xi.pos == GOALKEEPER].athlete_id):
            gk = 1.0
    return {"miss": miss, "att": att, "gk": gk, "n_hist": len(order)}


def match_features(players: pd.DataFrame, events: pd.DataFrame, team_map: pd.DataFrame,
                   lineups: dict | None = None) -> pd.DataFrame:
    """
    Features for each event. `lineups` maps event_id -> parsed lineup (for upcoming games);
    finished games use their own starters from `players`.
    """
    ids = dict(zip(team_map.espn_id, team_map.fd_name))
    by_team = {team_name: group for team_name, group in players.groupby("team")} if not players.empty else {}
    by_event = {key: group for key, group in players.groupby("event_id")} if not players.empty else {}
    rows = []
    for event_row in events.itertuples(index=False):
        home, away = ids.get(event_row.home_id), ids.get(event_row.away_id)
        if home is None or away is None:
            continue
        lu = lineups.get(event_row.event_id) if lineups is not None else by_event.get(event_row.event_id)
        if lu is None or lu.empty or not has_lineups(lu):
            continue
        features = {}
        for side, team, tid in (("h", home, event_row.home_id), ("a", away, event_row.away_id)):
            hist = by_team.get(team, pd.DataFrame())
            hist = hist[hist.date < event_row.date] if not hist.empty else hist
            xi = lu[(lu.team_id == tid) & lu.starter]
            team_feat = team_features(hist, xi) if not hist.empty else None
            if team_feat is None:
                break
            features.update({f"{side}_{key}": value for key, value in team_feat.items()})
        else:
            rows.append({"event_id": event_row.event_id, "date": event_row.date, "season": event_row.season,
                         "home": home, "away": away, **features})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ adjustment model
def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xh = df[HOME_DESIGN_COLS].to_numpy(float)
    Xa = df[AWAY_DESIGN_COLS].to_numpy(float)
    return Xh, Xa, np.log(df.xg_home.to_numpy(float)), np.log(df.xg_away.to_numpy(float))


def fit_coefs(train: pd.DataFrame, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    Xh, Xa, oh, oa = design(train)
    X = np.vstack([Xh, Xa])
    off = np.concatenate([oh, oa])
    goals = np.concatenate([train.hg.to_numpy(float), train.ag.to_numpy(float)])

    def objective(coefs):
        eta = off + X @ coefs
        mu = np.exp(eta)
        return np.sum(mu - goals * eta) + 0.5 * ridge * coefs @ coefs, X.T @ (mu - goals) + ridge * coefs

    return minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def multipliers(row, coefs: np.ndarray) -> tuple[float, float]:
    """
    Compute the home and away multipliers based on the lineup features and coefficients.

    Args:
        row: a row from the DataFrame containing the match and lineup features
        coefs: array of coefficients for the lineup features

    Returns:
        tuple of (home multiplier, away multiplier)
    """
    mh = np.exp(coefs @ np.array([getattr(row, col) for col in HOME_DESIGN_COLS], float))
    ma = np.exp(coefs @ np.array([getattr(row, col) for col in AWAY_DESIGN_COLS], float))
    return float(mh), float(ma)


def enhance(row, coefs: np.ndarray, max_goals: int = DEFAULT_MAX_GOALS) -> dict:
    mh, ma = multipliers(row, coefs)
    return summarize(row.xg_home * mh, row.xg_away * ma, row.rho, max_goals)


def walk_forward(feat: pd.DataFrame, base: pd.DataFrame, ridge: float = DEFAULT_RIDGE,
                 min_train: int = DEFAULT_MIN_TRAIN) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Refit coefficients before each match date on everything earlier; predict that date's games.

    Args:
        feat: DataFrame containing the lineup features for each match
        base: DataFrame containing the base match information
        ridge: ridge regularization parameter for fitting coefficients
        min_train: minimum number of training matches required to fit coefficients

    Returns:
        tuple of (DataFrame with predictions, final fitted coefficients)
    """
    df = feat.merge(base[[col_name for col_name in BASE_COLS if col_name in base]], on=["season", "home", "away"], how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    out, coefs = [], np.zeros(len(COEFS))
    for match_day, day_matches in df.groupby(df.date.dt.normalize()):
        train = df[df.date < match_day]
        if len(train) < min_train:
            continue
        coefs = fit_coefs(train, ridge)
        for row in day_matches.itertuples(index=False):
            pred = enhance(row, coefs)
            out.append({"event_id": row.event_id, **{f"lu_{key}": value for key, value in pred.items()}})
    ev = df.merge(pd.DataFrame(out), on="event_id", how="inner") if out else pd.DataFrame()
    final = fit_coefs(df, ridge) if len(df) >= min_train else coefs
    return ev, final


def compare(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Base vs lineup-enhanced on the same matches, overall and by season.

    goal_loglik is the average Poisson log-likelihood of both teams' goals (higher is better).
    It reacts to lineup information sooner than the W/D/L scores, which need many matches.
    """
    from scipy.stats import poisson
    rows = []
    for scope, subset in [("all", ev), *[(season_name, group) for season_name, group in ev.groupby("season")]]:
        Y = metrics.onehot(subset.result)
        for name, cols, xg in (("base model", metrics.PROB_COLS, ["xg_home", "xg_away"]),
                               ("with lineups", [f"lu_{prob_col}" for prob_col in metrics.PROB_COLS], ["lu_xg_home", "lu_xg_away"])):
            P = subset[cols].to_numpy()
            ll = (poisson.logpmf(subset.hg, subset[xg[0]]) + poisson.logpmf(subset.ag, subset[xg[1]])).mean()
            rows.append({"season": scope, "forecaster": name, "n": len(subset),
                         "rps": metrics.rps(P, Y).mean(), "log_loss": metrics.log_loss(P, Y).mean(),
                         "accuracy": (P.argmax(1) == Y.argmax(1)).mean(), "goal_loglik": ll})
    return pd.DataFrame(rows)


def save_coefs(path, coefs: np.ndarray, n_matches: int):
    path.write_text(json.dumps({"coefficients": dict(zip(COEFS, map(float, coefs))), "n_matches": n_matches}, indent=JSON_INDENT))


def load_coefs(path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array([json.loads(path.read_text())["coefficients"][key] for key in COEFS])
