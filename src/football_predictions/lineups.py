"""
Lineup-enhanced predictions.

For each team in a match, compare the announced starting XI with the team's recent usage, using
only matches played before kickoff:

  miss  share of the team's usual playing time missing from the XI (0 = full strength)
  att   log(attacking output of the XI / output of its usual XIs); goals + assists per 90,
        shrunk toward a position prior so one lucky goal doesn't make a player a star
  gk    log(goals-allowed rate of the XI keeper / goals-allowed rate of the usual keepers).
        Goals-allowed rate = 1 - save %, with save % shrunk toward the league average, so a few
        matches don't make a keeper great. 0 = the usual keeper; > 0 = a weaker keeper starts.

The base model's expected goals are then scaled:
  goals_for ~ Poisson(xg_base * exp(b1*miss_own + b2*att_own + b3*miss_opp + b4*gk_opp))
The four b's are fitted (ridge-shrunk toward 0, i.e. toward "no effect") on past matches, using
the base model's walk-forward expected goals as the offset. Each b is held to its sensible sign
(COEF_BOUNDS): a missing regular cannot help the team, and a weaker keeper cannot concede fewer goals. The evaluation is walk-forward too.

The history of a team keeps only its current players: the players with minutes for the team in
the season of the match, before the match, plus the announced XI. A player who left the club does
not count as missing. Before the team's first match of the season, the full history is kept.

Player impact: drop one player from the team's usual XI (the 11 players with the most recent
playing time), start the best backup at the same position instead, and push the change in
features through the fitted b's into the team's average expected goals for and against.
The result is the drop in expected points (3 x P(win) + P(draw)) and in expected goal
difference per match. Players outside the usual XI have no impact on the model.
"""

import json
from typing import cast

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
PRIOR_SAVE_PCT = 0.67           # league save % (EPL 2025-27: 0.672)
SHRINK_SHOTS = 40.0             # save % prior strength, in shots on target (about 10 matches)
MIN_ALLOWED_RATE = 0.05         # floor on 1 - save %, so the log ratio stays finite
COEF_BOUNDS = [(None, 0.0), (0.0, None), (0.0, None), (0.0, None)]   # COEFS order: miss_own <= 0, others >= 0
DEFAULT_RIDGE = 5.0
DEFAULT_MIN_TRAIN = 150         # matches needed before the coefficients are fitted
PLACEHOLDER_ID = "__backup__"   # an unknown backup: no playing time, position-prior output
WIN_POINTS = 3
DRAW_POINTS = 1
IMPACT_COLS = ["team", "athlete_id", "player", "pos", "share", "in_xi", "impact_pts", "impact_gd"]
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
    # mean per starter x 11: the usual XI output stays comparable when departed starters are filtered out
    usual = (starts.groupby("event_id").g.mean() * STARTING_XI * pd.Series(weights)).dropna().sum() / sum(weights.values())
    att = float(np.log(max(xi_att, MIN_ATTACK_OUTPUT) / max(usual, MIN_ATTACK_OUTPUT)))

    gk = keeper_feature(recent, xi)
    return {"miss": miss, "att": att, "gk": gk, "n_hist": len(order)}


def allowed_rate(keepers: pd.DataFrame, shrink_shots: float = SHRINK_SHOTS) -> pd.Series:
    """Per keeper (index athlete_id): 1 - save %, with save % shrunk toward PRIOR_SAVE_PCT."""
    save_pct = (keepers.saves + shrink_shots * PRIOR_SAVE_PCT) / (keepers.saves + keepers.goals_conceded + shrink_shots)
    return (1 - save_pct).clip(lower=MIN_ALLOWED_RATE)


def keeper_feature(recent: pd.DataFrame, xi: pd.DataFrame) -> float:
    """
    log(goals-allowed rate of the XI keeper / minutes-weighted rate of the recent keepers).
    recent: the team's weighted recent rows (column wmin); xi: the announced starters.
    """
    keepers = recent[recent.pos == GOALKEEPER].groupby("athlete_id").agg(
        wmin=("wmin", "sum"), saves=("saves", "sum"), goals_conceded=("goals_conceded", "sum"))
    keepers = keepers[keepers.wmin > 0]
    xi_keepers = xi[xi.pos == GOALKEEPER].athlete_id
    if keepers.empty or xi_keepers.empty:
        return 0.0
    rate = allowed_rate(keepers)
    usual = float((rate * keepers.wmin).sum() / keepers.wmin.sum())
    xi_rate = float(rate.reindex(xi_keepers).fillna(1 - PRIOR_SAVE_PCT).mean())
    return float(np.log(xi_rate / usual))


def current_players(hist: pd.DataFrame, season: str, keep_ids=()) -> pd.DataFrame:
    """
    Rows of `hist` (one team's earlier matches) for players with minutes for the team in `season`,
    plus the players in keep_ids. If the team has not played in `season` yet, return all of `hist`.
    """
    played = hist[(hist.season == season) & (hist.minutes > 0)].athlete_id
    if played.empty:
        return hist
    return hist[hist.athlete_id.isin(set(played) | set(keep_ids))]


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
            xi = lu[(lu.team_id == tid) & lu.starter]
            if not hist.empty:
                hist = current_players(hist[hist.date < event_row.date], str(event_row.season), xi.athlete_id)
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

    return minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", bounds=COEF_BOUNDS).x


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


# ------------------------------------------------------------------ player impact
def usual_xi(hist: pd.DataFrame, n_hist: int = HISTORY_MATCHES, decay: float = HISTORY_DECAY) -> pd.DataFrame:
    """
    Players in the team's last n_hist matches, ranked by decay-weighted minutes.
    Columns: athlete_id, player, pos, share (of the team's weighted minutes), in_xi (top STARTING_XI).
    """
    order = hist.drop_duplicates("event_id").sort_values("date", ascending=False).event_id.head(n_hist).tolist()
    weights = {eid: decay ** k for k, eid in enumerate(order)}
    recent = hist[hist.event_id.isin(order)]
    recent = recent.assign(wmin=recent.event_id.map(weights) * recent.minutes)
    latest = recent.sort_values("date").groupby("athlete_id").last()
    usage = (recent.groupby("athlete_id").wmin.sum() / (sum(weights.values()) * STARTING_XI * MATCH_MINUTES)).rename("share")
    usage = usage[usage > 0].sort_values(ascending=False).reset_index()
    usage["player"] = usage.athlete_id.map(latest.player)
    usage["pos"] = usage.athlete_id.map(recent.groupby("athlete_id").pos.agg(lambda positions: positions.mode().iat[0]))
    usage["in_xi"] = usage.index < STARTING_XI
    return usage


def team_xg(base: pd.DataFrame, season: str) -> pd.DataFrame:
    """Each team's average expected goals for and against in `season`, and the average rho, from base predictions."""
    games = base[base.season == season]
    sides = pd.concat([games[["home", "xg_home", "xg_away"]].set_axis(["team", "xg_for", "xg_against"], axis=1),
                       games[["away", "xg_away", "xg_home"]].set_axis(["team", "xg_for", "xg_against"], axis=1)])
    return sides.groupby("team").mean().assign(rho=games.rho.mean())


def _expected_points(xg_for: float, xg_against: float, rho: float) -> float:
    probs = summarize(xg_for, xg_against, rho)
    return WIN_POINTS * probs["p_home"] + DRAW_POINTS * probs["p_draw"]


def team_player_impact(hist: pd.DataFrame, coefs: np.ndarray, xg: pd.Series) -> pd.DataFrame:
    """
    Impact of each player in the team's recent matches: the drop in expected points and goal
    difference per match when the player misses the usual XI and the best same-position backup
    starts instead. xg holds the team's average xg_for, xg_against and rho.
    """
    usage = usual_xi(hist)
    xi = usage[usage.in_xi]
    base = team_features(hist, xi)
    if base is None:
        return usage.assign(impact_pts=np.nan, impact_gd=np.nan)
    miss_own, att_own, miss_opp, gk_opp = coefs
    base_pts = _expected_points(xg.xg_for, xg.xg_against, xg.rho)
    impact_pts, impact_gd = {}, {}
    bench = usage[~usage.in_xi]
    for idx, row in xi.iterrows():
        backups = bench[bench.pos == row.pos]
        backup = backups.head(1) if not backups.empty else pd.DataFrame({"athlete_id": [PLACEHOLDER_ID], "pos": [row.pos]})
        without = team_features(hist, pd.concat([xi.drop(idx), backup])[["athlete_id", "pos"]])
        if without is None:          # same history as `base`, so this cannot happen; keeps the type checker sure
            continue
        delta = {feature: without[feature] - base[feature] for feature in FEATURES}
        xg_for = xg.xg_for * np.exp(miss_own * delta["miss"] + att_own * delta["att"])
        xg_against = xg.xg_against * np.exp(miss_opp * delta["miss"] + gk_opp * delta["gk"])
        impact_pts[idx] = base_pts - _expected_points(xg_for, xg_against, xg.rho)
        impact_gd[idx] = (xg.xg_for - xg.xg_against) - (xg_for - xg_against)
    return usage.assign(impact_pts=pd.Series(impact_pts, dtype=float).reindex(usage.index, fill_value=0.0),
                        impact_gd=pd.Series(impact_gd, dtype=float).reindex(usage.index, fill_value=0.0))


def player_impact(players: pd.DataFrame, coefs: np.ndarray, xg_table: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Player impact for every team that played in `season`, as of its latest match. Only players
    with minutes for the team in `season` are included.
    xg_table: team_xg() output; teams missing from it are skipped.
    """
    current = set(players[players.season == season].team) & set(xg_table.index)
    frames = [team_player_impact(current_players(group, season), coefs, cast(pd.Series, xg_table.loc[team])).assign(team=team)
              for team, group in players[players.team.isin(current)].groupby("team")]
    if not frames:
        return pd.DataFrame(columns=IMPACT_COLS)
    return pd.concat(frames, ignore_index=True)[IMPACT_COLS]


def save_coefs(path, coefs: np.ndarray, n_matches: int):
    path.write_text(json.dumps({"coefficients": dict(zip(COEFS, map(float, coefs))), "n_matches": n_matches}, indent=JSON_INDENT))


def load_coefs(path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array([json.loads(path.read_text())["coefficients"][key] for key in COEFS])
