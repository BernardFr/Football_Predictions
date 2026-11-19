#!/usr/bin/env python3
"""
Dixon-Coles model for football match score prediction.
Reference: numpy minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
SLSQP (Sequential Least Squares Programming) is used for constrained optimization in this model.
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Match data column names
DATE_COL = "Date"
HOME_TEAM_COL = "HomeTeam"
AWAY_TEAM_COL = "AwayTeam"
HOME_GOALS_COL = "FTHG"
AWAY_GOALS_COL = "FTAG"
OUTCOME_IDX_COL = "OutcomeIdx"

# Match outcome indices (order used by predict_1x2 and RPS)
HOME_WIN_IDX = 0
DRAW_IDX = 1
AWAY_WIN_IDX = 2
N_OUTCOMES = 3

# Model defaults
DEFAULT_XI = 0.002  # Exponential time-decay rate (per day)
DEFAULT_MAX_GOALS = 7  # Max goals per team in the scoreline matrix
DC_ADJUSTED_GOALS = 2  # Dixon-Coles tau adjusts scorelines with 0 or 1 goals per team

# Optimizer settings
INIT_GAMMA = 0.25  # Initial guess for home advantage
INIT_RHO = -0.05  # Initial guess for low-score dependence
TEAM_PARAM_BOUNDS = (-3.0, 3.0)  # Bounds for alpha (attack) and beta (defense)
GAMMA_BOUNDS = (-1.0, 1.0)
RHO_BOUNDS = (-0.99, 0.99)
OPTIMIZER_METHOD = "SLSQP"  # Sequential Least Squares Programming for constrained optimization
OPTIMIZER_MAX_ITER = 500
OPTIMIZER_FTOL = 1e-8
TAU_MIN = 1e-10  # tau at or below this makes the likelihood invalid
INVALID_LIKELIHOOD_PENALTY = 1e10  # Returned when parameters produce an invalid tau

# Walk-forward cross-validation defaults
DEFAULT_MIN_TRAIN_MATCHES = 50
DEFAULT_STEP_DAYS = 7

# Demo settings
DEMO_SEED = 42
DEMO_TEAM_LST = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham", "Newcastle"]
DEMO_START_DATE = "2025-08-01"
DEMO_N_MATCHES = 180
DEMO_MATCH_FREQ = "2D"
DEMO_HOME_GOAL_RATE = 1.55
DEMO_AWAY_GOAL_RATE = 1.15
DEMO_XI_GRID = [0.000, 0.001, 0.002, 0.003, 0.005]
DEMO_MIN_TRAIN_MATCHES = 60
DEMO_STEP_DAYS = 14
DEMO_HOME_TEAM = "Arsenal"
DEMO_AWAY_TEAM = "Liverpool"


def tau_adjustment(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Computes the Dixon-Coles low-score dependence adjustment factor tau."""
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lam * rho)
    elif x == 1 and y == 0:
        return 1.0 + (mu * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def dc_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Negative weighted pseudo-log-likelihood for the Dixon-Coles model."""
    alpha = params[:n_teams]
    beta = params[n_teams : 2 * n_teams]
    gamma = params[-2]
    rho = params[-1]

    lam = np.exp(alpha[home_idx] + beta[away_idx] + gamma)
    mu = np.exp(alpha[away_idx] + beta[home_idx])

    tau = np.ones_like(lam)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    tau[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
    tau[mask_01] = 1.0 + lam[mask_01] * rho
    tau[mask_10] = 1.0 + mu[mask_10] * rho
    tau[mask_11] = 1.0 - rho

    if np.any(tau <= TAU_MIN):
        return INVALID_LIKELIHOOD_PENALTY

    log_lik = (
        poisson.logpmf(home_goals, lam)
        + poisson.logpmf(away_goals, mu)
        + np.log(tau)
    )
    return float(-np.sum(weights * log_lik))


def fit_dixon_coles(
    df: pd.DataFrame,
    xi: float = DEFAULT_XI,
    ref_date: str | pd.Timestamp | None = None,
    teams: list[str] | None = None,
) -> dict:
    """
    Fits Dixon-Coles parameters using MLE with exponential time decay.
    Arguments:
        df: DataFrame containing match data with columns "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG".
        xi: Exponential decay factor for weighting matches by recency.
        ref_date: Reference date for calculating time decay. Defaults to the latest date in the data.
        teams: List of team names to include in the model. If None, all teams in the data are used.
    Returns:
        Dictionary containing fitted parameters and optimization success flag.
    """
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    ref_dt = data[DATE_COL].max() if ref_date is None else pd.to_datetime(ref_date)

    days_elapsed = (ref_dt - data[DATE_COL]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * np.maximum(days_elapsed, 0))

    if teams is None:
        teams = sorted(list(set(data[HOME_TEAM_COL]).union(set(data[AWAY_TEAM_COL]))))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = data[HOME_TEAM_COL].map(team_to_idx).values
    away_idx = data[AWAY_TEAM_COL].map(team_to_idx).values
    home_goals = data[HOME_GOALS_COL].values
    away_goals = data[AWAY_GOALS_COL].values

    init_params = np.zeros(2 * n_teams + 2)
    init_params[-2] = INIT_GAMMA
    init_params[-1] = INIT_RHO

    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n_teams])}]  # Sum of alpha parameters should be zero for identifiability
    bounds = [TEAM_PARAM_BOUNDS] * (2 * n_teams) + [GAMMA_BOUNDS, RHO_BOUNDS]

    result = minimize(
        dc_log_likelihood,
        init_params,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method=OPTIMIZER_METHOD,
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": OPTIMIZER_MAX_ITER, "ftol": OPTIMIZER_FTOL},
    )

    params = result.x
    alpha_map = dict(zip(teams, params[:n_teams]))
    beta_map = dict(zip(teams, params[n_teams : 2 * n_teams]))

    return {
        "teams": teams,
        "alpha": alpha_map,
        "beta": beta_map,
        "gamma": float(params[-2]),
        "rho": float(params[-1]),
        "success": bool(result.success),
    }


def predict_score_matrix(
    model: dict, home_team: str, away_team: str, max_goals: int = DEFAULT_MAX_GOALS
) -> tuple[float, float, np.ndarray]:
    """Predicts expected goals (lambda, mu) and normalized exact scoreline probabilities."""
    alpha_h = model["alpha"].get(home_team, 0.0)
    beta_h = model["beta"].get(home_team, 0.0)
    alpha_a = model["alpha"].get(away_team, 0.0)
    beta_a = model["beta"].get(away_team, 0.0)
    gamma = model["gamma"]
    rho = model["rho"]

    lam = float(np.exp(alpha_h + beta_a + gamma))
    mu = float(np.exp(alpha_a + beta_h))

    home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_probs, away_probs)

    for x in range(DC_ADJUSTED_GOALS):
        for y in range(DC_ADJUSTED_GOALS):
            matrix[x, y] *= tau_adjustment(x, y, lam, mu, rho)

    matrix /= np.sum(matrix)
    return lam, mu, matrix


def predict_1x2(
    model: dict, home_team: str, away_team: str, max_goals: int = DEFAULT_MAX_GOALS
) -> np.ndarray:
    """Returns [P(Home Win), P(Draw), P(Away Win)] probabilities."""
    _, _, matrix = predict_score_matrix(model, home_team, away_team, max_goals=max_goals)
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.trace(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    return np.array([p_home, p_draw, p_away])


def ranked_probability_score(probs: np.ndarray, outcome_idx: int) -> float:
    """
    Computes Ranked Probability Score (RPS) for a 3-way match outcome (0=H, 1=D, 2=A).
    Lower RPS indicates better calibrated probabilistic predictions.
    """
    actual = np.zeros(N_OUTCOMES)
    actual[outcome_idx] = 1.0
    cum_probs = np.cumsum(probs)
    cum_actual = np.cumsum(actual)
    return float(0.5 * np.sum((cum_probs[:-1] - cum_actual[:-1]) ** 2))


def tune_xi_grid_search(
    df: pd.DataFrame,
    xi_grid: list[float] | np.ndarray,
    min_train_matches: int = DEFAULT_MIN_TRAIN_MATCHES,
    step_days: int = DEFAULT_STEP_DAYS,
) -> pd.DataFrame:
    """Walk-forward time-series cross-validation to find the optimal decay parameter xi."""
    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    data = data.sort_values(DATE_COL).reset_index(drop=True)

    all_teams = sorted(list(set(data[HOME_TEAM_COL]).union(set(data[AWAY_TEAM_COL]))))
    outcomes = np.where(
        data[HOME_GOALS_COL] > data[AWAY_GOALS_COL],
        HOME_WIN_IDX,
        np.where(data[HOME_GOALS_COL] == data[AWAY_GOALS_COL], DRAW_IDX, AWAY_WIN_IDX),
    )
    data[OUTCOME_IDX_COL] = outcomes

    start_date = pd.Timestamp(data[DATE_COL].iloc[min_train_matches - 1])
    max_date = pd.Timestamp(data[DATE_COL].max())

    results = []
    for xi in xi_grid:
        rps_scores = []
        current_date = start_date

        while current_date < max_date:
            next_date = current_date + pd.Timedelta(days=step_days)
            train_df = data[data[DATE_COL] <= current_date]
            val_df = data[(data[DATE_COL] > current_date) & (data[DATE_COL] <= next_date)]

            if len(val_df) > 0:
                model = fit_dixon_coles(
                    train_df, xi=float(xi), ref_date=current_date, teams=all_teams
                )
                for _, row in val_df.iterrows():
                    probs = predict_1x2(model, row[HOME_TEAM_COL], row[AWAY_TEAM_COL])
                    rps = ranked_probability_score(probs, int(row[OUTCOME_IDX_COL]))
                    rps_scores.append(rps)

            current_date = next_date

        mean_rps = float(np.mean(rps_scores)) if rps_scores else np.nan
        results.append({"xi": float(xi), "mean_rps": mean_rps, "n_eval_matches": len(rps_scores)})

    return pd.DataFrame(results).sort_values("mean_rps").reset_index(drop=True)


if __name__ == "__main__":
    # Generate synthetic league data to demonstrate the full pipeline
    rng = np.random.default_rng(DEMO_SEED)
    dates = pd.date_range(DEMO_START_DATE, periods=DEMO_N_MATCHES, freq=DEMO_MATCH_FREQ)

    records = []
    for d in dates:
        h, a = rng.choice(DEMO_TEAM_LST, size=2, replace=False)
        records.append({
            DATE_COL: d,
            HOME_TEAM_COL: h,
            AWAY_TEAM_COL: a,
            HOME_GOALS_COL: int(rng.poisson(DEMO_HOME_GOAL_RATE)),
            AWAY_GOALS_COL: int(rng.poisson(DEMO_AWAY_GOAL_RATE)),
        })
    match_df = pd.DataFrame(records)

    # 1. Grid search for optimal time-decay xi using walk-forward RPS
    cv_results = tune_xi_grid_search(
        match_df,
        xi_grid=DEMO_XI_GRID,
        min_train_matches=DEMO_MIN_TRAIN_MATCHES,
        step_days=DEMO_STEP_DAYS,
    )
    print("--- Walk-Forward Cross-Validation (Ranked Probability Score) ---")
    print(cv_results.to_string(index=False))

    # 2. Fit final model with the best xi
    best_xi = float(cv_results["xi"].iloc[0])
    final_model = fit_dixon_coles(match_df, xi=best_xi)
    print(f"\nBest xi: {best_xi} | Gamma: {final_model['gamma']:.4f} | Rho: {final_model['rho']:.4f}")

    # 3. Predict an upcoming match
    lam, mu, score_mat = predict_score_matrix(final_model, DEMO_HOME_TEAM, DEMO_AWAY_TEAM)
    probs_1x2 = predict_1x2(final_model, DEMO_HOME_TEAM, DEMO_AWAY_TEAM)
    print(f"\n{DEMO_HOME_TEAM} vs {DEMO_AWAY_TEAM} -> Expected Goals: {lam:.2f} - {mu:.2f}")
    print(f"1X2 Probabilities -> Home: {probs_1x2[0]:.1%}, Draw: {probs_1x2[1]:.1%}, Away: {probs_1x2[2]:.1%}")
