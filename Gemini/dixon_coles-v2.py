import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-scoring adjustment factor."""
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


def tau_vec(x, y, lam, mu, rho):
    """Vectorized version of tau for fast MLE optimization."""
    res = np.ones_like(lam, dtype=float)
    mask_00 = (x == 0) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_10 = (x == 1) & (y == 0)
    mask_11 = (x == 1) & (y == 1)

    res[mask_00] = 1.0 - (lam[mask_00] * mu[mask_00] * rho)
    res[mask_01] = 1.0 + (lam[mask_01] * rho)
    res[mask_10] = 1.0 + (mu[mask_10] * rho)
    res[mask_11] = 1.0 - rho
    return res


def fit_dixon_coles(df, xi=0.0018):
    """
    Fits Dixon-Coles parameters on a DataFrame with columns:
    ['home_team', 'away_team', 'home_goals', 'away_goals', 'days_ago']
    """
    teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    n_teams = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}

    home_idx = df['home_team'].map(team_to_idx).values
    away_idx = df['away_team'].map(team_to_idx).values
    x = df['home_goals'].values
    y = df['away_goals'].values
    weights = np.exp(-xi * df['days_ago'].values)

    def neg_log_likelihood(params):
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        gamma = params[-2]
        rho = params[-1]

        lam = np.exp(attack[home_idx] + defense[away_idx] + gamma)
        mu = np.exp(attack[away_idx] + defense[home_idx])

        tau_vals = tau_vec(x, y, lam, mu, rho)
        if np.any(tau_vals <= 0):
            return 1e10

        log_lik = weights * (
            np.log(tau_vals) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
        )
        return -np.sum(log_lik)

    init_params = np.zeros(2 * n_teams + 2)
    init_params[-2] = 0.25
    init_params[-1] = -0.05

    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:n_teams])}]
    bounds = [(-3, 3)] * (2 * n_teams) + [(0, 1.5), (-0.99, 0.99)]

    result = minimize(
        neg_log_likelihood,
        init_params,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    params = result.x
    return {
        'teams': teams,
        'attack': dict(zip(teams, params[:n_teams])),
        'defense': dict(zip(teams, params[n_teams : 2 * n_teams])),
        'gamma': params[-2],
        'rho': params[-1],
    }


def predict_match(model, home_team, away_team, max_goals=6):
    """Generates a scoreline probability matrix and 1X2 odds."""
    lam = np.exp(
        model['attack'][home_team]
        + model['defense'][away_team]
        + model['gamma']
    )
    mu = np.exp(model['attack'][away_team] + model['defense'][home_team])
    rho = model['rho']

    home_probs = poisson.pmf(np.arange(max_goals + 1), lam)
    away_probs = poisson.pmf(np.arange(max_goals + 1), mu)
    matrix = np.outer(home_probs, away_probs)

    for i in range(2):
        for j in range(2):
            matrix[i, j] *= tau(i, j, lam, mu, rho)

    matrix /= np.sum(matrix)

    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))

    return matrix, {'Home Win': home_win, 'Draw': draw, 'Away Win': away_win}


def ranked_probability_score(probs, outcome):
    """
    Computes the Ranked Probability Score (RPS) for a single 3-outcome match.
    probs: tuple or list of (p_home, p_draw, p_away)
    outcome: 'H' (Home Win), 'D' (Draw), or 'A' (Away Win)
    Lower RPS indicates better calibration (0 is perfect, 1 is worst).
    """
    if outcome == 'H':
        actual = [1.0, 0.0, 0.0]
    elif outcome == 'D':
        actual = [0.0, 1.0, 0.0]
    elif outcome == 'A':
        actual = [0.0, 0.0, 1.0]
    else:
        raise ValueError("outcome must be 'H', 'D', or 'A'")

    cum_pred = np.cumsum(probs)
    cum_actual = np.cumsum(actual)

    return 0.5 * np.sum((cum_pred[:-1] - cum_actual[:-1]) ** 2)


def backtest_xi(df, xi_grid, min_train_days=30, step_days=7):
    """
    Runs a rolling-window backtest over historical matches to tune the time-decay rate xi.
    df must contain: ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
    where 'date' is a pandas datetime column.
    """
    df = df.sort_values('date').reset_index(drop=True)
    min_date = df['date'].min()
    max_date = df['date'].max()

    results = {}

    for xi in xi_grid:
        rps_scores = []
        current_cutoff = min_date + pd.Timedelta(days=min_train_days)

        while current_cutoff < max_date:
            next_cutoff = current_cutoff + pd.Timedelta(days=step_days)

            train_df = df[df['date'] < current_cutoff].copy()
            test_df = df[(df['date'] >= current_cutoff) & (df['date'] < next_cutoff)].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                train_df['days_ago'] = (current_cutoff - train_df['date']).dt.days
                model = fit_dixon_coles(train_df, xi=xi)

                for _, row in test_df.iterrows():
                    h_team = row['home_team']
                    a_team = row['away_team']

                    # Skip matches involving teams not yet seen in training window
                    if h_team not in model['teams'] or a_team not in model['teams']:
                        continue

                    _, odds = predict_match(model, h_team, a_team)
                    probs = [odds['Home Win'], odds['Draw'], odds['Away Win']]

                    if row['home_goals'] > row['away_goals']:
                        actual = 'H'
                    elif row['home_goals'] == row['away_goals']:
                        actual = 'D'
                    else:
                        actual = 'A'

                    rps_scores.append(ranked_probability_score(probs, actual))

            current_cutoff = next_cutoff

        results[xi] = float(np.mean(rps_scores)) if rps_scores else np.nan

    best_xi = min(results, key=lambda k: results[k] if not np.isnan(results[k]) else float('inf'))
    return best_xi, results


if __name__ == '__main__':
    # Generate a synthetic multi-round season dataset for demonstration
    rng = np.random.default_rng(42)
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City']
    true_att = {'Arsenal': 0.35, 'Chelsea': 0.05, 'Liverpool': 0.30, 'Man City': 0.45}
    true_def = {'Arsenal': -0.25, 'Chelsea': 0.05, 'Liverpool': -0.20, 'Man City': -0.35}

    records = []
    start_date = pd.Timestamp('2025-08-15')
    for week in range(16):
        match_date = start_date + pd.Timedelta(days=week * 7)
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i != j and (i + j + week) % 2 == 0:
                    h, a = teams[i], teams[j]
                    lam = np.exp(true_att[h] + true_def[a] + 0.25)
                    mu = np.exp(true_att[a] + true_def[h])
                    records.append({
                        'date': match_date,
                        'home_team': h,
                        'away_team': a,
                        'home_goals': int(rng.poisson(lam)),
                        'away_goals': int(rng.poisson(mu)),
                    })

    season_df = pd.DataFrame(records)

    # Tune xi via rolling backtest
    candidate_xis = [0.0, 0.001, 0.0018, 0.003, 0.005]
    best_xi, rps_summary = backtest_xi(season_df, candidate_xis, min_train_days=28, step_days=7)

    print('--- Rolling Backtest RPS Results ---')
    for xi_val, mean_rps in rps_summary.items():
        print(f'xi = {xi_val:.4f} -> Mean RPS: {mean_rps:.4f}')
    print(f'Optimal xi: {best_xi:.4f}\n')

    # Fit final model on full dataset using optimal xi
    season_df['days_ago'] = (season_df['date'].max() - season_df['date']).dt.days
    final_model = fit_dixon_coles(season_df, xi=best_xi)
    _, match_odds = predict_match(final_model, 'Arsenal', 'Chelsea')
    print('Arsenal vs Chelsea Predicted Odds:', {k: round(v, 3) for k, v in match_odds.items()})
