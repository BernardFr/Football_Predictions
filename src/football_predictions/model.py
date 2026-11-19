"""
Time-weighted Dixon-Coles model with feeder divisions.

Expected goals:
    home: exp(base_scoring[div] + home_adv[div] + attack[home] - defence[away])
    away: exp(base_scoring[div]                 + attack[away] - defence[home])
Every team has one attack and one defence rating on a shared scale. Promoted and relegated
teams link the divisions, so a promoted side starts the season with a rating carried over from
its second-division matches, which sets an implicit "promotion penalty" fitted from the data.

Fitting is weighted Poisson maximum likelihood (weight = 0.5 ** (age / half_life)) with a ridge
penalty pulling ratings toward average. Then the Dixon-Coles rho, which corrects the
probabilities of 0-0, 1-0, 0-1 and 1-1, is fitted on top-division matches.
"""

from dataclasses import asdict, dataclass
from typing import Self, cast

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.stats import poisson

from .cups import CUP_DIV

DEFAULT_MAX_GOALS = 10            # score grid runs 0..DEFAULT_MAX_GOALS goals per side
DEFAULT_HALF_LIFE_DAYS = 180.0    # half-life for time decay in days
DEFAULT_RIDGE = 1.0               # ridge penalty for regularization
DEFAULT_FEEDER_WEIGHT = 1.0       # weight for feeder-division matches
DEFAULT_CUP_WEIGHT = 0.5          # weight for cup matches
DECAY_BASE = 0.5                  # weight = DECAY_BASE ** (age / half_life): halves every half-life
TOP_DIV_WEIGHT = 1.0
MIN_MATCH_WEIGHT = 1e-3           # matches weighted below this are dropped from the fit
MIN_FIT_MATCHES = 20
MIN_MEAN_GOALS = 0.1              # floor for the starting scoring level, so log() stays finite
INITIAL_HOME_ADV = 0.25           # starting home advantage on the log scale
OPTIMIZER = "L-BFGS-B"
OPTIMIZER_MAX_ITER = 2000
RHO_BOUNDS = (-0.25, 0.25)
RHO_OPTIMIZER = "bounded"
TAU_FLOOR = 1e-10                 # keeps the Dixon-Coles correction strictly positive
GD_INTERVAL_QUANTILES = (0.10, 0.90)   # 80% goal-difference interval
NO_ADJUSTMENT = 1.0               # expected-goals multiplier when `adjust` leaves a key out
AVERAGE_RATING = 0.0              # attack/defence of a league-average team
HIGHEST_PROB_SCORES_N = 3         # most likely scorelines reported, one column each
HIGHEST_PROB_SCORE_COLS = [f"highest_prob_score_{rank}" for rank in range(1, HIGHEST_PROB_SCORES_N + 1)]

@dataclass
class Hyper:
    """Hyperparameters for the model fitting process."""
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    ridge: float = DEFAULT_RIDGE
    feeder_weight: float = DEFAULT_FEEDER_WEIGHT
    cup_weight: float = DEFAULT_CUP_WEIGHT
    max_goals: int = DEFAULT_MAX_GOALS

    @classmethod
    def from_dict(cls, params: dict) -> "Hyper":
        return cls(**{key: value for key, value in params.items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Params:
    """
    Fitted model parameters; 
    to_vector()/from_vector() define the flat layout scipy's minimize needs.
    """
    attack: np.ndarray        # per team: attack rating
    defence: np.ndarray    # per team: defence rating
    base_scoring: np.ndarray   # per division: base scoring level (log scale)
    home_adv: np.ndarray   # per division: home advantage (log scale)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.attack, self.defence, self.base_scoring, self.home_adv])

    @classmethod
    def from_vector(cls, theta: np.ndarray, n_teams: int, n_divs: int) -> Self:
        """Split theta into views (no copies), so this is cheap inside the objective."""
        attack, defence, base_scoring, home_adv = np.split(theta, [n_teams, 2 * n_teams, 2 * n_teams + n_divs])
        return cls(attack, defence, base_scoring, home_adv)


@dataclass
class TrainingData:
    """
    Matches prepared for the optimizer: team/division lookups plus per-match and per-goal arrays.
    Each match contributes two goal rows: home goals (scored by home, conceded by away), then away goals.
    """
    team_lst: list[str]
    team_pos: dict[str, int]       # team name -> index into team_lst
    div_lst: list[str]
    div_pos: dict[str, int]        # division name -> index into div_lst
    home_idx: np.ndarray           # per match: home team index
    away_idx: np.ndarray           # per match: away team index
    div_idx: np.ndarray            # per match: division index
    goals: np.ndarray              # per goal row: goals scored
    scorer_idx: np.ndarray         # per goal row: scoring team index
    conceder_idx: np.ndarray       # per goal row: conceding team index
    goal_div_idx: np.ndarray       # per goal row: division index
    is_home: np.ndarray            # per goal row: 1.0 if scored by the home side at a non-neutral venue
    goal_weights: np.ndarray       # per goal row: match weight

    @classmethod
    def from_matches(cls, train: pd.DataFrame, weights: np.ndarray) -> Self:
        team_lst = sorted(set(train.home) | set(train.away))
        div_lst = sorted(set(train["div"]))
        team_pos = {team: i for i, team in enumerate(team_lst)}
        div_pos = {div_name: i for i, div_name in enumerate(div_lst)}
        home_idx = train.home.map(team_pos).to_numpy()
        away_idx = train.away.map(team_pos).to_numpy()
        div_idx = train["div"].map(div_pos).to_numpy()
        neutral = (train["neutral"].fillna(False).to_numpy(bool) if "neutral" in train else np.zeros(len(train), bool))
        return cls(
            team_lst=team_lst, team_pos=team_pos, div_lst=div_lst, div_pos=div_pos,
            home_idx=home_idx, away_idx=away_idx, div_idx=div_idx,
            goals=np.concatenate([train.hg.to_numpy(float), train.ag.to_numpy(float)]),
            scorer_idx=np.concatenate([home_idx, away_idx]),
            conceder_idx=np.concatenate([away_idx, home_idx]),
            goal_div_idx=np.concatenate([div_idx, div_idx]),
            is_home=np.concatenate([(~neutral).astype(float), np.zeros(len(train))]),   # no home advantage at neutral venues
            goal_weights=np.concatenate([weights, weights]),
        )

    @property
    def n_teams(self) -> int:
        return len(self.team_lst)

    @property
    def n_divs(self) -> int:
        return len(self.div_lst)


def _tau(x, y, lam, mu, rho) -> np.ndarray:
    """
    Dixon-Coles tau correction for low-scoring outcomes.
    • τ is the correction factor applied to the probability of low-scoring outcomes.
	• If x = 0, y = 0: τ = 1 - λ μ ρ
	• If x = 0, y = 1: τ = 1 - λ ρ (or variants depending on notation)
	• If x = 1, y = 0: τ = 1 - μ ρ
	• If x = 1, y = 1: τ = 1 + ρ
	• Otherwise: τ = 1

    • Role of ρ (Rho):
	• When ρ = 0, τ = 1, and the model collapses to a standard independent Poisson distribution.
	• A negative ρ (typically found in football data) increases the probability of 0-0 and 1-1 draws 
    while decreasing 1-0 and 0-1 outcomes, better aligning with real-world draw frequencies.


    Arguments:
        x: number of goals scored by the home team
        y: number of goals scored by the away team
        lam: expected goals for the home team
        mu: expected goals for the away team
        rho: Dixon-Coles correlation parameter
    Returns:
        tau correction factor for the given outcome
    """
    tau = np.ones_like(lam, dtype=float) # initialize tau with ones
    # np.where(condition, value_if_true, value_if_false) is used to apply the tau correction based on the outcome
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau) 
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return np.maximum(tau, TAU_FLOOR)


def compute_score_grid(lam: float, mu: float, rho: float, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """
    Compute the full score probability grid for given expected home and away goals and Dixon-Coles rho.
    Args:
        lam: expected goals for the home team
        mu: expected goals for the away team
        rho: Dixon-Coles correlation parameter
        max_goals: maximum number of goals to consider in the grid
    Returns:
        grid of score probabilities
    """
    goals = np.arange(max_goals + 1) # [0, 1, 2, ..., max_goals]
    # outer product of independent Poisson probabilities for expected home goals and expected away goals
    # i.e. grid_probs[i, j] = P(home goals = i) * P(away goals = j) for independent Poisson outcomes
    #  poisson.pmf(goals, lam)is the probability that the home team scores exactly 0, 1, 2, … goals in this match, given its expected goals lam (λ).
    grid_probs = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu)) # initial independent Poisson probabilities
    x, y = np.meshgrid(goals, goals, indexing="ij") # create coordinate matrices for the score grid
    # x = [[0, 1, 2, ..., max_goals],
    #      [0, 1, 2, ..., max_goals],
    #      ...]
    # y = [[0, 0, 0, ..., 0],
    #      [1, 1, 1, ..., 1],
    #      ...]

    # full_like: create arrays of the same shape as grid_probs filled with lam and mu respectively for the tau correction
    # apply Dixon-Coles tau correction to the independent Poisson probabilities
    grid_probs = grid_probs * _tau(x, y, np.full_like(grid_probs, lam), np.full_like(grid_probs, mu), rho)
    return grid_probs / grid_probs.sum()


def highest_prob_scorelines(grid_probs: np.ndarray, n: int = HIGHEST_PROB_SCORES_N) -> list[str]:
    """The n most likely scorelines with their probabilities, most likely first, e.g. ["1-1 12%", "1-0 11%", "2-0 10%"]."""
    flat_idx = np.argsort(grid_probs, axis=None)[::-1][:n]
    home_goals, away_goals = np.unravel_index(flat_idx, grid_probs.shape)
    return [f"{h}-{a} {grid_probs[h, a]:.0%}" for h, a in zip(home_goals, away_goals)]


def summarize(lam: float, mu: float, rho: float, max_goals: int = DEFAULT_MAX_GOALS) -> dict:
    """
    Expected goals -> W/D/L probabilities, pick, goal-difference forecast and interval.
    Args:
        lam: expected goals for the home team
        mu: expected goals for the away team
        rho: Dixon-Coles correlation parameter
        max_goals: maximum number of goals to consider in the grid
    Returns:
        dictionary with probabilities, pick, expected goal difference, and likely score
        - expected goals for home team
        - expected goals for away team
        - home win probability
        - draw probability
        - away win probability
        - most likely winner: Home, Draw, Away
        - Confidence of the most likely winner
        - expected goal difference
        - goal difference 80% confidence interval (lower bound, upper bound)
        - most likely score combination (home goals, away goals)
        - probability of the most likely score combination
        - the HIGHEST_PROB_SCORES_N most likely scorelines with their probabilities, one column each (text, for display)
        - 80% confidence interval for the most likely score combination (home goals, away goals)

    grid_probs contains the probability of each score combination (home goals (rows) x away goals (columns))
    home win probability is the sum of the lower triangular part of the grid (excluding the diagonal) ... use -1 to eliminate the diagonal
    away win probability is the sum of the upper triangular part of the grid (excluding the diagonal) ... use 1 to eliminate the diagonal
    diagonal represents draw probabilities

    gd contains the possible goal differences (home goals - away goals)
    0  -1  -2
    1   0  -1
    2   1   0

    gd_vals contains the possible goal differences (home goals - away goals) in sorted order: ..., -2, -1, 0, 1, 2, ...

    gd_p contains the probability distribution of the goal differences corresponding to gd_vals

    cdf contains the cumulative distribution function of the goal differences

    """
    grid_probs = compute_score_grid(lam, mu, rho, max_goals)
    grid_size = grid_probs.shape[0]
    p_h, p_d, p_a = np.tril(grid_probs, -1).sum(), np.trace(grid_probs), np.triu(grid_probs, 1).sum() # home, draw, away probabilities
    gd = np.subtract.outer(np.arange(grid_size), np.arange(grid_size)).ravel() # goal differences for all score combinations
    gd_vals = np.arange(-(grid_size - 1), grid_size) # possible goal differences
    gd_p = np.bincount(gd + grid_size - 1, weights=grid_probs.ravel(), minlength=2 * grid_size - 1) # probability distribution of goal differences
    cdf = np.cumsum(gd_p) # cumulative distribution function of goal differences
    lo = int(gd_vals[np.searchsorted(cdf, GD_INTERVAL_QUANTILES[0])]) # lower bound of goal difference interval
    hi = int(gd_vals[np.searchsorted(cdf, GD_INTERVAL_QUANTILES[1])]) # upper bound of goal difference interval
    # unravel transforms the flat index  into row and column indices (home and away goals)
    x, y = np.unravel_index(grid_probs.argmax(), grid_probs.shape) # get the row and column indices of the most likely score combination (home and away goals)
    probs = {"H": p_h, "D": p_d, "A": p_a}  # dictionary of outcome probabilities (home win, draw, away win)
    pick = max(probs, key=probs.__getitem__) # most likely outcome (H/D/A)
    return {
        "xg_home": float(lam), "xg_away": float(mu),
        "p_home": float(p_h), "p_draw": float(p_d), "p_away": float(p_a),
        "pick": pick, "confidence": float(probs[pick]),
        "exp_gd": float(gd_vals @ gd_p), "gd_lo80": lo, "gd_hi80": hi,
        "likely_score": f"{x}-{y}", "p_likely_score": float(grid_probs[x, y]),
        **dict(zip(HIGHEST_PROB_SCORE_COLS, highest_prob_scorelines(grid_probs))),
        "rho": float(rho),
    } # return summary dictionary


class DixonColes:
    """
    Dixon-Coles model for football match predictions.
    """
    def __init__(self, hyper: Hyper | None = None):
        self.hyper = hyper or Hyper() # set hyperparameters, use default if none provided
        self.fitted = False

    # ---------------------------------------------------------------- fit
    def fit(self, matches: pd.DataFrame, asof, top_div: str, warm: "DixonColes | None" = None) -> Self:
        """
        Fit the Dixon-Coles model to historical match data.

        Parameters:
        - matches: DataFrame containing historical match data with columns 'date', 'home', 'away', 'hg', 'ag', 'div'
        - asof: Timestamp to consider matches up to
        - top_div: Name of the top division
        - warm: Optional pre-fitted DixonColes model for warm start

        Returns:
        - self: Fitted DixonColes model instance
        """
        asof = pd.Timestamp(asof)
        hyper = self.hyper
        train = matches[matches.date < asof]
        age = (asof - train.date).dt.days.to_numpy(float)
        weights = DECAY_BASE ** (age / hyper.half_life_days) # apply exponential decay based on match age
        div_arr = train["div"].to_numpy()
        # adjust weights based on division importance
        # weight is TOP_DIV_WEIGHT for top division, hyper.cup_weight for cup division, hyper.feeder_weight for feeder divisions   
        weights = weights * np.where(div_arr == top_div, TOP_DIV_WEIGHT, np.where(div_arr == CUP_DIV, hyper.cup_weight, hyper.feeder_weight))
        keep = weights > MIN_MATCH_WEIGHT # filter out matches with very low weight
        train, weights = train[keep], weights[keep]
        if len(train) < MIN_FIT_MATCHES or not (train["div"] == top_div).any():
            raise ValueError("not enough matches to fit")

        data = TrainingData.from_matches(train, weights)
        n_teams, n_divs = data.n_teams, data.n_divs
        ridge = hyper.ridge # ridge regularization parameter for the model

        def objective(theta):
            """ Compute the Poisson regression loss and gradient for the given parameter vector."""
            params = Params.from_vector(theta, n_teams, n_divs)
            attack, defence, base_scoring, home_adv = params.attack, params.defence, params.base_scoring, params.home_adv
            eta = (base_scoring[data.goal_div_idx] + home_adv[data.goal_div_idx] * data.is_home
                   + attack[data.scorer_idx] - defence[data.conceder_idx])
            lam = np.exp(eta)
            resid = data.goal_weights * (lam - data.goals)
            loss = np.sum(data.goal_weights * (lam - data.goals * eta)) + 0.5 * ridge * (attack @ attack + defence @ defence)
            grad = np.concatenate([
                np.bincount(data.scorer_idx, resid, n_teams) + ridge * attack,
                -np.bincount(data.conceder_idx, resid, n_teams) + ridge * defence,
                np.bincount(data.goal_div_idx, resid, n_divs),
                np.bincount(data.goal_div_idx, resid * data.is_home, n_divs),
            ])
            return loss, grad

        init = Params(attack=np.zeros(n_teams), defence=np.zeros(n_teams),
                      base_scoring=np.full(n_divs, np.log(max(data.goals.mean(), MIN_MEAN_GOALS))),
                      home_adv=np.full(n_divs, INITIAL_HOME_ADV))
        if warm is not None and warm.fitted:   # start from the previous fit's ratings where they exist
            for team, i in data.team_pos.items():
                if team in warm.team_index:
                    j = warm.team_index[team]
                    init.attack[i], init.defence[i] = warm.params.attack[j], warm.params.defence[j]
            for div_name, i in data.div_pos.items():
                if div_name in warm.div_index:
                    j = warm.div_index[div_name]
                    init.base_scoring[i], init.home_adv[i] = warm.params.base_scoring[j], warm.params.home_adv[j]

        res = minimize(objective, init.to_vector(), jac=True, method=OPTIMIZER, options={"maxiter": OPTIMIZER_MAX_ITER})
        self.params = Params.from_vector(res.x, n_teams, n_divs)
        attack, defence, base_scoring, home_adv = self.params.attack, self.params.defence, self.params.base_scoring, self.params.home_adv
        home_idx, away_idx, div_idx = data.home_idx, data.away_idx, data.div_idx

        self.teams, self.team_index = data.team_lst, data.team_pos
        self.divs, self.div_index = data.div_lst, data.div_pos
        self.top_div, self.asof = top_div, asof
        lg = train[train["div"] != CUP_DIV]   # a team's division comes from its league games, not cup ties
        self.last_div = cast(dict[str, str], (pd.concat([lg[["date", "home", "div"]].rename(columns={"home": "team"}),
                                                         lg[["date", "away", "div"]].rename(columns={"away": "team"})])
                                              .sort_values("date").groupby("team")["div"].last().to_dict()))

        # Dixon-Coles low-score correction on top-division matches
        top = (train["div"] == top_div).to_numpy()
        lam_h = np.exp(base_scoring[div_idx] + home_adv[div_idx] + attack[home_idx] - defence[away_idx])[top]
        lam_a = np.exp(base_scoring[div_idx] + attack[away_idx] - defence[home_idx])[top]
        top_hg, top_ag, top_weights = train.hg.to_numpy()[top], train.ag.to_numpy()[top], weights[top]
        rho_fit = cast(OptimizeResult, minimize_scalar(lambda rho: -np.sum(top_weights * np.log(_tau(top_hg, top_ag, lam_h, lam_a, rho))),
                                                 bounds=RHO_BOUNDS, method=RHO_OPTIMIZER))
        self.rho = float(rho_fit.x)
        self.n_matches = int(len(train))
        self.converged = bool(res.success)
        self.fitted = True
        return self

    # ---------------------------------------------------------------- predict
    def _team(self, team: str) -> tuple[float, float]:
        if team in self.team_index:
            j = self.team_index[team]
            return self.params.attack[j], self.params.defence[j]
        # unseen team: assume a typical feeder-division side (or league average if no feeder data)
        feeders = [self.team_index[team_name] for team_name, div_name in self.last_div.items() if div_name != self.top_div]
        if feeders:
            return float(np.mean(self.params.attack[feeders])), float(np.mean(self.params.defence[feeders]))
        return AVERAGE_RATING, AVERAGE_RATING

    def expected_goals(self, home: str, away: str, div: str | None = None,
                       adjust: dict | None = None) -> tuple[float, float]:
        j = self.div_index[div or self.top_div]
        home_attack, home_def = self._team(home)
        away_attack, away_def = self._team(away)
        lam = np.exp(self.params.base_scoring[j] + self.params.home_adv[j] + home_attack - away_def)
        mu = np.exp(self.params.base_scoring[j] + away_attack - home_def)
        if adjust:   # optional multipliers, e.g. from lineup information
            lam *= adjust.get("home_attack", NO_ADJUSTMENT) * adjust.get("away_defence", NO_ADJUSTMENT)
            mu *= adjust.get("away_attack", NO_ADJUSTMENT) * adjust.get("home_defence", NO_ADJUSTMENT)
        return float(lam), float(mu)

    def score_grid(self, lam: float, mu: float) -> np.ndarray:
        return compute_score_grid(lam, mu, self.rho, self.hyper.max_goals)

    def predict(self, home: str, away: str, div: str | None = None, adjust: dict | None = None) -> dict:
        lam, mu = self.expected_goals(home, away, div, adjust)
        return {"home": home, "away": away, **summarize(lam, mu, self.rho, self.hyper.max_goals)}

    def ratings(self) -> pd.DataFrame:
        """Team ratings, with expected goal difference vs. an average top-division team at a neutral venue."""
        j = self.div_index[self.top_div]
        neutral_scoring = self.params.base_scoring[j] + self.params.home_adv[j] / 2   # half the home edge: neutral venue
        top_ix = [self.team_index[team_name] for team_name, div_name in self.last_div.items() if div_name == self.top_div]
        avg_attack, avg_def = np.mean(self.params.attack[top_ix]), np.mean(self.params.defence[top_ix])
        df = pd.DataFrame({
            "team": self.teams, "last_division": [self.last_div.get(team_name) for team_name in self.teams],
            "attack": self.params.attack, "defence": self.params.defence,
        })
        df["gf_vs_avg"] = np.exp(neutral_scoring + df.attack - avg_def)
        df["ga_vs_avg"] = np.exp(neutral_scoring + avg_attack - df.defence)
        df["gd_vs_avg"] = df.gf_vs_avg - df.ga_vs_avg
        return df.sort_values("gd_vs_avg", ascending=False).reset_index(drop=True)
