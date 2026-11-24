"""
The grade of a prediction: how much better its score probabilities did than a baseline, once the result is known.

The baseline is the league's own scoreline frequencies (e.g. the previous seasons). A prediction is graded on
three parts. Each part is the ranked probability score (RPS) of an ordered distribution, so probability placed
near the actual outcome earns credit, not only probability placed on it:

    result           home win > draw > away win                        weight 50%
    goal difference  -max_goals .. +max_goals, home minus away         weight 25%
    goals            home goals and away goals, 0 .. max_goals each    weight 25%

Skill of a part = 1 - RPS_model / RPS_baseline: 0 = no better than the baseline, 1 = certain and right,
below 0 = worse than the baseline. The grade is the weighted skill, x100 (e.g. +25).
A group of games (a matchweek, a team's games, the season) adds up the RPS over its games before the ratio, so one
odd game does not dominate it.

Matchweeks are not in the data, so assign_matchweeks() infers them: a new matchweek starts after MATCHWEEK_GAP_DAYS or more
days with no games, or on a date when a team would play a second time in the matchweek.
"""

import numpy as np
import pandas as pd

from .model import DEFAULT_MAX_GOALS, compute_score_grid

PART_WEIGHTS = {"result": 0.50, "goal_diff": 0.25, "goals": 0.25}
GRADE_SCALE = 100
MODEL_PREFIX, BASE_PREFIX, SKILL_PREFIX = "rps_", "base_rps_", "skill_"
GRADE_COL = "grade"
MATCHWEEK_GAP_DAYS = 2              # this many days or more between game dates start a new matchweek
PART_LABELS = {"result": "result skill", "goal_diff": "goal diff skill", "goals": "goals skill"}
DRAW_LABEL = "Draw"


def ordinal_rps(probs: np.ndarray, actual_index: int) -> float:
    """RPS of one ordered distribution (unnormalized: only ratios of RPS are used)."""
    observed = (np.arange(len(probs)) >= actual_index).astype(float)
    return float(((np.cumsum(probs) - observed) ** 2).sum())


def part_rps(grid: np.ndarray, home_goals: int, away_goals: int) -> dict[str, float]:
    """RPS per part of a score grid (rows = home goals, columns = away goals) against the actual score."""
    max_goals = grid.shape[0] - 1
    home_goals, away_goals = min(int(home_goals), max_goals), min(int(away_goals), max_goals)
    result_probs = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])   # H, D, A
    result_index = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    gd_probs = np.array([np.trace(grid, offset=-gd) for gd in range(-max_goals, max_goals + 1)])
    return {"result": ordinal_rps(result_probs, result_index),
            "goal_diff": ordinal_rps(gd_probs, home_goals - away_goals + max_goals),
            "goals": ordinal_rps(grid.sum(axis=1), home_goals) + ordinal_rps(grid.sum(axis=0), away_goals)}


def skill(model_rps: float, base_rps: float) -> float:
    return 1 - model_rps / base_rps if base_rps > 0 else 0.0


class Grader:
    """Grades settled predictions (columns xg_home, xg_away, rho, hg, ag) against a baseline score grid."""

    def __init__(self, baseline_grid: np.ndarray) -> None:
        self.baseline_grid = baseline_grid
        self.max_goals = baseline_grid.shape[0] - 1

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, max_goals: int = DEFAULT_MAX_GOALS) -> "Grader":
        """Baseline = how often each scoreline happened in ``matches`` (goals above max_goals are pooled)."""
        grid = np.zeros((max_goals + 1, max_goals + 1))
        np.add.at(grid, (matches.hg.clip(upper=max_goals).astype(int), matches.ag.clip(upper=max_goals).astype(int)), 1)
        return cls(grid / grid.sum())

    def grade(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Per game: the RPS of the model and of the baseline, the skill per part and the grade."""
        rows = []
        for pred in predictions.itertuples(index=False):
            grid = compute_score_grid(pred.xg_home, pred.xg_away, pred.rho, self.max_goals)
            model = part_rps(grid, pred.hg, pred.ag)
            base = part_rps(self.baseline_grid, pred.hg, pred.ag)
            rows.append({**{MODEL_PREFIX + part: model[part] for part in PART_WEIGHTS},
                         **{BASE_PREFIX + part: base[part] for part in PART_WEIGHTS}})
        graded = pd.DataFrame(rows, index=predictions.index, columns=[prefix + part for prefix in (MODEL_PREFIX, BASE_PREFIX)
                                                                      for part in PART_WEIGHTS])
        for part in PART_WEIGHTS:
            graded[SKILL_PREFIX + part] = [skill(model_rps, base_rps) for model_rps, base_rps
                                           in zip(graded[MODEL_PREFIX + part], graded[BASE_PREFIX + part])]
        graded[GRADE_COL] = self.weighted(graded)
        return graded

    @staticmethod
    def weighted(skills: pd.DataFrame | pd.Series) -> pd.Series | float:
        return GRADE_SCALE * sum(weight * skills[SKILL_PREFIX + part] for part, weight in PART_WEIGHTS.items())

    @classmethod
    def pooled(cls, graded: pd.DataFrame) -> pd.Series:
        """A group of games (e.g. a matchweek): skill per part from the summed RPS, and the grade."""
        totals = graded.sum()
        skills = pd.Series({SKILL_PREFIX + part: skill(totals[MODEL_PREFIX + part], totals[BASE_PREFIX + part])
                            for part in PART_WEIGHTS})
        skills[GRADE_COL] = cls.weighted(skills)
        return skills


def assign_matchweeks(games: pd.DataFrame, gap_days: int = MATCHWEEK_GAP_DAYS) -> pd.Series:
    """Matchweek number (1, 2, ...) of each game, aligned with ``games`` (columns date, home, away)."""
    matchweek_of_date: dict[pd.Timestamp, int] = {}
    matchweek_no, seen, previous = 0, set(), None
    for game_date, day_games in games.groupby("date", sort=True):
        teams = set(day_games.home) | set(day_games.away)
        if previous is None or (game_date - previous).days >= gap_days or teams & seen:
            matchweek_no, seen = matchweek_no + 1, set()
        seen |= teams
        matchweek_of_date[game_date] = matchweek_no
        previous = game_date
    return games.date.map(matchweek_of_date)


class SeasonGrades:
    """
    Grades of a season's settled predictions (e.g. the walk-forward backtest): per game, per matchweek, per team
    and for the season. Skills and grades are x100. Each group pools its games (see the module docstring).
    """

    def __init__(self, games: pd.DataFrame, grader: Grader) -> None:
        self.games = games.sort_values(["date", "home"]).reset_index(drop=True)
        self.graded = grader.grade(self.games)
        self.matchweeks = assign_matchweeks(self.games)
        self.hits = self.games.pick == self.games.result

    def _pooled(self, mask: pd.Series) -> dict:
        pooled = Grader.pooled(self.graded[mask])
        return {"games": int(mask.sum()), "picks right": self.hits[mask].mean(), "grade": pooled[GRADE_COL],
                **{label: GRADE_SCALE * pooled[SKILL_PREFIX + part] for part, label in PART_LABELS.items()}}

    def season(self) -> pd.DataFrame:
        first, last = self.games.date.min(), self.games.date.max()
        return pd.DataFrame([{"first game": first, "last game": last, "matchweeks": int(self.matchweeks.max()),
                              **self._pooled(pd.Series(True, index=self.games.index))}])

    def by_matchweek(self) -> pd.DataFrame:
        rows = []
        for matchweek_no, dates in self.games.date.groupby(self.matchweeks):
            rows.append({"matchweek": matchweek_no, "first game": dates.min(), "last game": dates.max(),
                         **self._pooled(self.matchweeks == matchweek_no)})
        return pd.DataFrame(rows)

    def by_team(self) -> pd.DataFrame:
        """Teams from most to least predictable: the pooled grade of the games they played."""
        teams = sorted(set(self.games.home) | set(self.games.away))
        table = pd.DataFrame([{"team": team, **self._pooled((self.games.home == team) | (self.games.away == team))}
                              for team in teams])
        table = table.sort_values("grade", ascending=False).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        return table

    def by_game(self) -> pd.DataFrame:
        games = self.games
        label = lambda code_col: [outcome_label(row.home, row.away, getattr(row, code_col))   # noqa: E731
                                  for row in games.itertuples()]
        return pd.DataFrame({
            "matchweek": self.matchweeks, "date": games.date, "home": games.home, "away": games.away,
            "pick": label("pick"), "confidence": games.confidence, "predicted score": games.likely_score,
            "actual score": games.hg.astype(int).astype(str) + "-" + games.ag.astype(int).astype(str),
            "result": label("result"), "pick right": self.hits, "grade": self.graded[GRADE_COL],
            **{label_text: GRADE_SCALE * self.graded[SKILL_PREFIX + part] for part, label_text in PART_LABELS.items()},
        })


def outcome_label(home: str, away: str, code: str) -> str:
    """The team name for a home or away win, or DRAW_LABEL."""
    return {"H": home, "A": away}.get(code, DRAW_LABEL)
