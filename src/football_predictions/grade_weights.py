"""
Tune the grade weights (PART_WEIGHTS in grading.py): the weights that best tell a better forecaster from a worse one.

The grade is a scoring rule, not a model setting. A higher grade is not the goal: the pooled grade is linear in the
weights, so "maximize the grade" always puts 100% on the part the model happens to be best at. Instead, the study
asks: with which weights does one matchweek's grade rank two forecasters in their true order most reliably?

Method
    1. Panel. Per game, a score grid for each forecaster: the model, model variants (other hyperparameters),
       the bookmakers (H/D/A odds turned into expected goals), the model blended toward the baseline, and the
       baseline itself. Only games with odds are used, so every forecaster grades the same games.
    2. Truth. Each forecaster's long-run quality is its mean log score of the exact scoreline over the evaluation
       seasons. This score uses the full grid and is none of the three parts, so the test is not circular.
       A pair of forecasters is used only when its log-score gap is significant (paired t >= MIN_REFERENCE_T).
    3. Objective. For weights w, the per-matchweek grade difference of each pair (better minus worse) gives a
       paired z = mean / standard error. The objective is the mean z over pairs. Concordance (the share of
       matchweeks that rank a pair right) is reported next to it.
    4. Search. Every point of the weight simplex at SIMPLEX_STEP. A grade is a dot product of pooled part skills,
       so the search is fast once the part RPS are known.
    5. Validation. Leave one evaluation season out: choose w on the others, score it on the season left out. A
       bootstrap over matchweeks gives a confidence interval on the out-of-sample gain over the current weights.
       The study recommends new weights only when that interval is above 0. The current season is a final check.

The study only recommends. To apply its weights, edit PART_WEIGHTS in grading.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .grading import (BASE_PREFIX, GRADE_COL, MODEL_PREFIX, PART_WEIGHTS, Grader, assign_matchweeks,
                      part_rps)
from .metrics import implied_probs
from .model import compute_score_grid, implied_expected_goals
from .report import _sheet

PARTS = list(PART_WEIGHTS)
MODEL, BOOKMAKERS, BASELINE = "model", "bookmakers", "baseline"
BLEND_NAME = "model {share:.0%} + baseline {alpha:.0%}"
DEFAULT_BLEND_ALPHAS = (0.25, 0.5, 0.75)     # share of the baseline in each blended forecaster
DEFAULT_SIMPLEX_STEP = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_SEED = 2026
CONFIDENCE_LEVEL = 0.95
MIN_REFERENCE_T = 2.0                        # pairs with a smaller log-score t-statistic have no reliable true order
LOG_SCORE_FLOOR = 1e-12
PERCENT = 100
WEIGHT_DECIMALS = 6                          # rounding of simplex points, so 0.15 + 0.35 + 0.5 is exactly 1
ODDS_COLS = ["odds_h", "odds_d", "odds_a"]
GAME_KEYS = ["date", "home", "away"]
XG_COLS = ["xg_home", "xg_away", "rho"]
OBJECTIVE_COL, CONCORDANCE_COL = "mean z", "concordance"
CANDIDATE_CURRENT, CANDIDATE_BEST, CANDIDATE_EQUAL = "current", "best", "equal"
REPORT_FILE = "{league}_grade_weights.xlsx"


def weights_label(weights: dict[str, float]) -> str:
    return " / ".join(f"{part} {weight:.0%}" for part, weight in weights.items())


def variant_name(override: dict) -> str:
    """The forecaster name of a model variant, e.g. "half_life_days=150"."""
    return ", ".join(f"{key}={value}" for key, value in override.items())


def weights_of(row: pd.Series) -> dict[str, float]:
    """The weight of each part in a row that has one column per part."""
    return {part: float(row[part]) for part in PARTS}


def simplex_grid(step: float = DEFAULT_SIMPLEX_STEP) -> pd.DataFrame:
    """Every weight vector over PARTS with entries that are multiples of ``step`` and add up to 1."""
    n_steps = round(1 / step)
    rows = [(i, j, n_steps - i - j) for i in range(n_steps + 1) for j in range(n_steps + 1 - i)]
    return pd.DataFrame(np.round(np.array(rows) / n_steps, WEIGHT_DECIMALS), columns=PARTS)


class ForecasterPanel:
    """
    Score grids of every forecaster for one season's games, and their part RPS and log score against the actual scores.
    Input frames are walk-forward backtests (columns date, home, away, xg_home, xg_away, rho, hg, ag, odds_h/d/a).
    """

    def __init__(self, baseline_grid: np.ndarray, blend_alphas=DEFAULT_BLEND_ALPHAS) -> None:
        self.baseline_grid = baseline_grid
        self.max_goals = baseline_grid.shape[0] - 1
        self.blend_alphas = tuple(blend_alphas)

    def scores(self, season: str, model_bt: pd.DataFrame, variant_bts: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """One row per game and forecaster: season, matchweek, game, forecaster, RPS per part, baseline RPS, log score."""
        games = model_bt.sort_values(GAME_KEYS).reset_index(drop=True)
        games = games.assign(matchweek=assign_matchweeks(games))
        for name, variant_bt in variant_bts.items():
            games = games.merge(variant_bt[GAME_KEYS + XG_COLS].rename(columns={col: f"{name}|{col}" for col in XG_COLS}),
                                on=GAME_KEYS, how="inner")
        games = games[games[ODDS_COLS].notna().all(axis=1)].reset_index(drop=True)
        bookie_probs = implied_probs(games.odds_h, games.odds_d, games.odds_a)
        rows = []
        for game_no, (_, game) in enumerate(games.iterrows()):
            base = part_rps(self.baseline_grid, int(game.hg), int(game.ag))
            for forecaster, grid in self._grids(game, list(variant_bts), bookie_probs[game_no]).items():
                rows.append({"season": season, "matchweek": int(game.matchweek), "game": game_no, "forecaster": forecaster,
                             **{MODEL_PREFIX + part: value for part, value in part_rps(grid, int(game.hg), int(game.ag)).items()},
                             **{BASE_PREFIX + part: value for part, value in base.items()},
                             "log_score": self._log_score(grid, int(game.hg), int(game.ag))})
        return pd.DataFrame(rows)

    def _grids(self, game: pd.Series, variant_names: list[str], bookie_probs: np.ndarray) -> dict[str, np.ndarray]:
        model = compute_score_grid(game.xg_home, game.xg_away, game.rho, self.max_goals)
        grids = {MODEL: model}
        for name in variant_names:
            grids[name] = compute_score_grid(game[f"{name}|xg_home"], game[f"{name}|xg_away"], game[f"{name}|rho"],
                                             self.max_goals)
        p_home, _, p_away = bookie_probs
        grids[BOOKMAKERS] = compute_score_grid(*implied_expected_goals(p_home, p_away, game.rho, self.max_goals),
                                               game.rho, self.max_goals)
        for alpha in self.blend_alphas:
            grids[BLEND_NAME.format(share=1 - alpha, alpha=alpha)] = (1 - alpha) * model + alpha * self.baseline_grid
        grids[BASELINE] = self.baseline_grid
        return grids

    def _log_score(self, grid: np.ndarray, home_goals: int, away_goals: int) -> float:
        return float(np.log(max(grid[min(home_goals, self.max_goals), min(away_goals, self.max_goals)], LOG_SCORE_FLOOR)))


@dataclass
class SkillCube:
    """Pooled skill per forecaster, matchweek and part (array forecasters x matchweeks x parts), plus per-game log scores."""
    forecaster_lst: list[str]
    matchweeks: pd.DataFrame        # one row per matchweek: season, matchweek
    skills: np.ndarray
    game_log_scores: pd.DataFrame   # one row per game, one column per forecaster

    @classmethod
    def from_scores(cls, scores: pd.DataFrame) -> "SkillCube":
        forecaster_lst = list(dict.fromkeys(scores.forecaster))
        totals = scores.groupby(["forecaster", "season", "matchweek"])[
            [MODEL_PREFIX + part for part in PARTS] + [BASE_PREFIX + part for part in PARTS]].sum()
        skill_frame = pd.DataFrame({part: 1 - totals[MODEL_PREFIX + part] / totals[BASE_PREFIX + part] for part in PARTS})
        matchweeks = scores[["season", "matchweek"]].drop_duplicates().sort_values(["season", "matchweek"])
        matchweek_index = pd.MultiIndex.from_frame(matchweeks)
        skills = np.stack([skill_frame.loc[forecaster].reindex(matchweek_index).to_numpy() for forecaster in forecaster_lst])
        game_log_scores = scores.pivot_table(index=["season", "game"], columns="forecaster", values="log_score")
        return cls(forecaster_lst, matchweeks.reset_index(drop=True), skills, game_log_scores[forecaster_lst])

    def select(self, season_lst: list[str]) -> "SkillCube":
        keep = self.matchweeks.season.isin(season_lst).to_numpy()
        games = self.game_log_scores[self.game_log_scores.index.get_level_values("season").isin(season_lst)]
        return SkillCube(self.forecaster_lst, self.matchweeks[keep].reset_index(drop=True), self.skills[:, keep], games)

    @property
    def season_lst(self) -> list[str]:
        return list(dict.fromkeys(self.matchweeks.season))


@dataclass
class PairSet:
    """Forecaster pairs with a reliable true order (better first), from the paired t-test of their log scores."""
    table: pd.DataFrame     # better, worse, log score gap, t

    @classmethod
    def from_cube(cls, cube: SkillCube, min_t: float = MIN_REFERENCE_T) -> "PairSet":
        log_scores = cube.game_log_scores
        rows = []
        for i, first in enumerate(cube.forecaster_lst):
            for second in cube.forecaster_lst[i + 1:]:
                gap = (log_scores[first] - log_scores[second]).to_numpy()
                t_stat = gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))) if gap.std() > 0 else 0.0
                better, worse = (first, second) if t_stat > 0 else (second, first)
                rows.append({"better": better, "worse": worse, "log score gap": abs(gap.mean()), "t": abs(t_stat)})
        table = pd.DataFrame(rows)
        table["used"] = table.t >= min_t
        return cls(table.sort_values("t", ascending=False).reset_index(drop=True))

    def differences(self, cube: SkillCube) -> np.ndarray:
        """Skill of the better minus the worse forecaster: pairs x matchweeks x parts."""
        used = self.table[self.table.used]
        position = {forecaster: i for i, forecaster in enumerate(cube.forecaster_lst)}
        return cube.skills[used.better.map(position).to_numpy()] - cube.skills[used.worse.map(position).to_numpy()]


def mean_z(grade_diffs: np.ndarray) -> np.ndarray:
    """Mean over pairs of the paired z of grade differences; grade_diffs is pairs x matchweeks (x weight vectors)."""
    n_matchweeks = grade_diffs.shape[1]
    return (grade_diffs.mean(axis=1) / (grade_diffs.std(axis=1, ddof=1) / np.sqrt(n_matchweeks))).mean(axis=0)


def evaluate(diffs: np.ndarray, weight_frame: pd.DataFrame) -> pd.DataFrame:
    """The weight vectors of ``weight_frame`` with their objective (mean z) and concordance on ``diffs``."""
    grade_diffs = diffs @ weight_frame[PARTS].to_numpy().T        # pairs x matchweeks x weight vectors
    return weight_frame.assign(**{OBJECTIVE_COL: mean_z(grade_diffs), CONCORDANCE_COL: (grade_diffs > 0).mean(axis=(0, 1))})


def fisher_weights(diffs: np.ndarray) -> dict[str, float]:
    """
    Closed-form check: w ∝ Σ⁻¹ μ of the pair skill differences (Fisher's discriminant), negatives set to 0, sum 1.
    """
    stacked = diffs.reshape(-1, len(PARTS))
    raw = np.clip(np.linalg.pinv(np.cov(stacked.T)) @ stacked.mean(axis=0), 0, None)
    return dict(zip(PARTS, raw / raw.sum() if raw.sum() > 0 else np.full(len(PARTS), np.nan)))


@dataclass
class StudyResult:
    recommended: dict[str, float]
    best: dict[str, float]
    recommendation: str
    summary: pd.DataFrame
    folds: pd.DataFrame
    holdout: pd.DataFrame
    simplex: pd.DataFrame
    pairs: pd.DataFrame
    forecasters: pd.DataFrame
    correlation: pd.DataFrame


class WeightStudy:
    """Leave-one-season-out search for the grade weights that best rank forecasters (see the module docstring)."""

    def __init__(self, current: dict[str, float] = PART_WEIGHTS, step: float = DEFAULT_SIMPLEX_STEP,
                 bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES, seed: int = DEFAULT_SEED) -> None:
        self.current = pd.DataFrame([current])[PARTS]
        self.simplex = simplex_grid(step)
        self.bootstrap_samples, self.seed = bootstrap_samples, seed

    def run(self, evaluation: SkillCube, holdout: SkillCube | None = None) -> StudyResult:
        pairs = PairSet.from_cube(evaluation)
        diffs = pairs.differences(evaluation)
        if not len(diffs):
            raise ValueError("no forecaster pair has a reliable true order: add evaluation seasons")
        simplex = evaluate(diffs, self.simplex).sort_values(OBJECTIVE_COL, ascending=False).reset_index(drop=True)
        best = weights_of(simplex.iloc[0])
        folds, oos_grades, current_grades = self._folds(evaluation, diffs)
        gain, ci_low, ci_high = self._bootstrap_gain(np.concatenate(oos_grades, axis=1), np.concatenate(current_grades, axis=1))
        improves = ci_low > 0
        recommended = best if improves else weights_of(self.current.iloc[0])
        recommendation = (f"change PART_WEIGHTS to {weights_label(best)}: the out-of-sample gain is significant"
                          if improves else f"keep PART_WEIGHTS at {weights_label(recommended)}: no weights do "
                                           f"significantly better out of sample")
        candidates = pd.concat([self.current.assign(weights=CANDIDATE_CURRENT),
                                pd.DataFrame([best]).assign(weights=CANDIDATE_BEST)], ignore_index=True)
        summary = pd.DataFrame([{
            "recommendation": recommendation, "evaluation seasons": ", ".join(evaluation.season_lst),
            "matchweeks": len(evaluation.matchweeks), "pairs used": len(diffs),
            "best weights (all seasons)": weights_label(best),
            "Fisher check": weights_label(fisher_weights(diffs)),
            "out-of-sample gain (mean z)": gain, f"gain CI low ({CONFIDENCE_LEVEL:.0%})": ci_low,
            f"gain CI high ({CONFIDENCE_LEVEL:.0%})": ci_high}]).T.reset_index()
        summary.columns = ["item", "value"]
        return StudyResult(
            recommended=recommended, best=best, recommendation=recommendation, summary=summary, folds=folds,
            holdout=self._holdout(holdout, pairs, candidates), simplex=simplex, pairs=pairs.table,
            forecasters=evaluation.game_log_scores.mean().rename("mean log score").sort_values(ascending=False)
                                                  .rename_axis("forecaster").reset_index(),
            correlation=self._correlation(evaluation, diffs))

    def _folds(self, evaluation: SkillCube, diffs: np.ndarray) -> tuple[pd.DataFrame, list, list]:
        rows, oos_grades, current_grades = [], [], []
        seasons = evaluation.matchweeks.season.to_numpy()
        current = self.current.iloc[0][PARTS].to_numpy(dtype=float)
        for season in evaluation.season_lst:
            train, test = diffs[:, seasons != season], diffs[:, seasons == season]
            fold_best = evaluate(train, self.simplex).sort_values(OBJECTIVE_COL, ascending=False).iloc[0][PARTS]
            scored = evaluate(test, pd.concat([self.current, fold_best.to_frame().T], ignore_index=True))
            rows.append({"season left out": season, "weights chosen on the others": weights_label(weights_of(fold_best)),
                         "mean z, current": scored[OBJECTIVE_COL].iloc[0], "mean z, chosen": scored[OBJECTIVE_COL].iloc[1],
                         "concordance, current": scored[CONCORDANCE_COL].iloc[0],
                         "concordance, chosen": scored[CONCORDANCE_COL].iloc[1]})
            oos_grades.append(test @ fold_best.to_numpy(dtype=float))
            current_grades.append(test @ current)
        return pd.DataFrame(rows), oos_grades, current_grades

    def _bootstrap_gain(self, oos_grades: np.ndarray, current_grades: np.ndarray) -> tuple[float, float, float]:
        """Out-of-sample gain in mean z over the current weights, with a bootstrap interval over matchweeks."""
        rng = np.random.default_rng(self.seed)
        n_matchweeks = oos_grades.shape[1]
        samples = rng.integers(0, n_matchweeks, size=(self.bootstrap_samples, n_matchweeks))
        gains = np.array([mean_z(oos_grades[:, idx]) - mean_z(current_grades[:, idx]) for idx in samples])
        tail = (1 - CONFIDENCE_LEVEL) / 2
        return (float(mean_z(oos_grades) - mean_z(current_grades)),
                float(np.quantile(gains, tail)), float(np.quantile(gains, 1 - tail)))

    @staticmethod
    def _holdout(holdout: SkillCube | None, pairs: PairSet, candidates: pd.DataFrame) -> pd.DataFrame:
        if holdout is None or len(holdout.matchweeks) < 2:
            return pd.DataFrame()
        scored = evaluate(pairs.differences(holdout), candidates)
        return scored.assign(season=", ".join(holdout.season_lst), matchweeks=len(holdout.matchweeks))

    @staticmethod
    def _correlation(evaluation: SkillCube, diffs: np.ndarray) -> pd.DataFrame:
        """How alike the parts are: correlation of the model's matchweek skills, and of the pair skill differences."""
        model_skills = pd.DataFrame(evaluation.skills[evaluation.forecaster_lst.index(MODEL)], columns=PARTS)
        pair_diffs = pd.DataFrame(diffs.reshape(-1, len(PARTS)), columns=PARTS)
        return pd.concat([model_skills.corr().assign(of="model matchweek skill"),
                          pair_diffs.corr().assign(of="pair skill difference")]).rename_axis("part").reset_index()


def ranking_sensitivity(graded: pd.DataFrame, games: pd.DataFrame, candidates: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    How much the published tables move under other weights: Spearman correlation of the team ranking and of the
    matchweek grades with those of the current weights, and the largest change of a team's rank.
    ``graded`` is Grader.grade() of ``games`` (columns date, home, away).
    """
    matchweeks = assign_matchweeks(games)
    teams = sorted(set(games.home) | set(games.away))
    team_masks = {team: (games.home == team) | (games.away == team) for team in teams}

    def tables(weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
        team_grades = pd.Series({team: Grader.pooled(graded[mask], weights)[GRADE_COL] for team, mask in team_masks.items()})
        matchweek_grades = pd.Series({matchweek_no: Grader.pooled(graded[matchweeks == matchweek_no], weights)[GRADE_COL]
                                      for matchweek_no in sorted(matchweeks.unique())})
        return team_grades, matchweek_grades

    current_teams, current_matchweeks = tables(PART_WEIGHTS)
    current_rank = current_teams.rank(ascending=False)
    rows = []
    for name, weights in candidates.items():
        team_grades, matchweek_grades = tables(weights)
        rows.append({"weights": name, "parts": weights_label(weights),
                     "season grade": float(Grader.pooled(graded, weights)[GRADE_COL]),
                     "team ranking vs current (Spearman)": team_grades.corr(current_teams, method="spearman"),
                     "largest team rank change": int((team_grades.rank(ascending=False) - current_rank).abs().max()),
                     "matchweek grades vs current (Spearman)": matchweek_grades.corr(current_matchweeks, method="spearman")})
    return pd.DataFrame(rows)


def sensitivity_candidates(best: dict[str, float]) -> dict[str, dict[str, float]]:
    """The weights the sensitivity table compares: current, best, equal, and each part alone."""
    candidate_dict = {CANDIDATE_CURRENT: dict(PART_WEIGHTS), CANDIDATE_BEST: best,
                      CANDIDATE_EQUAL: {part: 1 / len(PARTS) for part in PARTS}}
    candidate_dict |= {f"{part} only": {other: float(other == part) for other in PARTS} for part in PARTS}
    return candidate_dict


def write_report(path: Path, result: StudyResult, sensitivity: pd.DataFrame | None) -> None:
    """The study as an Excel workbook: summary, folds, holdout, sensitivity, simplex, forecasters and pairs, correlation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    simplex = result.simplex.assign(**{f"{part} %": PERCENT * result.simplex[part] for part in PARTS}).drop(columns=PARTS)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        _sheet(xw, "Summary", [("Grade weights study (the study only recommends: edit PART_WEIGHTS in grading.py "
                                "to apply)", result.summary),
                               ("Leave one season out", result.folds),
                               ("Final check on the current season", result.holdout)])
        if sensitivity is not None:
            _sheet(xw, "Sensitivity", [(None, sensitivity)])
        _sheet(xw, "Simplex", [(None, simplex)])
        _sheet(xw, "Forecasters", [("Mean log score of the exact scoreline (higher is better)", result.forecasters),
                                   (f"Pairs (used when t >= {MIN_REFERENCE_T})", result.pairs)])
        _sheet(xw, "Correlation", [(None, result.correlation)])
