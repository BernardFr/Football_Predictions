#!/usr/local/bin/python3
"""Grade-weight study: odds -> expected goals, simplex, objective, forecaster ranking on simulated seasons, report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_predictions import grade_weights as gw
from football_predictions.grading import GRADE_COL, PART_WEIGHTS, SKILL_PREFIX, Grader
from football_predictions.model import compute_score_grid, implied_expected_goals

MAX_GOALS = 8
RHO = -0.08
N_TEAMS = 20
WEEKS_PER_SEASON = 19
EVALUATION_SEASONS = ["2023-24", "2024-25", "2025-26"]
HOLDOUT_SEASON = "2026-27"
MODEL_NOISE, VARIANT_NOISE = 0.10, 0.35      # log-scale noise on the true expected goals
SEED = 7
GOALS_SHARE_FLOOR = 0.8                      # sampling noise can give a little weight to the parts with no signal
REPORT_PATH = Path(__file__).parent / "grade_weights_report.xlsx"


def result_probs(grid: np.ndarray) -> np.ndarray:
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def test_implied_expected_goals_recover_the_grid_behind_the_probabilities():
    p_home, _, p_away = result_probs(compute_score_grid(1.9, 0.8, RHO, MAX_GOALS))
    assert implied_expected_goals(p_home, p_away, RHO, MAX_GOALS) == pytest.approx((1.9, 0.8), abs=1e-3)


def test_simplex_points_add_up_to_one_and_include_the_current_weights():
    simplex = gw.simplex_grid(0.05)
    assert len(simplex) == 231 and np.allclose(simplex.sum(axis=1), 1)
    assert ((simplex[gw.PARTS] == pd.Series(PART_WEIGHTS)).all(axis=1)).any()


def test_pooled_grade_uses_the_weights_given():
    skills = pd.Series({SKILL_PREFIX + "result": 0.1, SKILL_PREFIX + "goal_diff": 0.2, SKILL_PREFIX + "goals": 0.4})
    assert Grader.weighted(skills) == pytest.approx(100 * sum(weight * skills[SKILL_PREFIX + part] for part, weight in PART_WEIGHTS.items()))
    assert Grader.weighted(skills, {"result": 0, "goal_diff": 0, "goals": 1}) == pytest.approx(40)


def test_objective_prefers_the_part_that_carries_the_signal():
    rng = np.random.default_rng(SEED)
    diffs = rng.normal(0, 1, size=(4, 60, 3))
    diffs[:, :, 2] += 1.0                        # only "goals" separates better from worse
    scored = gw.evaluate(diffs, gw.simplex_grid(0.1)).sort_values(gw.OBJECTIVE_COL, ascending=False)
    assert scored.iloc[0]["goals"] >= GOALS_SHARE_FLOOR
    assert gw.fisher_weights(diffs)["goals"] >= GOALS_SHARE_FLOOR


def simulated_season(season: str, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A model backtest and one noisier variant, from known expected goals, with fair bookmaker odds."""
    strength = rng.normal(0, 0.3, N_TEAMS)
    start = pd.Timestamp(f"{season[:4]}-08-16")
    rows = []
    for week in range(WEEKS_PER_SEASON):
        order = rng.permutation(N_TEAMS)
        for home, away in zip(order[::2], order[1::2]):
            true_home, true_away = np.exp(0.3 + strength[home] - strength[away] * 0.5), np.exp(0.1 + strength[away] - strength[home] * 0.5)
            truth = compute_score_grid(true_home, true_away, RHO, MAX_GOALS)
            flat = rng.choice(truth.size, p=truth.ravel())
            home_goals, away_goals = divmod(flat, MAX_GOALS + 1)
            rows.append({"date": start + pd.Timedelta(weeks=week), "home": f"T{home}", "away": f"T{away}",
                         "true_home": true_home, "true_away": true_away, "rho": RHO, "hg": home_goals, "ag": away_goals,
                         **dict(zip(gw.ODDS_COLS, 1 / result_probs(truth)))})
    games = pd.DataFrame(rows)
    model = games.assign(xg_home=games.true_home * np.exp(rng.normal(0, MODEL_NOISE, len(games))),
                         xg_away=games.true_away * np.exp(rng.normal(0, MODEL_NOISE, len(games))))
    variant = games.assign(xg_home=games.true_home * np.exp(rng.normal(0, VARIANT_NOISE, len(games))),
                           xg_away=games.true_away * np.exp(rng.normal(0, VARIANT_NOISE, len(games))))
    return model, variant


@pytest.fixture(scope="module")
def study():
    rng = np.random.default_rng(SEED)
    baseline = compute_score_grid(1.5, 1.2, RHO, MAX_GOALS)
    panel = gw.ForecasterPanel(baseline)
    score_lst, holdout_games = [], None
    for season in [*EVALUATION_SEASONS, HOLDOUT_SEASON]:
        model, variant = simulated_season(season, rng)
        score_lst.append(panel.scores(season, model, {"noisy": variant}))
        holdout_games = model
    cube = gw.SkillCube.from_scores(pd.concat(score_lst, ignore_index=True))
    result = gw.WeightStudy(bootstrap_samples=200).run(cube.select(EVALUATION_SEASONS), cube.select([HOLDOUT_SEASON]))
    return cube, result, baseline, holdout_games


def test_panel_has_every_forecaster_on_the_same_games(study):
    cube, *_ = study
    assert cube.forecaster_lst[:3] == [gw.MODEL, "noisy", gw.BOOKMAKERS] and cube.forecaster_lst[-1] == gw.BASELINE
    assert cube.skills.shape == (len(cube.forecaster_lst), 4 * WEEKS_PER_SEASON, 3)
    assert np.allclose(cube.skills[-1], 0)       # the baseline graded against itself


def test_true_order_puts_the_truth_first_and_less_baseline_above_more(study):
    _, result, *_ = study
    ranking = result.forecasters.forecaster.tolist()
    blends = [gw.BLEND_NAME.format(share=1 - alpha, alpha=alpha) for alpha in gw.DEFAULT_BLEND_ALPHAS]
    assert ranking[0] == gw.BOOKMAKERS and ranking.index(gw.MODEL) < ranking.index("noisy")
    assert [ranking.index(name) for name in [gw.MODEL, *blends, gw.BASELINE]] == sorted(
        ranking.index(name) for name in [gw.MODEL, *blends, gw.BASELINE])
    assert result.pairs.used.any()


def test_study_finds_weights_at_least_as_good_as_the_current_ones_and_reports_folds(study):
    _, result, *_ = study
    simplex = result.simplex.set_index(gw.PARTS)
    assert simplex[gw.OBJECTIVE_COL].iloc[0] >= simplex.loc[tuple(PART_WEIGHTS.values()), gw.OBJECTIVE_COL]
    assert result.folds["season left out"].tolist() == EVALUATION_SEASONS
    assert sum(result.recommended.values()) == pytest.approx(1)
    assert result.holdout.weights.tolist() == [gw.CANDIDATE_CURRENT, gw.CANDIDATE_BEST]


def test_sensitivity_of_the_current_weights_to_themselves_is_perfect(study):
    _, result, baseline, games = study
    sensitivity = gw.ranking_sensitivity(Grader(baseline).grade(games), games, gw.sensitivity_candidates(result.best))
    current = sensitivity.set_index("weights").loc[gw.CANDIDATE_CURRENT]
    assert current["team ranking vs current (Spearman)"] == pytest.approx(1)
    assert current["largest team rank change"] == 0
    assert current["season grade"] == pytest.approx(Grader.pooled(Grader(baseline).grade(games))[GRADE_COL])


def test_report_has_every_sheet(study):
    from openpyxl import load_workbook
    _, result, baseline, games = study
    sensitivity = gw.ranking_sensitivity(Grader(baseline).grade(games), games, gw.sensitivity_candidates(result.best))
    gw.write_report(REPORT_PATH, result, sensitivity)
    assert load_workbook(REPORT_PATH).sheetnames == ["Summary", "Sensitivity", "Simplex", "Forecasters", "Correlation"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
