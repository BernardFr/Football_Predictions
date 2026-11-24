#!/usr/local/bin/python3
"""Prediction grades: skill per part against a baseline score grid, per game and pooled over a matchweek."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_predictions.grading import (GRADE_COL, GRADE_SCALE, SKILL_PREFIX, Grader, SeasonGrades, assign_matchweeks,
                                         ordinal_rps, part_rps)
from football_predictions.model import compute_score_grid

MAX_GOALS = 6
BASE_XG, RHO = (1.5, 1.2), -0.05


def certain_grid(home_goals: int, away_goals: int) -> np.ndarray:
    grid = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    grid[home_goals, away_goals] = 1
    return grid


def prediction(xg_home: float, xg_away: float, hg: int, ag: int) -> dict:
    return {"xg_home": xg_home, "xg_away": xg_away, "rho": RHO, "hg": hg, "ag": ag}


@pytest.fixture
def grader() -> Grader:
    return Grader(compute_score_grid(*BASE_XG, RHO, MAX_GOALS))


def test_ordinal_rps_is_zero_when_certain_and_right_and_grows_with_distance():
    assert ordinal_rps(np.array([0, 1, 0]), 1) == 0
    near, far = ordinal_rps(np.array([0, 1, 0, 0]), 2), ordinal_rps(np.array([1, 0, 0, 0]), 2)
    assert 0 < near < far


def test_part_rps_of_a_certain_right_grid_is_zero_for_every_part():
    assert part_rps(certain_grid(2, 1), 2, 1) == {"result": 0, "goal_diff": 0, "goals": 0}


def test_goals_above_the_grid_count_as_the_last_row():
    assert part_rps(certain_grid(MAX_GOALS, 0), MAX_GOALS + 3, 0)["goals"] == 0


def test_baseline_prediction_grades_zero_and_a_sharper_right_one_grades_above_zero(grader):
    graded = grader.grade(pd.DataFrame([prediction(*BASE_XG, 2, 0), prediction(2.6, 0.5, 2, 0), prediction(0.5, 2.6, 2, 0)]))
    assert graded[GRADE_COL].iloc[0] == pytest.approx(0)
    assert graded[GRADE_COL].iloc[1] > 0 > graded[GRADE_COL].iloc[2]
    assert (graded[GRADE_COL] <= GRADE_SCALE).all()


def test_near_miss_on_goal_difference_beats_a_far_miss(grader):
    near, far = grader.grade(pd.DataFrame([prediction(2.0, 1.0, 3, 1), prediction(1.0, 2.0, 3, 1)]))[SKILL_PREFIX + "goal_diff"]
    assert near > far


def test_pooled_matchweek_grade_sums_rps_before_the_ratio(grader):
    graded = grader.grade(pd.DataFrame([prediction(2.6, 0.5, 2, 0), prediction(0.8, 1.9, 0, 0)]))
    pooled = Grader.pooled(graded)
    totals = graded.sum()
    assert pooled[SKILL_PREFIX + "result"] == pytest.approx(1 - totals.rps_result / totals.base_rps_result)
    assert pooled[GRADE_COL] == pytest.approx(Grader.weighted(pooled))


def test_baseline_from_matches_is_the_scoreline_frequencies():
    matches = pd.DataFrame({"hg": [1, 1, 0, 12], "ag": [0, 0, 0, 1]})
    grid = Grader.from_matches(matches, MAX_GOALS).baseline_grid
    assert grid.sum() == pytest.approx(1)
    assert grid[1, 0] == pytest.approx(0.5) and grid[MAX_GOALS, 1] == pytest.approx(0.25)


def fixtures(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "home", "away"]).assign(date=lambda frame: pd.to_datetime(frame.date))


def test_matchweeks_split_on_gaps_and_on_a_team_playing_twice():
    games = fixtures([("2026-08-22", "A", "B"), ("2026-08-23", "C", "D"), ("2026-08-24", "E", "F"),   # weekend
                      ("2026-08-25", "B", "C"), ("2026-08-26", "D", "A"),                              # midweek: B plays again
                      ("2026-08-29", "A", "C")])                                                       # 3-day gap
    assert assign_matchweeks(games).tolist() == [1, 1, 1, 2, 2, 3]


def season_games() -> pd.DataFrame:
    rows = [("2026-08-22", "A", "B", 2.0, 0.8, 2, 0), ("2026-08-22", "C", "D", 1.2, 1.4, 1, 1),
            ("2026-08-29", "A", "C", 1.8, 1.0, 0, 3), ("2026-08-29", "B", "D", 1.4, 1.3, 1, 0)]
    games = pd.DataFrame(rows, columns=["date", "home", "away", "xg_home", "xg_away", "hg", "ag"])
    games = games.assign(date=pd.to_datetime(games.date), rho=RHO, confidence=0.5, likely_score="1-1",
                         pick=np.where(games.xg_home > games.xg_away, "H", "A"),
                         result=np.where(games.hg > games.ag, "H", np.where(games.hg == games.ag, "D", "A")))
    return games


def test_season_grades_per_game_matchweek_team_and_season(grader):
    grades = SeasonGrades(season_games(), grader)
    by_game, by_matchweek, by_team, season = grades.by_game(), grades.by_matchweek(), grades.by_team(), grades.season()
    assert len(by_game) == 4 and by_game["matchweek"].tolist() == [1, 1, 2, 2]
    assert by_matchweek.games.tolist() == [2, 2] and season.matchweeks.iloc[0] == 2 and season.games.iloc[0] == 4
    assert by_game.pick.iloc[0] == "A" and by_game.result.iloc[1] == "Draw"
    assert by_team["rank"].tolist() == [1, 2, 3, 4] and by_team.grade.is_monotonic_decreasing
    assert set(by_team.team) == {"A", "B", "C", "D"} and (by_team.games == 2).all()
    assert season.grade.iloc[0] == pytest.approx(Grader.pooled(grades.graded)[GRADE_COL])


def test_report_has_grade_on_backtest_sheet_and_grades_and_predictability_sheets(grader):
    from openpyxl import load_workbook
    from football_predictions import report
    backtest = season_games().assign(p_home=0.45, p_draw=0.27, p_away=0.28, exp_gd=0.3, gd_lo80=-1, gd_hi80=2)
    path = Path(__file__).parent / "grades_report.xlsx"
    report.write(path, "Test", None, backtest, "2026-27", None, None, None, 0.6, 0.45, grader=grader)
    workbook = load_workbook(path)
    assert {"Grades", "Predictability"} <= set(workbook.sheetnames)
    assert any(report.GRADE_TITLE in str(cell.value) for cell in workbook["Backtest"]["A"])
    predictability = [row for row in workbook["Predictability"].iter_rows(values_only=True)]
    assert predictability[1][:2] == ("rank", "team") and len(predictability) == 2 + 4
    report.write(path, "Test", None, backtest, "2026-27", None, None, None, 0.6, 0.45)   # no grader: no grade sheets
    assert "Grades" not in load_workbook(path).sheetnames


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
