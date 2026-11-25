#!/usr/local/bin/python3
"""The website: every page builds from small synthetic frames, and each page shows what it should."""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from football_predictions import website
from football_predictions.grading import Grader
from football_predictions.histogram import ScorelineStats, chart_fragment, scoreline_table

SITE_DIR = Path(__file__).resolve().parent / "test_site"
SEASON = "2026-27"
PREVIOUS_SEASON = "2025-26"
UPDATED = datetime(2026, 11, 24, 9, 30).astimezone()


def prediction(date, home, away, pick, p_home, p_draw, p_away, **extra) -> dict:
    return {"date": pd.Timestamp(date), "season": SEASON, "home": home, "away": away, "pick": pick,
            "confidence": max(p_home, p_draw, p_away), "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "exp_gd": 0.4, "gd_lo80": -1, "gd_hi80": 2, "likely_score": "1-1",
            "highest_prob_score_1": "1-1 13%", "highest_prob_score_2": "1-0 11%", "highest_prob_score_3": "2-1 9%",
            "xg_home": 1.5, "xg_away": 1.1, "rho": -0.05, "odds_h": 2.0, "odds_d": 3.5, "odds_a": 4.0, **extra}


def ledger_frame() -> pd.DataFrame:
    settled = dict(variant="base")
    rows = [
        prediction("2026-11-14", "Old", "Town", "H", .5, .3, .2, hg=2, ag=0, result="H", **settled),   # an earlier matchweek
        prediction("2026-11-21", "Arsenal", "Chelsea", "H", .5, .3, .2, hg=1, ag=1, result="D", **settled),
        prediction("2026-11-22", "Leeds", "Fulham", "A", .2, .3, .5, hg=0, ag=2, result="A", **settled),
        prediction("2026-11-22", "Leeds", "Fulham", "A", .2, .3, .5, hg=0, ag=2, result="A", variant="lineup"),
        prediction("2026-11-23", "Hull", "Everton", "H", .5, .3, .2, hg=None, ag=None, result=None, **settled),
    ]
    rows[1]["likely_score"] = "1-1"     # Arsenal v Chelsea: exact score, wrong pick
    return pd.DataFrame(rows)


def site_data(**overrides) -> website.SiteData:
    upcoming = pd.DataFrame([prediction("2026-11-29", "Spurs", "<Villa>", "A", .3, .25, .45),
                             prediction("2026-11-28", "Brighton", "Wolves", "H", .62, .2, .18, odds_h=None)])
    impact = pd.DataFrame({"team": ["Arsenal", "Arsenal", "Chelsea"], "athlete_id": [1, 2, 3],
                           "player": ["Saka", "Rice", "Palmer"], "pos": ["F", "M", "M"], "share": [.1, .09, .1],
                           "in_xi": [True, True, True], "impact_pts": [.04, .001, .05], "impact_gd": [.06, .002, .07]})
    scorers = pd.DataFrame({"team": ["Arsenal", "Chelsea"], "player": ["Saka", "Palmer"], "goals": [3, 5],
                            "first_goal": [1, 2], "equalizer": [0, 1], "go_ahead": [2, 1], "combined": [3, 4]})
    matches = pd.DataFrame({"date": pd.to_datetime(["2025-09-01", "2026-09-01", "2026-09-08"]),
                            "season": [PREVIOUS_SEASON, SEASON, SEASON], "home": ["A", "B", "C"], "away": ["D", "E", "F"],
                            "hg": [1, 2, 0], "ag": [0, 2, 1]})
    seasons = [PREVIOUS_SEASON, SEASON]
    stats = ScorelineStats("Test League", "football_data", seasons, matches, scoreline_table(matches, seasons),
                           {str(season_name): int(count) for season_name, count in matches.groupby("season").size().items()})
    fields = dict(league_name="Test League", season=SEASON, high=.6, medium=.45, upcoming=upcoming, ledger=ledger_frame(),
                  player_impact=impact, valuable_scorers=scorers, scorelines=stats)
    return website.SiteData(**{**fields, **overrides})


def build(data: website.SiteData, folder: str) -> dict[str, str]:
    site_dir = SITE_DIR / folder
    website.Website(data, updated=UPDATED).build(site_dir)
    return {path.name: path.read_text(encoding="utf-8") for path in site_dir.iterdir()}


def test_every_page_and_static_file_is_written_with_navigation():
    pages = build(site_data(), "full")
    assert set(pages) == {"index.html", "season.html", "players.html", "scores.html", "style.css", "players.js"}
    for name in ("index.html", "season.html", "players.html", "scores.html"):
        assert not re.search(r"\$\{[a-z_]+\}", pages[name])                # every placeholder filled
        assert f'<a href="{name}" aria-current="page">' in pages[name]  # its own tab is marked
        assert pages[name].count("<nav>") == 1 and "<footer>" in pages[name]
    assert "No graded games yet" in pages["season.html"]                # no grader in site_data()


def test_predictions_page_lists_upcoming_games_by_date_and_escapes_names():
    page = build(site_data(), "full")["index.html"]
    assert page.index("Brighton") < page.index("Spurs")       # sorted by date, not by confidence
    assert "&lt;Villa&gt;" in page and "<Villa>" not in page
    assert "1-1 13% · 1-0 11% · 2-1 9%" in page
    assert website.MISSING in page                             # Brighton has no bookmaker odds


def test_last_matchweek_is_the_base_models_settled_games_of_the_latest_days():
    played = website.last_matchweek(ledger_frame())
    assert list(played.home) == ["Arsenal", "Leeds"]            # not the earlier matchweek, the lineup row or the unplayed game
    page = build(site_data(), "full")["index.html"]
    assert "Last matchweek: Sat 21 Nov to Sun 22 Nov" in page
    assert "<strong>1 of 2</strong> picks right (50%)" in page
    assert "<strong>1</strong> exact score " in page
    assert len(re.findall(r'<tr class="hit">', page)) == 1 and len(re.findall(r'<tr class="miss">', page)) == 1


def test_last_matchweek_has_a_grade_per_game_and_a_matchweek_grade_when_a_grader_is_given():
    baseline = pd.DataFrame({"hg": [1, 0, 2, 1], "ag": [1, 0, 1, 2]})
    page = build(site_data(grader=Grader.from_matches(baseline)), "graded")["index.html"]
    assert "<th class=\"num\">Grade</th>" in page
    assert len(re.findall(r'<td class="num">[+-]\d+</td></tr>', page)) == 2
    assert re.search(r"Matchweek grade <strong>[+-]\d+</strong> \(result [+-]\d+, goal difference [+-]\d+, goals [+-]\d+\)", page)
    assert "Grade</th>" not in build(site_data(), "full")["index.html"]   # no grader, no column


def test_signed_never_shows_minus_zero():
    assert [website.signed(value) for value in (-0.3, 0.4, 12.6, -3.5)] == ["+0", "+0", "+13", "-4"]


def test_season_page_shows_season_grade_teams_matchweeks_and_games():
    ledger = ledger_frame()
    backtest = ledger[(ledger.variant == "base") & ledger.result.notna()]
    baseline = pd.DataFrame({"hg": [1, 0, 2, 1], "ag": [1, 0, 1, 2]})
    page = build(site_data(backtest=backtest, grader=Grader.from_matches(baseline)), "season")["season.html"]
    assert re.search(r'<span class="grade-value">[+-]\d+</span> season grade over 3 games in 2 matchweeks', page)
    for team in ("Arsenal", "Chelsea", "Leeds", "Fulham", "Old", "Town"):
        assert f"<td>{team}</td>" in page
    assert "Sat 21 Nov to Sun 22 Nov" in page                             # matchweek 2's dates
    assert "All 3 games, latest first" in page


def test_players_page_has_one_section_per_team_with_its_own_players():
    page = build(site_data(), "full")["players.html"]
    assert '<option value="arsenal">Arsenal</option>' in page and '<option value="chelsea">Chelsea</option>' in page
    arsenal = page[page.index('id="team-arsenal"'):page.index('id="team-chelsea"')]
    assert "Saka (-0.04 pts, -0.06 GD)" in arsenal and "Palmer" not in arsenal
    assert "How to read these tables" in page


def test_scores_page_ticks_only_the_current_season():
    stats = site_data().scorelines
    assert stats is not None
    fragment = chart_fragment(stats, {SEASON})
    assert f'"label": "{PREVIOUS_SEASON}", "color": ' in fragment
    assert re.search(rf'"label": "{PREVIOUS_SEASON}".*?"visible": false', fragment)
    assert re.search(rf'"label": "{SEASON}".*?"visible": true', fragment)
    assert '"visible": false' not in chart_fragment(stats)     # the standalone chart ticks every series
    assert '"visible": true' in build(site_data(), "full")["scores.html"]


def test_pages_build_with_no_data_and_say_so():
    empty = site_data(upcoming=None, ledger=pd.DataFrame(), player_impact=None, valuable_scorers=None, scorelines=None)
    pages = build(empty, "empty")
    assert "No games in the next days" in pages["index.html"]
    assert "No scored predictions yet" in pages["index.html"]
    assert "No player data yet" in pages["players.html"]
    assert "No results yet" in pages["scores.html"]


def test_template_with_an_unfilled_placeholder_stops_the_build(tmp_path):
    for name in website.STATIC_FILES + [website.FOOTER_TEMPLATE, "predictions.html", "players.html", "scores.html"]:
        (tmp_path / name).write_text((website.TEMPLATE_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "season.html").write_text("<p>${not_a_field}</p>", encoding="utf-8")
    with pytest.raises(KeyError, match="not_a_field"):
        website.Website(site_data(), website.SiteTemplates(tmp_path), UPDATED).build(tmp_path / "site")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
