#!/usr/local/bin/python3
"""Error paths around network failures: missing seasons, unavailable ESPN feeds."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from football_predictions import cups, sources
from football_predictions.main import Run
from football_predictions.sources import COLUMNS, FootballData, OpenFootball

HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500
SEASON = "2024-25"
DIVISION = "E0"


def _failing_download(status_code: int):
    """A stand-in for sources._download that fails with the given HTTP status."""
    def download(*_):
        response = requests.Response()
        response.status_code = status_code
        raise requests.HTTPError(f"HTTP {status_code}", response=response)
    return download


@pytest.mark.parametrize("source_cls", [FootballData, OpenFootball])
def test_missing_season_returns_empty_table(tmp_path, monkeypatch, source_cls):
    monkeypatch.setattr(sources, "_download", _failing_download(HTTP_NOT_FOUND))
    result = source_cls(tmp_path).results(SEASON, DIVISION, refresh=True)
    assert result.empty and list(result.columns) == COLUMNS


@pytest.mark.parametrize("source_cls", [FootballData, OpenFootball])
def test_other_http_errors_propagate(tmp_path, monkeypatch, source_cls):
    monkeypatch.setattr(sources, "_download", _failing_download(HTTP_SERVER_ERROR))
    with pytest.raises(requests.HTTPError):
        source_cls(tmp_path).results(SEASON, DIVISION, refresh=True)


def _broken_espn_client() -> Mock:
    """A stand-in for EspnClient whose feed is down: every call raises ConnectionError."""
    client = Mock()
    client.sync_season.side_effect = client.upcoming.side_effect = ConnectionError("feed down")
    return client


def test_cup_load_warns_and_continues_when_espn_fails(caplog):
    result = cups.load(_broken_espn_client(), ["eng.fa"], [SEASON], ["Arsenal"])
    assert result.empty and list(result.columns) == COLUMNS + cups.EXTRA_COLUMNS
    assert "ConnectionError: feed down" in caplog.text


def test_espn_schedule_failure_keeps_main_fixtures(tmp_path, monkeypatch, caplog):
    run = Run.__new__(Run)   # skip __init__: only the attributes _espn_fixtures reads are needed
    run.league = SimpleNamespace(espn="eng.1")
    run.cfg = SimpleNamespace(lineups={}, upcoming_days=7)
    run.season = SEASON
    run.source = FootballData(tmp_path)
    monkeypatch.setattr(run, "_espn", _broken_espn_client)
    fixtures = pd.DataFrame({"home": ["Arsenal"], "away": ["Chelsea"]})
    result = run._espn_fixtures(fixtures, model=None)
    assert result is fixtures
    assert "ESPN schedule unavailable (feed down)" in caplog.text


def _recording_run(espn_slug: str | None, lineup_model) -> tuple[Run, list[str]]:
    """A Run whose steps only record their names, in the order they run."""
    run, calls = Run.__new__(Run), []
    run.league = SimpleNamespace(espn=espn_slug)
    for step in ("settle", "backtest", "predict", "report", "site"):
        setattr(run, step, lambda *_, step=step, **__: calls.append(step))
    run._espn = Mock
    setattr(run, "_lineup_model", lambda client: (calls.append("lineup_model"), lineup_model(client)))
    return run, calls


def test_matchweek_runs_every_step_in_order():
    run, calls = _recording_run("eng.1", lambda client: None)
    run.matchweek()
    assert calls == ["settle", "backtest", "predict", "lineup_model", "report", "site"]


def test_matchweek_warns_and_still_builds_report_and_site_when_espn_fails(caplog):
    def broken(client):
        raise ConnectionError("feed down")
    run, calls = _recording_run("eng.1", broken)
    run.matchweek()
    assert calls[-2:] == ["report", "site"]
    assert "ESPN player data not refreshed" in caplog.text


def test_matchweek_skips_espn_for_a_league_without_a_slug():
    run, calls = _recording_run(None, lambda client: None)
    run.matchweek()
    assert "lineup_model" not in calls
