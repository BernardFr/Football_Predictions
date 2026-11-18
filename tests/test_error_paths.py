#!/usr/local/bin/python3
"""Error paths around network failures: missing seasons, unavailable ESPN feeds."""

from types import SimpleNamespace

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
    def download(url, path, timeout=None):
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


class _BrokenEspnClient:
    def sync_season(self, *args, **kwargs):
        raise ConnectionError("feed down")

    def upcoming(self, *args, **kwargs):
        raise ConnectionError("feed down")


def test_cup_load_warns_and_continues_when_espn_fails():
    messages = []
    result = cups.load(_BrokenEspnClient(), ["eng.fa"], [SEASON], ["Arsenal"], log=messages.append)
    assert result.empty and list(result.columns) == COLUMNS + cups.EXTRA_COLUMNS
    assert any("ConnectionError: feed down" in message for message in messages)


def test_espn_schedule_failure_keeps_main_fixtures(tmp_path, monkeypatch, capsys):
    run = Run.__new__(Run)   # skip __init__: only the attributes _espn_fixtures reads are needed
    run.league = SimpleNamespace(espn="eng.1")
    run.cfg = SimpleNamespace(lineups={}, upcoming_days=7)
    run.season = SEASON
    run.source = FootballData(tmp_path)
    monkeypatch.setattr(run, "_espn", _BrokenEspnClient)
    fixtures = pd.DataFrame({"home": ["Arsenal"], "away": ["Chelsea"]})
    result = run._espn_fixtures(fixtures, model=None)
    assert result is fixtures
    assert "ESPN schedule unavailable (feed down)" in capsys.readouterr().out
