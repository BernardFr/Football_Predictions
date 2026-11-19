"""
Data sources -> one standard match table.

Standard columns: date, season, div, home, away, hg, ag, odds_h, odds_d, odds_a
(odds are decimal bookmaker odds; NaN when the source has none).
"""

import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import season_start

COLUMNS = ["date", "season", "div", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a"]
ODDS_COLS = ["odds_h", "odds_d", "odds_a"]
UA = {"User-Agent": "Mozilla/5.0 (Football_Predictions; personal research)"}
DOWNLOAD_TIMEOUT_SECONDS = 60
HTTP_NOT_FOUND = 404
CSV_ENCODINGS = ("utf-8-sig", "latin-1")    # tried in order
MIN_DECIMAL_ODDS = 1                        # decimal odds at or below this are invalid
FOOTBALL_DATA_URL = "https://www.football-data.co.uk"
FOOTBALL_DATA_FIXTURES = "fixtures.csv"
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

# closing odds first (the sharpest benchmark), then opening; market average, then Pinnacle, then Bet365
ODDS_PRIORITY = [("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA"),
                 ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("B365H", "B365D", "B365A")]


def _download(url: str, path: Path, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> None:
    response = requests.get(url, headers=UA, timeout=timeout)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


class FootballData:
    """football-data.co.uk season CSVs (results + odds) and fixtures.csv (upcoming games + odds)."""
    name = "football_data"
    BASE = FOOTBALL_DATA_URL

    def __init__(self, cache_dir: Path):
        self.cache = Path(cache_dir) / self.name

    @staticmethod
    def code(season: str) -> str:
        start_year = season_start(season)
        return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        raw = path.read_bytes()
        for enc in CSV_ENCODINGS:
            try:
                return pd.read_csv(io.StringIO(raw.decode(enc)), on_bad_lines="skip")
            except UnicodeDecodeError:
                continue
        raise ValueError(f"cannot decode {path}")

    @staticmethod
    def _odds(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=df.index, columns=ODDS_COLS)
        for cols in ODDS_PRIORITY:
            if all(col_name in df for col_name in cols):
                vals = df[list(cols)].apply(pd.to_numeric, errors="coerce")
                fill = out.odds_h.isna() & vals.notna().all(axis=1) & (vals > MIN_DECIMAL_ODDS).all(axis=1)
                out.loc[fill, :] = vals.loc[fill].to_numpy()
        return out

    def _standard(self, df: pd.DataFrame, season: str, div: str, played: bool) -> pd.DataFrame:
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])
        if played:
            df = df.dropna(subset=["FTHG", "FTAG"])
        if df.empty:
            return _empty()
        out = pd.DataFrame({
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "season": season, "div": div,
            "home": df["HomeTeam"].str.strip(), "away": df["AwayTeam"].str.strip(),
            "hg": pd.to_numeric(df["FTHG"], errors="coerce") if played else np.nan,
            "ag": pd.to_numeric(df["FTAG"], errors="coerce") if played else np.nan,
        })
        out = pd.concat([out, self._odds(df)], axis=1)
        return out.dropna(subset=["date"])[COLUMNS].reset_index(drop=True)

    def results(self, season: str, div: str, refresh: bool = False) -> pd.DataFrame:
        path = self.cache / self.code(season) / f"{div}.csv"
        if refresh or not path.exists():
            try:
                _download(f"{self.BASE}/mmz4281/{self.code(season)}/{div}.csv", path)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == HTTP_NOT_FOUND:
                    return _empty()    # season not started / not published yet
                raise
        return self._standard(self._read(path), season, div, played=True)

    def fixtures(self, season: str, div: str, today: date, days: int) -> pd.DataFrame:
        path = self.cache / FOOTBALL_DATA_FIXTURES
        _download(f"{self.BASE}/{FOOTBALL_DATA_FIXTURES}", path)
        df = self._read(path)
        df = df[df["Div"].astype(str).str.strip() == div]
        out = self._standard(df, season, div, played=False)
        start = pd.Timestamp(today)
        return out[(out.date >= start) & (out.date <= start + pd.Timedelta(days=days))].reset_index(drop=True)


class OpenFootball:
    """github.com/openfootball/football.json — results and full schedules, no odds. Can lag by weeks."""
    name = "openfootball"
    BASE = OPENFOOTBALL_URL

    def __init__(self, cache_dir: Path):
        self.cache = Path(cache_dir) / self.name

    def _load(self, season: str, div: str, refresh: bool) -> list[dict]:
        path = self.cache / season / f"{div}.json"
        if refresh or not path.exists():
            try:
                _download(f"{self.BASE}/{season}/{div}.json", path)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == HTTP_NOT_FOUND:
                    return []
                raise
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "matches" in data:
            return data["matches"]
        return [match for round_entry in data.get("rounds", []) for match in round_entry.get("matches", [])]

    def _frame(self, matches, season, div, played) -> pd.DataFrame:
        rows = []
        for match in matches:
            ft = (match.get("score") or {}).get("ft") if isinstance(match.get("score"), dict) else None
            if bool(ft) != played:
                continue
            rows.append({"date": pd.Timestamp(match["date"]), "season": season, "div": div,
                         "home": match["team1"], "away": match["team2"],
                         "hg": ft[0] if ft else np.nan, "ag": ft[1] if ft else np.nan,
                         "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan})
        return pd.DataFrame(rows, columns=COLUMNS) if rows else _empty()

    def results(self, season: str, div: str, refresh: bool = False) -> pd.DataFrame:
        return self._frame(self._load(season, div, refresh), season, div, played=True)

    def fixtures(self, season: str, div: str, today: date, days: int) -> pd.DataFrame:
        df = self._frame(self._load(season, div, refresh=False), season, div, played=False)
        start = pd.Timestamp(today)
        return df[(df.date >= start) & (df.date <= start + pd.Timedelta(days=days))].reset_index(drop=True)


SOURCES = {cls.name: cls for cls in (FootballData, OpenFootball)}


def make_source(name: str, cache_dir: Path):
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; choose from {list(SOURCES)}")
    return SOURCES[name](cache_dir)


def load_matches(source, seasons: list[str], divisions: list[str], refresh_season: str | None = None) -> pd.DataFrame:
    frames = [source.results(season_name, div, refresh=(season_name == refresh_season)) for season_name in seasons for div in divisions]
    frames = [piece for piece in frames if not piece.empty]
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    df["hg"] = df.hg.astype(int)
    df["ag"] = df.ag.astype(int)
    return df.sort_values(["date", "div", "home"]).reset_index(drop=True)
