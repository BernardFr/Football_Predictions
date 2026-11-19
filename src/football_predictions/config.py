"""Load src/football_predictions.toml into typed objects."""

import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "football_predictions.toml"
SEASON_START_MONTH = 7          # a season starting in year Y runs from July Y to June Y+1
AUTO_SEASON = "auto"            # current_season value meaning "work it out from today's date"
DEFAULT_SOURCE = "football_data"
DEFAULT_DATA_DIR = "../data"
DEFAULT_STATE_DIR = "../state"
DEFAULT_REPORTS_DIR = "../reports"
DEFAULT_HISTORY_SEASONS = 2
DEFAULT_UPCOMING_DAYS = 10
DEFAULT_CONFIDENCE = {"high": 0.6, "medium": 0.45}

def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start(season: str) -> int:
    return int(season[:4])


def shift_season(season: str, offset: int) -> str:
    return season_label(season_start(season) + offset)


def auto_season(today: date | None = None) -> str:
    today = today or date.today()
    return season_label(today.year if today.month >= SEASON_START_MONTH else today.year - 1)


@dataclass
class League:
    key: str
    name: str
    current_season: str
    enabled: bool
    divisions: dict[str, list[str]]   # source name -> [top, feeder, ...]
    espn: str | None = None           # ESPN league slug for lineups, e.g. "eng.1"
    cups: list[str] = field(default_factory=list)   # ESPN slugs of domestic cups used as extra data

    def divs(self, source: str) -> list[str]:
        if source not in self.divisions:
            raise KeyError(f"league {self.key} has no division codes for source {source!r}")
        return self.divisions[source]


@dataclass
class Config:
    root: Path
    source: str
    data_dir: Path
    state_dir: Path
    reports_dir: Path
    history_seasons: int
    upcoming_days: int
    model: dict
    confidence: dict
    tune_grid: dict
    leagues: list[League] = field(default_factory=list)
    lineups: dict = field(default_factory=dict)
    today: date = field(default_factory=date.today)
    now: datetime | None = None   # UTC, naive; None = the real clock (set only when replaying a past date)

    def league(self, key: str) -> League:
        for lg in self.leagues:
            if lg.key.lower() == key.lower():
                return lg
        raise KeyError(f"unknown league {key!r}; known: {[league.key for league in self.leagues]}")


def read_toml_config(path: str | Path = DEFAULT_CONFIG_PATH,
                     today: date | None = None, now: datetime | None = None) -> Config:
    today = today or date.today()
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    general = raw["general"]
    root = path.parent
    leagues = []
    for lg in raw.get("leagues", []):
        cs = lg.get("current_season", AUTO_SEASON)
        divisions = {key: value for key, value in lg.items() if isinstance(value, list) and key != "cups"}
        leagues.append(League(lg["key"], lg.get("name", lg["key"]),
                              auto_season(today) if cs == AUTO_SEASON else cs,
                              bool(lg.get("enabled", True)), divisions, lg.get("espn"), list(lg.get("cups", []))))
    return Config(
        root=root,
        source=general.get("source", DEFAULT_SOURCE),
        data_dir=(root / general.get("data_dir", DEFAULT_DATA_DIR)).resolve(),
        state_dir=(root / general.get("state_dir", DEFAULT_STATE_DIR)).resolve(),
        reports_dir=(root / general.get("reports_dir", DEFAULT_REPORTS_DIR)).resolve(),
        history_seasons=int(general.get("history_seasons", DEFAULT_HISTORY_SEASONS)),
        upcoming_days=int(general.get("upcoming_days", DEFAULT_UPCOMING_DAYS)),
        model=raw.get("model", {}),
        confidence=raw.get("confidence", dict(DEFAULT_CONFIDENCE)),
        tune_grid=raw.get("tune", {}),
        leagues=leagues,
        lineups=raw.get("lineups", {}),
        today=today,
        now=now,
    )
