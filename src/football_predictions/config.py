"""Load src/football_predictions.toml into typed objects."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start(season: str) -> int:
    return int(season[:4])


def shift_season(season: str, k: int) -> str:
    return season_label(season_start(season) + k)


def auto_season(today: date | None = None) -> str:
    today = today or date.today()
    return season_label(today.year if today.month >= 7 else today.year - 1)


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

    def league(self, key: str) -> League:
        for lg in self.leagues:
            if lg.key.lower() == key.lower():
                return lg
        raise KeyError(f"unknown league {key!r}; known: {[l.key for l in self.leagues]}")


def load_config(path: str | Path = Path(__file__).resolve().parent.parent / "football_predictions.toml", today: date | None = None) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    g = raw["general"]
    root = path.parent
    leagues = []
    for lg in raw.get("leagues", []):
        cs = lg.get("current_season", "auto")
        divisions = {k: v for k, v in lg.items() if isinstance(v, list) and k != "cups"}
        leagues.append(League(lg["key"], lg.get("name", lg["key"]),
                              auto_season(today) if cs == "auto" else cs,
                              bool(lg.get("enabled", True)), divisions, lg.get("espn"), list(lg.get("cups", []))))
    return Config(
        root=root,
        source=g.get("source", "football_data"),
        data_dir=(root / g.get("data_dir", "../data")).resolve(),
        state_dir=(root / g.get("state_dir", "../state")).resolve(),
        reports_dir=(root / g.get("reports_dir", "../reports")).resolve(),
        history_seasons=int(g.get("history_seasons", 2)),
        upcoming_days=int(g.get("upcoming_days", 10)),
        model=raw.get("model", {}),
        confidence=raw.get("confidence", {"high": 0.6, "medium": 0.45}),
        tune_grid=raw.get("tune", {}),
        leagues=leagues,
        lineups=raw.get("lineups", {}),
    )
