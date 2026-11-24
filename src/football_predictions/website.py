"""
Static website: the latest predictions, the last round's results, impact players and scoring statistics.

    football-predictions site [--league EPL]

Site layout (``site/``, set by ``site_dir`` in src/football_predictions.toml)::

    index.html    Predictions: upcoming games, then the last round, predicted vs. actual   <- templates/predictions.html
    season.html   Season predictions (placeholder)                                         <- templates/season.html
    players.html  Impact players, one team at a time                                       <- templates/players.html
    scores.html   Scoring statistics: the scoreline histogram, this season ticked           <- templates/scores.html
    style.css, players.js                                                                  <- copied as-is

Each page comes from its template in ``templates/`` (Python ``string.Template``, ``${placeholder}``).
``templates/footer.html`` goes into every page. A placeholder with no value stops the build.
The build reads the files in state/<league>/ and the cached results. It downloads nothing.
GitHub Pages publishes site/ when it is pushed (.github/workflows/pages.yml).
"""

import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

import pandas as pd

from custom_logger import custom_logger as logger

from . import report
from .histogram import ScorelineStats, chart_fragment
from .ledger import BASE_VARIANT
from .model import HIGHEST_PROB_SCORE_COLS

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
FOOTER_TEMPLATE = "footer.html"
STATIC_FILES = ["style.css", "players.js"]
LAST_ROUND_DAYS = 4             # the last round = settled games up to this many days before the latest result
MISSING = "–"
SCORE_SEPARATOR = " · "
DAY_FORMAT = "%a %d %b"
UPDATED_FORMAT = "%d %b %Y, %H:%M %Z"
NOTE_SEPARATOR = ": "           # report table titles read "<heading>: <note>"
HIT_CLASS, MISS_CLASS = "hit", "miss"
HIT_MARK, MISS_MARK = "✓", "✗"
NUM_CLASS = "num"
CURRENT_PAGE_ATTR = ' aria-current="page"'


@dataclass
class SiteData:
    """Everything the pages show. A frame left as None (or empty) shows as a short note."""
    league_name: str
    season: str
    high: float                 # confidence band thresholds
    medium: float
    upcoming: pd.DataFrame | None = None
    ledger: pd.DataFrame | None = None
    player_impact: pd.DataFrame | None = None
    valuable_scorers: pd.DataFrame | None = None
    defence_records: pd.DataFrame | None = None
    scorelines: ScorelineStats | None = None


class SiteTemplates:
    """The page templates in ``template_dir``, editable offline."""

    def __init__(self, template_dir: Path = TEMPLATE_DIR) -> None:
        self.template_dir = template_dir

    def render(self, name: str, **fields: str) -> str:
        """Fill template ``name``. Raises KeyError that names a placeholder with no value."""
        template = Template((self.template_dir / name).read_text(encoding="utf-8"))
        try:
            return template.substitute(fields)
        except KeyError as exc:
            raise KeyError(f"{self.template_dir / name} has placeholder {exc} that website.py does not fill") from exc

    def copy_static(self, name: str, site_dir: Path) -> None:
        shutil.copyfile(self.template_dir / name, site_dir / name)


# --------------------------------------------------------------------------------------- helpers
def html_table(frame: pd.DataFrame, col_classes: dict[str, str] | None = None,
               row_classes: list[str] | None = None, header: bool = True) -> str:
    """An HTML table of ``frame``, every value escaped. col_classes: column -> CSS class of its cells."""
    col_classes = col_classes or {}

    def attr(col: str) -> str:
        return f' class="{col_classes[col]}"' if col in col_classes else ""

    head = ("<thead><tr>" + "".join(f"<th{attr(col)}>{escape(str(col))}</th>" for col in frame.columns)
            + "</tr></thead>") if header else ""
    rows = []
    for i, values in enumerate(frame.itertuples(index=False)):
        row_attr = f' class="{row_classes[i]}"' if row_classes else ""
        cells = "".join(f"<td{attr(col)}>{escape(str(value))}</td>" for col, value in zip(frame.columns, values))
        rows.append(f"<tr{row_attr}>{cells}</tr>")
    return f'<div class="table-wrap"><table>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def note(text: str) -> str:
    return f'<p class="hint">{escape(text)}</p>'


def pct(value) -> str:
    return MISSING if pd.isna(value) else f"{value:.0%}"


def day(value) -> str:
    return pd.Timestamp(value).strftime(DAY_FORMAT)


def outcome_label(home: str, away: str, code: str) -> str:
    """The team name for a home or away win, or report.DRAW_LABEL."""
    return {"H": home, "A": away}.get(code, report.DRAW_LABEL)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def last_round(ledger: pd.DataFrame | None, days: int = LAST_ROUND_DAYS) -> pd.DataFrame:
    """The base model's settled predictions for games up to ``days`` days before the latest result."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    settled = ledger[(ledger.variant == BASE_VARIANT) & ledger.result.notna()]
    if settled.empty:
        return settled
    latest = settled.date.max()
    return settled[settled.date > latest - pd.Timedelta(days=days)].sort_values(["date", "home"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------- pages
class Page(ABC):
    """One page of the site: its file, its navigation label, its template and the values it fills."""
    filename: str
    nav_label: str
    template: str

    def __init__(self, data: SiteData) -> None:
        self.data = data

    @abstractmethod
    def fields(self) -> dict[str, str]:
        """The page's own placeholder values (HTML)."""


class PredictionsPage(Page):
    filename, nav_label, template = "index.html", "Predictions", "predictions.html"

    def fields(self) -> dict[str, str]:
        played = last_round(self.data.ledger)
        return {"current_table": self.current_table(), "last_round_title": self.last_round_title(played),
                "last_round_summary": self.last_round_summary(played), "last_round_table": self.last_round_table(played)}

    def current_table(self) -> str:
        upcoming = self.data.upcoming
        if upcoming is None or upcoming.empty:
            return note("No games in the next days (international break or end of season?).")
        disp = report.display_predictions(upcoming, self.data.high, self.data.medium).sort_values(["date", "home"])
        score_cols = [col for col in HIGHEST_PROB_SCORE_COLS if col in disp]
        has_odds = set(report.BOOKMAKER_COLS) <= set(disp)
        table = pd.DataFrame({
            "Date": disp.date.map(day), "Home": disp.home, "Away": disp.away, "Pick": disp.pick,
            "Confidence": disp.confidence.map(pct), "Band": disp.band,
            "Home win": disp.p_home.map(pct), "Draw": disp.p_draw.map(pct), "Away win": disp.p_away.map(pct),
            "Expected GD": disp.exp_gd.map("{:+.2f}".format), "GD 80% range": disp.gd_range_80,
            "Most likely scores": disp[score_cols].fillna("").agg(SCORE_SEPARATOR.join, axis=1),
            "Bookmaker H / D / A": disp.apply(
                lambda row: " / ".join(pct(row[col]) for col in report.BOOKMAKER_COLS)
                if has_odds and pd.notna(row.bk_home) else MISSING, axis=1),
        })
        num = dict.fromkeys(["Confidence", "Home win", "Draw", "Away win", "Expected GD"], NUM_CLASS)
        return html_table(table, num)

    @staticmethod
    def last_round_title(played: pd.DataFrame) -> str:
        if played.empty:
            return "Last round"
        first, last = played.date.min(), played.date.max()
        return f"Last round: {day(first)}" + ("" if first == last else f" to {day(last)}")

    @staticmethod
    def _scores(played: pd.DataFrame) -> pd.Series:
        return played.hg.astype(int).astype(str) + "-" + played.ag.astype(int).astype(str)

    def last_round_summary(self, played: pd.DataFrame) -> str:
        if played.empty:
            return note("No scored predictions yet.")
        hits = int((played.pick == played.result).sum())
        exact = int((played.likely_score == self._scores(played)).sum())
        return (f'<p class="summary-line"><strong>{hits} of {len(played)}</strong> picks right ({hits / len(played):.0%}). '
                f"<strong>{exact}</strong> exact score{'' if exact == 1 else 's'} (most likely score).</p>")

    def last_round_table(self, played: pd.DataFrame) -> str:
        if played.empty:
            return ""
        hit = (played.pick == played.result).to_numpy()
        table = pd.DataFrame({
            "Date": played.date.map(day), "Home": played.home, "Away": played.away,
            "Pick": [outcome_label(row.home, row.away, row.pick) for row in played.itertuples()],
            "Confidence": played.confidence.map(pct), "Predicted score": played.likely_score,
            "Actual score": self._scores(played),
            "Result": [outcome_label(row.home, row.away, row.result) for row in played.itertuples()],
            "Pick right?": [HIT_MARK if is_hit else MISS_MARK for is_hit in hit],
        })
        classes = {"Confidence": NUM_CLASS, "Pick": "pick", "Pick right?": "mark"}
        return html_table(table, classes, [HIT_CLASS if is_hit else MISS_CLASS for is_hit in hit])


class SeasonPage(Page):
    filename, nav_label, template = "season.html", "Season predictions", "season.html"

    def fields(self) -> dict[str, str]:
        return {}


class PlayersPage(Page):
    """Impact players: the Excel report's 'Impact Players' tables, one team at a time."""
    filename, nav_label, template = "players.html", "Impact players", "players.html"

    def report_tables(self) -> list[tuple[str, str, pd.DataFrame]]:
        """(heading, note, frame) per table; each frame has a label column, then one column per team."""
        frames: list[tuple[str | None, pd.DataFrame]] = []
        if self.data.player_impact is not None and not self.data.player_impact.empty:
            frames += report.impact_frames(self.data.player_impact)
        if self.data.valuable_scorers is not None and not self.data.valuable_scorers.empty:
            frames += report.scorer_frames(self.data.valuable_scorers, self.data.season)
        if self.data.defence_records is not None and not self.data.defence_records.empty:
            frames += report.defence_frames(self.data.defence_records, self.data.season)
        tables = []
        for title, frame in frames:
            heading, _, text = (title or "").partition(NOTE_SEPARATOR)
            tables.append((heading, text, frame))
        return tables

    def fields(self) -> dict[str, str]:
        tables = self.report_tables()
        teams = sorted({str(team) for _, _, frame in tables for team in frame.columns[1:]})
        if not teams:
            return {"team_options": "", "team_sections": note("No player data yet: run `football-predictions "
                                                              "lineups-backtest`."), "table_notes": ""}
        options = "".join(f'<option value="{slug(team)}">{escape(team)}</option>' for team in teams)
        sections = "".join(self.team_section(team, tables) for team in teams)
        notes = "".join(f"<li><strong>{escape(heading)}</strong>: {escape(text)}</li>"
                        for heading, text, _ in tables if text)
        return {"team_options": options, "team_sections": sections, "table_notes": f"<ul>{notes}</ul>"}

    @staticmethod
    def team_section(team: str, tables: list[tuple[str, str, pd.DataFrame]]) -> str:
        cards = []
        for heading, _, frame in tables:
            label_col = frame.columns[0]
            rows = frame[[label_col, team]] if team in frame else pd.DataFrame(columns=[label_col, team])
            rows = rows[rows[team].astype(str).str.len() > 0]
            body = html_table(rows, {label_col: "label"}, header=False) if not rows.empty else note("None this season.")
            cards.append(f'<div class="card"><h3>{escape(heading)}</h3>{body}</div>')
        return (f'<section class="team" id="team-{slug(team)}"><h2>{escape(team)}</h2>'
                f'<div class="cards">{"".join(cards)}</div></section>')


class ScoresPage(Page):
    """Scoring statistics: the scoreline histogram, with only the current season ticked at first."""
    filename, nav_label, template = "scores.html", "Scoring statistics", "scores.html"

    def fields(self) -> dict[str, str]:
        stats = self.data.scorelines
        if stats is None:
            return {"chart_title": "Final scores", "chart": note("No results yet.")}
        visible = {self.data.season} if self.data.season in stats.seasons else None
        return {"chart_title": escape(stats.title), "chart": chart_fragment(stats, visible)}


# ----------------------------------------------------------------------------------------- site
class Website:
    """Builds every page into a folder."""
    PAGE_CLASSES: tuple[type[Page], ...] = (PredictionsPage, SeasonPage, PlayersPage, ScoresPage)

    def __init__(self, data: SiteData, templates: SiteTemplates | None = None, updated: datetime | None = None) -> None:
        self.data = data
        self.templates = templates or SiteTemplates()
        self.updated = updated or datetime.now().astimezone()

    def build(self, site_dir: Path) -> list[Path]:
        """Write the pages and their static files into ``site_dir``. Returns the pages written."""
        site_dir.mkdir(parents=True, exist_ok=True)
        pages = [page_class(self.data) for page_class in self.PAGE_CLASSES]
        common = {"league_name": escape(self.data.league_name), "season": escape(self.data.season),
                  "updated": self.updated.strftime(UPDATED_FORMAT),
                  "footer": self.templates.render(FOOTER_TEMPLATE, year=str(self.updated.year))}
        written = []
        for page in pages:
            html = self.templates.render(page.template, nav=self.nav(pages, page), **common, **page.fields())
            path = site_dir / page.filename
            path.write_text(html, encoding="utf-8")
            written.append(path)
        for name in STATIC_FILES:
            self.templates.copy_static(name, site_dir)
        logger.info(f"website -> {site_dir} ({', '.join(path.name for path in written)})")
        return written

    @staticmethod
    def nav(pages: list[Page], current: Page) -> str:
        return "".join(f'<a href="{page.filename}"{CURRENT_PAGE_ATTR if page is current else ""}>'
                       f"{escape(page.nav_label)}</a>" for page in pages)
