"""
Histogram of final scorelines (home-away) in the top division: the past two seasons and this one.

    football-score-histogram [--league EPL] [--seasons 3] [--top N] [--source openfootball]

Scorelines run left to right from the heaviest home defeat to the biggest home win (by goal
difference). Within one goal difference, defeats put the highest-scoring result first
(..., 1-2, 0-1) and wins put it last (1-0, 2-1, ...); draws go 0-0, 1-1, 2-2, ...

Writes, in reports/:
  <league>_score_histogram.html  interactive chart: tick any season(s) and/or all seasons combined
  <league>_score_histogram.xlsx  tables + a native Excel chart (Excel's chart filter toggles series)
  <league>_score_histogram.png   all seasons combined as bars, each season as markers
Seasons are compared as a share of matches played, since the current season is only part-way through.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from custom_logger import custom_logger as logger
from custom_logger import setup_logger

from .config import DEFAULT_CONFIG_PATH, Config, League, read_toml_config, shift_season
from .report import DATE_FORMAT, FLOAT_FORMAT, HEADER_COLOR, HEADER_FONT_COLOR, PCT_FORMAT, TITLE_FONT_SIZE
from .sources import load_matches, make_source

PROG_NAME = "football-score-histogram"
DEFAULT_LEAGUE = "EPL"
DEFAULT_SEASONS = 3
DEFAULT_TOP = 0                      # 0 = show every scoreline
FILE_STEM = "{league}_score_histogram"
SEASON_COLORS = ["#DCE6F2", "#B4C9E4", "#7FA3CF", "#4A78B5", "#1F4E8C"]   # light (oldest) .. dark (current)
COMBINED = "All seasons"
COMBINED_LABEL = "All seasons combined"
COMBINED_COLOR = "#E07B39"
OTHER = "Other"
DRAW = "Draw"
GRID_CAP = 6                         # goals at or above this are pooled as "6+" in the home x away grid

# Excel
SHEET_SCORELINES = "Scorelines"
SHEET_SUMMARY = "Summary"
SHEET_GRID = "Home x away grid"
SHEET_MATCHES = "Matches"
FIRST_COL_WIDTH = 22
COL_WIDTH = 14
HEADER_ROW_HEIGHT = 32
FREEZE_SCORELINES = "B3"
FREEZE_MATCHES = "A2"
AXIS_PCT_FORMAT = "0%"
CHART_GAP_WIDTH = 40
CHART_MIN_WIDTH = 30
CHART_WIDTH_PER_SCORE = 0.9
CHART_HEIGHT = 12
CHART_ROWS_BELOW_TABLE = 6

# PNG
PNG_MIN_WIDTH_IN = 12
PNG_WIDTH_PER_SCORE_IN = 0.36
PNG_WIDTH_MARGIN_IN = 2
PNG_HEIGHT_IN = 5.5
PNG_DPI = 150
BAR_WIDTH = 0.75
BAR_ALPHA = 0.85
MARKER_SIZE = 22
MARKER_EDGE_WIDTH = 0.5
GRIDLINE_COLOR = "#e5e5e5"
GRIDLINE_WIDTH = 0.8
DRAW_BAND_COLOR = "#f1f1f1"
LABEL_COLOR = "#666666"
LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE_PNG = 13
REGION_LABEL_OFFSET = 0.7            # gap between the draw band and the "Home defeats"/"Home wins" labels

# HTML
HTML_MIN_WIDTH_PX = 720
HTML_PX_PER_SCORE = 22
SHARE_DECIMALS = 6


def season_colors(n_seasons: int) -> list[str]:
    return SEASON_COLORS[-n_seasons:] if n_seasons <= len(SEASON_COLORS) else SEASON_COLORS[:1] * (n_seasons - len(SEASON_COLORS)) + SEASON_COLORS


def order_key(score: str) -> tuple:
    """Heaviest home defeat first, biggest home win last (see module docstring for ties)."""
    if score == OTHER:
        return (np.inf, 0)
    home_goals, away_goals = map(int, score.split("-"))
    gd, total = home_goals - away_goals, home_goals + away_goals
    return (gd, -total) if gd < 0 else (gd, total)


def result_of(score: str) -> str:
    if score == OTHER:
        return ""
    home_goals, away_goals = map(int, score.split("-"))
    return "Home win" if home_goals > away_goals else DRAW if home_goals == away_goals else "Home defeat"


def scoreline_table(matches: pd.DataFrame, seasons: list[str], top: int = DEFAULT_TOP) -> pd.DataFrame:
    """
    One row per scoreline in display order. top > 0 keeps only the `top` most common scorelines
    and pools the rest as 'Other' (placed last).
    """
    matches = matches.assign(score=matches.hg.astype(int).astype(str) + "-" + matches.ag.astype(int).astype(str))
    counts = matches.pivot_table(index="score", columns="season", values="home", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=seasons, fill_value=0)
    counts[COMBINED] = counts.sum(axis=1)
    if top and len(counts) > top:
        keep = counts[COMBINED].sort_values(ascending=False).index[:top]
        rest = counts.drop(index=keep).sum().to_frame(OTHER).T
        counts = pd.concat([counts.loc[keep], rest])
    counts = counts.loc[sorted(counts.index, key=order_key)]
    share = counts / counts.sum()
    out = pd.DataFrame({"scoreline": counts.index})
    out["goal difference"] = [None if score == OTHER else int(score.split("-")[0]) - int(score.split("-")[1]) for score in out.scoreline]
    out["result"] = [result_of(score) for score in out.scoreline]
    for series_name in seasons + [COMBINED]:
        out[f"{series_name} matches"] = counts[series_name].to_numpy()
    for series_name in seasons + [COMBINED]:
        out[f"{series_name} %"] = share[series_name].to_numpy()
    return out


def grid(matches: pd.DataFrame, cap: int = GRID_CAP) -> pd.DataFrame:
    """Home goals (rows) x away goals (columns), all seasons; cap+ is pooled."""
    lab = lambda goals: str(goals) if goals < cap else f"{cap}+"     # noqa: E731
    order = [lab(goals) for goals in range(cap + 1)]
    table = pd.crosstab(matches.hg.clip(upper=cap).map(lab), matches.ag.clip(upper=cap).map(lab))
    table = table.reindex(index=order, columns=order, fill_value=0)
    table.index.name, table.columns.name = "home goals \\ away goals", None
    return table


def summary(matches: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    rows = []
    for series_name, subset in [*((series_name, matches[matches.season == series_name]) for series_name in seasons), (COMBINED, matches)]:
        rows.append({"season": series_name, "matches": len(subset), "goals per match": (subset.hg + subset.ag).mean(),
                     "home win %": (subset.hg > subset.ag).mean(), "draw %": (subset.hg == subset.ag).mean(),
                     "away win %": (subset.hg < subset.ag).mean(), "0-0 %": ((subset.hg == 0) & (subset.ag == 0)).mean(),
                     "last match": subset.date.max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- Excel
def write_excel(path: Path, title: str, table: pd.DataFrame, grd: pd.DataFrame, summ: pd.DataFrame,
                matches: pd.DataFrame, seasons: list[str]):
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head = PatternFill("solid", fgColor=HEADER_COLOR)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name=SHEET_SCORELINES, index=False, startrow=1)
        summ.to_excel(xw, sheet_name=SHEET_SUMMARY, index=False, startrow=1)
        grd.to_excel(xw, sheet_name=SHEET_GRID, startrow=1)
        matches.to_excel(xw, sheet_name=SHEET_MATCHES, index=False)

        for name, df, note in ((SHEET_SCORELINES, table, f"{title}: share of matches by final score, "
                                                         f"heaviest home defeat (left/top) to biggest home win"),
                               (SHEET_SUMMARY, summ, f"{title}: per-season summary"),
                               (SHEET_GRID, grd, f"{title}: matches by home and away goals, all seasons")):
            ws = xw.sheets[name]
            ws["A1"] = note
            ws["A1"].font = Font(bold=True, size=TITLE_FONT_SIZE)
            ncols = len(df.columns) + (1 if name == SHEET_GRID else 0)
            for j in range(1, ncols + 1):
                cell = ws.cell(row=2, column=j)
                cell.font, cell.fill = Font(bold=True, color=HEADER_FONT_COLOR), head
                cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
                ws.column_dimensions[get_column_letter(j)].width = FIRST_COL_WIDTH if j == 1 else COL_WIDTH
            ws.row_dimensions[2].height = HEADER_ROW_HEIGHT
            for j, col in enumerate(df.columns, start=1 + (name == SHEET_GRID)):
                fmt = (PCT_FORMAT if str(col).endswith("%") else
                       FLOAT_FORMAT if pd.api.types.is_float_dtype(df[col]) and col != "goal difference" else
                       DATE_FORMAT if pd.api.types.is_datetime64_any_dtype(df[col]) else None)
                if fmt:
                    for i in range(len(df)):
                        ws.cell(row=3 + i, column=j).number_format = fmt
            ws.freeze_panes = FREEZE_SCORELINES
        xw.sheets[SHEET_MATCHES].freeze_panes = FREEZE_MATCHES

        # native Excel chart: one series per season plus all seasons combined (% of matches).
        # Excel's chart filter button (or clicking the legend in newer Excel) shows/hides series.
        ws = xw.sheets[SHEET_SCORELINES]
        n_rows = len(table)
        col_lst = list(table.columns)
        chart = BarChart()
        chart.type, chart.grouping = "col", "clustered"
        chart.title = f"{title}: scorelines, heaviest home defeat to biggest home win"
        chart.y_axis.title, chart.x_axis.title = "Share of matches", "Final score (home-away)"
        chart.y_axis.number_format = AXIS_PCT_FORMAT
        chart.y_axis.majorGridlines = None
        colors = season_colors(len(seasons)) + [COMBINED_COLOR]
        for k, series_name in enumerate(seasons + [COMBINED]):
            col = col_lst.index(f"{series_name} %") + 1
            chart.add_data(Reference(ws, min_col=col, min_row=2, max_row=2 + n_rows), titles_from_data=True)
            chart.series[-1].graphicalProperties.solidFill = colors[k].lstrip("#")
            chart.series[-1].graphicalProperties.line.noFill = True
        chart.set_categories(Reference(ws, min_col=1, min_row=3, max_row=2 + n_rows))
        chart.gapWidth, chart.overlap = CHART_GAP_WIDTH, 0
        chart.width, chart.height = max(CHART_MIN_WIDTH, round(CHART_WIDTH_PER_SCORE * n_rows)), CHART_HEIGHT
        chart.legend.position = "b"
        ws.add_chart(chart, f"A{n_rows + CHART_ROWS_BELOW_TABLE}")


# ----------------------------------------------------------------------------------------- PNG
def write_png(path: Path, title: str, table: pd.DataFrame, seasons: list[str], counts: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(max(PNG_MIN_WIDTH_IN, PNG_WIDTH_PER_SCORE_IN * len(table) + PNG_WIDTH_MARGIN_IN),
                                    PNG_HEIGHT_IN), dpi=PNG_DPI)
    _shade_draws(ax, table)
    total = sum(counts.values())
    ax.bar(x, table[f"{COMBINED} %"] * 100, BAR_WIDTH, color=COMBINED_COLOR, alpha=BAR_ALPHA, edgecolor="none",
           label=f"{COMBINED_LABEL} ({total} matches)", zorder=2)
    for color, series_name in zip(season_colors(len(seasons)), seasons):
        y = table[f"{series_name} %"].to_numpy() * 100
        seen = y > 0                       # no marker where that season had no such score
        ax.scatter(x[seen], y[seen], s=MARKER_SIZE, color=color, edgecolor=f"#{HEADER_COLOR}", linewidth=MARKER_EDGE_WIDTH,
                   label=f"{series_name} ({counts[series_name]} matches)", zorder=3)
    ax.set_xticks(x, table.scoreline, rotation=90)
    ax.set_ylabel("Share of matches (%)")
    ax.set_xlabel("Final score (home-away): heaviest home defeat → biggest home win")
    ax.set_title(title, loc="left", fontsize=TITLE_FONT_SIZE_PNG, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRIDLINE_COLOR, linewidth=GRIDLINE_WIDTH)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _shade_draws(ax, table):
    draws = np.flatnonzero(table.result.to_numpy() == DRAW)
    if len(draws):
        ax.axvspan(draws.min() - 0.5, draws.max() + 0.5, color=DRAW_BAND_COLOR, zorder=0)
        ymax = 1.0
        for label, xpos, ha in (("Home defeats", draws.min() - REGION_LABEL_OFFSET, "right"), ("Draws", (draws.min() + draws.max()) / 2, "center"),
                                ("Home wins", draws.max() + REGION_LABEL_OFFSET, "left")):
            ax.text(xpos, ymax, label, transform=ax.get_xaxis_transform(), ha=ha, va="bottom",
                    fontsize=LABEL_FONT_SIZE, color=LABEL_COLOR)


# ---------------------------------------------------------------------------------------- HTML
# The chart alone, without a page around it: the standalone HTML file and the website's Scoring
# statistics page both embed it. The host page defines the colors --fg, --bg, --muted, --line, --band.
CHART_FRAGMENT = """<div class="score-chart">
<style>
  .score-chart .controls { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin-bottom:12px; }
  .score-chart .controls label { display:inline-flex; align-items:center; gap:6px; cursor:pointer; user-select:none; }
  .score-chart .series-swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
  .score-chart .sep { width:1px; height:20px; background:var(--line); }
  .score-chart button { font:inherit; padding:3px 10px; border:1px solid var(--line); border-radius:6px;
           background:transparent; color:var(--fg); cursor:pointer; }
  .score-chart button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .score-chart .wrap { position:relative; height:520px; overflow-x:auto; }
  .score-chart .inner { position:relative; height:100%; min-width:__MINW__px; }
  .score-chart .chart-footer { margin-top:10px; color:var(--muted); font-size:12px; }
</style>
<div class="controls">
  <span id="series"></span>
  <span class="sep"></span>
  <button id="all">All</button><button id="none">None</button>
  <span class="sep"></span>
  <span>Show:</span>
  <button id="m-share" aria-pressed="true">% of matches</button><button id="m-count" aria-pressed="false">Match count</button>
</div>
<div class="wrap"><div class="inner"><canvas id="c"></canvas></div></div>
<p class="chart-footer">__FOOTER__</p>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let metric = "share";

const drawBand = {
  id: "drawBand",
  beforeDatasetsDraw(chart) {
    const idx = DATA.results.map((r, i) => r === "Draw" ? i : -1).filter(i => i >= 0);
    if (!idx.length) return;
    const x = chart.scales.x, {top, bottom} = chart.chartArea, ctx = chart.ctx;
    const step = x.getPixelForValue(1) - x.getPixelForValue(0);
    const left = x.getPixelForValue(idx[0]) - step / 2, right = x.getPixelForValue(idx[idx.length - 1]) + step / 2;
    ctx.save(); ctx.fillStyle = css("--band"); ctx.fillRect(left, top, right - left, bottom - top);
    ctx.fillStyle = css("--muted"); ctx.font = "12px sans-serif"; ctx.textBaseline = "top";
    ctx.textAlign = "right"; ctx.fillText("◀ home defeats", left - 6, top + 2);
    ctx.textAlign = "center"; ctx.fillText("draws", (left + right) / 2, top + 2);
    ctx.textAlign = "left"; ctx.fillText("home wins ▶", right + 6, top + 2);
    ctx.restore();
  }
};

const datasets = DATA.series.map(s => ({
  label: s.label, data: s.share.map(v => v * 100), backgroundColor: s.color, borderWidth: 0,
  hidden: !s.visible, categoryPercentage: 0.85, barPercentage: 0.95,
}));

const chart = new Chart(document.getElementById("c"), {
  type: "bar",
  data: { labels: DATA.labels, datasets },
  plugins: [drawBand],
  options: {
    responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
    layout: { padding: { top: 22 } },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => {
        const s = DATA.series[c.datasetIndex], i = c.dataIndex;
        return `${s.label}: ${(s.share[i] * 100).toFixed(1)}% (${s.count[i]} of ${s.n} matches)`;
      }, title: items => `${items[0].label}  ·  ${DATA.results[items[0].dataIndex] || "pooled rare scores"}` } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: css("--fg"), autoSkip: false, maxRotation: 90, minRotation: 0 } },
      y: { beginAtZero: true, grace: "6%", grid: { color: css("--line") }, ticks: { color: css("--muted"), callback: v => metric === "share" ? v + "%" : v },
           title: { display: true, text: "Share of matches", color: css("--muted") } }
    }
  }
});

const box = document.getElementById("series");
DATA.series.forEach((s, i) => {
  const l = document.createElement("label");
  l.innerHTML = `<input type="checkbox" ${s.visible ? "checked" : ""}><span class="series-swatch" style="background:${s.color}"></span>${s.label} <span style="color:var(--muted)">(${s.n})</span>`;
  l.querySelector("input").addEventListener("change", e => { chart.setDatasetVisibility(i, e.target.checked); chart.update(); });
  box.appendChild(l); box.appendChild(document.createTextNode(" "));
});
const setAll = on => { box.querySelectorAll("input").forEach((el, i) => { el.checked = on; chart.setDatasetVisibility(i, on); }); chart.update(); };
document.getElementById("all").onclick = () => setAll(true);
document.getElementById("none").onclick = () => setAll(false);
const setMetric = m => {
  metric = m;
  document.getElementById("m-share").setAttribute("aria-pressed", m === "share");
  document.getElementById("m-count").setAttribute("aria-pressed", m === "count");
  chart.data.datasets.forEach((d, i) => d.data = m === "share" ? DATA.series[i].share.map(v => v * 100) : DATA.series[i].count);
  chart.options.scales.y.title.text = m === "share" ? "Share of matches" : "Number of matches";
  chart.update();
};
document.getElementById("m-share").onclick = () => setMetric("share");
document.getElementById("m-count").onclick = () => setMetric("count");
</script>
</div>
"""

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ scorelines</title>
<style>
  :root { --bg:#ffffff; --fg:#1d2733; --muted:#5f6b7a; --line:#e3e7ec; --band:rgba(120,130,140,.09); }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14181d; --fg:#e6eaef; --muted:#9aa5b1; --line:#2a3139; --band:rgba(200,210,220,.07); }
  }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  main { max-width:1400px; margin:0 auto; padding:24px 16px 32px; }
  h1 { font-size:20px; margin:0 0 4px; }
  p.sub { margin:0 0 16px; color:var(--muted); }
</style>
</head>
<body>
<main>
  <h1>__TITLE__: final scores</h1>
  <p class="sub">Left to right: heaviest home defeat to biggest home win. The shaded band is draws.</p>
  __CHART__
</main>
</body>
</html>
"""


def chart_fragment(stats: "ScorelineStats", visible: set[str] | None = None) -> str:
    """The interactive chart as an HTML fragment. visible: the seasons ticked at first (None = every series)."""
    color_lst = season_colors(len(stats.seasons)) + [COMBINED_COLOR]
    series = []
    for color, series_name in zip(color_lst, stats.seasons + [COMBINED]):
        series.append({"label": COMBINED_LABEL if series_name == COMBINED else series_name, "color": color,
                       "n": int(sum(stats.counts.values())) if series_name == COMBINED else int(stats.counts[series_name]),
                       "share": [round(float(value), SHARE_DECIMALS) for value in stats.table[f"{series_name} %"]],
                       "count": [int(value) for value in stats.table[f"{series_name} matches"]],
                       "visible": visible is None or series_name in visible})
    data = {"labels": stats.table.scoreline.tolist(), "results": stats.table.result.tolist(), "series": series}
    return (CHART_FRAGMENT.replace("__DATA__", json.dumps(data))
                          .replace("__MINW__", str(max(HTML_MIN_WIDTH_PX, HTML_PX_PER_SCORE * len(stats.table))))
                          .replace("__FOOTER__", stats.footer))


def write_html(path: Path, stats: "ScorelineStats"):
    path.write_text(HTML.replace("__TITLE__", stats.title).replace("__CHART__", chart_fragment(stats)), encoding="utf-8")


# ---------------------------------------------------------------------------------------- data
@dataclass
class ScorelineStats:
    """The top division's final scores over several seasons, ending with the current one."""
    league_name: str
    source_name: str
    seasons: list[str]          # oldest first; only seasons with at least one match
    matches: pd.DataFrame
    table: pd.DataFrame         # scoreline_table()
    counts: dict[str, int]      # matches per season

    @classmethod
    def load(cls, cfg: Config, league: League, source, n_seasons: int = DEFAULT_SEASONS, top: int = DEFAULT_TOP,
             refresh: bool = True) -> "ScorelineStats | None":
        """Results played before cfg.today; None when there are none. refresh re-downloads the current season."""
        top_div = league.divs(source.name)[0]
        seasons = [shift_season(league.current_season, -k) for k in range(n_seasons - 1, -1, -1)]
        matches = load_matches(source, seasons, [top_div], refresh_season=league.current_season if refresh else None)
        matches = matches[(matches["div"] == top_div) & (matches.date < pd.Timestamp(cfg.today))]
        if matches.empty:
            logger.warning(f"no {league.name} results found for {seasons}")
            return None
        seasons = [season_name for season_name in seasons if (matches.season == season_name).any()]
        return cls(league.name, source.name, seasons, matches, scoreline_table(matches, seasons, top),
                   {str(season_name): int(count) for season_name, count in matches.groupby("season").size().items()})

    @property
    def title(self) -> str:
        return f"{self.league_name}, {self.seasons[0]} to {self.seasons[-1]}"

    @property
    def footer(self) -> str:
        return (f"Source: {self.source_name}, results up to {self.matches.date.max():%d %b %Y}. "
                + ", ".join(f"{season_name}: {self.counts[season_name]} matches" for season_name in self.seasons) + ".")


# ---------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    setup_logger(DEFAULT_CONFIG_PATH.stem)   # football_predictions.toml -> football_predictions.log
    ap = argparse.ArgumentParser(prog=PROG_NAME, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--league", default=DEFAULT_LEAGUE)
    ap.add_argument("--seasons", type=int, default=DEFAULT_SEASONS,
                    help=f"number of seasons, ending with the current one (default {DEFAULT_SEASONS})")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP,
                    help="keep only the N most common scorelines and pool the rest as 'Other' (default: show all)")
    ap.add_argument("--source", help="override the data source in the config file")
    ap.add_argument("--today", help="pretend today is this date (YYYY-MM-DD)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached results; don't re-download this season")
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    cfg = read_toml_config(args.config, today)
    lg = cfg.league(args.league)
    source = make_source(args.source or cfg.source, cfg.data_dir)
    stats = ScorelineStats.load(cfg, lg, source, args.seasons, args.top, refresh=not args.no_refresh)
    if stats is None:
        return 1

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.reports_dir / FILE_STEM.format(league=lg.key)
    raw = stats.matches[["date", "season", "home", "away", "hg", "ag"]].rename(
        columns={"hg": "home goals", "ag": "away goals"}).reset_index(drop=True)
    write_html(stem.with_suffix(".html"), stats)
    write_excel(stem.with_suffix(".xlsx"), stats.title, stats.table, grid(stats.matches),
                summary(stats.matches, stats.seasons), raw, stats.seasons)
    write_png(stem.with_suffix(".png"), stats.title, stats.table, stats.seasons, stats.counts)

    logger.info(f"{stats.title}: " + ", ".join(f"{season_name} {stats.counts[season_name]} matches" for season_name in stats.seasons))
    logger.info("order: " + " ".join(stats.table.scoreline))
    logger.info(f"chart (interactive) -> {stem.with_suffix('.html')}\nexcel -> {stem.with_suffix('.xlsx')}\n"
                f"image -> {stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
