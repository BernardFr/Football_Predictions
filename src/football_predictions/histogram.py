"""Histogram of final scorelines (home-away) in the top division: the past two seasons and this one.

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
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, shift_season
from .sources import load_matches, make_source

SEASON_COLORS = ["#DCE6F2", "#B4C9E4", "#7FA3CF", "#4A78B5", "#1F4E8C"]   # light (oldest) .. dark (current)
COMBINED = "All seasons"
COMBINED_COLOR = "#E07B39"
OTHER = "Other"


def season_colors(k: int) -> list[str]:
    return SEASON_COLORS[-k:] if k <= len(SEASON_COLORS) else SEASON_COLORS[:1] * (k - len(SEASON_COLORS)) + SEASON_COLORS


def order_key(score: str) -> tuple:
    """Heaviest home defeat first, biggest home win last (see module docstring for ties)."""
    if score == OTHER:
        return (np.inf, 0)
    h, a = map(int, score.split("-"))
    gd, total = h - a, h + a
    return (gd, -total) if gd < 0 else (gd, total)


def result_of(score: str) -> str:
    if score == OTHER:
        return ""
    h, a = map(int, score.split("-"))
    return "Home win" if h > a else "Draw" if h == a else "Home defeat"


def scoreline_table(m: pd.DataFrame, seasons: list[str], top: int = 0) -> pd.DataFrame:
    """One row per scoreline in display order. top > 0 keeps only the `top` most common scorelines
    and pools the rest as 'Other' (placed last)."""
    m = m.assign(score=m.hg.astype(int).astype(str) + "-" + m.ag.astype(int).astype(str))
    counts = m.pivot_table(index="score", columns="season", values="home", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=seasons, fill_value=0)
    counts[COMBINED] = counts.sum(axis=1)
    if top and len(counts) > top:
        keep = counts[COMBINED].sort_values(ascending=False).index[:top]
        rest = counts.drop(index=keep).sum().to_frame(OTHER).T
        counts = pd.concat([counts.loc[keep], rest])
    counts = counts.loc[sorted(counts.index, key=order_key)]
    share = counts / counts.sum()
    out = pd.DataFrame({"scoreline": counts.index})
    out["goal difference"] = [None if s == OTHER else int(s.split("-")[0]) - int(s.split("-")[1]) for s in out.scoreline]
    out["result"] = [result_of(s) for s in out.scoreline]
    for s in seasons + [COMBINED]:
        out[f"{s} matches"] = counts[s].to_numpy()
    for s in seasons + [COMBINED]:
        out[f"{s} %"] = share[s].to_numpy()
    return out


def grid(m: pd.DataFrame, cap: int = 6) -> pd.DataFrame:
    """Home goals (rows) x away goals (columns), all seasons; cap+ is pooled."""
    lab = lambda g: str(g) if g < cap else f"{cap}+"     # noqa: E731
    order = [lab(g) for g in range(cap + 1)]
    g = pd.crosstab(m.hg.clip(upper=cap).map(lab), m.ag.clip(upper=cap).map(lab))
    g = g.reindex(index=order, columns=order, fill_value=0)
    g.index.name, g.columns.name = "home goals \\ away goals", None
    return g


def summary(m: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    rows = []
    for s, x in [*((s, m[m.season == s]) for s in seasons), (COMBINED, m)]:
        rows.append({"season": s, "matches": len(x), "goals per match": (x.hg + x.ag).mean(),
                     "home win %": (x.hg > x.ag).mean(), "draw %": (x.hg == x.ag).mean(),
                     "away win %": (x.hg < x.ag).mean(), "0-0 %": ((x.hg == 0) & (x.ag == 0)).mean(),
                     "last match": x.date.max()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- Excel
def write_excel(path: Path, title: str, table: pd.DataFrame, grd: pd.DataFrame, summ: pd.DataFrame,
                matches: pd.DataFrame, seasons: list[str]):
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head = PatternFill("solid", fgColor="1F3864")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Scorelines", index=False, startrow=1)
        summ.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
        grd.to_excel(xw, sheet_name="Home x away grid", startrow=1)
        matches.to_excel(xw, sheet_name="Matches", index=False)

        for name, df, note in (("Scorelines", table, f"{title}: share of matches by final score, "
                                                     f"heaviest home defeat (left/top) to biggest home win"),
                               ("Summary", summ, f"{title}: per-season summary"),
                               ("Home x away grid", grd, f"{title}: matches by home and away goals, all seasons")):
            ws = xw.sheets[name]
            ws["A1"] = note
            ws["A1"].font = Font(bold=True, size=12)
            ncols = len(df.columns) + (1 if name == "Home x away grid" else 0)
            for j in range(1, ncols + 1):
                c = ws.cell(row=2, column=j)
                c.font, c.fill = Font(bold=True, color="FFFFFF"), head
                c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
                ws.column_dimensions[get_column_letter(j)].width = 22 if j == 1 else 14
            ws.row_dimensions[2].height = 32
            for j, col in enumerate(df.columns, start=1 + (name == "Home x away grid")):
                fmt = ("0.0%" if str(col).endswith("%") else
                       "0.00" if pd.api.types.is_float_dtype(df[col]) and col != "goal difference" else
                       "yyyy-mm-dd" if pd.api.types.is_datetime64_any_dtype(df[col]) else None)
                if fmt:
                    for i in range(len(df)):
                        ws.cell(row=3 + i, column=j).number_format = fmt
            ws.freeze_panes = "B3"
        xw.sheets["Matches"].freeze_panes = "A2"

        # native Excel chart: one series per season plus all seasons combined (% of matches).
        # Excel's chart filter button (or clicking the legend in newer Excel) shows/hides series.
        ws = xw.sheets["Scorelines"]
        n = len(table)
        cols = list(table.columns)
        chart = BarChart()
        chart.type, chart.grouping = "col", "clustered"
        chart.title = f"{title}: scorelines, heaviest home defeat to biggest home win"
        chart.y_axis.title, chart.x_axis.title = "Share of matches", "Final score (home-away)"
        chart.y_axis.number_format = "0%"
        chart.y_axis.majorGridlines = None
        colors = season_colors(len(seasons)) + [COMBINED_COLOR]
        for k, s in enumerate(seasons + [COMBINED]):
            col = cols.index(f"{s} %") + 1
            chart.add_data(Reference(ws, min_col=col, min_row=2, max_row=2 + n), titles_from_data=True)
            chart.series[-1].graphicalProperties.solidFill = colors[k].lstrip("#")
            chart.series[-1].graphicalProperties.line.noFill = True
        chart.set_categories(Reference(ws, min_col=1, min_row=3, max_row=2 + n))
        chart.gapWidth, chart.overlap = 40, 0
        chart.width, chart.height = max(30, round(0.9 * n)), 12
        chart.legend.position = "b"
        ws.add_chart(chart, f"A{n + 6}")


# ----------------------------------------------------------------------------------------- PNG
def write_png(path: Path, title: str, table: pd.DataFrame, seasons: list[str], counts: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(max(12, 0.36 * len(table) + 2), 5.5), dpi=150)
    _shade_draws(ax, table)
    total = sum(counts.values())
    ax.bar(x, table[f"{COMBINED} %"] * 100, 0.75, color=COMBINED_COLOR, alpha=0.85, edgecolor="none",
           label=f"All seasons combined ({total} matches)", zorder=2)
    for c, s in zip(season_colors(len(seasons)), seasons):
        y = table[f"{s} %"].to_numpy() * 100
        seen = y > 0                       # no marker where that season had no such score
        ax.scatter(x[seen], y[seen], s=22, color=c, edgecolor="#1F3864", linewidth=0.5,
                   label=f"{s} ({counts[s]} matches)", zorder=3)
    ax.set_xticks(x, table.scoreline, rotation=90)
    ax.set_ylabel("Share of matches (%)")
    ax.set_xlabel("Final score (home-away): heaviest home defeat → biggest home win")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _shade_draws(ax, table):
    draws = np.flatnonzero(table.result.to_numpy() == "Draw")
    if len(draws):
        ax.axvspan(draws.min() - 0.5, draws.max() + 0.5, color="#f1f1f1", zorder=0)
        ymax = 1.0
        for label, xpos, ha in (("Home defeats", draws.min() - 0.7, "right"), ("Draws", (draws.min() + draws.max()) / 2, "center"),
                                ("Home wins", draws.max() + 0.7, "left")):
            ax.text(xpos, ymax, label, transform=ax.get_xaxis_transform(), ha=ha, va="bottom",
                    fontsize=9, color="#666666")


# ---------------------------------------------------------------------------------------- HTML
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ scorelines</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
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
  .controls { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; margin-bottom:12px; }
  .controls label { display:inline-flex; align-items:center; gap:6px; cursor:pointer; user-select:none; }
  .swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }
  .sep { width:1px; height:20px; background:var(--line); }
  button { font:inherit; padding:3px 10px; border:1px solid var(--line); border-radius:6px;
           background:transparent; color:var(--fg); cursor:pointer; }
  button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .wrap { position:relative; height:520px; overflow-x:auto; }
  .inner { position:relative; height:100%; min-width:__MINW__px; }
  footer { margin-top:10px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<main>
  <h1>__TITLE__: final scores</h1>
  <p class="sub">Left to right: heaviest home defeat to biggest home win. The shaded band is draws.</p>
  <div class="controls" id="controls">
    <span id="series"></span>
    <span class="sep"></span>
    <button id="all">All</button><button id="none">None</button>
    <span class="sep"></span>
    <span>Show:</span>
    <button id="m-share" aria-pressed="true">% of matches</button><button id="m-count" aria-pressed="false">Match count</button>
  </div>
  <div class="wrap"><div class="inner"><canvas id="c"></canvas></div></div>
  <footer>__FOOTER__</footer>
</main>
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
  l.innerHTML = `<input type="checkbox" ${s.visible ? "checked" : ""}><span class="swatch" style="background:${s.color}"></span>${s.label} <span style="color:var(--muted)">(${s.n})</span>`;
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
</body>
</html>
"""


def write_html(path: Path, title: str, table: pd.DataFrame, seasons: list[str], counts: dict, footer: str):
    colors = season_colors(len(seasons)) + [COMBINED_COLOR]
    series = []
    for c, s in zip(colors, seasons + [COMBINED]):
        series.append({"label": "All seasons combined" if s == COMBINED else s, "color": c,
                       "n": int(sum(counts.values())) if s == COMBINED else int(counts[s]),
                       "share": [round(float(v), 6) for v in table[f"{s} %"]],
                       "count": [int(v) for v in table[f"{s} matches"]],
                       "visible": True})
    data = {"labels": table.scoreline.tolist(), "results": table.result.tolist(), "series": series}
    html = (HTML.replace("__TITLE__", title)
                .replace("__DATA__", json.dumps(data))
                .replace("__MINW__", str(max(720, 22 * len(table))))
                .replace("__FOOTER__", footer))
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="football-score-histogram", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "football_predictions.toml"))
    ap.add_argument("--league", default="EPL")
    ap.add_argument("--seasons", type=int, default=3, help="number of seasons, ending with the current one (default 3)")
    ap.add_argument("--top", type=int, default=0,
                    help="keep only the N most common scorelines and pool the rest as 'Other' (default: show all)")
    ap.add_argument("--source", help="override the data source in the config file")
    ap.add_argument("--today", help="pretend today is this date (YYYY-MM-DD)")
    ap.add_argument("--no-refresh", action="store_true", help="use cached results; don't re-download this season")
    a = ap.parse_args(argv)

    today = date.fromisoformat(a.today) if a.today else date.today()
    cfg = load_config(a.config, today)
    lg = cfg.league(a.league)
    source = make_source(a.source or cfg.source, cfg.data_dir)
    top_div = lg.divs(source.name)[0]
    seasons = [shift_season(lg.current_season, -k) for k in range(a.seasons - 1, -1, -1)]
    m = load_matches(source, seasons, [top_div], refresh_season=None if a.no_refresh else lg.current_season)
    m = m[(m["div"] == top_div) & (m.date < pd.Timestamp(today))]
    if m.empty:
        print(f"no {lg.name} results found for {seasons}")
        return 1
    seasons = [s for s in seasons if (m.season == s).any()]
    counts = m.groupby("season").size().to_dict()

    table = scoreline_table(m, seasons, a.top)
    title = f"{lg.name}, {seasons[0]} to {seasons[-1]}"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.reports_dir / f"{lg.key}_score_histogram"
    raw = m[["date", "season", "home", "away", "hg", "ag"]].rename(
        columns={"hg": "home goals", "ag": "away goals"}).reset_index(drop=True)
    footer = (f"Source: {source.name}, results up to {m.date.max():%d %b %Y}. "
              + ", ".join(f"{s}: {counts[s]} matches" for s in seasons) + ".")
    write_html(stem.with_suffix(".html"), title, table, seasons, counts, footer)
    write_excel(stem.with_suffix(".xlsx"), title, table, grid(m), summary(m, seasons), raw, seasons)
    write_png(stem.with_suffix(".png"), title, table, seasons, counts)

    print(f"{title}: " + ", ".join(f"{s} {counts[s]} matches" for s in seasons))
    print("order: " + " ".join(table.scoreline))
    print(f"chart (interactive) -> {stem.with_suffix('.html')}\nexcel -> {stem.with_suffix('.xlsx')}\n"
          f"image -> {stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
