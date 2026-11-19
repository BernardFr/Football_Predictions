"""Excel report: upcoming predictions, backtest, calibration, live track record, team ratings."""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import metrics
from .lineups import FEATURE_COLS
from .model import HIGHEST_PROB_SCORE_COLS

PCT = {"p_home", "p_draw", "p_away", "confidence", "bk_home", "bk_draw", "bk_away", "accuracy",
       "hit_rate", "mean_confidence", "mean_forecast", "observed_freq", "gd_80_coverage",
       "p_likely_score", "p_base_h", "p_base_d", "p_base_a"}
HEADER_COLOR = "1F3864"
HEADER_FONT_COLOR = "FFFFFF"
HEAD = PatternFill("solid", fgColor=HEADER_COLOR)
TITLE_FONT_SIZE = 12
PCT_FORMAT = "0.0%"
FLOAT_FORMAT = "0.00"
DATE_FORMAT = "yyyy-mm-dd"
WIDTH_SAMPLE_ROWS = 200        # rows scanned to size a column
COL_WIDTH_PADDING = 2
MIN_COL_WIDTH = 9
MAX_COL_WIDTH = 34
ROWS_BETWEEN_TABLES = 2
FREEZE_BELOW_HEADER = "A2"
DRAW_LABEL = "Draw"
BOOKMAKER_COLS = ["bk_home", "bk_draw", "bk_away"]
PREDICTION_COLS = ["date", "home", "away", "pick_label", "confidence", "band", "p_home", "p_draw", "p_away",
                   "exp_gd", "gd_range_80", *HIGHEST_PROB_SCORE_COLS, "xg_home", "xg_away"]
LINEUP_BASE_COLS = ["base_p_home", "base_p_draw", "base_p_away"]
STATS_TOP_N = 3
# (title, column it ranks, show the largest values first?, columns shown next to the team)
STATS_RANKINGS = [
    ("Won more games than predicted", "wins_vs_pred", True, ["wins", "exp_wins"]),
    ("Won fewer games than predicted", "wins_vs_pred", False, ["wins", "exp_wins"]),
    ("Lost fewer games than predicted", "losses_vs_pred", False, ["losses", "exp_losses"]),
    ("Lost more games than predicted", "losses_vs_pred", True, ["losses", "exp_losses"]),
    ("Goal difference above prediction", "gd_vs_pred", True, ["gd", "exp_gd"]),
    ("Goal difference below prediction", "gd_vs_pred", False, ["gd", "exp_gd"]),
    ("Best attack (most goals scored)", "goals_for", True, ["goals_against"]),
    ("Best defence (fewest goals conceded)", "goals_against", False, ["goals_for"]),
    ("Worst attack (fewest goals scored)", "goals_for", False, ["goals_against"]),
    ("Worst defence (most goals conceded)", "goals_against", True, ["goals_for"]),
]


def _sheet(xw, name: str, frames: list[tuple[str | None, pd.DataFrame]]):
    row = 0
    for title, df in frames:
        if df is None or df.empty:
            continue
        if title:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name=name, startrow=row, index=False, header=False)
            row += 1
        df.to_excel(xw, sheet_name=name, startrow=row, index=False)
        ws = xw.sheets[name]
        if title:
            ws.cell(row=row, column=1).font = Font(bold=True, size=TITLE_FONT_SIZE)
        for j, col in enumerate(df.columns, start=1):
            cell = ws.cell(row=row + 1, column=j)
            cell.font, cell.fill = Font(bold=True, color=HEADER_FONT_COLOR), HEAD
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            fmt = PCT_FORMAT if col in PCT else (FLOAT_FORMAT if pd.api.types.is_float_dtype(df[col]) else None)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                fmt = DATE_FORMAT
            if fmt:
                for i in range(len(df)):
                    ws.cell(row=row + 2 + i, column=j).number_format = fmt
            width = max(len(str(col)), *(len(str(value)) for value in df[col].head(WIDTH_SAMPLE_ROWS))) if len(df) else len(str(col))
            ws.column_dimensions[get_column_letter(j)].width = min(max(width + COL_WIDTH_PADDING, MIN_COL_WIDTH), MAX_COL_WIDTH)
        row += len(df) + 1 + ROWS_BETWEEN_TABLES   # + 1 for the header row
    if name in xw.sheets:
        xw.sheets[name].freeze_panes = FREEZE_BELOW_HEADER if frames and frames[0][0] is None else None


def display_predictions(df: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    disp = df.copy()
    disp["pick_label"] = disp.apply(lambda row: row.home if row.pick == "H" else (row.away if row.pick == "A" else DRAW_LABEL), axis=1)
    disp["band"] = metrics.band(disp.confidence, high, medium)
    disp["gd_range_80"] = disp.apply(lambda row: f"{row.gd_lo80:+d} to {row.gd_hi80:+d}", axis=1)
    cols = list(PREDICTION_COLS)
    if set(metrics.ODDS_COLS) <= set(disp) and disp.odds_h.notna().any():
        has = disp[metrics.ODDS_COLS].notna().all(axis=1)
        bookie_probs = metrics.implied_probs(disp.odds_h[has], disp.odds_d[has], disp.odds_a[has])
        for i, col_name in enumerate(BOOKMAKER_COLS):
            disp.loc[has, col_name] = bookie_probs[:, i]
        cols += BOOKMAKER_COLS
    if "adjusted" in disp:
        cols.append("adjusted")
    if "schedule" in disp:
        cols.append("schedule")
    return disp[[col_name for col_name in cols if col_name in disp]].rename(columns={"pick_label": "pick"})


def stats_frames(backtest: pd.DataFrame, label: str, top_n: int = STATS_TOP_N) -> list[tuple[str | None, pd.DataFrame]]:
    """
    Top-n teams per ranking in STATS_RANKINGS: teams that beat (or fell short of) the model's
    walk-forward pre-match predictions, then best and worst attack and defence.
    """
    table = metrics.team_vs_prediction(backtest)
    frames: list[tuple[str | None, pd.DataFrame]] = []
    for title, col, largest_first, shown in STATS_RANKINGS:
        top = table.sort_values(col, ascending=not largest_first).head(top_n)
        frames.append((f"{label}: {title} (top {top_n})",
                       top[["team", "games", *shown, col]].reset_index(drop=True)))
    return frames


def write(path: Path, league_name: str, upcoming, backtest, season, ledger, ratings, tuning, high, medium,
          lineup_comparison=None, lineup_coefs=None, lineup_upcoming=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        if upcoming is not None and not upcoming.empty:
            _sheet(xw, "Upcoming", [(None, display_predictions(upcoming, high, medium))])
        else:
            _sheet(xw, "Upcoming", [(None, pd.DataFrame({"note": ["No upcoming fixtures in the window"]}))])
        if backtest is not None and not backtest.empty:
            _sheet(xw, "Backtest", [
                (f"{league_name} {season}: walk-forward backtest, {len(backtest)} matches", metrics.summary(backtest)),
                (f"Hit rate by confidence band (High >= {high:.0%}, Medium >= {medium:.0%})",
                 metrics.confidence_bands(backtest, high, medium)),
            ])
            _sheet(xw, "Calibration", [("Pooled reliability: does a 30% forecast happen 30% of the time?",
                                        metrics.calibration(backtest))])
            det = display_predictions(backtest, high, medium)
            det["result"] = backtest.hg.astype(int).astype(str) + "-" + backtest.ag.astype(int).astype(str)
            det["correct"] = backtest.pick.to_numpy() == backtest.result.to_numpy()
            _sheet(xw, "Backtest detail", [(None, det)])
            _sheet(xw, "Stats", stats_frames(backtest, f"{league_name} {season}"))
        if lineup_upcoming is not None and not lineup_upcoming.empty:
            lu = display_predictions(lineup_upcoming, high, medium)
            for col_name in LINEUP_BASE_COLS + FEATURE_COLS:
                if col_name in lineup_upcoming:
                    lu[col_name] = lineup_upcoming[col_name].to_numpy()
            _sheet(xw, "Lineup predictions", [(
                "Adjusted for announced XIs. miss = share of usual playing time missing; att = log attacking "
                "output vs usual; gk = regular keeper not starting (h = home, a = away)", lu)])
        if lineup_comparison is not None and not lineup_comparison.empty:
            frames: list[tuple[str | None, pd.DataFrame]] = [("Walk-forward test: base model vs. lineup-adjusted, same matches (lower RPS is better)",
                       lineup_comparison)]
            if lineup_coefs:
                frames.append(("Fitted effects on log expected goals (latest fit)",
                               pd.DataFrame([lineup_coefs["coefficients"]]).assign(n_matches=lineup_coefs.get("n_matches"))))
            _sheet(xw, "Lineup backtest", frames)
        if ledger is not None and not ledger.empty:
            settled = ledger[ledger.result.notna()].copy()
            frames = []
            for variant, group in settled.groupby("variant"):
                group = group.assign(hg=group.hg.astype(int), ag=group.ag.astype(int))
                frames.append((f"Live '{variant}' predictions made before kickoff, scored on results", metrics.summary(group)))
            frames.append(("All logged predictions", ledger))
            _sheet(xw, "Live track record", frames)
        if ratings is not None and not ratings.empty:
            _sheet(xw, "Team ratings", [("Expected goals vs. an average top-division team, neutral venue", ratings)])
        if tuning is not None and not tuning.empty:
            _sheet(xw, "Tuning", [("Hyperparameter grid, walk-forward RPS on the previous season (lower is better)", tuning)])
