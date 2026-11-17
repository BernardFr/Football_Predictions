"""Excel report: upcoming predictions, backtest, calibration, live track record, team ratings."""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import metrics

PCT = {"p_home", "p_draw", "p_away", "confidence", "bk_home", "bk_draw", "bk_away", "accuracy",
       "hit_rate", "mean_confidence", "mean_forecast", "observed_freq", "gd_80_coverage",
       "p_likely_score", "p_base_h", "p_base_d", "p_base_a"}
HEAD = PatternFill("solid", fgColor="1F3864")


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
            ws.cell(row=row, column=1).font = Font(bold=True, size=12)
        for j, col in enumerate(df.columns, start=1):
            c = ws.cell(row=row + 1, column=j)
            c.font, c.fill = Font(bold=True, color="FFFFFF"), HEAD
            c.alignment = Alignment(wrap_text=True, vertical="center")
            fmt = "0.0%" if col in PCT else ("0.00" if pd.api.types.is_float_dtype(df[col]) else None)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                fmt = "yyyy-mm-dd"
            if fmt:
                for i in range(len(df)):
                    ws.cell(row=row + 2 + i, column=j).number_format = fmt
            width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) if len(df) else len(str(col))
            ws.column_dimensions[get_column_letter(j)].width = min(max(width + 2, 9), 34)
        row += len(df) + 3
    if name in xw.sheets:
        xw.sheets[name].freeze_panes = "A2" if frames and frames[0][0] is None else None


def display_predictions(df: pd.DataFrame, high: float, medium: float) -> pd.DataFrame:
    d = df.copy()
    d["pick_label"] = d.apply(lambda r: r.home if r.pick == "H" else (r.away if r.pick == "A" else "Draw"), axis=1)
    d["band"] = metrics.band(d.confidence, high, medium)
    d["gd_range_80"] = d.apply(lambda r: f"{r.gd_lo80:+d} to {r.gd_hi80:+d}", axis=1)
    cols = ["date", "home", "away", "pick_label", "confidence", "band", "p_home", "p_draw", "p_away",
            "exp_gd", "gd_range_80", "likely_score", "xg_home", "xg_away"]
    if {"odds_h", "odds_d", "odds_a"} <= set(d) and d.odds_h.notna().any():
        has = d[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
        k = metrics.implied_probs(d.odds_h[has], d.odds_d[has], d.odds_a[has])
        for i, c in enumerate(["bk_home", "bk_draw", "bk_away"]):
            d.loc[has, c] = k[:, i]
        cols += ["bk_home", "bk_draw", "bk_away"]
    if "adjusted" in d:
        cols.append("adjusted")
    if "schedule" in d:
        cols.append("schedule")
    return d[[c for c in cols if c in d]].rename(columns={"pick_label": "pick"})


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
        if lineup_upcoming is not None and not lineup_upcoming.empty:
            lu = display_predictions(lineup_upcoming, high, medium)
            for c in ("base_p_home", "base_p_draw", "base_p_away", "h_miss", "h_att", "h_gk", "a_miss", "a_att", "a_gk"):
                if c in lineup_upcoming:
                    lu[c] = lineup_upcoming[c].to_numpy()
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
            for variant, g in settled.groupby("variant"):
                g = g.assign(hg=g.hg.astype(int), ag=g.ag.astype(int))
                frames.append((f"Live '{variant}' predictions made before kickoff, scored on results", metrics.summary(g)))
            frames.append(("All logged predictions", ledger))
            _sheet(xw, "Live track record", frames)
        if ratings is not None and not ratings.empty:
            _sheet(xw, "Team ratings", [("Expected goals vs. an average top-division team, neutral venue", ratings)])
        if tuning is not None and not tuning.empty:
            _sheet(xw, "Tuning", [("Hyperparameter grid, walk-forward RPS on the previous season (lower is better)", tuning)])
