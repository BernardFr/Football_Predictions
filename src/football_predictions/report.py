"""Excel report: upcoming predictions, backtest, grades, predictability, calibration, player impact, live track record, team ratings."""

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import metrics
from .grading import Grader, SeasonGrades
from .defence import CLEAN_SHEET_MIN_MINUTES, MIN_OFF_MINUTES
from .lineups import FEATURE_COLS
from .model import HIGHEST_PROB_SCORE_COLS

PCT = {"picks right", "p_home", "p_draw", "p_away", "confidence", "bk_home", "bk_draw", "bk_away", "accuracy",
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
FREEZE_FIRST_COLUMN = "B1"     # team-column sheets: keep the label column visible
DRAW_LABEL = "Draw"
BOOKMAKER_COLS = ["bk_home", "bk_draw", "bk_away"]
PREDICTION_COLS = ["date", "home", "away", "pick_label", "confidence", "band", "p_home", "p_draw", "p_away",
                   "exp_gd", "gd_range_80", *HIGHEST_PROB_SCORE_COLS, "xg_home", "xg_away"]
LINEUP_BASE_COLS = ["base_p_home", "base_p_draw", "base_p_away"]
STATS_TOP_N = 3
IMPACT_TOP_N = 3
NO_IMPACT_POINTS = 0.005       # |impact| below this many expected points per match counts as no impact
POSITION_ORDER = {"G": 0, "D": 1, "M": 2, "F": 3}
SCORERS_TOP_N = 3
TOP_SCORERS_N = 5
GOALKEEPER_ROWS = ["Keeper", "Matches", "Goals conceded", "Conceded per 90 (rank)", "Clean sheets (rank)"]
GRADE_TITLE = ("prediction grade vs. the previous seasons' scorelines (0 = no better, 100 = certain and right; "
               "result 80%, goals 20%)")
PREDICTABILITY_TITLE = ("Teams from most to least predictable: the grade of the predictions for the games they played "
                        "(0 = no better than the previous seasons' scorelines)")
SCORER_CLASS_LABELS = {"first_goal": "First goal", "equalizer": "Equalizer", "go_ahead": "Go-ahead", "combined": "Combined"}
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


def _team_columns(label_name: str, labels: list, name_dict: dict[Any, list]) -> pd.DataFrame:
    """One column per team (sorted), led by a label column so tables on one sheet line up."""
    table = pd.DataFrame({team: pd.Series(name_dict[team], dtype=object) for team in sorted(name_dict)})
    table = table.reindex(range(len(labels))).fillna("")
    table.insert(0, label_name, labels)
    return table


def impact_frames(impact: pd.DataFrame, top_n: int = IMPACT_TOP_N,
                  threshold: float = NO_IMPACT_POINTS) -> list[tuple[str | None, pd.DataFrame]]:
    """One column per team: its usual XI, its top_n players by impact, then its players with no impact."""
    xi_dict, top_dict, none_dict = {}, {}, {}
    for team, group in impact.groupby("team"):
        xi = group[group.in_xi].assign(order=lambda frame: frame.pos.map(POSITION_ORDER))
        xi = xi.sort_values(["order", "share"], ascending=[True, False])
        xi_dict[team] = [f"{row.player} ({row.pos})" for row in xi.itertuples()]
        top = group[group.impact_pts >= threshold].nlargest(top_n, "impact_pts")
        top = top.assign(pts_change=-top.impact_pts, gd_change=-top.impact_gd)     # what the team loses without the player
        top_dict[team] = [f"{row.player} ({row.pts_change:+.2f} pts, {row.gd_change:+.2f} GD)" for row in top.itertuples()]
        none_dict[team] = group[group.impact_pts.abs() < threshold].sort_values("share", ascending=False).player.tolist()
    n_xi, n_none = (max(map(len, name_dict.values()), default=0) for name_dict in (xi_dict, none_dict))
    return [("Usual XI: the 11 players with the most minutes in the last matches (recent matches count more)",
             _team_columns("#", list(range(1, n_xi + 1)), xi_dict)),
            (f"Most impact (top {top_n}): change in the team's expected points and goal difference per match "
             f"if the player misses the usual XI and the best backup starts", _team_columns("rank", list(range(1, top_n + 1)), top_dict)),
            (f"No impact (under {threshold} pts per match): played in recent matches, mostly outside the usual XI",
             _team_columns("#", list(range(1, n_none + 1)), none_dict))]


def scorer_frames(scorer_df: pd.DataFrame, season: str, top_n: int = SCORERS_TOP_N,
                  top_scorers_n: int = TOP_SCORERS_N) -> list[tuple[str | None, pd.DataFrame]]:
    """One column per team: its top_scorers_n scorers, then its top_n players for each class of valuable goal."""
    top_scorers = {team: [f"{row.player} ({row.goals})" for row in group.nlargest(top_scorers_n, "goals").itertuples()]
                   for team, group in scorer_df.groupby("team")}
    labels, name_dict = [], {team: [] for team in scorer_df.team.unique()}
    for col, label in SCORER_CLASS_LABELS.items():
        labels += [f"{label} {rank}" for rank in range(1, top_n + 1)]
        for team, group in scorer_df.groupby("team"):
            top = group[group[col] > 0].sort_values([col, "goals"], ascending=False).head(top_n)
            names = [f"{row.player} ({getattr(row, col)})" for row in top.itertuples()]
            name_dict[team] += names + [""] * (top_n - len(names))
    return [(f"Top {top_scorers_n} scorers {season} (goals, own goals excluded)",
             _team_columns("rank", list(range(1, top_scorers_n + 1)), top_scorers)),
            (f"Valuable scorers {season} (goals): first goal = team's first of the match; equalizer = levels "
             f"the score; go-ahead = puts the team ahead from level; combined = sum of the three. Own goals excluded", _team_columns("category", labels, name_dict))]


def defence_frames(records: pd.DataFrame, season: str) -> list[tuple[str | None, pd.DataFrame]]:
    """One column per team: its usual keeper's record, then its usual defenders' records, with league ranks."""
    keepers, defenders = records[records.pos == "G"], records[records.pos == "D"]
    keeper_dict = {row.team: [row.player, row.matches, row.conceded,
                              f"{row.per90:.2f} (rank {row.per90_rank} of {row.n_ranked})",
                              f"{row.clean_sheets} (rank {row.clean_sheets_rank} of {row.n_ranked})"]
                   for row in keepers.itertuples()}
    defender_dict = {team: [f"{row.player}: {row.conceded} in {row.matches} matches, {row.per90:.2f} per 90 "
                            f"(#{row.per90_rank}), clean sheets {row.clean_sheets} (#{row.clean_sheets_rank})"
                            for row in group.sort_values("minutes", ascending=False).itertuples()]
                     for team, group in defenders.groupby("team")}
    on_off_dict = {team: [_on_off_text(row) for row in group.sort_values("minutes", ascending=False).itertuples()]
                   for team, group in defenders.groupby("team")}
    n_defenders = max(map(len, defender_dict.values()), default=0)
    note = (f"goals conceded while on the pitch; clean sheet = {CLEAN_SHEET_MIN_MINUTES}+ minutes and none conceded; "
            f"rank 1 = fewest conceded per 90 / most clean sheets")
    frames: list[tuple[str | None, pd.DataFrame]] = []
    if keeper_dict:
        frames.append((f"Goalkeepers {season} (usual XI keeper, ranked among all {len(keepers)} usual XI keepers): {note}",
                       _team_columns("stat", GOALKEEPER_ROWS, keeper_dict)))
    if defender_dict:
        frames.append((f"Defenders {season} (usual XI defenders, ranked among all {len(defenders)} usual XI defenders): {note}",
                       _team_columns("#", list(range(1, n_defenders + 1)), defender_dict)))
        n_on_off = int(defenders.on_off.notna().sum())
        frames.append((f"Defenders on/off {season}: team goals conceded per 90 with the defender on vs. off the pitch "
                       f"(matches missed included). + = the team concedes less with the defender. Rank among the "
                       f"{n_on_off} usual XI defenders with {MIN_OFF_MINUTES}+ minutes off the pitch",
                       _team_columns("#", list(range(1, n_defenders + 1)), on_off_dict)))
    return frames


def _on_off_text(row) -> str:
    if pd.isna(row.on_off):
        return f"{row.player}: {row.per90:.2f} on, too few minutes off ({int(row.off_minutes)})"
    return f"{row.player}: {row.per90:.2f} on vs {row.off_per90:.2f} off ({row.on_off:+.2f}, #{int(row.on_off_rank)})"


def write(path: Path, league_name: str, upcoming, backtest, season, ledger, ratings, tuning, high, medium,
          lineup_comparison=None, lineup_coefs=None, lineup_upcoming=None, player_impact=None,
          valuable_scorers=None, defence_records=None, grader: Grader | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        if upcoming is not None and not upcoming.empty:
            by_confidence = display_predictions(upcoming, high, medium).sort_values("confidence", ascending=False)
            _sheet(xw, "Upcoming", [(None, by_confidence)])
        else:
            _sheet(xw, "Upcoming", [(None, pd.DataFrame({"note": ["No upcoming fixtures in the window"]}))])
        if backtest is not None and not backtest.empty:
            grades = SeasonGrades(backtest, grader) if grader is not None else None
            _sheet(xw, "Backtest", [
                (f"{league_name} {season}: walk-forward backtest, {len(backtest)} matches", metrics.summary(backtest)),
                (f"Hit rate by confidence band (High >= {high:.0%}, Medium >= {medium:.0%})",
                 metrics.confidence_bands(backtest, high, medium)),
                *([(GRADE_TITLE, grades.season())] if grades else []),
            ])
            if grades:
                _sheet(xw, "Grades", [(f"{league_name} {season}: {GRADE_TITLE}", grades.season()),
                                      ("Per matchweek (matchweeks inferred from the dates; each team plays once per matchweek)",
                                       grades.by_matchweek()),
                                      ("Per game", grades.by_game())])
                _sheet(xw, "Predictability", [(PREDICTABILITY_TITLE, grades.by_team())])
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
                "output vs usual; gk = keeper quality vs usual, > 0 = weaker keeper (h = home, a = away)", lu)])
        if lineup_comparison is not None and not lineup_comparison.empty:
            frames: list[tuple[str | None, pd.DataFrame]] = [("Walk-forward test: base model vs. lineup-adjusted, same matches (lower RPS is better)",
                       lineup_comparison)]
            if lineup_coefs:
                frames.append(("Fitted effects on log expected goals (latest fit)",
                               pd.DataFrame([lineup_coefs["coefficients"]]).assign(n_matches=lineup_coefs.get("n_matches"))))
            _sheet(xw, "Lineup backtest", frames)
        impact_tables = []
        if player_impact is not None and not player_impact.empty:
            impact_tables += impact_frames(player_impact)
        if valuable_scorers is not None and not valuable_scorers.empty:
            impact_tables += scorer_frames(valuable_scorers, season)
        if defence_records is not None and not defence_records.empty:
            impact_tables += defence_frames(defence_records, season)
        if impact_tables:
            _sheet(xw, "Impact Players", impact_tables)
            xw.sheets["Impact Players"].freeze_panes = FREEZE_FIRST_COLUMN
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
