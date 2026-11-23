"""
Defensive record of keepers and defenders: goals conceded while the player is on the pitch.

  conceded       opponent goals (own goals included) scored between the minutes the player came on and went off
  per90          conceded per 90 minutes played, so players with fewer minutes compare fairly
  clean_sheets   matches where the player played CLEAN_SHEET_MIN_MINUTES or more and the team
                 conceded nothing while the player was on the pitch
  off_per90      the team's goals conceded per 90 minutes while the player was off the pitch,
                 in all the team's matches of the season (matches the player missed included)
  on_off         off_per90 - per90: goals per 90 the team saves with the player on the pitch
                 (positive = the team concedes less with the player). Empty when the player
                 was off the pitch for less than MIN_OFF_MINUTES, because the sample is too small.

Ranks compare the players of one position in the teams' usual XIs (rank 1 = best: fewest
conceded per 90, most clean sheets, largest on_off). These numbers mostly measure the team's defence, not
only the player.
"""

import pandas as pd

from .espn import MATCH_MINUTES

CLEAN_SHEET_MIN_MINUTES = 60
MIN_OFF_MINUTES = 270          # 3 full matches off the pitch before on/off is shown
DEFENSIVE_POSITIONS = ["G", "D"]
RANK_METHOD = "min"            # tied players share the better rank
RECORD_COLS = ["team", "athlete_id", "player", "pos", "matches", "minutes", "conceded", "per90",
               "clean_sheets", "off_minutes", "off_per90", "on_off", "per90_rank", "clean_sheets_rank",
               "n_ranked", "on_off_rank", "n_on_off_ranked"]


def player_records(players: pd.DataFrame, goal_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per player and team in `season`: matches, minutes, goals conceded while on the pitch, clean sheets."""
    played = players[(players.season == season) & (players.minutes > 0)]
    if played.empty:
        return pd.DataFrame()
    goals = goal_df[["event_id", "team", "minute"]].rename(columns={"team": "scoring_team"})
    on_pitch = played[["event_id", "team", "athlete_id", "on_minute", "off_minute"]].merge(goals, on="event_id")
    on_pitch = on_pitch[(on_pitch.scoring_team != on_pitch.team) & (on_pitch.minute > on_pitch.on_minute)
                        & (on_pitch.minute <= on_pitch.off_minute)]
    conceded = on_pitch.groupby(["event_id", "athlete_id"]).size().rename("conceded")
    per_match = played.join(conceded, on=["event_id", "athlete_id"]).fillna({"conceded": 0})
    per_match["clean_sheet"] = (per_match.conceded == 0) & (per_match.minutes >= CLEAN_SHEET_MIN_MINUTES)
    records = per_match.groupby(["team", "athlete_id"]).agg(
        player=("player", "last"), matches=("event_id", "size"), minutes=("minutes", "sum"),
        conceded=("conceded", "sum"), clean_sheets=("clean_sheet", "sum")).reset_index()
    records["conceded"] = records.conceded.astype(int)
    records["per90"] = records.conceded / records.minutes * MATCH_MINUTES
    team_record = _team_records(players[players.season == season], goal_df)
    records = records.join(team_record, on="team")
    records["off_minutes"] = records.team_minutes - records.minutes
    off_conceded = records.team_conceded - records.conceded
    records["off_per90"] = (off_conceded / records.off_minutes * MATCH_MINUTES).where(records.off_minutes >= MIN_OFF_MINUTES)
    records["on_off"] = records.off_per90 - records.per90
    return records.drop(columns=["team_minutes", "team_conceded"])


def _team_records(season_players: pd.DataFrame, goal_df: pd.DataFrame) -> pd.DataFrame:
    """Per team: minutes played (MATCH_MINUTES per match) and goals conceded in its matches."""
    team_events = season_players[["team", "event_id"]].drop_duplicates()
    against = team_events.merge(goal_df[["event_id", "team"]].rename(columns={"team": "scoring_team"}), on="event_id")
    conceded = against[against.scoring_team != against.team].groupby("team").size()
    return pd.DataFrame({"team_minutes": team_events.groupby("team").size() * MATCH_MINUTES,
                         "team_conceded": conceded}).fillna({"team_conceded": 0})


def usual_xi_records(records: pd.DataFrame, impact: pd.DataFrame) -> pd.DataFrame:
    """
    Records of the keepers and defenders in each team's usual XI (from lineups.player_impact),
    ranked against the other players of the same position.
    """
    xi = impact[impact.in_xi & impact.pos.isin(DEFENSIVE_POSITIONS)][["team", "athlete_id", "pos"]]
    table = xi.merge(records, on=["team", "athlete_id"], how="inner")
    if table.empty:
        return pd.DataFrame(columns=RECORD_COLS)
    by_pos = table.groupby("pos")
    table["per90_rank"] = by_pos.per90.rank(method=RANK_METHOD).astype(int)
    table["clean_sheets_rank"] = by_pos.clean_sheets.rank(method=RANK_METHOD, ascending=False).astype(int)
    table["n_ranked"] = by_pos.pos.transform("size")
    table["on_off_rank"] = by_pos.on_off.rank(method=RANK_METHOD, ascending=False).astype("Int64")
    table["n_on_off_ranked"] = by_pos.on_off.transform("count")
    return table[RECORD_COLS]
