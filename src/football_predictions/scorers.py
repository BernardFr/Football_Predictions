"""
Valuable scorers: goals that change the state of a match.

Each goal is classified by the score just before it, from the scoring team's side:

  first_goal  the team had not scored yet (0-0, 0-1, 0-2, ...)
  equalizer   the team was one goal behind (0-1, 1-2, ...), so the goal levels the score
  go_ahead    the score was level (0-0, 1-1, ...), so the goal puts the team ahead

A goal can be in more than one class: a goal at 0-0 is a first goal and a go-ahead goal.
combined is the sum of the three counts (equal weight), so such a goal counts twice.
Own goals change the score but are not credited to a player.
"""

import pandas as pd

from .espn import STATE_POST, parse_goals

GOAL_CLASSES = ["first_goal", "equalizer", "go_ahead"]
COMBINED = "combined"
SCORER_COLS = ["team", "player", "goals", *GOAL_CLASSES, COMBINED]


def goal_table(client, slug: str, events: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Goals in every finished event, in match order, with the football-data team name attached."""
    ids = dict(zip(team_map.espn_id, team_map.fd_name))
    frames = [parse_goals(client.summary(slug, event_row.event_id), str(event_row.event_id)).assign(season=event_row.season)
              for event_row in events[events.state == STATE_POST].itertuples(index=False)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    goal_df = pd.concat(frames, ignore_index=True)
    goal_df["team"] = goal_df.team_id.map(ids)
    return goal_df


def classify(goal_df: pd.DataFrame) -> pd.DataFrame:
    """Add the GOAL_CLASSES flags to each goal, from the score just before it."""
    flags = []
    for _, match_goals in goal_df.groupby("event_id", sort=False):
        score = dict.fromkeys(match_goals.team_id.unique(), 0)
        for goal in match_goals.itertuples():
            goals_for = score[goal.team_id]
            goals_against = sum(score.values()) - goals_for
            flags.append({"index": goal.Index, "first_goal": goals_for == 0,
                          "equalizer": goals_for == goals_against - 1, "go_ahead": goals_for == goals_against})
            score[goal.team_id] += 1
    return goal_df.join(pd.DataFrame(flags).set_index("index"))


def valuable_scorers(goal_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per player in `season`: goals scored, how many fall in each of GOAL_CLASSES, and their sum."""
    goals = classify(goal_df)
    goals = goals[(goals.season == season) & ~goals.own_goal & goals.team.notna()]
    if goals.empty:
        return pd.DataFrame(columns=SCORER_COLS)
    table = goals.groupby(["team", "athlete_id"]).agg(player=("player", "last"), goals=("player", "size"),
                                                      **{name: (name, "sum") for name in GOAL_CLASSES})
    table[COMBINED] = table[GOAL_CLASSES].sum(axis=1)
    return table.reset_index()[SCORER_COLS]
