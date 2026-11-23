"""
Map ESPN team ids to football-data team names by matching fixtures on date.

Two fixtures on the same date with similar home and away names are the same game; votes are
counted over a season so one odd pairing can't create a wrong mapping. Overrides can be put in
state/<league>/team_map_overrides.csv (columns espn_name, fd_name).
"""

import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

import pandas as pd

ALIASES = [(r"\bman\b", "manchester"), (r"\butd\b", "united"), (r"nott'?m", "nottingham"),
           (r"\bwolves\b", "wolverhampton wanderers"), (r"\bspurs\b", "tottenham hotspur"),
           (r"\bqpr\b", "queens park rangers"), (r"sheffield weds", "sheffield wednesday"),
           (r"west brom\b", "west bromwich albion"), (r"\bmgladbach\b", "monchengladbach"),
           (r"\bath\b", "athletic"), (r"\bsociedad\b", "real sociedad")]
NAME_MATCH_THRESHOLD = 0.8     # match_name: minimum similarity to accept a name
NAME_MATCH_MARGIN = 0.1        # match_name: required lead over the runner-up
CONTAINED_SIMILARITY = 0.9     # score when one name's tokens are a subset of the other's
FIXTURE_MATCH_THRESHOLD = 0.8  # build: minimum home + away similarity (out of 2) for a vote
FIXTURE_MATCH_MARGIN = 0.15    # build: required lead over the runner-up fixture
DATE_TOLERANCE_DAYS = 1        # fixtures may be dated a day apart across sources
OVERRIDE_VOTES = -1            # marks a mapping that came from the overrides file
DROP = r"\b(fc|afc|cf|sc|ac|ssc|as|us|sv|vfb|vfl|tsg|1\.|fsv|rcd|cd|ud|sd|rc|ogc|losc|stade|calcio|club|de|the|and|&)\b"


def tokens(name: str) -> list[str]:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for pat, replacement in ALIASES:
        text = re.sub(pat, replacement, text)
    text = re.sub(DROP, " ", text)
    return re.sub(r"[^a-z0-9 ]", " ", text).split()


def similarity(name_a: str, name_b: str) -> float:
    """0..1. Handles 'Man United' vs 'Manchester United FC', 'Leeds' vs 'Leeds United'."""
    ta, tb = tokens(name_a), tokens(name_b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / len(sa | sb)
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    contained = CONTAINED_SIMILARITY if (sa <= sb or sb <= sa) else 0.0       # 'leeds' within 'leeds united'
    return max(jac, seq, contained)


def match_name(name: str, candidate_lst, threshold: float = NAME_MATCH_THRESHOLD,
               margin: float = NAME_MATCH_MARGIN) -> str | None:
    """Best football-data name for an ESPN name by similarity alone, if it is clear-cut."""
    scored = sorted(((similarity(name, candidate), candidate) for candidate in set(candidate_lst)), reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1]


def build(espn_events: pd.DataFrame, fd_matches: pd.DataFrame, overrides: Path | None = None) -> pd.DataFrame:
    """Return espn_id, espn_name, fd_name, votes."""
    votes: dict[str, Counter] = defaultdict(Counter)
    names = {}
    fd = fd_matches.assign(day=fd_matches.date.dt.normalize())
    by_day = {match_day: group for match_day, group in fd.groupby("day")}
    for event in espn_events.itertuples(index=False):
        home_id, away_id = str(event.home_id), str(event.away_id)
        home_name, away_name = str(event.home_name), str(event.away_name)
        names[home_id], names[away_id] = home_name, away_name
        day = cast(pd.Timestamp, event.date).normalize()
        near = [by_day[match_day] for match_day in (day - pd.Timedelta(days=DATE_TOLERANCE_DAYS), day, day + pd.Timedelta(days=DATE_TOLERANCE_DAYS)) if match_day in by_day]
        if not near:
            continue
        cands = pd.concat(near)
        scores = [(similarity(home_name, str(fixture.home)) + similarity(away_name, str(fixture.away)), str(fixture.home), str(fixture.away))
                  for fixture in cands.itertuples(index=False)]
        scores.sort(reverse=True)
        best = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best[0] >= FIXTURE_MATCH_THRESHOLD and best[0] - second >= FIXTURE_MATCH_MARGIN:
            votes[home_id][best[1]] += 1
            votes[away_id][best[2]] += 1
    rows = [{"espn_id": espn_id, "espn_name": names.get(espn_id), "fd_name": counter.most_common(1)[0][0],
             "votes": counter.most_common(1)[0][1], "total": sum(counter.values())} for espn_id, counter in votes.items()]
    tm = pd.DataFrame(rows, columns=["espn_id", "espn_name", "fd_name", "votes", "total"])
    if overrides is not None and Path(overrides).exists():
        ov = pd.read_csv(overrides)
        for override_row in ov.itertuples(index=False):
            ids = [espn_id for espn_id, team_name in names.items() if team_name == override_row.espn_name]
            tm = tm[~tm.espn_id.isin(ids)]
            override_df = pd.DataFrame({"espn_id": ids, "espn_name": override_row.espn_name, "fd_name": override_row.fd_name,
                                        "votes": OVERRIDE_VOTES, "total": OVERRIDE_VOTES})
            tm = pd.concat([tm, override_df], ignore_index=True)
    return tm.sort_values("espn_name").reset_index(drop=True)
