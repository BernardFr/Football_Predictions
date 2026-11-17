"""Map ESPN team ids to football-data team names by matching fixtures on date.

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
DROP = r"\b(fc|afc|cf|sc|ac|ssc|as|us|sv|vfb|vfl|tsg|1\.|fsv|rcd|cd|ud|sd|rc|ogc|losc|stade|calcio|club|de|the|and|&)\b"


def tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for pat, rep in ALIASES:
        s = re.sub(pat, rep, s)
    s = re.sub(DROP, " ", s)
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def similarity(a: str, b: str) -> float:
    """0..1. Handles 'Man United' vs 'Manchester United FC', 'Leeds' vs 'Leeds United'."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / len(sa | sb)
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    contained = 0.9 if (sa <= sb or sb <= sa) else 0.0       # 'leeds' within 'leeds united'
    return max(jac, seq, contained)


def match_name(name: str, candidates, threshold: float = 0.8, margin: float = 0.1) -> str | None:
    """Best football-data name for an ESPN name by similarity alone, if it is clear-cut."""
    scored = sorted(((similarity(name, c), c) for c in set(candidates)), reverse=True)
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
    by_day = {d: g for d, g in fd.groupby("day")}
    for e in espn_events.itertuples(index=False):
        home_id, away_id = str(e.home_id), str(e.away_id)
        home_name, away_name = str(e.home_name), str(e.away_name)
        names[home_id], names[away_id] = home_name, away_name
        day = cast(pd.Timestamp, e.date).normalize()
        near = [by_day[d] for d in (day - pd.Timedelta(days=1), day, day + pd.Timedelta(days=1)) if d in by_day]
        if not near:
            continue
        cands = pd.concat(near)
        scores = [(similarity(home_name, str(r.home)) + similarity(away_name, str(r.away)), str(r.home), str(r.away))
                  for r in cands.itertuples(index=False)]
        scores.sort(reverse=True)
        best = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best[0] >= 0.8 and best[0] - second >= 0.15:
            votes[home_id][best[1]] += 1
            votes[away_id][best[2]] += 1
    rows = [{"espn_id": k, "espn_name": names.get(k), "fd_name": c.most_common(1)[0][0],
             "votes": c.most_common(1)[0][1], "total": sum(c.values())} for k, c in votes.items()]
    tm = pd.DataFrame(rows, columns=["espn_id", "espn_name", "fd_name", "votes", "total"])
    if overrides is not None and Path(overrides).exists():
        ov = pd.read_csv(overrides)
        for r in ov.itertuples(index=False):
            ids = [k for k, n in names.items() if n == r.espn_name]
            tm = tm[~tm.espn_id.isin(ids)]
            tm = pd.concat([tm, pd.DataFrame([{"espn_id": i, "espn_name": r.espn_name, "fd_name": r.fd_name,
                                               "votes": -1, "total": -1} for i in ids])], ignore_index=True)
    return tm.sort_values("espn_name").reset_index(drop=True)
