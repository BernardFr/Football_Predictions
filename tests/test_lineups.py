"""Lineup layer, tested offline: a synthetic league written into ESPN's JSON shape in the cache."""

import json

import numpy as np
import pandas as pd
import pytest

from football_predictions import lineups, ledger, teammap
from football_predictions.espn import EspnClient, has_lineups, parse_lineups, position_group
from football_predictions.metrics import outcome
from football_predictions.model import summarize

SLUG = "eng.1"
ESPN_NAMES = ["Manchester United", "Manchester City", "Nottingham Forest", "Tottenham Hotspur",
              "Wolverhampton Wanderers", "Brighton & Hove Albion", "Leeds United", "Arsenal",
              "Newcastle United", "AFC Bournemouth", "Sheffield Wednesday", "West Bromwich Albion"]
FD_NAMES = ["Man United", "Man City", "Nott'm Forest", "Tottenham", "Wolves", "Brighton", "Leeds",
            "Arsenal", "Newcastle", "Bournemouth", "Sheffield Weds", "West Brom"]


def test_similarity_pairs():
    for e, f in zip(ESPN_NAMES, FD_NAMES):
        best = max(FD_NAMES, key=lambda x: teammap.similarity(e, x))
        assert best == f, (e, best)


def test_position_groups():
    assert [position_group(a) for a in ["G", "CD-L", "RB", "DM", "AM-R", "F", "CF-L", "LW", None]] == \
           ["G", "D", "D", "M", "M", "F", "F", "F", "M"]


def roster_entry(pid, name, pos, starter, goals=0, assists=0, sub_in=None, sub_out=None):
    e = {"athlete": {"id": pid, "displayName": name}, "position": {"abbreviation": pos}, "starter": starter,
         "subbedIn": {"didSub": sub_in is not None, "clock": {"displayValue": f"{sub_in}'"}} if sub_in else {"didSub": False},
         "subbedOut": {"didSub": sub_out is not None, "clock": {"displayValue": f"{sub_out}'"}} if sub_out else {"didSub": False},
         "stats": [{"name": "totalGoals", "value": goals}, {"name": "goalAssists", "value": assists}]}
    return e


def build_league(tmp, seed=3, seasons=("2023-24", "2024-25")):
    """Write ESPN-shaped scoreboards/summaries into the cache; return fd matches and true strengths."""
    rng = np.random.default_rng(seed)
    n = len(ESPN_NAMES)
    strength = np.linspace(0.45, -0.45, n)
    squads = {}
    for t in range(n):
        pos = ["G", "G"] + ["CD"] * 5 + ["M"] * 5 + ["F"] * 4
        squads[t] = [(f"{t}-{k}", f"P{t}_{k}", p) for k, p in enumerate(pos)]
    sc_w = {"G": 0.0, "CD": 0.2, "M": 0.7, "F": 1.5}
    fd_rows, cal = [], {}
    root = tmp / "espn" / SLUG
    eid = 1000
    for season in seasons:
        y = int(season[:4])
        day0 = pd.Timestamp(f"{y}-08-10")
        pairs = [(h, a) for h in range(n) for a in range(n) if h != a]
        rng.shuffle(pairs)
        days = {}
        for k, (h, a) in enumerate(pairs):
            d = day0 + pd.Timedelta(days=2 * (k // 6))
            eid += 1
            ros, lam = [], {}
            info = {}
            for side, t in (("home", h), ("away", a)):
                sq = squads[t]
                star_out = rng.random() < 0.3
                gk_out = rng.random() < 0.15
                gk = sq[1] if gk_out else sq[0]
                fwd = [sq[13] if star_out else sq[12], sq[14], sq[15]]   # P12 is the star striker
                xi = [gk] + sq[2:6] + sq[7:10] + fwd
                info[side] = (t, star_out, gk_out, xi)
            for side, opp in (("home", "away"), ("away", "home")):
                t, star_out, _, _ = info[side]
                _, _, opp_gk_out, _ = info[opp]
                o = info[opp][0]
                lam[side] = np.exp(0.25 + (0.25 if side == "home" else 0) + strength[t] - strength[o]
                                   - (0.5 if star_out else 0) + (0.4 if opp_gk_out else 0))
            goals = {s: rng.poisson(lam[s]) for s in lam}
            for side in ("home", "away"):
                t, star_out, _, xi = info[side]
                w = np.array([4.0 if p[0].endswith("-12") else sc_w[p[2]] for p in xi])
                scorer = rng.choice(len(xi), size=goals[side], p=w / w.sum())
                g = np.bincount(scorer, minlength=len(xi))
                roster = [roster_entry(p[0], p[1], p[2], True, int(g[i]), sub_out=70 if i == 10 else None)
                          for i, p in enumerate(xi)]
                bench = [p for p in squads[t] if p not in xi]
                roster.append(roster_entry(bench[0][0], bench[0][1], bench[0][2], False, sub_in=70))
                roster += [roster_entry(p[0], p[1], p[2], False) for p in bench[1:]]
                ros.append({"homeAway": side, "team": {"id": str(t + 1), "displayName": ESPN_NAMES[t]}, "roster": roster})
            ev = {"id": str(eid), "date": f"{d:%Y-%m-%d}T15:00Z", "competitions": [{
                "competitors": [{"homeAway": "home", "team": {"id": str(h + 1), "displayName": ESPN_NAMES[h]}, "score": str(goals["home"])},
                                {"homeAway": "away", "team": {"id": str(a + 1), "displayName": ESPN_NAMES[a]}, "score": str(goals["away"])}],
                "status": {"type": {"state": "post", "completed": True}}}]}
            days.setdefault(f"{d:%Y%m%d}", []).append(ev)
            (root / "summary").mkdir(parents=True, exist_ok=True)
            (root / "summary" / f"{eid}.json").write_text(json.dumps({"rosters": ros}))
            fd_rows.append({"date": d, "season": season, "div": "E0", "home": FD_NAMES[h], "away": FD_NAMES[a],
                            "hg": goals["home"], "ag": goals["away"], "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
                            "true_xh": np.exp(0.5 + strength[h] - strength[a]), "true_xa": np.exp(0.25 + strength[a] - strength[h])})
        (root / "scoreboard").mkdir(parents=True, exist_ok=True)
        calendar = [f"{pd.Timestamp(k):%Y-%m-%d}T07:00Z" for k in days]
        (root / "scoreboard" / f"{y}0701.json").write_text(json.dumps({"leagues": [{"calendar": calendar}], "events": []}))
        for k, evs in days.items():
            (root / "scoreboard" / f"{k}.json").write_text(json.dumps({"leagues": [{"calendar": calendar}], "events": evs}))
    return pd.DataFrame(fd_rows)


def test_parse_lineups_minutes_and_stats():
    s = {"rosters": [{"team": {"id": "1"}, "roster": [
        roster_entry("a", "A", "F", True, goals=2, assists=1, sub_out=70),
        roster_entry("b", "B", "F", False, sub_in=70),
        roster_entry("c", "C", "G", False)]}]}
    lu = parse_lineups(s, "9").set_index("athlete_id")
    assert lu.loc["a", "minutes"] == 70 and lu.loc["b", "minutes"] == 20 and lu.loc["c", "minutes"] == 0
    assert lu.loc["a", "ga"] == 3 and lu.loc["a", "pos"] == "F"
    pre = {"rosters": [{"team": {"id": str(t)}, "roster": [{"athlete": {"id": f"{t}{i}"}, "starter": True,
                                                            "position": {"abbreviation": "M"}} for i in range(11)]}
                       for t in (1, 2)]}
    assert has_lineups(parse_lineups(pre, "10"))            # pre-match roster: starters only, no subs/stats
    assert not has_lineups(parse_lineups({"rosters": []}, "11"))


def test_end_to_end_recovers_lineup_effects(tmp_path):
    fd = build_league(tmp_path)
    client = EspnClient(tmp_path, delay=0)
    events = pd.concat([client.sync_season(SLUG, s, log=None) for s in ("2023-24", "2024-25")], ignore_index=True)
    assert client.requests_made == 0 and len(events) == len(fd)

    tm = teammap.build(events, fd)
    assert dict(zip(tm.espn_name, tm.fd_name)) == dict(zip(ESPN_NAMES, FD_NAMES))

    players = lineups.player_table(client, SLUG, events, tm)
    feat = lineups.match_features(players, events, tm)
    assert len(feat) > 0.8 * len(fd)
    assert feat.h_gk.isin([0.0, 1.0]).all() and feat.h_miss.between(0, 1).all()

    # base model = true strengths without lineup effects (what a team-strength model knows)
    base = fd.assign(xg_home=fd.true_xh, xg_away=fd.true_xa, rho=0.0, result=outcome(fd.hg, fd.ag))
    probs = pd.DataFrame([summarize(r.xg_home, r.xg_away, 0.0) for r in base.itertuples()])
    base = pd.concat([base, probs[["p_home", "p_draw", "p_away", "pick", "confidence", "exp_gd", "gd_lo80", "gd_hi80"]]], axis=1)

    ev, b = lineups.walk_forward(feat, base, ridge=1.0, min_train=100)
    coef = dict(zip(lineups.COEFS, b))
    assert coef["att_own"] > 0.2           # star striker missing -> lower att -> fewer goals
    assert coef["gk_opp"] > 0.1            # opponent's backup keeper -> more goals
    comp = lineups.compare(ev).set_index(["season", "forecaster"])
    # out-of-sample goals are predicted better; W/D/L scores need far more matches to show it
    assert comp.loc[("all", "with lineups"), "goal_loglik"] > comp.loc[("all", "base model"), "goal_loglik"]


def test_ledger_variants(tmp_path):
    path = tmp_path / "ledger.csv"
    pred = pd.DataFrame([{"date": pd.Timestamp("2026-11-21"), "season": "2026-27", "home": "A", "away": "B",
                          "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "pick": "H", "confidence": 0.5,
                          "exp_gd": 0.4, "gd_lo80": -1, "gd_hi80": 2}])
    ledger.record(path, pred)
    ledger.record(path, pred.assign(p_home=0.6), variant="lineup")
    assert ledger.settle(path, pd.DataFrame([{"season": "2026-27", "home": "A", "away": "B", "hg": 0, "ag": 0}])) == 2
    led = ledger.load(path)
    assert set(led.variant) == {"base", "lineup"} and (led.result == "D").all()


def test_match_name_is_conservative():
    fd = ["Man United", "Man City", "Tottenham", "Wolves", "Brighton", "Leeds", "Sunderland"]
    assert teammap.match_name("Manchester United", fd) == "Man United"
    assert teammap.match_name("Brighton & Hove Albion", fd) == "Brighton"
    assert teammap.match_name("Wolverhampton Wanderers", fd) == "Wolves"
    assert teammap.match_name("Burnley", fd) is None          # no confident match -> skipped, not guessed


def test_upcoming_schedule_from_espn(tmp_path):
    build_league(tmp_path)
    sb = tmp_path / "espn" / SLUG / "scoreboard"
    last = max(p.stem for p in sb.glob("2024*.json") if p.stem != "20240701")
    data = json.loads((sb / f"{last}.json").read_text())
    for e in data["events"]:
        c = e["competitions"][0]
        c["status"] = {"type": {"state": "pre", "completed": False}}
        for t in c["competitors"]:
            t.pop("score", None)
    (sb / f"{last}.json").write_text(json.dumps(data))
    day = pd.Timestamp(last)
    client = EspnClient(tmp_path, delay=0, now=day - pd.Timedelta(days=2))
    up = client.upcoming(SLUG, "2024-25", days=5)
    assert len(up) == len(data["events"]) and (up.state == "pre").all() and client.requests_made == 0
    assert EspnClient(tmp_path, delay=0, now=day + pd.Timedelta(days=1)).upcoming(SLUG, "2024-25", 5).empty


# ---------------------------------------------------------------- cups
def _cup_event(eid, home, away, hs, as_, status="STATUS_FULL_TIME", desc="Full Time", neutral=False, lines=None):
    comp = {"neutralSite": neutral, "status": {"type": {"state": "post", "completed": True, "name": status,
                                                         "description": desc}},
            "competitors": [{"homeAway": "home", "team": {"id": home[0], "displayName": home[1]}, "score": str(hs)},
                            {"homeAway": "away", "team": {"id": away[0], "displayName": away[1]}, "score": str(as_)}]}
    if lines:
        for c, ls in zip(comp["competitors"], lines):
            c["linescores"] = [{"value": v} for v in ls]
    return {"id": eid, "date": "2025-01-11T15:00Z", "competitions": [comp]}


def test_cup_scores_at_90_minutes_and_neutral():
    from football_predictions.espn import parse_scoreboard
    evs = parse_scoreboard({"events": [
        _cup_event("1", ("1", "Arsenal"), ("2", "Leeds"), 3, 1),
        _cup_event("2", ("1", "Arsenal"), ("2", "Leeds"), 2, 1, "STATUS_FINAL_AET", "After Extra Time",
                   lines=[[1, 0, 1, 0], [0, 1, 0, 0]]),                       # 1-1 at 90, 2-1 a.e.t.
        _cup_event("3", ("1", "Arsenal"), ("2", "Leeds"), 3, 2, "STATUS_FINAL_AET", "After Extra Time"),
        _cup_event("4", ("1", "Arsenal"), ("2", "Leeds"), 1, 1, "STATUS_FINAL_PEN", "Final - Pens", neutral=True),
    ]})
    got = [(e["home_90"], e["away_90"], e["extra_time"], e["neutral"]) for e in evs]
    assert got == [(3, 1, False, False), (1, 1, True, False), (2, 2, True, False), (1, 1, True, True)]


def test_neutral_and_cup_weight_in_model():
    from football_predictions.model import DixonColes, Hyper
    rng = np.random.default_rng(5)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for k in range(6):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=k * 60 + len(rows) % 50),
                                 "season": "2025-26", "div": "E0", "home": h, "away": a,
                                 "hg": rng.poisson(1.6), "ag": rng.poisson(1.1), "neutral": False})
    league = pd.DataFrame(rows)
    # cup games at a neutral venue where the "home" side has no edge
    cup = pd.DataFrame([{"date": pd.Timestamp("2026-01-10"), "season": "2025-26", "div": "CUP", "home": "T0",
                         "away": "T1", "hg": rng.poisson(1.2), "ag": rng.poisson(1.2), "neutral": True}
                        for _ in range(300)])
    both = pd.concat([league, cup], ignore_index=True)
    off = DixonColes(Hyper(cup_weight=0.0)).fit(both, "2026-06-01", "E0")
    base = DixonColes(Hyper()).fit(league, "2026-06-01", "E0")
    assert off.predict("T2", "T3")["p_home"] == pytest.approx(base.predict("T2", "T3")["p_home"], abs=1e-6)
    on = DixonColes(Hyper(cup_weight=1.0)).fit(both, "2026-06-01", "E0")
    assert "CUP" in on.div_index and on.last_div["T0"] == "E0"           # cups don't change a team's division
    assert on.home_adv[on.div_index["E0"]] > 0.2                          # league home edge survives


def test_cups_keep_only_rated_teams(tmp_path):
    from football_predictions import cups
    slug, root = "eng.fa", tmp_path / "espn" / "eng.fa" / "scoreboard"
    root.mkdir(parents=True)
    cal = {"leagues": [{"calendar": ["2025-01-11T08:00Z"]}], "events": []}
    (root / "20240701.json").write_text(json.dumps(cal))
    day = dict(cal, events=[
        _cup_event("1", ("10", "Manchester United"), ("11", "Leeds United"), 2, 0),
        _cup_event("2", ("12", "Brighton & Hove Albion"), ("13", "Tamworth"), 3, 0),      # non-league: skipped
        _cup_event("3", ("14", "Arsenal"), ("10", "Manchester United"), 1, 1, "STATUS_FINAL_PEN", "Pens", neutral=True),
    ])
    (root / "20250111.json").write_text(json.dumps(day))
    client = EspnClient(tmp_path, delay=0, now=pd.Timestamp("2025-03-01"))
    out = cups.load(client, [slug], ["2024-25"], ["Man United", "Leeds", "Brighton", "Arsenal"], log=None)
    assert out[["home", "away", "hg", "ag", "neutral"]].values.tolist() == [
        ["Man United", "Leeds", 2, 0, False], ["Arsenal", "Man United", 1, 1, True]]
    assert (out["div"] == "CUP").all() and client.requests_made == 0
