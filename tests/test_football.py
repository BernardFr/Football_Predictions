from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_predictions import backtest, ledger, metrics
from football_predictions.config import auto_season, shift_season
from football_predictions.model import DixonColes, Hyper
from football_predictions.sources import COLUMNS, FootballData


def synthetic(seed=0, n_seasons=3, top=8, feeder=8):
    """Two divisions; true strengths known; 2 teams swap divisions each season."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(top + feeder)]
    strength = {t: s for t, s in zip(teams, np.linspace(0.5, -0.7, len(teams)))}
    upper, lower = teams[:top], teams[top:]
    rows, d0 = [], pd.Timestamp("2023-08-01")
    for s in range(n_seasons):
        season = f"{2023 + s}-{(24 + s) % 100:02d}"
        for div, group in (("D1", upper), ("D2", lower)):
            k = 0
            for h in group:
                for a in group:
                    if h == a:
                        continue
                    lam = np.exp(0.2 + 0.25 + strength[h] - strength[a] * 0.8)
                    mu = np.exp(0.2 + strength[a] - strength[h] * 0.8)
                    rows.append({"date": d0 + pd.Timedelta(days=365 * s + 3 * k), "season": season, "div": div,
                                 "home": h, "away": a, "hg": rng.poisson(lam), "ag": rng.poisson(mu),
                                 "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan})
                    k += 1
        upper, lower = upper[:-2] + lower[:2], lower[2:] + upper[-2:]   # relegate 2, promote 2
    return pd.DataFrame(rows, columns=COLUMNS), strength


def test_fit_recovers_strength_order_across_divisions():
    df, strength = synthetic()
    m = DixonColes(Hyper(half_life_days=2000, ridge=0.1)).fit(df, "2027-01-01", "D1")
    r = m.ratings().set_index("team")
    true = pd.Series(strength)
    rho = np.corrcoef(true[r.index], r.gd_vs_avg)[0, 1]
    assert rho > 0.9                                   # one scale across both divisions
    assert m.home_adv[m.div_index["D1"]] > 0


def test_prediction_is_a_proper_distribution():
    df, _ = synthetic()
    m = DixonColes().fit(df, "2027-01-01", "D1")
    p = m.predict("T0", "T7")
    assert p["p_home"] + p["p_draw"] + p["p_away"] == pytest.approx(1.0)
    assert p["pick"] == "H" and p["gd_lo80"] <= round(p["exp_gd"]) <= p["gd_hi80"]
    assert m.score_grid(1.4, 1.1).sum() == pytest.approx(1.0)
    q = m.predict("T0", "T7", adjust={"home_attack": 0.8})
    assert q["xg_home"] == pytest.approx(0.8 * p["xg_home"])
    assert m.predict("Unknown FC", "T0")["p_away"] > 0.4      # unseen team gets a weak prior


def test_rps_values():
    P = np.array([[1, 0, 0], [1 / 3, 1 / 3, 1 / 3], [0, 0, 1]], float)
    Y = metrics.onehot(np.array(["H", "H", "H"]))
    assert metrics.rps(P, Y) == pytest.approx([0.0, 0.5 * ((2 / 3) ** 2 + (1 / 3) ** 2), 1.0])


def test_implied_probs_remove_margin():
    p = metrics.implied_probs([1.9], [3.6], [4.2])
    assert p.sum() == pytest.approx(1.0) and p[0, 0] > p[0, 2]


def test_walk_forward_has_no_lookahead():
    df, _ = synthetic()
    season = "2025-26"
    out = backtest.walk_forward(df, season, "D1", Hyper())
    first = out.date.min()
    m = DixonColes(Hyper()).fit(df, first, "D1")
    r0 = out[out.date == first].iloc[0]
    assert m.predict(str(r0.home), str(r0.away))["p_home"] == pytest.approx(r0.p_home)
    s = metrics.summary(out)
    assert s["rps"].iloc[0] < s["rps"].iloc[1]        # model beats base rate on synthetic data


def test_football_data_parsing(tmp_path):
    csv = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgCH,AvgCD,AvgCA\n"
           "E0,15/08/2026,20:00,Arsenal,Coventry,3,0,H,1.3,5.5,9,1.28,5.8,10\n"
           "E0,16/08/2026,15:00,Leeds,Spurs,1,1,D,2.8,3.4,2.5,,,\n"
           ",,,,,,,,,,,,,\n")
    fd = FootballData(tmp_path)
    p = fd.cache / "2627" / "E0.csv"
    p.parent.mkdir(parents=True)
    p.write_bytes(csv.encode("latin-1"))
    df = fd.results("2026-27", "E0")
    assert list(df.columns) == COLUMNS and len(df) == 2
    assert df.date[0] == pd.Timestamp("2026-08-15")
    assert df.odds_h[0] == 1.28 and df.odds_h[1] == 2.8      # closing odds preferred, opening as fallback


def test_ledger_record_and_settle(tmp_path):
    path = tmp_path / "ledger.csv"
    pred = pd.DataFrame([{"date": pd.Timestamp("2026-11-21"), "season": "2026-27", "home": "A", "away": "B",
                          "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "pick": "H", "confidence": 0.5,
                          "exp_gd": 0.4, "gd_lo80": -1, "gd_hi80": 2}])
    ledger.record(path, pred)
    ledger.record(path, pred.assign(p_home=0.55))            # re-prediction replaces the unsettled row
    led = ledger.load(path)
    assert len(led) == 1 and led.p_home[0] == 0.55
    res = pd.DataFrame([{"season": "2026-27", "home": "A", "away": "B", "hg": 2, "ag": 1}])
    assert ledger.settle(path, res) == 1
    ledger.record(path, pred.assign(p_home=0.9))             # settled rows are never overwritten
    led = ledger.load(path)
    assert len(led) == 1 and led.p_home[0] == 0.55 and led.result[0] == "H"


def test_seasons():
    assert auto_season(date(2026, 11, 15)) == "2026-27"
    assert auto_season(date(2027, 3, 1)) == "2026-27"
    assert shift_season("2026-27", -2) == "2024-25"


def test_score_histogram_tables():
    from football_predictions.histogram import grid, order_key, scoreline_table, summary
    scores = ["2-1", "1-0", "0-0", "1-2", "0-1", "3-0", "0-3", "2-2", "1-1", "3-1", "2-0", "1-3", "0-2"]
    assert sorted(scores, key=order_key) == ["0-3", "1-3", "0-2", "1-2", "0-1", "0-0", "1-1", "2-2",
                                             "1-0", "2-1", "2-0", "3-1", "3-0"]
    m = pd.DataFrame({"season": ["2025-26"] * 4 + ["2026-27"] * 2, "home": list("abcdef"), "away": list("ghijkl"),
                      "hg": [1, 1, 2, 0, 1, 7], "ag": [1, 1, 0, 0, 1, 0],
                      "date": pd.date_range("2026-01-01", periods=6)})
    t = scoreline_table(m, ["2025-26", "2026-27"]).set_index("scoreline")
    assert list(t.index) == ["0-0", "1-1", "2-0", "7-0"]
    assert t.loc["1-1", "2025-26 matches"] == 2 and t.loc["1-1", "2026-27 %"] == 0.5
    assert t.loc["1-1", "All seasons matches"] == 3 and t.loc["7-0", "goal difference"] == 7
    assert t[[c for c in t if str(c).endswith("%")]].sum().round(9).eq(1).all()
    pooled = scoreline_table(m, ["2025-26", "2026-27"], top=1)
    assert pooled.scoreline.tolist() == ["1-1", "Other"] and pooled["All seasons matches"].tolist() == [3, 3]
    g = grid(m)
    assert g.loc["6+", "0"] == 1 and g.to_numpy().sum() == 6
    s = summary(m, ["2025-26", "2026-27"]).set_index("season")
    assert s.loc["All seasons", "matches"] == 6 and s.loc["2025-26", "draw %"] == 0.75
