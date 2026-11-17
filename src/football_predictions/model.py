"""Time-weighted Dixon-Coles model with feeder divisions.

Expected goals:
    home: exp(mu_div + home_div + att[home] - def[away])
    away: exp(mu_div            + att[away] - def[home])
Every team has one attack and one defence rating on a shared scale. Promoted and relegated
teams link the divisions, so a promoted side starts the season with a rating carried over from
its second-division matches, which sets an implicit "promotion penalty" fitted from the data.

Fitting is weighted Poisson maximum likelihood (weight = 0.5 ** (age / half_life)) with a ridge
penalty pulling ratings toward average. Then the Dixon-Coles rho, which corrects the
probabilities of 0-0, 1-0, 0-1 and 1-1, is fitted on top-division matches.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


@dataclass
class Hyper:
    half_life_days: float = 180.0
    ridge: float = 1.0
    feeder_weight: float = 1.0
    cup_weight: float = 0.5
    max_goals: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "Hyper":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.maximum(t, 1e-10)


def score_grid(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    p = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    x, y = np.meshgrid(g, g, indexing="ij")
    p = p * _tau(x, y, np.full_like(p, lam), np.full_like(p, mu), rho)
    return p / p.sum()


def summarize(lam: float, mu: float, rho: float, max_goals: int = 10) -> dict:
    """Expected goals -> W/D/L probabilities, pick, goal-difference forecast and interval."""
    p = score_grid(lam, mu, rho, max_goals)
    n = p.shape[0]
    p_h, p_d, p_a = np.tril(p, -1).sum(), np.trace(p), np.triu(p, 1).sum()
    gd = np.subtract.outer(np.arange(n), np.arange(n)).ravel()
    gd_vals = np.arange(-(n - 1), n)
    gd_p = np.bincount(gd + n - 1, weights=p.ravel(), minlength=2 * n - 1)
    cdf = np.cumsum(gd_p)
    lo = int(gd_vals[np.searchsorted(cdf, 0.10)])
    hi = int(gd_vals[np.searchsorted(cdf, 0.90)])
    x, y = np.unravel_index(p.argmax(), p.shape)
    probs = {"H": p_h, "D": p_d, "A": p_a}
    pick = max(probs, key=probs.get)
    return {
        "xg_home": float(lam), "xg_away": float(mu),
        "p_home": float(p_h), "p_draw": float(p_d), "p_away": float(p_a),
        "pick": pick, "confidence": float(probs[pick]),
        "exp_gd": float(gd_vals @ gd_p), "gd_lo80": lo, "gd_hi80": hi,
        "likely_score": f"{x}-{y}", "p_likely_score": float(p[x, y]), "rho": float(rho),
    }


class DixonColes:
    def __init__(self, hyper: Hyper | None = None):
        self.hyper = hyper or Hyper()
        self.fitted = False

    # ---------------------------------------------------------------- fit
    def fit(self, matches: pd.DataFrame, asof, top_div: str, warm: "DixonColes | None" = None) -> "DixonColes":
        asof = pd.Timestamp(asof)
        h = self.hyper
        m = matches[matches.date < asof]
        age = (asof - m.date).dt.days.to_numpy(float)
        w = 0.5 ** (age / h.half_life_days)
        div = m["div"].to_numpy()
        w = w * np.where(div == top_div, 1.0, np.where(div == "CUP", h.cup_weight, h.feeder_weight))
        keep = w > 1e-3
        m, w = m[keep], w[keep]
        if len(m) < 20 or not (m["div"] == top_div).any():
            raise ValueError("not enough matches to fit")

        teams = sorted(set(m.home) | set(m.away))
        divs = sorted(set(m["div"]))
        ti = {t: i for i, t in enumerate(teams)}
        di = {d: i for i, d in enumerate(divs)}
        T, D = len(teams), len(divs)
        hi = m.home.map(ti).to_numpy()
        ai = m.away.map(ti).to_numpy()
        dv = m["div"].map(di).to_numpy()

        y = np.concatenate([m.hg.to_numpy(float), m.ag.to_numpy(float)])
        sc = np.concatenate([hi, ai])
        cc = np.concatenate([ai, hi])
        dd = np.concatenate([dv, dv])
        neutral = (m["neutral"].fillna(False).to_numpy(bool) if "neutral" in m else np.zeros(len(m), bool))
        ih = np.concatenate([(~neutral).astype(float), np.zeros(len(m))])   # no home advantage at neutral venues
        ww = np.concatenate([w, w])
        ridge = h.ridge

        def unpack(th):
            return th[:T], th[T:2 * T], th[2 * T:2 * T + D], th[2 * T + D:]

        def f(th):
            att, de, mu, ha = unpack(th)
            eta = mu[dd] + ha[dd] * ih + att[sc] - de[cc]
            lam = np.exp(eta)
            r = ww * (lam - y)
            loss = np.sum(ww * (lam - y * eta)) + 0.5 * ridge * (att @ att + de @ de)
            g = np.concatenate([
                np.bincount(sc, r, T) + ridge * att,
                -np.bincount(cc, r, T) + ridge * de,
                np.bincount(dd, r, D),
                np.bincount(dd, r * ih, D),
            ])
            return loss, g

        th0 = np.zeros(2 * T + 2 * D)
        th0[2 * T:2 * T + D] = np.log(max(y.mean(), 0.1))
        th0[2 * T + D:] = 0.25
        if warm is not None and warm.fitted:
            for t, i in ti.items():
                if t in warm.team_index:
                    j = warm.team_index[t]
                    th0[i], th0[T + i] = warm.att[j], warm.defence[j]
            for d, i in di.items():
                if d in warm.div_index:
                    j = warm.div_index[d]
                    th0[2 * T + i], th0[2 * T + D + i] = warm.mu[j], warm.home_adv[j]

        res = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        att, de, mu, ha = unpack(res.x)

        self.teams, self.team_index = teams, ti
        self.divs, self.div_index = divs, di
        self.att, self.defence, self.mu, self.home_adv = att, de, mu, ha
        self.top_div, self.asof = top_div, asof
        lg = m[m["div"] != "CUP"]   # a team's division comes from its league games, not cup ties
        self.last_div = (pd.concat([lg[["date", "home", "div"]].rename(columns={"home": "team"}),
                                    lg[["date", "away", "div"]].rename(columns={"away": "team"})])
                         .sort_values("date").groupby("team")["div"].last().to_dict())

        # Dixon-Coles low-score correction on top-division matches
        top = (m["div"] == top_div).to_numpy()
        lam_h = np.exp(mu[dv] + ha[dv] + att[hi] - de[ai])[top]
        lam_a = np.exp(mu[dv] + att[ai] - de[hi])[top]
        x, yy, wt = m.hg.to_numpy()[top], m.ag.to_numpy()[top], w[top]
        r = minimize_scalar(lambda p: -np.sum(wt * np.log(_tau(x, yy, lam_h, lam_a, p))),
                            bounds=(-0.25, 0.25), method="bounded")
        self.rho = float(r.x)
        self.n_matches = int(len(m))
        self.converged = bool(res.success)
        self.fitted = True
        return self

    # ---------------------------------------------------------------- predict
    def _team(self, team: str) -> tuple[float, float]:
        if team in self.team_index:
            j = self.team_index[team]
            return self.att[j], self.defence[j]
        # unseen team: assume a typical feeder-division side (or league average if no feeder data)
        feeders = [self.team_index[t] for t, d in self.last_div.items() if d != self.top_div]
        if feeders:
            return float(np.mean(self.att[feeders])), float(np.mean(self.defence[feeders]))
        return 0.0, 0.0

    def expected_goals(self, home: str, away: str, div: str | None = None,
                       adjust: dict | None = None) -> tuple[float, float]:
        j = self.div_index[div or self.top_div]
        ah, dh = self._team(home)
        aa, da = self._team(away)
        lam = np.exp(self.mu[j] + self.home_adv[j] + ah - da)
        mu = np.exp(self.mu[j] + aa - dh)
        if adjust:   # optional multipliers, e.g. from lineup information
            lam *= adjust.get("home_attack", 1.0) * adjust.get("away_defence", 1.0)
            mu *= adjust.get("away_attack", 1.0) * adjust.get("home_defence", 1.0)
        return float(lam), float(mu)

    def score_grid(self, lam: float, mu: float) -> np.ndarray:
        return score_grid(lam, mu, self.rho, self.hyper.max_goals)

    def predict(self, home: str, away: str, div: str | None = None, adjust: dict | None = None) -> dict:
        lam, mu = self.expected_goals(home, away, div, adjust)
        return {"home": home, "away": away, **summarize(lam, mu, self.rho, self.hyper.max_goals)}

    def ratings(self) -> pd.DataFrame:
        """Team ratings, with expected goal difference vs. an average top-division team at a neutral venue."""
        j = self.div_index[self.top_div]
        mu = self.mu[j] + self.home_adv[j] / 2
        top_ix = [self.team_index[t] for t, d in self.last_div.items() if d == self.top_div]
        a0, d0 = np.mean(self.att[top_ix]), np.mean(self.defence[top_ix])
        df = pd.DataFrame({
            "team": self.teams, "last_division": [self.last_div.get(t) for t in self.teams],
            "attack": self.att, "defence": self.defence,
        })
        df["gf_vs_avg"] = np.exp(mu + df.attack - d0)
        df["ga_vs_avg"] = np.exp(mu + a0 - df.defence)
        df["gd_vs_avg"] = df.gf_vs_avg - df.ga_vs_avg
        return df.sort_values("gd_vs_avg", ascending=False).reset_index(drop=True)
