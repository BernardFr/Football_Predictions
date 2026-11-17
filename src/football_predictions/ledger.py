"""Live track record: every prediction made before kickoff is stored, then scored when the result arrives.

Key = (season, home, away, variant): a home/away pairing happens once per league season, so
rescheduled games keep their row and the latest pre-kickoff prediction wins. `variant` is "base"
(the team-strength model) or "lineup" (adjusted for the announced starting XIs).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import outcome

KEY = ["season", "home", "away", "variant"]


def load(path: Path) -> pd.DataFrame:
    if Path(path).exists():
        led = pd.read_csv(path, parse_dates=["date", "predicted_at"])
        if "variant" not in led:
            led["variant"] = "base"
        return led
    return pd.DataFrame()


def record(path: Path, preds: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    led = load(path)
    preds = preds.assign(variant=variant, predicted_at=pd.Timestamp.now().floor("s"), hg=np.nan, ag=np.nan, result=None)
    if not led.empty:
        settled = led[led.result.notna()]
        fresh = preds.merge(settled[KEY], on=KEY, how="left", indicator=True)
        preds = preds[(fresh["_merge"] == "left_only").to_numpy()]
        led = pd.concat([led[~led.set_index(KEY).index.isin(preds.set_index(KEY).index)], preds], ignore_index=True)
    else:
        led = preds
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    led.sort_values("date").to_csv(path, index=False)
    return led


def settle(path: Path, results: pd.DataFrame) -> int:
    led = load(path)
    if led.empty:
        return 0
    k = ["season", "home", "away"]
    res = results[k + ["hg", "ag"]].rename(columns={"hg": "hg_new", "ag": "ag_new"})
    m = led.merge(res, on=k, how="left")
    m["result"] = m["result"].astype(object)
    new = m.result.isna() & m.hg_new.notna()
    m.loc[new, "hg"] = m.loc[new, "hg_new"]
    m.loc[new, "ag"] = m.loc[new, "ag_new"]
    m.loc[new, "result"] = outcome(m.loc[new, "hg"], m.loc[new, "ag"])
    m.drop(columns=["hg_new", "ag_new"]).to_csv(path, index=False)
    return int(new.sum())
