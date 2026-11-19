"""
Live track record: every prediction made before kickoff is stored, then scored when the result arrives.

Key = (season, home, away, variant): a home/away pairing happens once per league season, so
rescheduled games keep their row and the latest pre-kickoff prediction wins. `variant` is "base"
(the team-strength model) or "lineup" (adjusted for the announced starting XIs).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .csvio import read_csv, write_csv
from .metrics import outcome

MATCH_KEY = ["season", "home", "away"]
KEY = MATCH_KEY + ["variant"]
BASE_VARIANT = "base"
DATE_COLS = ["date", "predicted_at"]
TIMESTAMP_RESOLUTION = "s"


def load(path: Path) -> pd.DataFrame:
    if Path(path).exists():
        led = read_csv(path, parse_dates=DATE_COLS)
        if "variant" not in led:
            led["variant"] = BASE_VARIANT
        return led
    return pd.DataFrame()


def record(path: Path, preds: pd.DataFrame, variant: str = BASE_VARIANT) -> pd.DataFrame:
    led = load(path)
    preds = preds.assign(variant=variant, predicted_at=pd.Timestamp.now().floor(TIMESTAMP_RESOLUTION), hg=np.nan, ag=np.nan, result=None)
    if not led.empty:
        settled = led[led.result.notna()]
        fresh = preds.merge(settled[KEY], on=KEY, how="left", indicator=True)
        preds = preds[(fresh["_merge"] == "left_only").to_numpy()]
        led = pd.concat([led[~led.set_index(KEY).index.isin(preds.set_index(KEY).index)], preds], ignore_index=True)
    else:
        led = preds
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_csv(led.sort_values("date"), path)
    return led


def settle(path: Path, results: pd.DataFrame) -> int:
    led = load(path)
    if led.empty:
        return 0
    match_key = MATCH_KEY
    res = results[match_key + ["hg", "ag"]].rename(columns={"hg": "hg_new", "ag": "ag_new"})
    merged = led.merge(res, on=match_key, how="left")
    merged["result"] = merged["result"].astype(object)
    new = merged.result.isna() & merged.hg_new.notna()
    merged.loc[new, "hg"] = merged.loc[new, "hg_new"]
    merged.loc[new, "ag"] = merged.loc[new, "ag_new"]
    merged.loc[new, "result"] = outcome(merged.loc[new, "hg"], merged.loc[new, "ag"])
    write_csv(merged.drop(columns=["hg_new", "ag_new"]), path)
    return int(new.sum())
