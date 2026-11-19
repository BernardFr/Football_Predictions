"""
CSV read/write that keeps scoreline text (e.g. "1-1") from being turned into a date by Excel.

Scoreline columns are written with a leading apostrophe ("'1-1") and the apostrophe is removed on read.
Files written before this convention (no apostrophe) read back unchanged.
"""

from pathlib import Path

import pandas as pd

SCORE_COLS = ("likely_score", "lu_likely_score")
TEXT_PREFIX = "'"


def _prefix(value):
    return f"{TEXT_PREFIX}{value}" if isinstance(value, str) else value


def _unprefix(value):
    return value.removeprefix(TEXT_PREFIX) if isinstance(value, str) else value


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in SCORE_COLS:
        if col in out:
            out[col] = out[col].map(_prefix)
    out.to_csv(path, index=False)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    for col in SCORE_COLS:
        if col in df:
            df[col] = df[col].map(_unprefix)
    return df
