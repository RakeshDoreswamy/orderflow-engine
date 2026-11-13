"""EOD provider contract. A provider returns one row per trading day for one symbol:

    date (datetime64, tz-naive, exchange-local date), open, high, low, close, volume
    optional: delivery_qty, delivery_pct   (NSE publishes these in the security-wise delivery file)

Providers read what they are given. They never fabricate rows and never fill gaps.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

REQUIRED = ["date", "open", "high", "low", "close", "volume"]
OPTIONAL = ["delivery_qty", "delivery_pct"]


class EODProvider(Protocol):
    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame: ...


def validate_eod(df: pd.DataFrame) -> pd.DataFrame:
    """Check the contract and return a clean, date-sorted copy. Raises ValueError on a broken frame."""
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"EOD frame is missing columns: {miss}")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    for c in REQUIRED[1:] + [c for c in OPTIONAL if c in d.columns]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("date").reset_index(drop=True)
    if d["date"].duplicated().any():
        raise ValueError("EOD frame has duplicate dates")
    bad = d[["open", "high", "low", "close"]].isna().any(axis=1) | (d["high"] < d["low"]) | (d["volume"] < 0)
    bad |= (d["high"] < d[["open", "close"]].max(axis=1) - 1e-9) | (d["low"] > d[["open", "close"]].min(axis=1) + 1e-9)
    if bad.any():
        raise ValueError(f"EOD frame has {int(bad.sum())} inconsistent rows (first: {d.loc[bad, 'date'].iloc[0].date()})")
    return d


def slice_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)
