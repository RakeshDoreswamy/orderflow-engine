"""Offline EOD providers: read files YOU downloaded. No network access, no credentials.

* ``CsvProvider``: one CSV per symbol, ``<root>/<SYMBOL>.csv``, common column names are accepted (case-insensitive).
* ``parse_bhavcopy``: NSE "security-wise delivery position" file (sec_bhavdata_full_DDMMYYYY.csv style).
  Column names are matched by name, not position; unknown columns are ignored. Check the layout of the file you
  downloaded: NSE has changed file formats before, and this parser raises if the columns it needs are absent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import slice_dates, validate_eod

_ALIASES = {
    "date": ["date", "timestamp", "date1", "tradingdate"],
    "open": ["open", "open_price"], "high": ["high", "high_price"], "low": ["low", "low_price"],
    "close": ["close", "close_price", "last_price_close"], "volume": ["volume", "ttl_trd_qnty", "total_traded_quantity", "tottrdqty"],
    "delivery_qty": ["delivery_qty", "deliv_qty", "deliverable_qty"],
    "delivery_pct": ["delivery_pct", "deliv_per", "pct_deliverable", "delivery_percentage"],
}


def _norm(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("%", "pct").replace(".", "")


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename vendor columns to the EOD contract. Raises if a required column cannot be found."""
    cols = {_norm(c): c for c in raw.columns}
    out = {}
    for std, names in _ALIASES.items():
        for n in names:
            if n in cols:
                out[std] = raw[cols[n]]
                break
    df = pd.DataFrame(out)
    miss = [c for c in ["date", "open", "high", "low", "close", "volume"] if c not in df.columns]
    if miss:
        raise ValueError(f"cannot find columns for {miss}; file has {list(raw.columns)}")
    if "delivery_pct" in df.columns:
        df["delivery_pct"] = pd.to_numeric(df["delivery_pct"].astype(str).str.strip().replace({"-": None, "": None}), errors="coerce")
    if "delivery_qty" in df.columns:
        df["delivery_qty"] = pd.to_numeric(df["delivery_qty"].astype(str).str.strip().replace({"-": None, "": None}), errors="coerce")
    return df


class CsvProvider:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        p = self.root / f"{symbol.upper()}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (this provider only reads files you placed there)")
        df = validate_eod(standardise(pd.read_csv(p)))
        return slice_dates(df, start, end)


def parse_bhavcopy(path: str | Path, symbol: str | None = None, series: str = "EQ") -> pd.DataFrame:
    """Parse one day's delivery file into the EOD contract (all symbols, or one)."""
    raw = pd.read_csv(path, skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    nc = {_norm(c): c for c in raw.columns}
    if "symbol" not in nc:
        raise ValueError("bhavcopy file has no SYMBOL column")
    if "series" in nc:
        raw = raw[raw[nc["series"]].astype(str).str.strip() == series]
    raw = raw[raw[nc["symbol"]].astype(str).str.strip().str.upper() == symbol.upper()] if symbol else raw
    df = standardise(raw)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="mixed", dayfirst=True)
    df.insert(0, "symbol", raw[nc["symbol"]].astype(str).str.strip().to_numpy())
    return df.reset_index(drop=True)


def build_series_from_bhavcopies(paths: list[str | Path], symbol: str) -> pd.DataFrame:
    """Stack daily bhavcopy files into one validated series for ``symbol``. Missing days stay missing."""
    parts = [parse_bhavcopy(p, symbol) for p in paths]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError(f"{symbol} not found in any of the files")
    return validate_eod(pd.concat(parts).drop(columns="symbol"))
