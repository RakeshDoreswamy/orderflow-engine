"""Ticks -> time / tick / volume bars and footprint (price x {bid_vol, ask_vol, trades}).

Aggressor convention: side +1 trades at the ask (``ask_vol``), side -1 at the bid (``bid_vol``).
If the side column is an ESTIMATE (``estimated=True``) every derived column gets the ``_est`` suffix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def _suffix(estimated: bool) -> str:
    return "_est" if estimated else ""


def assign_time_bars(ts: pd.Series, freq: str, session_open: str) -> pd.Series:
    """Start timestamp of the bar each trade belongs to, anchored at the session open of its own day
    (so 45-minute bars start 09:15, 10:00, ...). Uses only the trade's own timestamp (no lookahead)."""
    h, m = (int(x) for x in session_open.split(":"))
    day_open = ts.dt.normalize() + pd.Timedelta(hours=h, minutes=m)
    step = pd.Timedelta(freq)
    return day_open + ((ts - day_open) // step) * step


def time_bars(ticks: pd.DataFrame, cfg: Config, freq: str | None = None,
              side_col: str = "side_true", estimated: bool = False) -> pd.DataFrame:
    """OHLCV bars with buy/sell volume and delta from the given side column."""
    freq = freq or cfg.bars.time_bar_freq
    sx = _suffix(estimated)
    t = ticks.copy()
    t["bar"] = assign_time_bars(t["ts"], freq, cfg.market.session_open)
    t["_buy"] = np.where(t[side_col] > 0, t["qty"], 0)
    t["_sell"] = np.where(t[side_col] < 0, t["qty"], 0)
    g = t.groupby("bar", sort=True)
    out = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                        "close": g["price"].last(), "volume": g["qty"].sum(), "trades": g["qty"].size(),
                        f"ask_vol{sx}": g["_buy"].sum(), f"bid_vol{sx}": g["_sell"].sum()})
    out[f"delta{sx}"] = out[f"ask_vol{sx}"] - out[f"bid_vol{sx}"]
    return out.reset_index()


def _chunk_bars(ticks: pd.DataFrame, group: np.ndarray) -> pd.DataFrame:
    g = ticks.groupby(group, sort=True)
    out = pd.DataFrame({"start": g["ts"].first(), "end": g["ts"].last(), "open": g["price"].first(),
                        "high": g["price"].max(), "low": g["price"].min(), "close": g["price"].last(),
                        "volume": g["qty"].sum(), "trades": g["qty"].size()})
    return out.reset_index(drop=True)


def tick_bars(ticks: pd.DataFrame, n_trades: int) -> pd.DataFrame:
    """A new bar every ``n_trades`` trades. The last bar may be incomplete."""
    return _chunk_bars(ticks, np.arange(len(ticks)) // n_trades)


def volume_bars(ticks: pd.DataFrame, volume: int) -> pd.DataFrame:
    """A new bar after ``volume`` shares have traded (a trade belongs to the bar where it starts
    its cumulative volume, so a bar's volume is ``volume`` give or take one trade)."""
    cum_before = ticks["qty"].cumsum().to_numpy() - ticks["qty"].to_numpy()
    return _chunk_bars(ticks, (cum_before // volume).astype(int))


def footprint(ticks: pd.DataFrame, cfg: Config, tick_size: float | None = None,
              freq: str | None = None, row_ticks: int | None = None,
              side_col: str = "side_true", estimated: bool = False) -> pd.DataFrame:
    """Long-format footprint: one row per (bar, price row).

    Columns: bar, row_price, bid_vol, ask_vol, delta, total_vol, trades (+ unclassified volume
    when some trades have side 0). ``row_price`` is the LOWER edge of the row; a row spans
    ``row_ticks`` ticks. Conservation: sum of volumes equals sum of trade quantities.
    """
    tick_size = tick_size or cfg.instrument.default_tick_size
    row_ticks = row_ticks or cfg.bars.footprint_row_ticks
    freq = freq or cfg.bars.time_bar_freq
    sx = _suffix(estimated)
    t = ticks[["ts", "price", "qty", side_col]].copy()
    t["bar"] = assign_time_bars(t["ts"], freq, cfg.market.session_open)
    ticks_idx = np.rint(t["price"].to_numpy() / tick_size).astype(np.int64)
    t["row_price"] = np.round((ticks_idx // row_ticks) * row_ticks * tick_size, 6)
    t["_ask"] = np.where(t[side_col] > 0, t["qty"], 0)
    t["_bid"] = np.where(t[side_col] < 0, t["qty"], 0)
    t["_unk"] = np.where(t[side_col] == 0, t["qty"], 0)
    g = t.groupby(["bar", "row_price"], sort=True)
    out = pd.DataFrame({f"bid_vol{sx}": g["_bid"].sum(), f"ask_vol{sx}": g["_ask"].sum(),
                        "total_vol": g["qty"].sum(), "trades": g["qty"].size()})
    out[f"delta{sx}"] = out[f"ask_vol{sx}"] - out[f"bid_vol{sx}"]
    unk = g["_unk"].sum()
    if unk.any():
        out[f"unclassified_vol{sx}"] = unk
    return out.reset_index()
