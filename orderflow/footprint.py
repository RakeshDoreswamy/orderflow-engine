"""Footprint analytics on the long footprint from ``data.bars.footprint``.

Rules (plain English):
* POC of a bar: the price row with the most total volume (ties -> closest to the middle).
* Value area of a bar: expansion from the POC until ``profile.value_area_pct`` of the bar's volume.
* Buy imbalance at row P: ask[P] >= ratio * bid[P - 1 row] and ask[P] >= min volume.
  Sell imbalance at row P: bid[P] >= ratio * ask[P + 1 row] and bid[P] >= min volume.
  (Diagonal comparison. The lowest row has no buy test and the highest has no sell test.)
* Stacked imbalance: >= ``stacked_min_levels`` consecutive rows with the same-side imbalance.
* Finished high: zero ask volume at the bar's high row (unfinished otherwise); finished low: zero bid
  volume at the low row. Unfinished extremes are 'poor' highs/lows in footprint terms (the TPO version
  lives in tpo.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .profile import value_area_idx


def _runs(flag: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros(len(flag), bool)
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return out


def analyze_footprint(fp: pd.DataFrame, cfg: Config, tick_size: float, row_ticks: int | None = None,
                      estimated: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (bars, rows). ``rows`` = every row of every bar on a contiguous price grid (zero-volume rows
    included) with is_poc, in_va, buy_imb, sell_imb, stacked_buy, stacked_sell. ``bars`` = one line per bar
    with poc, vah, val, high/low row and finished/unfinished status, zero_rows count."""
    sx = "_est" if estimated else ""
    row_ticks = row_ticks or cfg.bars.footprint_row_ticks
    step = row_ticks * tick_size
    f, ratio, minv = cfg.footprint, cfg.footprint.imbalance_ratio, cfg.footprint.imbalance_min_volume
    rows_out, bars_out = [], []
    for bar, g in fp.groupby("bar", sort=True):
        k = np.rint((g["row_price"].to_numpy() - g["row_price"].min()) / step).astype(int)
        n = k.max() + 1
        bid, ask = np.zeros(n), np.zeros(n)
        bid[k], ask[k] = g[f"bid_vol{sx}"].to_numpy(), g[f"ask_vol{sx}"].to_numpy()
        tot = bid + ask + (np.zeros(n) if f"unclassified_vol{sx}" not in g else np.bincount(
            k, weights=g[f"unclassified_vol{sx}"].to_numpy(), minlength=n))
        price = np.round(g["row_price"].min() + np.arange(n) * step, 6)
        buy, sell = np.zeros(n, bool), np.zeros(n, bool)
        buy[1:] = (ask[1:] >= ratio * bid[:-1]) & (ask[1:] >= minv)
        sell[:-1] = (bid[:-1] >= ratio * ask[1:]) & (bid[:-1] >= minv)
        lo, hi, poc = value_area_idx(tot, cfg.profile.value_area_pct)
        rows_out.append(pd.DataFrame({"bar": bar, "row_price": price, f"bid_vol{sx}": bid, f"ask_vol{sx}": ask,
                                      "total_vol": tot, "is_poc": np.arange(n) == poc,
                                      "in_va": (np.arange(n) >= lo) & (np.arange(n) <= hi),
                                      "buy_imb": buy, "sell_imb": sell,
                                      "stacked_buy": _runs(buy, f.stacked_min_levels),
                                      "stacked_sell": _runs(sell, f.stacked_min_levels)}))
        traded = np.flatnonzero(tot > 0)
        hr, lr = traded.max(), traded.min()
        fin_hi, fin_lo = ask[hr] == 0, bid[lr] == 0
        bars_out.append({"bar": bar, "poc": price[poc], "vah": price[hi], "val": price[lo],
                         "high_row": price[hr], "low_row": price[lr],
                         "high_finished": bool(fin_hi), "low_finished": bool(fin_lo),
                         "poor_high": not fin_hi, "poor_low": not fin_lo,
                         "zero_rows": int((tot[lr:hr + 1] == 0).sum())})
    return pd.DataFrame(bars_out), pd.concat(rows_out, ignore_index=True)
