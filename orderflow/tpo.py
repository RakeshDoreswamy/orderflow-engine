"""Market profile (TPO): 30-minute letters, initial balance, extensions, single prints, poor/excess
extremes, and a heuristic open-type classifier (rules in config; to be TESTED, not believed).
Only COMPLETED periods should be used for decisions: an unfinished period is still changing."""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data.bars import assign_time_bars
from .profile import value_area_idx

LETTERS = string.ascii_uppercase + string.ascii_lowercase


@dataclass
class TPO:
    matrix: pd.DataFrame        # index: row price (ascending), columns: period letters, bool
    period_starts: list
    ibh: float
    ibl: float
    extensions: list            # [(letter, 'up'|'down')], first period that broke the IB each way
    poc: float
    vah: float
    val: float
    single_prints: list         # [(low_price, high_price, is_tail)]
    poor_high: bool
    poor_low: bool
    excess_high: bool
    excess_low: bool


def build_tpo(ticks: pd.DataFrame, cfg: Config, tick_size: float) -> TPO:
    c = cfg.tpo
    step = c.row_ticks * tick_size
    t = ticks[["ts", "price"]].copy()
    t["p"] = assign_time_bars(t["ts"], f"{c.period_min}min", cfg.market.session_open)
    t["k"] = np.rint(t["price"] / tick_size).astype(np.int64) // c.row_ticks
    lo, hi = t["k"].min(), t["k"].max()
    starts = sorted(t["p"].unique())
    m = np.zeros((hi - lo + 1, len(starts)), bool)
    per = t.groupby("p")["k"].agg(["min", "max"])
    for j, s in enumerate(starts):
        m[per.loc[s, "min"] - lo:per.loc[s, "max"] - lo + 1, j] = True
    prices = np.round((np.arange(lo, hi + 1)) * step, 6)
    letters = [LETTERS[j] for j in range(len(starts))]
    mat = pd.DataFrame(m, index=prices, columns=letters)
    cnt = m.sum(axis=1)
    ibn = min(c.ib_periods, len(starts))
    ibh = prices[np.flatnonzero(m[:, :ibn].any(axis=1)).max()]
    ibl = prices[np.flatnonzero(m[:, :ibn].any(axis=1)).min()]
    ext = []
    for j in range(ibn, len(starts)):
        rows = np.flatnonzero(m[:, j])
        if prices[rows.max()] > ibh and not any(d == "up" for _, d in ext):
            ext.append((letters[j], "up"))
        if prices[rows.min()] < ibl and not any(d == "down" for _, d in ext):
            ext.append((letters[j], "down"))
    l, h, poc = value_area_idx(cnt.astype(float), cfg.profile.value_area_pct)
    singles, i, n = [], 0, len(cnt)
    while i < n:
        if cnt[i] == 1:
            j = i
            while j + 1 < n and cnt[j + 1] == 1:
                j += 1
            if j - i + 1 >= c.single_print_min_rows:
                singles.append((prices[i], prices[j], bool(i == 0 or j == n - 1)))
            i = j + 1
        else:
            i += 1

    def tail(side: str) -> bool:
        r = 0
        arr = cnt[::-1] if side == "high" else cnt
        while r < n and arr[r] == 1:
            r += 1
        return r >= c.excess_min_rows

    return TPO(mat, [pd.Timestamp(s) for s in starts], ibh, ibl, ext, prices[poc], prices[h], prices[l],
               singles, bool(cnt[-1] >= c.poor_min_tpos), bool(cnt[0] >= c.poor_min_tpos),
               tail("high"), tail("low"))


def open_type(ticks: pd.DataFrame, cfg: Config) -> str:
    """Heuristic open classification from the first ``window_min`` minutes (see config open_type.*):
    open_drive, open_test_drive, open_rejection_reverse or open_auction (everything else)."""
    c = cfg.open_type
    t0 = ticks["ts"].iloc[0]
    w = ticks[ticks["ts"] < t0.normalize() + pd.Timedelta(
        hours=int(cfg.market.session_open[:2]), minutes=int(cfg.market.session_open[3:]) + c.window_min)]
    if w.empty:
        return "unknown"
    o, h, l, last = w["price"].iloc[0], w["price"].max(), w["price"].min(), w["price"].iloc[-1]
    rng, net = h - l, last - o
    if rng == 0 or net == 0:
        return "open_auction"
    up = net > 0
    counter = (o - l) if up else (h - o)
    big = abs(net) >= c.drive_net_frac * rng
    if big and counter <= c.drive_counter_frac * rng:
        return "open_drive"
    if big and counter <= c.test_counter_max_frac * rng:
        return "open_test_drive"
    t_counter = w.loc[w["price"].idxmin() if up else w["price"].idxmax(), "ts"]
    t_main = w.loc[w["price"].idxmax() if up else w["price"].idxmin(), "ts"]
    if counter > c.test_counter_max_frac * rng and abs(net) >= c.reverse_net_frac * rng and t_counter < t_main:
        return "open_rejection_reverse"
    return "open_auction"
