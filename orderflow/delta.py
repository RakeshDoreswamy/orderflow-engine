"""Delta family: per-bar delta, intrabar extremes, CVD, confirmed swings, divergences, flips.

Aggressor convention: ask_vol (side +1) minus bid_vol (side -1). If sides are estimates, pass
``estimated=True`` and every output column ends in ``_est``.
Causality: swings are only KNOWN ``swing_right`` bars after they happen, and signals are stamped at
that confirmation bar, never at the swing bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data.bars import assign_time_bars, time_bars


def delta_bars(ticks: pd.DataFrame, cfg: Config, freq: str | None = None, side_col: str = "side_true",
               estimated: bool = False) -> pd.DataFrame:
    """Time bars + delta, delta_pct, delta_max/min (running delta inside the bar, 0 included),
    delta_at_high / delta_at_low (net delta of trades printed exactly at the bar's high / low) and
    session-anchored CVD (resets each trading day) at the bar close."""
    freq = freq or cfg.bars.time_bar_freq
    sx = "_est" if estimated else ""
    bars = time_bars(ticks, cfg, freq, side_col, estimated)
    t = ticks[["ts", "price", "qty", side_col]].copy()
    t["bar"] = assign_time_bars(t["ts"], freq, cfg.market.session_open)
    t["signed"] = t[side_col].astype(int) * t["qty"]
    g = t.groupby("bar")
    t["run"] = g["signed"].cumsum()
    t["hi"], t["lo"] = g["price"].transform("max"), g["price"].transform("min")
    t["at_hi"] = np.where(t["price"] == t["hi"], t["signed"], 0)
    t["at_lo"] = np.where(t["price"] == t["lo"], t["signed"], 0)
    g = t.groupby("bar")
    extra = pd.DataFrame({f"delta_max{sx}": g["run"].max().clip(lower=0),
                          f"delta_min{sx}": g["run"].min().clip(upper=0),
                          f"delta_at_high{sx}": g["at_hi"].sum(), f"delta_at_low{sx}": g["at_lo"].sum()})
    out = bars.merge(extra, left_on="bar", right_index=True)
    out[f"delta_pct{sx}"] = out[f"delta{sx}"] / out["volume"]
    out[f"cvd{sx}"] = out.groupby(out["bar"].dt.date)[f"delta{sx}"].cumsum()
    return out


def cvd_anchored(bars: pd.DataFrame, anchor_ts: pd.Timestamp, delta_col: str = "delta") -> pd.Series:
    """CVD that starts at 0 at ``anchor_ts`` (a bar start) and accumulates from there; NaN before it."""
    d = bars[delta_col].where(bars["bar"] >= anchor_ts)
    return d.cumsum().where(bars["bar"] >= anchor_ts)


def ohlcv_delta_est(bars: pd.DataFrame) -> pd.Series:
    """Bar-level estimate from OHLCV only: volume * (2*close - high - low) / (high - low).
    Crude: assumes volume splits by where the close sits in the range. Zero-range bars -> NaN."""
    rng = (bars["high"] - bars["low"]).replace(0, np.nan)
    return bars["volume"] * (2 * bars["close"] - bars["high"] - bars["low"]) / rng


def confirmed_swings(high: np.ndarray, low: np.ndarray, left: int, right: int) -> pd.DataFrame:
    """Swing high at i: high[i] > every high in the ``left`` bars before and >= every high in the
    ``right`` bars after (mirror for lows). It becomes known at confirm_idx = i + right."""
    h, l = np.asarray(high, float), np.asarray(low, float)
    rows = []
    for i in range(left, len(h) - right):
        if h[i] > h[i - left:i].max() and h[i] >= h[i + 1:i + right + 1].max():
            rows.append((i, i + right, "high", h[i]))
        if l[i] < l[i - left:i].min() and l[i] <= l[i + 1:i + right + 1].min():
            rows.append((i, i + right, "low", l[i]))
    return pd.DataFrame(rows, columns=["idx", "confirm_idx", "kind", "price"]).sort_values(
        ["confirm_idx", "kind"]).reset_index(drop=True)


def cvd_divergences(bars: pd.DataFrame, cfg: Config, estimated: bool = False) -> pd.DataFrame:
    """Divergences between consecutive confirmed swings of the same kind within one session.
    bearish: higher price high but LOWER CVD at the swing.  bullish: lower price low but HIGHER CVD."""
    sx = "_est" if estimated else ""
    sw = confirmed_swings(bars["high"].to_numpy(), bars["low"].to_numpy(),
                          cfg.delta.swing_left, cfg.delta.swing_right)
    cvd = bars[f"cvd{sx}"].to_numpy()
    day = bars["bar"].dt.date.to_numpy()
    out = []
    for kind in ("high", "low"):
        s = sw[sw["kind"] == kind].reset_index(drop=True)
        for a, b in zip(s.index[:-1], s.index[1:]):
            r1, r2 = s.loc[a], s.loc[b]
            if day[r1.idx] != day[r2.idx]:
                continue
            if kind == "high" and r2.price > r1.price and cvd[r2.idx] < cvd[r1.idx]:
                out.append(("bearish", r1.idx, r2.idx, r2.confirm_idx))
            if kind == "low" and r2.price < r1.price and cvd[r2.idx] > cvd[r1.idx]:
                out.append(("bullish", r1.idx, r2.idx, r2.confirm_idx))
    df = pd.DataFrame(out, columns=["kind", "first_idx", "second_idx", "confirm_idx"])
    df = df.astype({"first_idx": int, "second_idx": int, "confirm_idx": int})
    df["confirm_bar"] = pd.Series(bars["bar"].to_numpy()[df["confirm_idx"].to_numpy()], dtype=bars["bar"].dtype)
    return df.sort_values("confirm_idx").reset_index(drop=True)


def flags(bars: pd.DataFrame, cfg: Config, tick_size: float, estimated: bool = False) -> pd.DataFrame:
    """Adds delta_flip (+1: delta turned positive, -1: turned negative, 0: none) and
    delta_no_progress (big one-sided delta but price barely moved in that direction)."""
    sx = "_est" if estimated else ""
    out = bars.copy()
    pct = out[f"delta_pct{sx}"].to_numpy()
    prev = np.r_[np.nan, pct[:-1]]
    m = cfg.delta.flip_min_abs_pct
    ok = (np.abs(pct) >= m) & (np.abs(prev) >= m) & (np.sign(pct) != np.sign(prev))
    out[f"delta_flip{sx}"] = np.where(ok, np.sign(pct), 0).astype(int)
    move = (out["close"] - out["open"]).to_numpy() * np.sign(pct)
    out[f"delta_no_progress{sx}"] = (np.abs(pct) >= cfg.delta.no_progress_min_abs_pct) & (
        move <= cfg.delta.no_progress_max_move_ticks * tick_size)
    return out
