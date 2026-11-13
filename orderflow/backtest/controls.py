"""Validation controls for the backtest engine, on bars built from scratch (independent of the order-flow generator).

* Negative control: a driftless random walk with random-side signals. There is no edge, so the net result must be
  about minus (costs + slippage). A backtester that shows profit here is broken or leaking the future.
* Positive control: the same, but after each signal the price is nudged in the signal's direction. The engine must
  find a clearly positive net result. If it cannot see an edge you planted, a null result on real data means little.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def control_bars(cfg: Config, n_days: int = 20, seed: int = 1, drift_ticks: float = 0.0, drift_minutes: int = 15,
                 signal_every_min: int = 40, stop_ticks: int = 10, target_ticks: int = 15, tick_size: float = 0.05,
                 sigma_ticks: float = 6.0, start_price: float = 1000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (1-minute bars, signals). Signal side is random; stop/target are set from the signal bar's close."""
    rng = np.random.default_rng(seed)
    tz = cfg.market.timezone
    h, m = (int(x) for x in cfg.market.session_open.split(":"))
    nmin = 375
    bars, sigs = [], []
    price = start_price
    for d in pd.bdate_range("2026-03-02", periods=n_days):
        t0 = pd.Timestamp(d).tz_localize(tz) + pd.Timedelta(hours=h, minutes=m)
        inc = rng.normal(0, sigma_ticks * tick_size, nmin)
        side = np.zeros(nmin, int)
        for k in range(15, nmin - 60, signal_every_min):
            s = int(rng.choice([-1, 1]))
            side[k] = s
            inc[k + 1:k + 1 + drift_minutes] += s * drift_ticks * tick_size
        close = price + np.cumsum(inc)
        opn = np.concatenate([[price], close[:-1]])
        wick = np.abs(rng.normal(0, 0.5 * sigma_ticks * tick_size, (2, nmin)))
        high = np.maximum(opn, close) + wick[0]
        low = np.minimum(opn, close) - wick[1]
        for i in range(nmin):
            bars.append({"bar": t0 + pd.Timedelta(minutes=i), "open": opn[i], "high": high[i], "low": low[i], "close": close[i]})
        for k in np.flatnonzero(side):
            c, s = close[k], side[k]
            sigs.append({"time": t0 + pd.Timedelta(minutes=int(k) + 1), "side": s, "stop": c - s * stop_ticks * tick_size,
                         "target": c + s * target_ticks * tick_size, "setup": "control"})
        price = close[-1]
    return pd.DataFrame(bars), pd.DataFrame(sigs)
