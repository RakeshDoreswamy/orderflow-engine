"""TOY daily-bar generator for tests and teaching. NOT market data.

A random walk with planted "effort without result" episodes: a lead-in drift (down before an at-low episode,
up before an at-high one), then a few days of high volume, high delivery % and a compressed range. After the
episode the walk is memoryless: nothing here makes price reverse, so no profit can be read from these labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class SyntheticEOD:
    eod: pd.DataFrame      # contract columns + delivery_qty, delivery_pct
    labels: pd.DataFrame   # kind, start, end (dates)


def generate_eod(cfg: Config, seed: int | None = None, n_days: int | None = None) -> SyntheticEOD:
    s = cfg.synthetic_eod
    n = int(n_days or s.n_days)
    rng = np.random.default_rng(int(s.seed if seed is None else seed))
    dates = pd.bdate_range(s.start_date, periods=n)
    L, lead = int(s.episode_len), int(s.lead_in_days)
    seg = n // int(s.n_episodes)
    drift = np.zeros(n)
    in_ep = np.zeros(n, bool)
    labels = []
    for k in range(int(s.n_episodes)):
        lo = k * seg + lead + 25 * (k == 0)
        hi = (k + 1) * seg - L - 6
        if hi <= lo:
            continue
        a = int(rng.integers(lo, hi))
        side = "low" if rng.random() < 0.5 else "high"
        drift[a - lead:a] = (-1 if side == "low" else 1) * s.lead_in_drift_pct / 100
        in_ep[a:a + L] = True
        labels.append({"kind": f"absorption_at_{side}", "start": dates[a], "end": dates[a + L - 1]})
    sig = s.daily_vol_pct / 100 * np.where(in_ep, s.episode_range_mult, 1.0)
    ret = drift + rng.normal(0, 1, n) * sig
    gap = rng.normal(0, s.gap_pct / 100, n) * np.where(in_ep, s.episode_range_mult, 1.0)
    close = np.empty(n)
    opn = np.empty(n)
    prev = float(s.start_price)
    for i in range(n):
        opn[i] = prev * (1 + gap[i])
        close[i] = opn[i] * (1 + ret[i])
        prev = close[i]
    wick = np.abs(rng.normal(0, 0.5 * sig, (2, n)))
    high = np.maximum(opn, close) * (1 + wick[0])
    low = np.minimum(opn, close) * (1 - wick[1])
    vol = s.base_volume * np.exp(rng.normal(0, s.volume_logsd, n)) * np.where(in_ep, s.episode_vol_mult, 1.0)
    dp = np.clip(rng.normal(s.base_deliv_pct, s.deliv_sd, n) + np.where(in_ep, s.episode_deliv_add, 0.0), 5, 95)
    eod = pd.DataFrame({"date": dates, "open": opn.round(2), "high": high.round(2), "low": low.round(2),
                        "close": close.round(2), "volume": vol.round(0), "delivery_pct": dp.round(2)})
    eod["delivery_qty"] = (eod["volume"] * eod["delivery_pct"] / 100).round(0)
    # rounding must not break high/low consistency
    eod["high"] = eod[["high", "open", "close"]].max(axis=1)
    eod["low"] = eod[["low", "open", "close"]].min(axis=1)
    return SyntheticEOD(eod, pd.DataFrame(labels, columns=["kind", "start", "end"]))


def generate_eod_series(cfg: Config, n: int, seed_base: int, n_days: int | None = None) -> list[SyntheticEOD]:
    return [generate_eod(cfg, seed=seed_base + i, n_days=n_days) for i in range(n)]
