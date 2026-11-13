"""Session and anchored VWAP with volume-weighted standard-deviation bands (all causal, cumulative)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def vwap_series(ticks: pd.DataFrame, cfg: Config, anchor_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """Per-trade VWAP, sigma and bands. Session VWAP restarts each trading day; with ``anchor_ts`` it
    starts at that timestamp (rows before it are NaN). sigma = sqrt(sum(q*p^2)/sum(q) - vwap^2)."""
    t = ticks[["ts", "price", "qty"]].copy()
    key = t["ts"].dt.date
    t["pq"], t["p2q"] = t["price"] * t["qty"], t["price"] ** 2 * t["qty"]
    mask = np.ones(len(t), bool) if anchor_ts is None else (t["ts"] >= anchor_ts).to_numpy()
    if anchor_ts is not None:
        key = pd.Series(0, index=t.index)
    for c in ("qty", "pq", "p2q"):
        t[c + "_c"] = t[c].where(mask, 0).groupby(key).cumsum()
    t["vwap"] = t["pq_c"] / t["qty_c"]
    var = (t["p2q_c"] / t["qty_c"] - t["vwap"] ** 2).clip(lower=0)
    t["sd"] = np.sqrt(var)
    for k in cfg.vwap.band_mults:
        t[f"upper_{k:g}"], t[f"lower_{k:g}"] = t["vwap"] + k * t["sd"], t["vwap"] - k * t["sd"]
    cols = ["ts", "vwap", "sd"] + [c for k in cfg.vwap.band_mults for c in (f"upper_{k:g}", f"lower_{k:g}")]
    return t[cols]  # before the anchor cumulative volume is 0, so vwap/sd/bands are NaN
