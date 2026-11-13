"""EOD order-flow PROXIES (daily bars + optional NSE delivery %).

Daily data has no aggressor side, so everything here is an estimate and every estimated column ends in ``_est``.
All baselines use only PREVIOUS days (shift(1)), so a row never depends on a later row; a swing becomes known at
``confirm_idx = idx + swing_right`` and flow summaries use bars up to and including that day only.

Hypothesis, not a result: heavy volume and heavy delivery with a small range, near a prior low/high, may mean the
move was absorbed. The synthetic score only shows the rule finds what it describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .delta import confirmed_swings, ohlcv_delta_est


def _z(s: pd.Series, n: int, warm: int) -> pd.Series:
    base = s.shift(1)
    sd = base.rolling(n, min_periods=warm).std().replace(0, np.nan)
    return (s - base.rolling(n, min_periods=warm).mean()) / sd


def flow_features(eod: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    c = cfg.eod
    d = eod.reset_index(drop=True).copy()
    d["vol_z"] = _z(np.log(d["volume"].where(d["volume"] > 0)), int(c.vol_lookback), int(c.vol_warmup))
    d["deliv_z"] = _z(d["delivery_pct"], int(c.deliv_lookback), int(c.deliv_warmup)) if "delivery_pct" in d else np.nan
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr_prior"] = tr.shift(1).rolling(int(c.atr_lookback), min_periods=int(c.atr_lookback)).mean()
    d["range_atr"] = (d["high"] - d["low"]) / d["atr_prior"]
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["clv"] = (2 * d["close"] - d["high"] - d["low"]) / rng
    d["delta_est"] = ohlcv_delta_est(d)
    n = int(c.pos_lookback)
    lo, hi = d["low"].shift(1).rolling(n, min_periods=n).min(), d["high"].shift(1).rolling(n, min_periods=n).max()
    d["pos_prior"] = (d["close"] - lo) / (hi - lo).replace(0, np.nan)
    return d


def effort_events(feat: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Days with heavy volume (and heavy delivery % when present) but a small range, placed by where the close
    sits in the prior range: ``absorption_at_low`` (pos <= pos_low_max) or ``absorption_at_high`` (pos >= pos_high_min)."""
    c = cfg.eod
    ok = (feat["vol_z"] >= c.effort_vol_z) & (feat["range_atr"] <= c.effort_range_atr)
    if feat["deliv_z"].notna().any():
        ok &= feat["deliv_z"] >= c.effort_deliv_z
    low, high = ok & (feat["pos_prior"] <= c.pos_low_max), ok & (feat["pos_prior"] >= c.pos_high_min)
    rows = []
    for m, side in ((low, "low"), (high, "high")):
        t = feat[m]
        rows.append(pd.DataFrame({"time": t["date"], "price": t["close"], "side": side, "strength": t["vol_z"],
                                  "rule": f"vol_z>={c.effort_vol_z}, range<={c.effort_range_atr}xATR, close near prior {side}"}))
    return pd.concat(rows).sort_values("time").reset_index(drop=True)


def swing_flow(feat: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """For each confirmed swing, summarise the flow over bars [idx-1 .. confirm_idx] (all known at confirmation).
    read: 'held_on_heavy_buying' (low), 'stalled_on_heavy_selling' (high), 'heavy_but_against' or 'unremarkable'."""
    c = cfg.eod
    sw = confirmed_swings(feat["high"].to_numpy(), feat["low"].to_numpy(), int(c.swing_left), int(c.swing_right))
    rows = []
    for r in sw.itertuples():
        a, b = max(int(r.idx) - 1, 0), int(r.confirm_idx)
        w = feat.iloc[a:b + 1]
        vz, dz = feat["vol_z"].iloc[r.idx], feat["deliv_z"].iloc[r.idx]
        heavy = bool(vz >= c.swing_heavy_vol_z and (np.isnan(dz) or dz >= c.swing_heavy_deliv_z))
        dsum = float(w["delta_est"].sum(skipna=True))
        if not heavy:
            read = "unremarkable"
        elif r.kind == "low":
            read = "held_on_heavy_buying" if dsum > 0 else "heavy_but_against"
        else:
            read = "stalled_on_heavy_selling" if dsum < 0 else "heavy_but_against"
        rows.append({"idx": int(r.idx), "confirm_idx": b, "swing_date": feat["date"].iloc[r.idx], "known_on": feat["date"].iloc[b],
                     "kind": r.kind, "price": float(r.price), "vol_z": float(vz), "deliv_z": float(dz),
                     "delta_sum_est": dsum, "heavy": heavy, "read": read})
    return pd.DataFrame(rows, columns=["idx", "confirm_idx", "swing_date", "known_on", "kind", "price", "vol_z", "deliv_z",
                                       "delta_sum_est", "heavy", "read"])
