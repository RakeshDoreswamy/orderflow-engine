"""Trade-side classification when the exchange gives no aggressor flag.

All three rules are CAUSAL: the label of trade i uses only trades/quotes up to and including i.
Output convention: +1 buyer-initiated, -1 seller-initiated, 0 unclassified.
Anything derived from these labels is an ESTIMATE and is named ``*_est`` downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def tick_rule(price: np.ndarray | pd.Series) -> np.ndarray:
    """Tick rule: an up-tick is a buy, a down-tick is a sell, a zero-tick copies the last
    non-zero direction. The very first trade is unclassified (0)."""
    p = np.asarray(price, dtype=float)
    d = np.zeros(len(p), dtype=np.int8)
    if len(p) > 1:
        d[1:] = np.sign(p[1:] - p[:-1]).astype(np.int8)
    idx = np.maximum.accumulate(np.where(d != 0, np.arange(len(p)), 0))
    return d[idx]


def quote_rule(price, bid, ask, eps: float = 0.0) -> np.ndarray:
    """Quote rule: at/above the ask is a buy, at/below the bid is a sell; strictly inside the
    spread compare to the midpoint; exactly at the midpoint (or missing quote) is 0."""
    p, b, a = (np.asarray(x, dtype=float) for x in (price, bid, ask))
    mid = (b + a) / 2.0
    out = np.zeros(len(p), dtype=np.int8)
    ok = np.isfinite(b) & np.isfinite(a)
    out[ok & (p > mid + eps)] = 1
    out[ok & (p < mid - eps)] = -1
    out[ok & (p >= a - eps)] = 1
    out[ok & (p <= b + eps)] = -1
    return out


def lee_ready(price, bid, ask, eps: float = 0.0) -> np.ndarray:
    """Lee-Ready: quote rule first; trades at the midpoint (or without a quote) fall back to the tick rule."""
    q = quote_rule(price, bid, ask, eps)
    return np.where(q != 0, q, tick_rule(price)).astype(np.int8)


def classify_ticks(ticks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a copy of ``ticks`` with side_tick_est / side_quote_est / side_lr_est columns."""
    eps = cfg.classify.quote_price_eps_ticks * cfg.instrument.default_tick_size
    out = ticks.copy()
    out["side_tick_est"] = tick_rule(out["price"])
    out["side_quote_est"] = quote_rule(out["price"], out["bid"], out["ask"], eps)
    out["side_lr_est"] = lee_ready(out["price"], out["bid"], out["ask"], eps)
    return out


def accuracy_report(df: pd.DataFrame, true_col: str = "side_true",
                    methods: dict[str, str] | None = None) -> pd.DataFrame:
    """Compare estimated sides with known truth.

    coverage          share of trades the rule labelled (non-zero)
    acc_classified    accuracy among labelled trades
    acc_all           accuracy counting unlabelled trades as wrong
    acc_all_vol       same, volume-weighted (what matters for delta)
    error_rate_all    1 - acc_all  (feed this to the stress test)
    """
    methods = methods or {"tick_rule": "side_tick_est", "quote_rule": "side_quote_est",
                          "lee_ready": "side_lr_est"}
    truth = df[true_col].to_numpy()
    q = df["qty"].to_numpy(dtype=float)
    rows = []
    for name, col in methods.items():
        est = df[col].to_numpy()
        lab = est != 0
        hit = est == truth
        rows.append({"method": name, "n": len(df), "coverage": lab.mean(),
                     "acc_classified": hit[lab].mean() if lab.any() else np.nan,
                     "acc_all": hit.mean(), "acc_all_vol": (q * hit).sum() / q.sum(),
                     "error_rate_all": 1 - hit.mean()})
    return pd.DataFrame(rows)


def flip_sides(side: np.ndarray, error_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Stress test helper: flip each label with probability ``error_rate`` (seeded rng)."""
    s = np.asarray(side).copy()
    m = rng.random(len(s)) < error_rate
    s[m] = -s[m]
    return s
