"""Does the bridge's verdict actually separate good ICT entries from bad ones? Compare realised R by verdict.

Feed it entries that already have an outcome (R multiple after costs, from the ICT engine's backtest or journal) and a
``verdict`` from ``ict_bridge``. It reports n and mean R per verdict with bootstrap intervals, plus the confirm-minus-reject
difference with an interval. Small groups are flagged. With no edge the difference interval should straddle zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _boot(x: np.ndarray, n: int, rng) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare_outcomes(df: pd.DataFrame, cfg: Config, r_col: str = "R", verdict_col: str = "verdict") -> dict:
    n = int(cfg.backtest.bootstrap_n)
    rng = np.random.default_rng(int(cfg.backtest.bootstrap_seed))
    rows = []
    for v in ("confirm", "neutral", "reject", "no_data"):
        x = df.loc[df[verdict_col] == v, r_col].dropna().to_numpy(float)
        lo, hi = _boot(x, n, rng)
        rows.append({"verdict": v, "n": len(x), "mean_R": float(x.mean()) if len(x) else float("nan"), "ci95": (lo, hi), "small_sample": len(x) < 30})
    c = df.loc[df[verdict_col] == "confirm", r_col].dropna().to_numpy(float)
    r = df.loc[df[verdict_col] == "reject", r_col].dropna().to_numpy(float)
    diff = None
    if len(c) and len(r):
        bc = rng.choice(c, size=(n, len(c))).mean(axis=1)
        br = rng.choice(r, size=(n, len(r))).mean(axis=1)
        d = bc - br
        diff = {"confirm_minus_reject": float(c.mean() - r.mean()), "ci95": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
    return {"by_verdict": pd.DataFrame(rows), "difference": diff}
