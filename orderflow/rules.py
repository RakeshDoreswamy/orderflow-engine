"""Harnesses for two popular rules of thumb. They MEASURE the rule on whatever data you pass in; they do not assume it.

80% rule (Dalton): if the market opens outside the prior day's value area, re-enters it and is ACCEPTED, it tends to
traverse the value area to the other side ~80% of the time. Exact definition used here (config ``rules.*``):
  open outside   : first trade >= VAH_top + k ticks (above) or <= VAL - k ticks (below), prior-day value area from the
                   volume profile (``value_area_pct`` of volume, pairwise expansion from the POC)
  accepted       : ``accept_periods`` consecutive ``period_freq`` bar CLOSES inside [VAL, VAH_top]; known at the end of the
                   last of those bars
  success        : after acceptance, price touches the opposite edge (VAL if it opened above, VAH_top if below) before the close
Other authors define acceptance with TPO periods or time spent; results depend on the definition, so keep it fixed
before looking at data.

Gap fill: a gap is open vs the previous close; it is filled if the same day's range touches the previous close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data.bars import time_bars
from .evaluate import wilson
from .profile import value_area, volume_profile


def rule80_day(prev_ticks: pd.DataFrame, ticks: pd.DataFrame, cfg: Config, tick_size: float) -> dict:
    r = cfg.rules
    prof = volume_profile(prev_ticks, tick_size, int(r.profile_row_ticks))
    va = value_area(prof, float(r.value_area_pct))
    val, top = float(va["val"]), float(va["vah"]) + int(r.profile_row_ticks) * tick_size
    k = r.open_outside_ticks * tick_size
    op = float(ticks["price"].iloc[0])
    row = {"date": ticks["ts"].iloc[0].date(), "open": op, "val": val, "vah_top": top, "side": "inside", "accepted": False, "success": np.nan}
    if op >= top + k:
        row["side"] = "above"
    elif op <= val - k:
        row["side"] = "below"
    else:
        return row
    bars = time_bars(ticks, cfg, freq=r.period_freq)
    bars = bars.reset_index() if "bar" not in bars.columns else bars
    n, step = int(r.accept_periods), pd.Timedelta(r.period_freq)
    inside = ((bars["close"] >= val) & (bars["close"] <= top)).to_numpy()
    for i in range(n - 1, len(bars)):
        if inside[i - n + 1:i + 1].all():
            t_acc = bars["bar"].iloc[i] + step
            after = ticks[ticks["ts"] >= t_acc]
            row.update(accepted=True, t_accept=t_acc)
            if len(after):
                row["success"] = bool((after["price"].min() <= val) if row["side"] == "above" else (after["price"].max() >= top))
            break
    return row


def rule80(days: list[tuple[pd.DataFrame, float]], cfg: Config) -> tuple[pd.DataFrame, dict]:
    """days = [(ticks, tick_size), ...] in date order. Returns per-day rows and a summary with a Wilson interval."""
    rows = [rule80_day(days[i - 1][0], days[i][0], cfg, days[i][1]) for i in range(1, len(days))]
    t = pd.DataFrame(rows)
    acc = t[(t["accepted"]) & t["success"].notna()] if len(t) else t
    k, n = int(acc["success"].sum()) if len(acc) else 0, len(acc)
    lo, hi = wilson(k, n)
    claimed = float(cfg.rules.claimed_probability)
    return t, {"days": len(t), "opened_outside": int((t["side"] != "inside").sum()) if len(t) else 0, "accepted": n, "successes": k,
               "rate": k / n if n else float("nan"), "ci95": (lo, hi), "claimed": claimed,
               "claimed_inside_ci": bool(lo <= claimed <= hi) if n else None, "small_sample": bool(n < 30)}


def gap_fill(eod: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day gap table and a bucketed summary (by |gap| % and direction) with Wilson intervals."""
    d = eod.reset_index(drop=True).copy()
    pc = d["close"].shift(1)
    d["gap_pct"] = (d["open"] - pc) / pc * 100
    d["dir"] = np.where(d["gap_pct"] > 0, "up", np.where(d["gap_pct"] < 0, "down", "none"))
    d["filled"] = np.where(d["dir"] == "up", d["low"] <= pc, np.where(d["dir"] == "down", d["high"] >= pc, np.nan))
    d = d[d["dir"] != "none"].dropna(subset=["gap_pct"])
    edges = list(cfg.rules.gap_buckets_pct) + [np.inf]
    d["bucket"] = pd.cut(d["gap_pct"].abs(), edges, right=False)
    out = []
    for (b, di), g in d.groupby(["bucket", "dir"], observed=True):
        k, n = int(g["filled"].sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"abs_gap_pct": str(b), "direction": di, "gaps": n, "filled": k, "fill_rate": k / n, "ci95": f"{lo:.2f}-{hi:.2f}", "small_sample": n < 30})
    return d, pd.DataFrame(out)


def toy_gapped_days(cfg: Config, n_days: int, seed: int = 5) -> list[tuple[pd.DataFrame, float]]:
    """Synthetic days with a random overnight gap added (the base generator opens each day at the previous close).
    TOY: it only lets the harness run; the gap process is a plain Gaussian."""
    from .data.synthetic import generate_days
    days = generate_days(cfg, n_days)
    rng = np.random.default_rng(seed)
    out, shift = [], 0.0
    for d in days:
        tk = d.tick_size
        shift += round(rng.normal(0, cfg.rules.toy_gap_pct / 100 * cfg.synthetic.start_price) / tk) * tk
        t = d.ticks.copy()
        t["price"] = (t["price"] + shift).round(6)
        out.append((t, tk))
    return out
