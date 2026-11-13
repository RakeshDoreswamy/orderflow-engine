"""Bar-level backtester with costs. One position at a time, fixed quantity, no compounding.

No-lookahead rules (each is unit-tested):
* A signal carries the time it became KNOWN. The entry fills at the OPEN of the first bar that starts at or after that
  time, plus adverse slippage.
* The stop/target are checked from the entry bar onward using that bar's high/low. If both lie inside one bar the
  stop is assumed to have been hit first (``backtest.both_hit_rule``), which can only hurt results.
* A time exit (max hold, or flat before the close) fills at the close of the last allowed bar, minus adverse slippage.
* Costs (costs.py) and slippage (``costs.<product>.slippage_ticks``) are applied to EVERY trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .costs import round_trip_cost

SIGNAL_COLS = ["time", "side", "stop", "target", "setup"]


def _flat_time(day_ts: pd.Timestamp, cfg: Config) -> pd.Timestamp:
    h, m = (int(x) for x in cfg.market.session_close.split(":"))
    return day_ts.normalize() + pd.Timedelta(hours=h, minutes=m) - pd.Timedelta(minutes=int(cfg.backtest.flat_before_close_min))


def _utc_ns(x):
    """datetime64[ns] in UTC for a Series/Timestamp (tz-aware or naive), so comparisons never depend on object dtype."""
    if isinstance(x, pd.Series):
        x = pd.DatetimeIndex(x)
    else:
        x = pd.DatetimeIndex([x])
    x = x.tz_convert("UTC").tz_localize(None) if x.tz is not None else x
    return x.to_numpy("datetime64[ns]")


def run_backtest(bars: pd.DataFrame, signals: pd.DataFrame, cfg: Config, tick_size: float) -> pd.DataFrame:
    """bars: columns bar (start ts), open, high, low, close. signals: SIGNAL_COLS (side +1/-1, absolute stop/target prices).
    Returns one row per trade."""
    b = cfg.backtest
    slip = getattr(cfg.costs, b.product).slippage_ticks * tick_size
    bars = bars.sort_values("bar").reset_index(drop=True)
    starts = _utc_ns(bars["bar"])
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sig = signals.sort_values("time").reset_index(drop=True)
    qty = int(b.qty)
    out, busy_until = [], None
    for s in sig.itertuples():
        i = int(np.searchsorted(starts, _utc_ns(s.time)[0], side="left"))
        if i >= len(bars):
            continue
        t_entry = bars["bar"].iloc[i]
        if busy_until is not None and t_entry <= busy_until:
            continue
        flat = _flat_time(t_entry, cfg)
        if t_entry >= flat:
            continue
        d = int(s.side)
        entry = o[i] + d * slip
        stop, target = float(s.stop), float(s.target)
        if not ((d > 0 and stop < entry < target) or (d < 0 and target < entry < stop)):
            continue  # stop/target on the wrong side of the actual fill: refuse the trade rather than fix it
        last_ok = min(bars["bar"].iloc[i] + pd.Timedelta(minutes=int(b.max_hold_min)), flat)
        exit_px, reason, j = None, None, i
        while j < len(bars) and bars["bar"].iloc[j] < last_ok and bars["bar"].iloc[j].date() == t_entry.date():
            hit_stop = (l[j] <= stop) if d > 0 else (h[j] >= stop)
            hit_tgt = (h[j] >= target) if d > 0 else (l[j] <= target)
            if hit_stop and (not hit_tgt or b.both_hit_rule == "stop"):
                exit_px, reason = stop - d * slip, "stop"
                break
            if hit_tgt:
                exit_px, reason = target - d * slip, "target"
                break
            j += 1
        if exit_px is None:
            j = max(j - 1, i)
            exit_px, reason = c[j] - d * slip, "time"
        t_exit = bars["bar"].iloc[j]  # start of the bar in which the exit happened
        gross = d * (exit_px - entry) * qty
        cost = round_trip_cost(d, entry, exit_px, qty, cfg, b.product)
        out.append({"setup": getattr(s, "setup", ""), "side": d, "signal_time": s.time, "entry_time": t_entry, "entry": entry,
                    "stop": stop, "target": target, "exit_time": t_exit, "exit": exit_px, "reason": reason, "qty": qty,
                    "gross": gross, "cost": cost, "net": gross - cost, "risk": abs(entry - stop) * qty})
        busy_until = bars["bar"].iloc[j]
    return pd.DataFrame(out, columns=["setup", "side", "signal_time", "entry_time", "entry", "stop", "target", "exit_time", "exit",
                                      "reason", "qty", "gross", "cost", "net", "risk"])


def summarize(trades: pd.DataFrame, cfg: Config) -> dict:
    """Plain performance facts with a bootstrap CI on the mean net P&L per trade. Small n is flagged, never hidden."""
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    net = trades["net"].to_numpy(float)
    rng = np.random.default_rng(int(cfg.backtest.bootstrap_seed))
    boots = rng.choice(net, size=(int(cfg.backtest.bootstrap_n), n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    eq = np.cumsum(net)
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"trades": n, "win_rate": float((net > 0).mean()), "gross_total": float(trades["gross"].sum()), "cost_total": float(trades["cost"].sum()),
            "net_total": float(net.sum()), "mean_net_per_trade": float(net.mean()), "mean_net_ci95": (float(lo), float(hi)),
            "mean_gross_per_trade": float(trades["gross"].mean()), "mean_cost_per_trade": float(trades["cost"].mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "max_drawdown": float((np.maximum.accumulate(eq) - eq).max()),
            "exits": trades["reason"].value_counts().to_dict(), "small_sample": bool(n < 30)}
