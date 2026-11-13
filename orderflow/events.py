"""Event detectors. Each returns a table with ``time`` (the first moment the rule could be known),
``price``, ``side``, ``strength`` and ``rule`` (the rule that fired, in words). All are causal: the row for time T
uses only data before T (checked by prefix tests). Thresholds live in config.yaml (events.*).

Raw hits can repeat while a condition persists; ``merge_hits`` turns them into episodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .delta import delta_bars


def merge_hits(hits: pd.DataFrame, gap_s: float, by: list[str] | None = None) -> pd.DataFrame:
    """Merge hits closer than ``gap_s`` seconds (per ``by`` group) into episodes: start, last, n, max strength."""
    if hits.empty:
        return pd.DataFrame(columns=["start", "last", "n", "strength"] + (by or []))
    out = []
    for key, g in (hits.groupby(by) if by else [(None, hits)]):
        g = g.sort_values("time")
        new = g["time"].diff().dt.total_seconds().fillna(1e18) > gap_s
        for _, e in g.groupby(new.cumsum()):
            row = {"start": e["time"].iloc[0], "last": e["time"].iloc[-1], "n": len(e), "strength": e["strength"].max(),
                   "price": e["price"].iloc[0]}
            if by:
                row.update(dict(zip(by, key if isinstance(key, tuple) else (key,))))
            out.append(row)
    return pd.DataFrame(out).sort_values("start").reset_index(drop=True)


def _day_open(ts: pd.Series, cfg: Config) -> pd.Series:
    h, m = (int(x) for x in cfg.market.session_open.split(":"))
    return ts.dt.normalize() + pd.Timedelta(hours=h, minutes=m)


def large_prints(ticks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rule: a trade whose size is >= large_print_mult x the median size of the previous large_print_lookback trades."""
    c = cfg.events
    med = ticks["qty"].shift(1).rolling(c.large_print_lookback, min_periods=c.large_print_lookback // 4).median()
    m = ticks["qty"] >= c.large_print_mult * med
    t = ticks[m]
    return pd.DataFrame({"time": t["ts"], "price": t["price"], "side": np.where(t["side_true"] > 0, "buy", "sell") if "side_true" in t else "",
                         "strength": (t["qty"] / med[m]).to_numpy(), "rule": f"qty >= {c.large_print_mult}x median of last {c.large_print_lookback} trades"}).reset_index(drop=True)


def absorption(ticks: pd.DataFrame, cfg: Config, tick_size: float, side_col: str = "side_true") -> pd.DataFrame:
    """Rule: in the trailing absorb_window_s, volume >= absorb_min_volume_mult x the usual window volume (expanding
    median of earlier windows), one-sided (|delta|/volume >= absorb_min_delta_frac) and price range <= absorb_max_range_ticks.
    side 'bid' = aggressive SELLING was absorbed; 'ask' = aggressive BUYING was absorbed. Evaluated every absorb_step_s."""
    c = cfg.events
    nb = int(round(c.absorb_window_s / c.absorb_step_s))
    out = []
    for _, d in ticks.groupby(ticks["ts"].dt.date):
        d0 = _day_open(d["ts"], cfg).iloc[0]
        sec = (d["ts"] - d0).dt.total_seconds().to_numpy()
        b = (sec // c.absorb_step_s).astype(int)
        n = b.max() + 1
        vol = np.bincount(b, weights=d["qty"].to_numpy(float), minlength=n)
        dlt = np.bincount(b, weights=(d[side_col].to_numpy() * d["qty"].to_numpy()).astype(float), minlength=n)
        hi = pd.Series(d["price"].to_numpy()).groupby(b).max().reindex(range(n))
        lo = pd.Series(d["price"].to_numpy()).groupby(b).min().reindex(range(n))
        last = pd.Series(d["price"].to_numpy()).groupby(b).last().reindex(range(n)).ffill()
        V = pd.Series(vol).rolling(nb, min_periods=nb).sum()
        D = pd.Series(dlt).rolling(nb, min_periods=nb).sum()
        rng = (hi.rolling(nb, min_periods=1).max() - lo.rolling(nb, min_periods=1).min()) / tick_size
        base = V.shift(1).expanding(min_periods=int(c.absorb_warmup_windows)).median()
        ok = (V >= c.absorb_min_volume_mult * base) & (D.abs() / V >= c.absorb_min_delta_frac) & (rng <= c.absorb_max_range_ticks)
        for i in np.flatnonzero(ok.to_numpy()):
            out.append({"time": d0 + pd.Timedelta(seconds=(i + 1) * c.absorb_step_s), "price": float(last.iloc[i]),
                        "side": "bid" if D.iloc[i] < 0 else "ask", "strength": float(V.iloc[i] / base.iloc[i]),
                        "rule": f"vol>={c.absorb_min_volume_mult}x usual, |delta|>={c.absorb_min_delta_frac:.0%}, range<={c.absorb_max_range_ticks} ticks"})
    return pd.DataFrame(out, columns=["time", "price", "side", "strength", "rule"])


def exhaustion(ticks: pd.DataFrame, cfg: Config, tick_size: float, side_col: str = "side_true") -> pd.DataFrame:
    """Rule (per completed bar): the bar's extreme is within exhaust_tol_ticks of the highest high (lowest low) of the
    last exhaust_lookback_bars; there WAS strong one-sided aggression in that lookback (max ask_vol / bid_vol >=
    exhaust_peak_mult x the usual bar volume); and this bar's total volume <= exhaust_dry_frac x the usual bar volume.
    side 'up' = buying exhausted at a high; 'down' = selling exhausted at a low. Usual = expanding median of earlier bars."""
    c = cfg.events
    K, tol = int(c.exhaust_lookback_bars), c.exhaust_tol_ticks * tick_size
    bars = delta_bars(ticks, cfg, freq=c.exhaust_bar_freq, side_col=side_col)
    step = pd.Timedelta(c.exhaust_bar_freq)
    out = []
    for _, d in bars.groupby(bars["bar"].dt.date):
        d = d.reset_index(drop=True)
        usual = d["volume"].shift(1).expanding(min_periods=int(c.exhaust_warmup_bars)).median()
        for i in range(K, len(d)):
            u = usual.iloc[i]
            if not np.isfinite(u) or d["volume"].iloc[i] > c.exhaust_dry_frac * u:
                continue
            prev = d.iloc[i - K:i]
            for side, ext_ok, peak in (("up", d["high"].iloc[i] >= max(prev["high"].max(), d["high"].iloc[i]) - tol and d["high"].iloc[i] >= prev["high"].max() - tol,
                                        prev["ask_vol"].max()),
                                       ("down", d["low"].iloc[i] <= prev["low"].min() + tol, prev["bid_vol"].max())):
                if ext_ok and peak >= c.exhaust_peak_mult * u:
                    out.append({"time": d["bar"].iloc[i] + step, "price": float(d["close"].iloc[i]), "side": side,
                                "strength": float(peak / max(d["volume"].iloc[i], 1)),
                                "rule": f"dry bar at the {'high' if side == 'up' else 'low'} after strong one-sided aggression"})
    return pd.DataFrame(out, columns=["time", "price", "side", "strength", "rule"])


def trapped(ticks: pd.DataFrame, cfg: Config, tick_size: float, side_col: str = "side_true") -> pd.DataFrame:
    """Rule (evaluated every trap_step_s on the trailing trap_window_min): the window's low (high) is at least
    trap_min_age_min old; price has since recovered (dropped) by >= trap_recovery_ticks; and aggressive SELLING (within trap_cluster_s of the extreme's time)
    (BUYING) within trap_near_ticks of that extreme was >= trap_min_vol_mult x the usual minute volume and >= trap_min_share
    of the window volume. 'sellers_trapped' = heavy selling at a low then a rally; 'buyers_trapped' = the mirror."""
    c = cfg.events
    W = c.trap_window_min * 60.0
    out = []
    for _, d in ticks.groupby(ticks["ts"].dt.date):
        d0 = _day_open(d["ts"], cfg).iloc[0]
        sec = (d["ts"] - d0).dt.total_seconds().to_numpy()
        px, q, sd = d["price"].to_numpy(), d["qty"].to_numpy(float), d[side_col].to_numpy()
        minute_vol = np.bincount((sec // 60).astype(int), weights=q)
        for g in np.arange(c.trap_warmup_min * 60.0, sec.max() + c.trap_step_s, c.trap_step_s):
            i0, i1 = np.searchsorted(sec, g - W), np.searchsorted(sec, g)
            if i1 - i0 < 20:
                continue
            usual = np.median(minute_vol[:max(int(g // 60), 1)])
            wp, wq, ws, wt = px[i0:i1], q[i0:i1], sd[i0:i1], sec[i0:i1]
            tot = wq.sum()
            near = c.trap_near_ticks * tick_size
            lo, hi, last = wp.min(), wp.max(), wp[-1]
            t_lo, t_hi = wt[np.argmax(wp == lo)], wt[np.argmax(wp == hi)]
            cl = c.trap_cluster_s
            sell_near = wq[(ws < 0) & (wp <= lo + near) & (np.abs(wt - t_lo) <= cl)].sum()
            buy_near = wq[(ws > 0) & (wp >= hi - near) & (np.abs(wt - t_hi) <= cl)].sum()
            if (g - t_lo >= c.trap_min_age_min * 60 and last - lo >= c.trap_recovery_ticks * tick_size
                    and sell_near >= c.trap_min_vol_mult * usual and sell_near >= c.trap_min_share * tot):
                out.append({"time": d0 + pd.Timedelta(seconds=float(g)), "price": float(last), "side": "sellers_trapped",
                            "strength": float(sell_near / usual), "rule": "heavy selling at the low, then recovery"})
            if (g - t_hi >= c.trap_min_age_min * 60 and hi - last >= c.trap_recovery_ticks * tick_size
                    and buy_near >= c.trap_min_vol_mult * usual and buy_near >= c.trap_min_share * tot):
                out.append({"time": d0 + pd.Timedelta(seconds=float(g)), "price": float(last), "side": "buyers_trapped",
                            "strength": float(buy_near / usual), "rule": "heavy buying at the high, then drop"})
    return pd.DataFrame(out, columns=["time", "price", "side", "strength", "rule"])


def delta_flip_events(ticks: pd.DataFrame, cfg: Config, side_col: str = "side_true") -> pd.DataFrame:
    """Rule (per completed bar): delta % changes sign vs the previous bar, both |delta %| >= flip_min_abs_pct, and
    this bar's volume >= flip_min_vol_frac x the usual bar volume (expanding median of earlier bars)."""
    c = cfg.events
    bars = delta_bars(ticks, cfg, freq=c.flip_bar_freq, side_col=side_col)
    step = pd.Timedelta(c.flip_bar_freq)
    out = []
    for _, d in bars.groupby(bars["bar"].dt.date):
        d = d.reset_index(drop=True)
        usual = d["volume"].shift(1).expanding(min_periods=int(c.flip_warmup_bars)).median()
        pct = d["delta_pct"].to_numpy()
        for i in range(1, len(d)):
            if (np.isfinite(usual.iloc[i]) and np.sign(pct[i]) != np.sign(pct[i - 1]) and min(abs(pct[i]), abs(pct[i - 1])) >= c.flip_min_abs_pct
                    and d["volume"].iloc[i] >= c.flip_min_vol_frac * usual.iloc[i]):
                out.append({"time": d["bar"].iloc[i] + step, "price": float(d["close"].iloc[i]),
                            "side": "to_buy" if pct[i] > 0 else "to_sell", "strength": float(min(abs(pct[i]), abs(pct[i - 1]))),
                            "rule": "delta % changed sign with both bars one-sided"})
    return pd.DataFrame(out, columns=["time", "price", "side", "strength", "rule"])
