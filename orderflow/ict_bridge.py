"""Bridge to ict-swing-engine: ask whether order-flow evidence CONFIRMS, REJECTS or is NEUTRAL about one of its entries.

Input: rows of the ICT scanner's ``scan.csv`` (symbol, entry, stop, t1, t2, sweep_time, mss_time, ...; the scanner is
long-side, 45-minute OTE, shorts mirror it; a ``side`` column of long/short is honoured if present).

Two evidence sources, same output schema:
  verdict_eod       daily bars (+ NSE delivery % when you have it), via swingflow. Works with what the scanner already stores.
  verdict_intraday  recorded ticks (Phase 5) or any tick frame, via the event detectors.

Rules of the bridge
* NO LOOKAHEAD: everything is truncated to ``as_of`` inside the function, so a verdict never uses data after the day/time it is
  asked for (tested by running with and without future data).
* Missing or short data gives ``no_data``; the bridge never guesses.
* The verdict is a transparent score: each piece of evidence is +1 / -1 and listed in ``evidence``. Thresholds are in
  config.yaml (bridge.*). Whether confirmed entries actually do better is an empirical question: use
  ``orderflow.bridge_eval.compare_outcomes`` on real outcomes before trusting any verdict. Nothing here has been shown to improve
  ICT entries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import events as E
from . import swingflow as S
from .config import Config

OUT_COLS = ["verdict", "score", "evidence", "source", "as_of"]
REQUIRED = ["symbol", "entry", "stop"]


def read_scan_csv(path) -> pd.DataFrame:
    """Read a scanner ``scan.csv`` and normalise it. ``side`` defaults to long (the scanner's side)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"scan file lacks columns {miss}; has {list(df.columns)}")
    if "side" not in df.columns:
        df["side"] = "long"
    df["side"] = df["side"].astype(str).str.lower().map({"long": "long", "buy": "long", "short": "short", "sell": "short"})
    if df["side"].isna().any():
        raise ValueError("side must be long/short")
    for c in ("sweep_time", "mss_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _verdict(score: int, ev: list[str], cfg: Config, source: str, as_of) -> dict:
    b = cfg.bridge
    v = "confirm" if score >= b.confirm_min_score else "reject" if score <= -b.reject_min_score else "neutral"
    return {"verdict": v, "score": int(score), "evidence": "; ".join(ev) if ev else "no flow evidence either way", "source": source, "as_of": as_of}


def _no_data(why: str, source: str, as_of) -> dict:
    return {"verdict": "no_data", "score": 0, "evidence": why, "source": source, "as_of": as_of}


def verdict_eod(entry: dict, eod: pd.DataFrame, cfg: Config, as_of=None) -> dict:
    """Evidence from daily bars up to and including ``as_of`` (default: the last bar). ``eod`` follows the EOD contract
    (date, open, high, low, close, volume, optional delivery_pct)."""
    b = cfg.bridge
    d = eod.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize() if getattr(d["date"].dt, "tz", None) is not None else pd.to_datetime(d["date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).tz_localize(None).normalize() if as_of is not None else d["date"].max()
    d = d[d["date"] <= as_of].sort_values("date").reset_index(drop=True)
    if len(d) < int(b.min_history_days):
        return _no_data(f"only {len(d)} daily bars up to {as_of.date()} (need {b.min_history_days})", "eod", as_of)
    long = entry.get("side", "long") == "long"
    f = S.flow_features(d, cfg)
    last = f["date"].iloc[-1]
    win_start = last - pd.Timedelta(days=int(b.lookback_days))
    score, ev = 0, []
    has_deliv = f["deliv_z"].notna().any()

    eff = S.effort_events(f, cfg)
    eff = eff[(eff["time"] > win_start)]
    want, against = ("low", "high") if long else ("high", "low")
    if (eff["side"] == want).any():
        score += 1
        ev.append(f"+1 effort without result near a prior {want} ({int((eff['side'] == want).sum())} day(s))")
    if (eff["side"] == against).any():
        score -= 1
        ev.append(f"-1 effort without result near a prior {against} ({int((eff['side'] == against).sum())} day(s))")

    sw = S.swing_flow(f, cfg)
    sweep = entry.get("sweep_time")
    if sweep is not None and pd.notna(sweep) and len(sw):
        sd = pd.Timestamp(sweep).tz_localize(None).normalize() if getattr(pd.Timestamp(sweep), "tzinfo", None) else pd.Timestamp(sweep).normalize()
        kind = "low" if long else "high"
        s = sw[(sw["kind"] == kind) & (sw["known_on"] <= as_of) & ((sw["swing_date"] - sd).abs() <= pd.Timedelta(days=int(b.swing_match_days)))]
        if len(s):
            r = s.iloc[(s["swing_date"] - sd).abs().argmin()]
            if r["read"] == ("held_on_heavy_buying" if long else "stalled_on_heavy_selling"):
                score += 1
                ev.append(f"+1 daily swing {kind} of {r['swing_date'].date()} formed on heavy volume with flow in the entry's favour ({r['read'].replace('_', ' ')})")
            elif r["read"] == "heavy_but_against":
                score -= 1
                ev.append(f"-1 daily swing {kind} of {r['swing_date'].date()} formed on heavy volume with flow against the entry")

    tail = f.tail(int(b.flow_days))
    vol = float(tail["volume"].sum())
    if vol > 0:
        ratio = float(tail["delta_est"].sum(skipna=True)) / vol
        if abs(ratio) >= b.flow_ratio_min:
            sup = (ratio > 0) == long
            score += 1 if sup else -1
            ev.append(f"{'+1' if sup else '-1'} net estimated delta over {len(tail)} days = {ratio:+.2f} of volume ({'supports' if sup else 'contradicts'})")
    if not has_deliv:
        ev.append("(no delivery % in this data: volume-only evidence, lower precision on toy data)")
    return _verdict(score, ev, cfg, "eod", as_of)


def verdict_intraday(entry: dict, ticks: pd.DataFrame, cfg: Config, tick_size: float, as_of=None, side_col: str = "side_true") -> dict:
    """Evidence from tick events between (sweep_time - pad) and ``as_of``. ``side_col`` should be an ESTIMATED side column
    (e.g. ``side_quote_est``) on live data; ``side_true`` exists only on synthetic data."""
    as_of = pd.Timestamp(as_of) if as_of is not None else ticks["ts"].max()
    t = ticks[ticks["ts"] <= as_of]
    sweep = entry.get("sweep_time")
    if t.empty or sweep is None or pd.isna(sweep):
        return _no_data("no ticks up to as_of or no sweep_time", "intraday", as_of)
    sweep = pd.Timestamp(sweep)
    if sweep.tzinfo is None:
        sweep = sweep.tz_localize(t["ts"].dt.tz)
    start = sweep - pd.Timedelta(minutes=int(cfg.bridge.intraday_pad_min))
    t = t[t["ts"].dt.date == as_of.date()]
    if len(t) < 500:
        return _no_data(f"only {len(t)} ticks that day up to as_of", "intraday", as_of)
    long = entry.get("side", "long") == "long"
    hits = {"absorption": E.absorption(t, cfg, tick_size, side_col), "exhaustion": E.exhaustion(t, cfg, tick_size, side_col),
            "trapped": E.trapped(t, cfg, tick_size, side_col)}
    good = {"absorption": "bid" if long else "ask", "exhaustion": "down" if long else "up", "trapped": "sellers_trapped" if long else "buyers_trapped"}
    bad = {"absorption": "ask" if long else "bid", "exhaustion": "up" if long else "down", "trapped": "buyers_trapped" if long else "sellers_trapped"}
    score, ev = 0, []
    for name, h in hits.items():
        h = h[(h["time"] >= start) & (h["time"] <= as_of)]
        if (h["side"] == good[name]).any():
            score += 1
            ev.append(f"+1 {name} ({good[name]}) at {h[h['side'] == good[name]]['time'].iloc[0]:%H:%M}")
        if (h["side"] == bad[name]).any():
            score -= 1
            ev.append(f"-1 {name} ({bad[name]}) at {h[h['side'] == bad[name]]['time'].iloc[0]:%H:%M}")
    return _verdict(score, ev, cfg, "intraday", as_of)


def bridge_scan(entries: pd.DataFrame, eod_by_symbol: dict[str, pd.DataFrame], cfg: Config, as_of=None) -> pd.DataFrame:
    """Add verdict columns to every scanner entry using daily data. A symbol without data gets ``no_data``."""
    rows = []
    for e in entries.to_dict("records"):
        d = eod_by_symbol.get(e["symbol"])
        rows.append(_no_data("no daily data for symbol", "eod", as_of) if d is None else verdict_eod(e, d, cfg, as_of))
    return pd.concat([entries.reset_index(drop=True), pd.DataFrame(rows, columns=OUT_COLS)], axis=1)
