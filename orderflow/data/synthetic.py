"""Seeded synthetic NSE-style tick stream with labelled events.

Why it exists: real aggressor-side data is scarce, so we generate a day where the TRUE
aggressor side of every trade is known. That lets us (a) teach, (b) test detectors, and
(c) measure how wrong the tick rule / quote rule / Lee-Ready are.

Columns of ``ticks``:
  ts          tz-aware IST timestamp
  price, qty  last trade price / size
  side_true   +1 buyer-initiated (lifted the offer), -1 seller-initiated (hit the bid)
  bid, ask    best quote ATTACHED to the trade. With probability ``stale_quote_prob`` this is the
              previous event's quote (models feed latency), so quote-based classifiers are not perfect.
  event_id    id of the labelled event the trade falls in (-1 = none)

This is a TOY model (random walk + depth consumption). It is for teaching and testing,
never for claiming anything about real NSE behaviour.
Not yet generated (needs an order book, Phase 3): iceberg refills, exhaustion, pulled spoofs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, load_config

_COMPOSITE = {"sweep_reversal": (("burst_down", 0.30), ("absorption_bid", 0.40), ("burst_up", 0.30))}


@dataclass
class SyntheticDay:
    symbol: str
    date: str
    tick_size: float
    ticks: pd.DataFrame
    events: pd.DataFrame


def _hhmm_to_seconds(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def _segments(cfg: Config) -> list[tuple[float, float, str, int, str]]:
    """Expand the config schedule into (start_s, end_s, kind, event_id, parent_kind)."""
    segs = []
    for eid, item in enumerate(cfg.synthetic.schedule):
        start = item["at_min"] * 60.0
        dur = item["dur_min"] * 60.0
        kind = item["kind"]
        if kind in _COMPOSITE:
            t = start
            for sub, frac in _COMPOSITE[kind]:
                segs.append((t, t + dur * frac, sub, eid, kind))
                t += dur * frac
        else:
            segs.append((start, start + dur, kind, eid, kind))
    return sorted(segs)


def generate_day(cfg: Config | None = None, symbol: str = "SYNTH", date: str = "2026-01-05",
                 seed: int | None = None) -> SyntheticDay:
    """Generate one session of ticks. Same seed + config -> identical output."""
    cfg = cfg or load_config()
    s = cfg.synthetic
    rng = np.random.default_rng(s.seed if seed is None else seed)
    tick = float(cfg.instrument.default_tick_size)
    open_s = _hhmm_to_seconds(cfg.market.session_open)
    length = _hhmm_to_seconds(cfg.market.session_close) - open_s
    segs = _segments(cfg)

    def draw_depth(mult: float = 1.0) -> int:
        return max(1, int(rng.lognormal(np.log(s.depth_median * mult), s.depth_sigma)))

    def draw_spread() -> int:
        return 2 if rng.random() < s.wide_spread_prob else 1

    bid_t = int(round(s.start_price / tick))
    spread = 1
    bid_sz, ask_sz = draw_depth(), draw_depth()
    prev_q = (bid_t, bid_t + spread)
    base_rate = s.events_per_min / 60.0
    t = 0.0
    si = 0
    rows = []
    while True:
        while si < len(segs) and t >= segs[si][1]:
            si += 1
        seg = segs[si] if si < len(segs) and segs[si][0] <= t else None
        kind = seg[2] if seg else "normal"
        eid = seg[3] if seg else -1
        rate = base_rate
        buy_p, qmult = s.buy_prob_normal, 1.0
        if kind in ("burst_up", "burst_down"):
            rate *= s.burst_intensity_mult
            buy_p = s.burst_buy_prob if kind == "burst_up" else 1 - s.burst_buy_prob
            qmult = s.burst_qty_mult
        elif kind in ("absorption_bid", "absorption_ask"):
            rate *= s.absorb_intensity_mult
            buy_p = (1 - s.absorb_aggressor_prob) if kind == "absorption_bid" else s.absorb_aggressor_prob
            qmult = s.absorb_qty_mult
        t += rng.exponential(1.0 / rate)
        if t >= length:
            break
        # a segment boundary may have been crossed while waiting; re-evaluate next loop is
        # unnecessary for a toy model: parameters apply to the event drawn after the wait.
        side = 1 if rng.random() < buy_p else -1
        qty = max(1, int(rng.lognormal(np.log(s.trade_qty_median), s.trade_qty_sigma) * qmult))
        ask_t = bid_t + spread
        cur_q = (bid_t, ask_t)
        seen = prev_q if rng.random() < s.stale_quote_prob else cur_q
        price_t = ask_t if side > 0 else bid_t
        rows.append((t, price_t * tick, qty, side, seen[0] * tick, seen[1] * tick, eid))
        prev_q = cur_q

        # absorption = price does not progress: BOTH sides are pinned, the absorbing side refills big
        pinned_bid = pinned_ask = kind in ("absorption_bid", "absorption_ask")
        if side > 0:
            ask_sz -= qty
            if ask_sz <= 0:
                if pinned_ask:
                    ask_sz = draw_depth(s.absorb_refill_mult if kind == "absorption_ask" else 1.0)
                else:
                    bid_t += 1
                    spread = draw_spread()
                    bid_sz, ask_sz = draw_depth(), draw_depth()
        else:
            bid_sz -= qty
            if bid_sz <= 0:
                if pinned_bid:
                    bid_sz = draw_depth(s.absorb_refill_mult if kind == "absorption_bid" else 1.0)
                else:
                    bid_t -= 1
                    spread = draw_spread()
                    bid_sz, ask_sz = draw_depth(), draw_depth()
        if kind == "normal" and rng.random() < s.move_prob_normal:
            bid_t += 1 if rng.random() < 0.5 else -1
            spread = draw_spread()
            bid_sz, ask_sz = draw_depth(), draw_depth()
        elif rng.random() < s.quote_refresh_prob:
            if not pinned_bid:
                bid_sz = draw_depth()
            if not pinned_ask:
                ask_sz = draw_depth()

    df = pd.DataFrame(rows, columns=["t", "price", "qty", "side_true", "bid", "ask", "event_id"])
    open_ts = pd.Timestamp(f"{date} 00:00", tz=cfg.market.timezone) + pd.Timedelta(seconds=open_s)
    df.insert(0, "ts", open_ts + pd.to_timedelta(df.pop("t").round(3), unit="s"))
    for c in ("price", "bid", "ask"):
        df[c] = df[c].round(4)
    df["side_true"] = df["side_true"].astype("int8")

    ev = []
    for eid, item in enumerate(cfg.synthetic.schedule):
        start = open_ts + pd.Timedelta(minutes=item["at_min"])
        end = start + pd.Timedelta(minutes=item["dur_min"])
        sel = df[df["event_id"] == eid]
        ev.append({"event_id": eid, "kind": item["kind"], "start": start, "end": end,
                   "n_trades": len(sel),
                   "price_lo": float(sel["price"].min()) if len(sel) else np.nan,
                   "price_hi": float(sel["price"].max()) if len(sel) else np.nan})
    return SyntheticDay(symbol, date, tick, df.reset_index(drop=True), pd.DataFrame(ev))
