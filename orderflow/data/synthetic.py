"""Seeded synthetic NSE-style tick + order book stream with labelled events.

Why it exists: real aggressor-side and depth data are scarce, so we generate a day where the TRUE
aggressor side of every trade and every planted event is known. That lets us (a) teach, (b) test
detectors, and (c) measure how wrong the classifiers and detectors are.

``ticks`` columns:
  ts          tz-aware IST timestamp
  price, qty  last trade price / size
  side_true   +1 buyer-initiated (lifted the offer), -1 seller-initiated (hit the bid)
  bid, ask    best quote ATTACHED to the trade. With probability ``stale_quote_prob`` this is the previous
              event's quote (models feed latency), so quote-based classifiers are not perfect.
  event_id    id of the schedule item the trade falls in (-1 = none)
``book`` columns: ts, trade_idx (index of the trade the snapshot follows, -1 for periodic snapshots),
  bid_px_1..L, bid_sz_1..L, ask_px_1..L, ask_sz_1..L  (level 1 = best).
``events``: one row per schedule item. ``labels``: fine-grained instances (composite sub-phases, every
  spoof/wall instance, every injected large print) used to score detectors.

This is a TOY model (random walk + depth consumption). It is for teaching and testing, never for
claiming anything about real NSE behaviour. Randomness is split: the main stream (trade side, size,
timing, level-1 depth) is unchanged from earlier phases so earlier days are byte-identical; everything
added in Phase 3 (deeper levels, injected prints) uses a second stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config, load_config

_COMPOSITE = {
    "sweep_reversal": (("burst_down", 0.30), ("absorption_bid", 0.40), ("burst_up", 0.30)),
    "trap_reversal_up": (("burst_up", 0.30), ("absorption_ask", 0.40), ("burst_down", 0.30)),
    "exhaustion_up": (("burst_up", 0.60), ("dry_up", 0.40)),
    "exhaustion_down": (("burst_down", 0.60), ("dry_up", 0.40)),
}
_PINNED = ("absorption_bid", "absorption_ask", "iceberg_bid", "iceberg_ask")
_NORMALISH = ("normal", "dry_up", "spoof_bid", "spoof_ask", "wall_bid", "wall_ask")


@dataclass
class SyntheticDay:
    symbol: str
    date: str
    tick_size: float
    ticks: pd.DataFrame
    events: pd.DataFrame
    book: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def book_ts_sorted(snaps: list) -> bool:
    t = [x[0] for x in snaps]
    return all(a <= b for a, b in zip(t, t[1:]))


def generate_day(cfg: Config | None = None, symbol: str = "SYNTH", date: str = "2026-01-05",
                 seed: int | None = None) -> SyntheticDay:
    """Generate one session. Same seed + config -> identical output."""
    cfg = cfg or load_config()
    s = cfg.synthetic
    seed0 = s.seed if seed is None else seed
    rng = np.random.default_rng(seed0)
    rng2 = np.random.default_rng(seed0 + 7919)     # Phase 3 randomness only
    tick = float(cfg.instrument.default_tick_size)
    open_s = _hhmm_to_seconds(cfg.market.session_open)
    length = _hhmm_to_seconds(cfg.market.session_close) - open_s
    segs = _segments(cfg)
    L = int(s.book_levels)

    def draw_depth(mult: float = 1.0) -> int:
        return max(1, int(rng.lognormal(np.log(s.depth_median * mult), s.depth_sigma)))

    def draw_spread() -> int:
        return 2 if rng.random() < s.wide_spread_prob else 1

    def draw_deep() -> list[int]:
        return [max(1, int(rng2.lognormal(np.log(s.depth_median * s.deep_depth_mult), s.depth_sigma)))
                for _ in range(L - 1)]

    st = {"bid_t": int(round(s.start_price / tick)), "spread": 1}
    bsz = [draw_depth()] + draw_deep()
    asz = [draw_depth()] + draw_deep()
    st["prev_q"] = (st["bid_t"], st["bid_t"] + st["spread"])
    override: dict[tuple[str, int], int] = {}      # ('b'|'a', price_tick) -> size (spoof / wall orders)
    ice: dict | None = None
    placed: set[int] = set()
    spoof: dict | None = None
    wall: dict | None = None
    last_pull_t = -1e9
    inst: list[dict] = []                          # spoof / wall instances for labels
    rows: list[tuple] = []
    snaps: list[tuple] = []
    base_rate = s.events_per_min / 60.0
    large_t = (sorted(rng2.uniform(600, length - 600, int(s.n_large_prints)).tolist())
               if s.n_large_prints > 0 else [])
    li = 0

    def level_size(side: str, k: int) -> int:
        pt = st["bid_t"] - k if side == "b" else st["bid_t"] + st["spread"] + k
        if (side, pt) in override:
            return override[(side, pt)]
        if ice is not None and ice["side"] == side and ice["ptick"] == pt and k == 0:
            return ice["disclosed"]
        return bsz[k] if side == "b" else asz[k]

    def snapshot(t_s: float, trade_idx: int) -> None:
        bt, at = st["bid_t"], st["bid_t"] + st["spread"]
        snaps.append((t_s, trade_idx,
                      *[(bt - k) * tick for k in range(L)], *[level_size("b", k) for k in range(L)],
                      *[(at + k) * tick for k in range(L)], *[level_size("a", k) for k in range(L)]))

    def redraw() -> None:
        bsz[:] = [draw_depth()] + draw_deep()
        asz[:] = [draw_depth()] + draw_deep()

    def process_trade(t_s: float, side: int, qty: int, stale_u: float, kind: str, eid: int) -> None:
        nonlocal ice
        bid_t, spread = st["bid_t"], st["spread"]
        ask_t = bid_t + spread
        cur_q = (bid_t, ask_t)
        seen = st["prev_q"] if stale_u < s.stale_quote_prob else cur_q
        price_t = ask_t if side > 0 else bid_t
        rows.append((t_s, price_t * tick, qty, side, seen[0] * tick, seen[1] * tick, eid))
        st["prev_q"] = cur_q
        pinned = kind in _PINNED
        sd, pt = ("a", ask_t) if side > 0 else ("b", bid_t)
        szs = asz if side > 0 else bsz
        if (sd, pt) in override:
            override[(sd, pt)] -= qty
            exhausted = override[(sd, pt)] <= 0
            if exhausted:
                del override[(sd, pt)]
        elif ice is not None and ice["side"] == sd and ice["ptick"] == pt:
            ice["hidden"] -= qty
            exhausted = ice["hidden"] <= 0
            if exhausted:
                ice = None
        else:
            szs[0] -= qty
            exhausted = szs[0] <= 0
        if exhausted:
            if pinned:
                szs[0] = draw_depth(s.absorb_refill_mult if kind == f"absorption_{'ask' if side > 0 else 'bid'}" else 1.0)
            else:
                st["bid_t"] += 1 if side > 0 else -1
                st["spread"] = draw_spread()
                redraw()
        if kind in _NORMALISH and rng.random() < s.move_prob_normal:
            st["bid_t"] += 1 if rng.random() < 0.5 else -1
            st["spread"] = draw_spread()
            redraw()
        elif rng.random() < s.quote_refresh_prob:
            if not pinned:
                bsz[0] = draw_depth()
            if not pinned:
                asz[0] = draw_depth()
        snapshot(t_s, len(rows) - 1)

    t = 0.0
    si = 0
    next_snap = 0.0
    snapshot(0.0, -1)

    def flush_snaps(upto: float) -> None:
        """Periodic snapshots: the book is unchanged between events, so record it every book_snapshot_sec."""
        nonlocal next_snap
        while next_snap + s.book_snapshot_sec <= upto:
            next_snap += s.book_snapshot_sec
            snapshot(next_snap, -1)
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
        elif kind in ("absorption_bid", "absorption_ask", "iceberg_bid", "iceberg_ask"):
            rate *= s.absorb_intensity_mult
            buy_p = (1 - s.absorb_aggressor_prob) if kind.endswith("_bid") else s.absorb_aggressor_prob
            qmult = s.absorb_qty_mult
        elif kind == "dry_up":
            rate *= s.dry_up_intensity_mult
        elif kind in ("spoof_bid", "spoof_ask"):
            buy_p = s.spoof_buy_prob if kind == "spoof_bid" else 1 - s.spoof_buy_prob
        elif kind in ("wall_bid", "wall_ask"):
            buy_p = s.wall_buy_prob if kind == "wall_ask" else 1 - s.wall_buy_prob
        t += rng.exponential(1.0 / rate)
        if t >= length:
            break
        # --- injected large prints happen BEFORE this event (second stream; none unless n_large_prints > 0) ---
        while li < len(large_t) and large_t[li] <= t:
            flush_snaps(large_t[li])
            q = int(s.trade_qty_median * s.large_print_mult * (1 + 0.5 * rng2.random()))
            process_trade(large_t[li], 1 if rng2.random() < 0.5 else -1, q, 1.0, kind, eid)
            inst.append({"kind": "large_print", "event_id": -1, "start": large_t[li], "end": large_t[li],
                         "price": rows[-1][1]})
            li += 1
        flush_snaps(t)
        # --- book-order lifecycle (Phase 3 kinds only; no main-stream draws) ---
        if ice is None and kind in ("iceberg_bid", "iceberg_ask") and eid not in placed:
            placed.add(eid)
            sd = "b" if kind == "iceberg_bid" else "a"
            ice = {"side": sd, "ptick": st["bid_t"] if sd == "b" else st["bid_t"] + st["spread"],
                   "disclosed": int(s.iceberg_disclosed), "hidden": int(s.iceberg_total)}
        elif ice is not None and kind not in ("iceberg_bid", "iceberg_ask"):
            ice = None
        if kind in ("spoof_bid", "spoof_ask"):
            sd = "b" if kind == "spoof_bid" else "a"
            if spoof is not None:
                pt = spoof["ptick"]
                dist = (st["bid_t"] - pt) if sd == "b" else (pt - (st["bid_t"] + st["spread"]))
                if dist <= s.spoof_pull_dist_ticks or t - spoof["t0"] >= s.spoof_max_life_s:
                    override.pop((sd, pt), None)
                    inst.append({"kind": f"spoof_{'bid' if sd == 'b' else 'ask'}", "event_id": eid, "start": spoof["t0"],
                                 "end": t, "price": pt * tick})
                    spoof, last_pull_t = None, t
            if spoof is None and t - last_pull_t >= s.spoof_interval_s and (seg[1] - t) > s.spoof_max_life_s:
                pt = st["bid_t"] - 2 if sd == "b" else st["bid_t"] + st["spread"] + 2
                override[(sd, pt)] = int(s.spoof_mult * s.depth_median)
                spoof = {"ptick": pt, "t0": t}
        elif spoof is not None:
            sd = "b" if (("b", spoof["ptick"]) in override) else "a"
            override.pop((sd, spoof["ptick"]), None)
            inst.append({"kind": f"spoof_{'bid' if sd == 'b' else 'ask'}", "event_id": eid, "start": spoof["t0"],
                         "end": t, "price": spoof["ptick"] * tick})
            spoof, last_pull_t = None, t
        if kind in ("wall_bid", "wall_ask"):
            if wall is None and eid not in placed:
                placed.add(eid)
                sd = "a" if kind == "wall_ask" else "b"
                pt = st["bid_t"] + st["spread"] + 2 if sd == "a" else st["bid_t"] - 2
                override[(sd, pt)] = int(s.wall_mult * s.depth_median)
                wall = {"side": sd, "ptick": pt, "t0": t}
        elif wall is not None:
            override.pop((wall["side"], wall["ptick"]), None)
            inst.append({"kind": f"wall_{'ask' if wall['side'] == 'a' else 'bid'}", "event_id": eid, "start": wall["t0"],
                         "end": t, "price": wall["ptick"] * tick})
            wall = None
        side = 1 if rng.random() < buy_p else -1
        qty = max(1, int(rng.lognormal(np.log(s.trade_qty_median), s.trade_qty_sigma) * qmult))
        process_trade(t, side, qty, rng.random(), kind, eid)

    df = pd.DataFrame(rows, columns=["t", "price", "qty", "side_true", "bid", "ask", "event_id"])
    open_ts = pd.Timestamp(f"{date} 00:00", tz=cfg.market.timezone) + pd.Timedelta(seconds=open_s)
    to_ts = lambda sec: open_ts + pd.to_timedelta(np.round(sec, 3), unit="s")  # noqa: E731
    df.insert(0, "ts", to_ts(df.pop("t").to_numpy()))
    for c in ("price", "bid", "ask"):
        df[c] = df[c].round(4)
    df["side_true"] = df["side_true"].astype("int8")
    assert df["ts"].is_monotonic_increasing and book_ts_sorted(snaps)

    cols = (["ts", "trade_idx"] + [f"bid_px_{k + 1}" for k in range(L)] + [f"bid_sz_{k + 1}" for k in range(L)]
            + [f"ask_px_{k + 1}" for k in range(L)] + [f"ask_sz_{k + 1}" for k in range(L)])
    book = pd.DataFrame(snaps, columns=["t"] + cols[1:])
    book.insert(0, "ts", to_ts(book.pop("t").to_numpy()))
    for k in range(L):
        for c in (f"bid_px_{k + 1}", f"ask_px_{k + 1}"):
            book[c] = book[c].round(4)

    ev, lab = [], []
    for eid, item in enumerate(cfg.synthetic.schedule):
        start = open_ts + pd.Timedelta(minutes=item["at_min"])
        end = start + pd.Timedelta(minutes=item["dur_min"])
        sel = df[df["event_id"] == eid]
        ev.append({"event_id": eid, "kind": item["kind"], "start": start, "end": end, "n_trades": len(sel),
                   "price_lo": float(sel["price"].min()) if len(sel) else np.nan,
                   "price_hi": float(sel["price"].max()) if len(sel) else np.nan})
        lab.append({"event_id": eid, "kind": item["kind"], "parent_kind": item["kind"], "start": start, "end": end,
                    "price": np.nan})
        if item["kind"] in _COMPOSITE:
            t0 = start
            for sub, frac in _COMPOSITE[item["kind"]]:
                t1 = t0 + pd.Timedelta(minutes=item["dur_min"] * frac)
                lab.append({"event_id": eid, "kind": sub, "parent_kind": item["kind"], "start": t0, "end": t1,
                            "price": np.nan})
                t0 = t1
    for r in inst:
        lab.append({"event_id": r["event_id"], "kind": r["kind"] + ("_instance" if r["kind"] != "large_print" else ""),
                    "parent_kind": r["kind"], "start": to_ts(np.array([r["start"]]))[0],
                    "end": to_ts(np.array([r["end"]]))[0], "price": round(r["price"], 4)})
    return SyntheticDay(symbol, date, tick, df.reset_index(drop=True), pd.DataFrame(ev), book, pd.DataFrame(lab))


_DAY_KINDS = ["burst_up", "burst_down", "absorption_bid", "absorption_ask", "sweep_reversal"]
EVAL_KINDS = ["burst_up", "burst_down", "absorption_bid", "absorption_ask", "sweep_reversal", "trap_reversal_up",
              "iceberg_bid", "iceberg_ask", "spoof_bid", "spoof_ask", "wall_bid", "wall_ask",
              "exhaustion_up", "exhaustion_down"]
_DUR = {"sweep_reversal": (12, 16), "trap_reversal_up": (12, 16), "exhaustion_up": (10, 14),
        "exhaustion_down": (10, 14), "spoof_bid": (6, 10), "spoof_ask": (6, 10), "wall_bid": (6, 10), "wall_ask": (6, 10)}


def generate_days(cfg: Config | None = None, n_days: int = 8, symbol: str = "SYNTH",
                  first_date: str = "2026-01-05") -> list[SyntheticDay]:
    """Several consecutive weekday sessions. Each day starts at the previous close and gets its own
    seeded random event schedule (3 events), so composite profiles, naked POCs and open types have
    something to vary over. Still a toy model: no claim about real markets."""
    cfg = cfg or load_config()
    days, price = [], cfg.synthetic.start_price
    dates = pd.bdate_range(first_date, periods=n_days)
    for i, d in enumerate(dates):
        rng = np.random.default_rng(cfg.synthetic.seed * 1000 + i)
        slots = sorted(rng.choice([30, 90, 150, 210, 270, 330], size=3, replace=False).tolist())
        sched = [{"kind": str(rng.choice(_DAY_KINDS)), "at_min": int(m), "dur_min": int(rng.integers(6, 15))}
                 for m in slots]
        c = cfg.with_overrides({"synthetic": {"start_price": float(price), "schedule": sched}})
        day = generate_day(c, symbol, d.strftime("%Y-%m-%d"), seed=cfg.synthetic.seed * 1000 + i)
        days.append(day)
        price = float(day.ticks["price"].iloc[-1])
    return days


def generate_eval_days(cfg: Config | None = None, n_days: int = 10, seed_base: int = 9000, n_events: int = 4,
                       kinds: list[str] | None = None, n_large_prints: int = 5, first_date: str = "2026-02-02"
                       ) -> list[SyntheticDay]:
    """Days for scoring detectors: ``n_events`` events per day from all Phase 3 kinds, drawn from slots
    that cannot overlap, plus injected large prints. Seeds are seed_base + day index, so calibration
    (e.g. seed_base=500) and held-out evaluation (seed_base=9000) use different days."""
    cfg = cfg or load_config()
    kinds = kinds or EVAL_KINDS
    days, price = [], cfg.synthetic.start_price
    for i, d in enumerate(pd.bdate_range(first_date, periods=n_days)):
        rng = np.random.default_rng(seed_base + i)
        slots = sorted(rng.choice([20, 80, 140, 200, 260, 320], size=n_events, replace=False).tolist())
        sched = []
        for m in slots:
            k = str(rng.choice(kinds))
            lo, hi = _DUR.get(k, (6, 12))
            sched.append({"kind": k, "at_min": int(m), "dur_min": int(rng.integers(lo, hi + 1))})
        c = cfg.with_overrides({"synthetic": {"start_price": float(price), "schedule": sched,
                                              "n_large_prints": n_large_prints}})
        day = generate_day(c, "SYNTH", d.strftime("%Y-%m-%d"), seed=seed_base + i)
        days.append(day)
        price = float(day.ticks["price"].iloc[-1])
    return days
