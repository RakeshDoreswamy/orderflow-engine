"""Score the detectors against the labels the synthetic generator planted.

HONESTY NOTES (also printed in the README):
* The generator and the detectors were written by the same author with the same mental model, so this is a
  CONSISTENCY test (does each detector find what its own rule describes?), not evidence about real markets.
* Thresholds are tuned only on calibration days (eval.calibration_seed_base); reported scores use different
  held-out days (eval.heldout_seed_base).
* Recall = labelled events with at least one detection inside [start, end + grace]. Precision = detection
  episodes that overlap an accepted labelled window. Intervals are Wilson 95%; tiny samples are flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import book as B
from . import events as E
from .config import Config
from .data.synthetic import SyntheticDay


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def run_detectors(day: SyntheticDay, cfg: Config) -> dict[str, pd.DataFrame]:
    """All detectors on one day (using TRUE sides; swap side_col to test with estimated sides)."""
    ts = day.tick_size
    ep = B.track_large_orders(day.book, day.ticks, cfg, ts)
    return {"large_print": E.large_prints(day.ticks, cfg), "absorption": E.absorption(day.ticks, cfg, ts),
            "iceberg": B.iceberg_refills(day.ticks, day.book, cfg, ts), "exhaustion": E.exhaustion(day.ticks, cfg, ts),
            "trapped": E.trapped(day.ticks, cfg, ts), "delta_flip": E.delta_flip_events(day.ticks, cfg),
            "wall": B.walls(ep), "spoof_like": B.spoof_flags(ep, cfg)}


def _win(lab: pd.DataFrame, kinds, top_level: bool, grace: pd.Timedelta, pre: pd.Timedelta = pd.Timedelta(0)):
    m = lab["kind"].isin(kinds)
    if top_level:
        m &= lab["parent_kind"] == lab["kind"]
    return [(r.start - pre, r.end + grace, r.kind, r.price) for r in lab[m].itertuples()]


def _in(t, w):
    return w[0] <= t <= w[1]


def score_day(day: SyntheticDay, hits: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Per-detector counts for one day: n_labels, n_detected, n_units, n_tp (units = hits or episodes)."""
    g = pd.Timedelta(minutes=cfg.eval.grace_min)
    gap = cfg.eval.episode_gap_s
    lab = day.labels
    rows = []

    def add(name, wins, accepted, units, side_map=None, extra=None):
        det = 0
        for w in wins:
            det += any(_in(t, w) for t in units["time"]) if len(units) else 0
        tp = 0
        for r in units.itertuples():
            tp += any(_in(r.time, a) for a in accepted)
        d = {"detector": name, "n_labels": len(wins), "n_detected": int(det), "n_units": len(units), "n_tp": int(tp)}
        if side_map is not None and len(units):
            ok = tot = 0
            for r in units.itertuples():
                for a in accepted:
                    if _in(r.time, a):
                        tot += 1
                        ok += side_map.get(a[2]) == r.side
                        break
            d.update(side_ok=ok, side_tot=tot)
        d.update(extra or {})
        rows.append(d)

    h = hits
    # large prints: exact trade match
    lp = lab[lab["kind"] == "large_print"]
    hp = h["large_print"]
    matched = sum(any(abs((t - s).total_seconds()) < 0.01 for t in hp["time"]) for s in lp["start"])
    tp = sum(any(abs((t - s).total_seconds()) < 0.01 for s in lp["start"]) for t in hp["time"])
    rows.append({"detector": "large_print", "n_labels": len(lp), "n_detected": int(matched), "n_units": len(hp), "n_tp": int(tp)})
    # episodes helper
    def episodes(df, by=None):
        e = E.merge_hits(df, gap, by)
        return e.rename(columns={"start": "time"}) if len(e) else pd.DataFrame(columns=["time"])

    a_top = _win(lab, ["absorption_bid", "absorption_ask"], True, g)
    a_acc = _win(lab, ["absorption_bid", "absorption_ask", "iceberg_bid", "iceberg_ask"], False, g)
    ea = E.merge_hits(h["absorption"], gap, ["side"])
    ea = ea.rename(columns={"start": "time"}) if len(ea) else pd.DataFrame(columns=["time", "side"])
    add("absorption", a_top, a_acc, ea, {"absorption_bid": "bid", "absorption_ask": "ask", "iceberg_bid": "bid", "iceberg_ask": "ask"})
    i_top = _win(lab, ["iceberg_bid", "iceberg_ask"], True, g)
    add("iceberg", i_top, i_top, h["iceberg"][["time", "side"]] if len(h["iceberg"]) else pd.DataFrame(columns=["time", "side"]),
        {"iceberg_bid": "bid", "iceberg_ask": "ask"})
    x_top = _win(lab, ["exhaustion_up", "exhaustion_down"], True, g)
    ex = E.merge_hits(h["exhaustion"], gap, ["side"])
    ex = ex.rename(columns={"start": "time"}) if len(ex) else pd.DataFrame(columns=["time", "side"])
    add("exhaustion", x_top, x_top, ex, {"exhaustion_up": "up", "exhaustion_down": "down"})
    t_top = _win(lab, ["sweep_reversal", "trap_reversal_up"], True, g)
    tr = E.merge_hits(h["trapped"], gap, ["side"])
    tr = tr.rename(columns={"start": "time"}) if len(tr) else pd.DataFrame(columns=["time", "side"])
    add("trapped", t_top, t_top, tr, {"sweep_reversal": "sellers_trapped", "trap_reversal_up": "buyers_trapped"})
    # delta flips: boundary instant = start of the final sub-phase of a sweep / trap
    bnd = lab[lab["parent_kind"].isin(["sweep_reversal", "trap_reversal_up"]) & lab["kind"].isin(["burst_up", "burst_down"])
              & (lab["kind"] != lab["parent_kind"])]
    bnd = bnd.sort_values("start").groupby("event_id").tail(1)
    fw = [(r.start - g, r.start + g, r.kind, np.nan) for r in bnd.itertuples()]
    add("delta_flip", fw, fw, h["delta_flip"][["time"]])
    # walls / spoof: price-matched instances
    inst_w = lab[lab["kind"].isin(["wall_bid_instance", "wall_ask_instance"])]
    inst_s = lab[lab["kind"].isin(["spoof_bid_instance", "spoof_ask_instance"])]
    inst_any = pd.concat([inst_w, inst_s])
    one = pd.Timedelta(seconds=1)

    def match(df, inst):
        return sum(any((abs(r.price - q.price) < 1e-6) and (q.start - one <= r.time <= q.end + one) for q in inst.itertuples())
                   for r in df.itertuples()) if len(df) else 0

    def found(df, inst):
        return sum(any((abs(r.price - q.price) < 1e-6) and (q.start - one <= r.time <= q.end + one) for r in df.itertuples())
                   for q in inst.itertuples()) if len(inst) else 0

    wl, sp = h["wall"], h["spoof_like"]
    rows.append({"detector": "wall", "n_labels": len(inst_w), "n_detected": found(wl, inst_w), "n_units": len(wl),
                 "n_tp": match(wl, inst_any)})
    rows.append({"detector": "spoof_like", "n_labels": len(inst_s), "n_detected": found(sp, inst_s), "n_units": len(sp),
                 "n_tp": match(sp, inst_s), "false_on_walls": match(sp, inst_w)})
    return pd.DataFrame(rows)


def score_days(days: list[SyntheticDay], cfg: Config) -> pd.DataFrame:
    """Sum the per-day counts and add recall / precision with Wilson intervals."""
    parts = [score_day(d, run_detectors(d, cfg), cfg) for d in days]
    tot = pd.concat(parts).groupby("detector", sort=False).sum(numeric_only=True).reset_index()
    out = []
    for r in tot.itertuples():
        rl, rh = wilson(int(r.n_detected), int(r.n_labels))
        pl, ph = wilson(int(r.n_tp), int(r.n_units))
        row = {"detector": r.detector, "labelled": int(r.n_labels), "found": int(r.n_detected),
               "recall": r.n_detected / r.n_labels if r.n_labels else np.nan, "recall_ci95": f"{rl:.2f}-{rh:.2f}",
               "detections": int(r.n_units), "true_pos": int(r.n_tp),
               "precision": r.n_tp / r.n_units if r.n_units else np.nan, "precision_ci95": f"{pl:.2f}-{ph:.2f}"}
        if hasattr(r, "side_tot") and r.side_tot:
            row["side_correct"] = r.side_ok / r.side_tot
        if hasattr(r, "false_on_walls"):
            row["false_on_genuine_walls"] = int(r.false_on_walls)
        row["note"] = "small sample" if r.n_labels < 20 or r.n_units < 20 else ""
        out.append(row)
    return pd.DataFrame(out)
