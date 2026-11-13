import numpy as np
import pandas as pd
import pytest

from orderflow.config import load_config
from orderflow.profile import value_area, volume_profile
from orderflow.rules import gap_fill, rule80, rule80_day

cfg = load_config()
TS = 0.05
TZ = "Asia/Kolkata"


def ticks_from(date, pts):
    """pts = [(minutes after 09:15, price, qty)]"""
    t0 = pd.Timestamp(f"{date} 09:15", tz=TZ)
    return pd.DataFrame({"ts": [t0 + pd.Timedelta(minutes=m) for m, _, _ in pts], "price": [p for _, p, _ in pts], "qty": [q for _, _, q in pts],
                         "side_true": 1})


def prev_day():
    px = np.round(np.arange(99.50, 100.55, TS), 2)
    w = np.exp(-((px - 100.0) / 0.25) ** 2) * 1000
    return ticks_from("2026-02-02", [(i, p, max(int(q), 1)) for i, (p, q) in enumerate(zip(px, w))])


def va():
    prof = volume_profile(prev_day(), TS, 1)
    v = value_area(prof, cfg.rules.value_area_pct)
    return v["val"], v["vah"] + TS


def day(pts):
    return ticks_from("2026-02-03", pts)


def test_value_area_is_sane():
    lo, top = va()
    assert 99.6 < lo < 100.0 < top < 100.4


def test_open_above_accepted_then_reaches_other_side():
    lo, top = va()
    mid = (lo + top) / 2
    # open well above, 30-min closes at 101 -> then two closes inside -> afterwards touch below VAL
    pts = [(0, top + 1.0, 10), (29, top + 1.0, 10), (35, mid, 10), (59, mid, 10), (65, mid, 10), (89, mid, 10), (100, lo - 0.1, 10)]
    r = rule80_day(prev_day(), day(pts), cfg, TS)
    assert r["side"] == "above" and r["accepted"] is True and r["success"] is True


def test_accepted_but_no_traverse_is_failure():
    lo, top = va()
    mid = (lo + top) / 2
    pts = [(0, top + 1.0, 10), (29, top + 1.0, 10), (35, mid, 10), (59, mid, 10), (65, mid, 10), (89, mid, 10), (120, mid, 10)]
    r = rule80_day(prev_day(), day(pts), cfg, TS)
    assert r["accepted"] is True and r["success"] is False


def test_never_accepted_and_inside_open():
    lo, top = va()
    out = rule80_day(prev_day(), day([(0, top + 1.0, 10), (40, top + 1.0, 10), (80, top + 1.2, 10)]), cfg, TS)
    assert out["side"] == "above" and out["accepted"] is False and np.isnan(out["success"])
    assert rule80_day(prev_day(), day([(0, (lo + top) / 2, 10), (40, lo, 10)]), cfg, TS)["side"] == "inside"


def test_acceptance_needs_known_time_before_success_counts():
    """A touch of the far edge BEFORE acceptance is known must not count as success."""
    lo, top = va()
    mid = (lo + top) / 2
    pts = [(0, top + 1.0, 10), (10, lo - 0.2, 10), (29, top + 1.0, 10), (35, mid, 10), (59, mid, 10), (65, mid, 10), (89, mid, 10), (120, mid, 10)]
    r = rule80_day(prev_day(), day(pts), cfg, TS)
    assert r["accepted"] is True and r["success"] is False


def test_rule80_summary_and_claim_check():
    lo, top = va()
    mid = (lo + top) / 2
    good = day([(0, top + 1.0, 10), (29, top + 1.0, 10), (35, mid, 10), (59, mid, 10), (65, mid, 10), (89, mid, 10), (100, lo - 0.1, 10)])
    t, s = rule80([(prev_day(), TS), (good, TS)], cfg)
    assert s["accepted"] == 1 and s["successes"] == 1 and s["small_sample"] and s["rate"] == 1.0


def test_gap_fill_hand_built():
    e = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4), "open": [100, 101, 99.0, 103.0], "high": [101, 102, 99.5, 104],
                      "low": [99, 100.5, 98, 102.5], "close": [100.5, 100, 99.2, 103.5], "volume": 1})
    d, s = gap_fill(e, cfg)
    # day2: gap up 0.5 (open 101 vs close 100.5), low 100.5 <= 100.5 -> filled; day3: gap down (99 vs 100) high 99.5 < 100 -> not filled
    # day4: gap up (103 vs 99.2), low 102.5 > 99.2 -> not filled
    got = d.set_index("date")["filled"].astype(bool).to_dict()
    assert got[pd.Timestamp("2024-01-02")] is True and got[pd.Timestamp("2024-01-03")] is False and got[pd.Timestamp("2024-01-04")] is False
    assert s["gaps"].sum() == 3 and s["filled"].sum() == 1
