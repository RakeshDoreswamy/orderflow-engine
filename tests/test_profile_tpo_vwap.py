import numpy as np
import pandas as pd

from orderflow.profile import naked_pocs, poc_index, profile_shape, value_area, value_area_idx, volume_profile
from orderflow.tpo import build_tpo, open_type
from orderflow.vwap import vwap_series

IST = "Asia/Kolkata"


def test_value_area_known_profile():
    assert value_area_idx(np.array([1, 2, 5, 10, 8, 3, 1]), 0.70) == (3, 5, 3)
    prof = pd.DataFrame({"price": np.arange(7) * 0.5 + 100, "vol": [1, 2, 5, 10, 8, 3, 1]})
    va = value_area(prof, 0.7)
    assert (va["poc"], va["val"], va["vah"]) == (101.5, 101.5, 102.5) and abs(va["share"] - 0.7) < 1e-9


def test_poc_tie_breaks_towards_middle():
    assert poc_index(np.array([9, 1, 1, 9, 1])) == 3


def test_volume_profile_and_composite_fixed_range(cfg, day):
    p = volume_profile(day.ticks, day.tick_size, 10)
    assert p.vol.sum() == day.ticks.qty.sum()
    half = volume_profile(day.ticks, day.tick_size, 10, end=pd.Timestamp("2026-01-05 12:00", tz=IST))
    assert half.vol.sum() == day.ticks[day.ticks.ts < pd.Timestamp("2026-01-05 12:00", tz=IST)].qty.sum()


def _g(c, sd, n=30):
    x = np.arange(n)
    return 1000 * np.exp(-0.5 * ((x - c) / sd) ** 2)


def test_profile_shapes(cfg):
    n = 30
    mk = lambda v: pd.DataFrame({"price": np.arange(n) * 0.5, "vol": v + 1.0})
    x = np.arange(n)
    assert profile_shape(mk(_g(14.5, 5)), cfg) == "D"
    p = _g(24, 3) + np.where(x < 20, 150 * x / 20, 0)
    assert profile_shape(mk(p), cfg) == "P"
    assert profile_shape(mk(p[::-1]), cfg) == "b"
    assert profile_shape(mk(_g(7, 2.5) + _g(23, 2.5)), cfg) == "double"
    assert profile_shape(mk(np.full(n, 100.0)), cfg) == "thin"


def test_naked_pocs():
    s = pd.DataFrame({"date": ["d1", "d2", "d3"], "poc": [100, 108, 101], "high": [105, 110, 103], "low": [95, 101, 99]})
    r = naked_pocs(s)
    assert r.naked.tolist() == [False, True, True] and r.first_tested[0] == "d3"


def _t(rows):
    return pd.DataFrame(rows, columns=["ts", "price"]).assign(ts=lambda d: pd.to_datetime(d.ts).dt.tz_localize(IST))


def test_tpo_letters_ib_extension_and_poor_high(cfg):
    # tick 0.05 x 10 = 0.50 rows. A:100-101  B:100.5-101.5  C:101-102  D:101.5-102.5
    rows = []
    for ts, lo, hi in [("09:20", 100.0, 101.0), ("09:50", 100.5, 101.5), ("10:20", 101.0, 102.0), ("10:50", 101.5, 102.5)]:
        rows += [(f"2026-01-05 {ts}:00", lo), (f"2026-01-05 {ts}:30", hi)]
    t = build_tpo(_t(rows), cfg, 0.05)
    cnt = t.matrix.sum(axis=1).tolist()
    assert t.matrix.columns.tolist() == ["A", "B", "C", "D"] and cnt == [1, 2, 3, 3, 2, 1]
    assert (t.ibl, t.ibh) == (100.0, 101.5) and t.extensions == [("C", "up")]
    assert t.poc == 101.0 and not t.poor_high and not t.poor_low and not t.excess_high
    rows2 = rows + [("2026-01-05 11:20:00", 102.5)]       # E re-tests the high -> 2 letters there
    assert build_tpo(_t(rows2), cfg, 0.05).poor_high


def test_tpo_excess_and_single_prints(cfg):
    rows = [("2026-01-05 09:20:00", 100.0), ("2026-01-05 09:40:00", 101.0),
            ("2026-01-05 09:50:00", 100.0), ("2026-01-05 10:10:00", 102.5)]
    t = build_tpo(_t(rows), cfg, 0.05)
    assert t.excess_high and not t.poor_high and t.single_prints == [(101.5, 102.5, True)]


def test_tpo_prefix_no_lookahead(cfg, day):
    full = build_tpo(day.ticks, cfg, day.tick_size)
    cut = pd.Timestamp("2026-01-05 12:15", tz=IST)
    part = build_tpo(day.ticks[day.ticks.ts < cut], cfg, day.tick_size)
    sub = full.matrix.loc[part.matrix.index, part.matrix.columns]
    assert (sub.to_numpy() == part.matrix.to_numpy()).all()
    assert (part.ibh, part.ibl) == (full.ibh, full.ibl)


def _w(prices, q0="2026-01-05 09:{:02d}:00"):
    return pd.DataFrame({"ts": [pd.Timestamp(q0.format(15 + i), tz=IST) for i in range(len(prices))], "price": prices})


def test_open_types(cfg):
    assert open_type(_w([100, 100.2, 100.4, 100.6, 100.8, 101.0]), cfg) == "open_drive"
    assert open_type(_w([100, 99.8, 100.2, 100.6, 100.8, 101.0]), cfg) == "open_test_drive"
    assert open_type(_w([100, 101.5, 100.5, 99.0, 98.0, 98.0]), cfg) == "open_rejection_reverse"
    assert open_type(_w([100, 100.5, 99.5, 100.4, 99.6, 100.05]), cfg) == "open_auction"


def test_vwap_hand_built_and_anchor(cfg):
    t = pd.DataFrame({"ts": pd.to_datetime(["2026-01-05 09:15:01", "2026-01-05 09:15:02", "2026-01-06 09:15:01"]).tz_localize(IST),
                      "price": [100.0, 102.0, 90.0], "qty": [10, 30, 5]})
    v = vwap_series(t, cfg)
    assert abs(v.vwap[1] - 101.5) < 1e-9 and abs(v.sd[1] - np.sqrt(0.75)) < 1e-9
    assert v.vwap[2] == 90.0                                       # new day resets
    assert abs(v.upper_2[1] - (101.5 + 2 * np.sqrt(0.75))) < 1e-9
    a = vwap_series(t, cfg, anchor_ts=t.ts[1])
    assert np.isnan(a.vwap[0]) and a.vwap[1] == 102.0


def test_vwap_prefix_no_lookahead(cfg, day):
    full = vwap_series(day.ticks, cfg)
    part = vwap_series(day.ticks.iloc[:5000], cfg)
    pd.testing.assert_frame_equal(part, full.iloc[:5000])
