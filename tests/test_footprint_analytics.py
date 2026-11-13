import pandas as pd

from orderflow.data.bars import footprint
from orderflow.footprint import analyze_footprint

IST = "Asia/Kolkata"


def _fp(bars):
    rows = []
    for i, rws in enumerate(bars):
        b = pd.Timestamp("2026-01-05 09:15", tz=IST) + pd.Timedelta(minutes=5 * i)
        for p, bid, ask in rws:
            rows.append((b, p, bid, ask, bid + ask))
    return pd.DataFrame(rows, columns=["bar", "row_price", "bid_vol", "ask_vol", "total_vol"])


def test_diagonal_and_stacked_imbalance(cfg):
    fp = _fp([
        [(100.00, 50, 0), (100.05, 10, 200), (100.10, 5, 300), (100.15, 5, 400), (100.20, 0, 20)],  # 0 stacked buys
        [(100.00, 40, 10), (100.05, 30, 100)],        # 1 ratio not met (100 < 3*40)
        [(100.00, 1, 0), (100.05, 0, 5)],             # 2 ratio met but below min volume
        [(100.00, 90, 5), (100.05, 5, 20)],           # 3 sell imbalance at 100.00 (90 >= 3*20)
    ])
    bars, rows = analyze_footprint(fp, cfg, 0.05)
    r = rows.set_index(["bar", "row_price"])
    b = lambda i: pd.Timestamp("2026-01-05 09:15", tz=IST) + pd.Timedelta(minutes=5 * i)
    assert r.loc[b(0)].buy_imb.tolist() == [False, True, True, True, True]
    assert r.loc[b(0)].stacked_buy.tolist() == [False, True, True, True, True]
    assert not r.loc[b(1)].buy_imb.any() and not r.loc[b(1)].sell_imb.any()
    assert not r.loc[b(2)].buy_imb.any()
    assert r.loc[b(3)].sell_imb.tolist() == [True, False]
    assert not r.loc[b(3)].stacked_sell.any()           # one level is not "stacked"


def test_poc_value_area_per_bar(cfg):
    bars, _ = analyze_footprint(_fp([[(100.00, 50, 0), (100.05, 10, 200), (100.10, 5, 300),
                                      (100.15, 5, 400), (100.20, 0, 20)]]), cfg, 0.05)
    assert (bars.poc[0], bars.val[0], bars.vah[0]) == (100.15, 100.05, 100.15)


def test_finished_unfinished_and_zero_rows(cfg):
    fp = _fp([[(100.00, 10, 30), (100.05, 40, 0)],                  # high row: no ask -> finished
              [(100.00, 0, 30), (100.05, 40, 5)],                   # low row: no bid -> finished low
              [(100.00, 5, 5), (100.10, 5, 5)]])                    # gap row at 100.05
    bars, _ = analyze_footprint(fp, cfg, 0.05)
    assert bars.high_finished.tolist() == [True, False, False]
    assert bars.low_finished.tolist() == [False, True, False]
    assert bars.poor_high.tolist() == [False, True, True]
    assert bars.zero_rows.tolist() == [0, 0, 1]


def test_prefix_no_lookahead(cfg, day):
    full = footprint(day.ticks, cfg, tick_size=day.tick_size)
    cut = pd.Timestamp("2026-01-05 12:00", tz=IST)
    part = footprint(day.ticks[day.ticks.ts < cut], cfg, tick_size=day.tick_size)
    a, ar = analyze_footprint(full[full.bar < cut], cfg, day.tick_size)
    b, br = analyze_footprint(part, cfg, day.tick_size)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ar, br)
