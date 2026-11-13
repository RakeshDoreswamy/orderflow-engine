import numpy as np
import pandas as pd

from orderflow.classify import classify_ticks
from orderflow.config import load_config
from orderflow.data.bars import assign_time_bars, footprint, tick_bars, time_bars, volume_bars


def _hand(cfg):
    ts = pd.to_datetime(["2026-01-05 09:15:10", "2026-01-05 09:16:00", "2026-01-05 09:17:30",
                         "2026-01-05 09:21:00", "2026-01-05 09:22:00"]).tz_localize("Asia/Kolkata")
    return pd.DataFrame({"ts": ts, "price": [100.00, 100.05, 100.05, 100.10, 100.00],
                         "qty": [10, 20, 5, 7, 3], "side_true": [1, -1, 1, 1, -1]})


def test_footprint_hand_built(cfg):
    fp = footprint(_hand(cfg), cfg, tick_size=0.05, freq="5min", row_ticks=1)
    r = fp.set_index(["bar", "row_price"])
    b0 = pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    assert r.loc[(b0, 100.0), "ask_vol"] == 10 and r.loc[(b0, 100.0), "bid_vol"] == 0
    assert r.loc[(b0, 100.05), "ask_vol"] == 5 and r.loc[(b0, 100.05), "bid_vol"] == 20
    assert r.loc[(b0, 100.05), "delta"] == -15 and r.loc[(b0, 100.05), "trades"] == 2
    b1 = pd.Timestamp("2026-01-05 09:20", tz="Asia/Kolkata")
    assert r.loc[(b1, 100.10), "ask_vol"] == 7


def test_footprint_row_aggregation(cfg):
    fp = footprint(_hand(cfg), cfg, tick_size=0.05, freq="5min", row_ticks=2)
    b0 = pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    row = fp[(fp["bar"] == b0) & (fp["row_price"] == 100.0)].iloc[0]  # 100.00 and 100.05 share a row
    assert row["ask_vol"] == 15 and row["bid_vol"] == 20


def test_estimated_columns_are_suffixed(cfg, day):
    t = classify_ticks(day.ticks.iloc[:2000], cfg)
    fp = footprint(t, cfg, tick_size=day.tick_size, side_col="side_lr_est", estimated=True)
    assert {"bid_vol_est", "ask_vol_est", "delta_est"} <= set(fp.columns)
    assert "delta" not in fp.columns
    tb = time_bars(t, cfg, side_col="side_lr_est", estimated=True)
    assert "delta_est" in tb.columns


def test_volume_conservation(cfg, day):
    fp = footprint(day.ticks, cfg, tick_size=day.tick_size)
    assert fp["total_vol"].sum() == day.ticks["qty"].sum()
    assert (fp["bid_vol"] + fp["ask_vol"] == fp["total_vol"]).all()
    tb = time_bars(day.ticks, cfg)
    assert tb["volume"].sum() == day.ticks["qty"].sum()
    assert (tb["delta"] == tb["ask_vol"] - tb["bid_vol"]).all()


def test_time_bars_anchor_to_open_for_45min(cfg):
    ts = pd.Series(pd.to_datetime(["2026-01-05 09:15:00", "2026-01-05 09:59:59", "2026-01-05 10:00:00"])
                   .tz_localize("Asia/Kolkata"))
    got = assign_time_bars(ts, "45min", "09:15")
    assert got.dt.strftime("%H:%M").tolist() == ["09:15", "09:15", "10:00"]


def test_tick_and_volume_bars(cfg, day):
    tb = tick_bars(day.ticks, 500)
    assert (tb["trades"].iloc[:-1] == 500).all()
    vb = volume_bars(day.ticks, 20000)
    assert vb["volume"].sum() == day.ticks["qty"].sum()
    slack = day.ticks["qty"].max()
    assert ((vb["volume"].iloc[:-1] - 20000).abs() <= slack).all()


def test_footprint_prefix_no_lookahead(cfg, day):
    full = footprint(day.ticks, cfg, tick_size=day.tick_size)
    cut_ts = pd.Timestamp("2026-01-05 11:00", tz="Asia/Kolkata")   # bar boundary
    part = footprint(day.ticks[day.ticks["ts"] < cut_ts], cfg, tick_size=day.tick_size)
    done = full[full["bar"] < cut_ts].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.reset_index(drop=True), done)
