import numpy as np
import pandas as pd
import pytest

from orderflow.backtest.costs import leg_cost, round_trip_cost
from orderflow.backtest.engine import run_backtest, summarize
from orderflow.config import load_config

cfg = load_config()
TS = 0.05
SLIP = cfg.costs.intraday.slippage_ticks * TS


def mk(rows, start="2026-02-02 10:00"):
    t0 = pd.Timestamp(start, tz="Asia/Kolkata")
    return pd.DataFrame([{"bar": t0 + pd.Timedelta(minutes=i), "open": o, "high": h, "low": l, "close": c} for i, (o, h, l, c) in enumerate(rows)])


def sig(t, side, stop, target):
    return pd.DataFrame([{"time": pd.Timestamp(t, tz="Asia/Kolkata"), "side": side, "stop": stop, "target": target, "setup": "x"}])


def test_leg_cost_hand_computed():
    # buy 100 @ 1000 = 100,000 turnover: brokerage capped at 20, txn 2.97, sebi 0.1, gst 18% of 23.07, stamp 3.0
    exp = 20 + 2.97 + 0.1 + 0.18 * (20 + 2.97 + 0.1) + 3.0
    assert leg_cost("buy", 1000, 100, cfg) == pytest.approx(exp)
    # sell adds STT 25 and no stamp
    exp_s = 20 + 2.97 + 0.1 + 0.18 * 23.07 + 25.0
    assert leg_cost("sell", 1000, 100, cfg) == pytest.approx(exp_s)
    assert round_trip_cost(1, 1000, 1000, 100, cfg) == pytest.approx(exp + exp_s)


def test_entry_is_next_bar_open_not_signal_bar():
    bars = mk([(100, 100.2, 99.9, 100.1), (100.1, 100.15, 100.0, 100.05), (100.05, 100.1, 100.0, 100.0), (100, 100.1, 99.95, 100.0)])
    # signal known at 10:01:30 -> first bar starting at/after is 10:02 (open 100.05)
    t = run_backtest(bars, sig("2026-02-02 10:01:30", 1, 99.0, 101.0), cfg, TS)
    assert len(t) == 1 and t["entry_time"].iloc[0] == bars["bar"].iloc[2]
    assert t["entry"].iloc[0] == pytest.approx(100.05 + SLIP)


def test_target_stop_both_and_time_exits():
    up = mk([(100, 100.1, 99.9, 100.0), (100, 101.2, 99.95, 101.0)])
    t = run_backtest(up, sig("2026-02-02 10:00", 1, 99.0, 101.0), cfg, TS)
    assert t["reason"].iloc[0] == "target" and t["exit"].iloc[0] == pytest.approx(101.0 - SLIP)
    dn = mk([(100, 100.1, 98.8, 99.0)])
    t = run_backtest(dn, sig("2026-02-02 10:00", 1, 99.0, 101.0), cfg, TS)
    assert t["reason"].iloc[0] == "stop" and t["exit"].iloc[0] == pytest.approx(99.0 - SLIP)
    both = mk([(100, 101.5, 98.5, 100)])
    t = run_backtest(both, sig("2026-02-02 10:00", 1, 99.0, 101.0), cfg, TS)
    assert t["reason"].iloc[0] == "stop"                      # conservative tie-break
    quiet = mk([(100, 100.1, 99.9, 100.0)] * 60)
    t = run_backtest(quiet, sig("2026-02-02 10:00", 1, 99.0, 101.0), cfg, TS)
    assert t["reason"].iloc[0] == "time" and (t["exit_time"].iloc[0] - t["entry_time"].iloc[0]) <= pd.Timedelta(minutes=cfg.backtest.max_hold_min)


def test_short_side_and_costs_make_flat_trade_lose():
    quiet = mk([(100, 100.1, 99.9, 100.0)] * 5)
    t = run_backtest(quiet, sig("2026-02-02 10:00", -1, 101.0, 99.0), cfg, TS)
    r = t.iloc[0]
    assert r["entry"] == pytest.approx(100 - SLIP) and r["exit"] == pytest.approx(100.0 + SLIP)
    assert r["gross"] < 0 and r["net"] == pytest.approx(r["gross"] - r["cost"]) and r["cost"] > 0


def test_one_position_at_a_time_and_bad_levels_refused():
    bars = mk([(100, 100.1, 99.9, 100.0)] * 30)
    s = pd.concat([sig("2026-02-02 10:00", 1, 99, 101), sig("2026-02-02 10:05", 1, 99, 101)], ignore_index=True)
    assert len(run_backtest(bars, s, cfg, TS)) == 1                      # second signal arrives while the first is still open
    assert len(run_backtest(bars, sig("2026-02-02 10:00", 1, 100.5, 101), cfg, TS)) == 0   # stop above the fill


def test_no_entry_near_close():
    bars = mk([(100, 100.1, 99.9, 100.0)] * 5, start="2026-02-02 15:20")
    assert len(run_backtest(bars, sig("2026-02-02 15:20", 1, 99, 101), cfg, TS)) == 0


def test_summarize_flags_small_sample_and_ci():
    bars = mk([(100, 100.1, 99.9, 100.0)] * 5)
    t = run_backtest(bars, sig("2026-02-02 10:00", 1, 99, 101), cfg, TS)
    s = summarize(t, cfg)
    assert s["trades"] == 1 and s["small_sample"] and s["mean_net_ci95"][0] <= s["mean_net_per_trade"] <= s["mean_net_ci95"][1]
    assert summarize(t.iloc[0:0], cfg) == {"trades": 0}


def test_negative_control_loses_about_the_costs():
    from orderflow.backtest.controls import control_bars
    bars, sigs = control_bars(cfg, n_days=30, seed=3, stop_ticks=30, target_ticks=45)
    t = run_backtest(bars, sigs, cfg, TS)
    s = summarize(t, cfg)
    assert s["trades"] >= 150  # costs per round trip are about 82 rupees for 100 shares of a 1000-rupee stock
    assert s["mean_net_ci95"][1] < 0                                   # no edge -> a loss after costs, with confidence
    gross_ci = np.percentile(np.random.default_rng(0).choice(t["gross"].to_numpy(), (2000, len(t))).mean(axis=1), [0.5, 99.5])
    assert gross_ci[0] < 0 < gross_ci[1] + 3 * SLIP * 100              # gross is indistinguishable from slippage-only drag


def test_positive_control_is_detected():
    from orderflow.backtest.controls import control_bars
    bars, sigs = control_bars(cfg, n_days=30, seed=3, drift_ticks=3, stop_ticks=30, target_ticks=45)
    s = summarize(run_backtest(bars, sigs, cfg, TS), cfg)
    assert s["mean_net_ci95"][0] > 0
