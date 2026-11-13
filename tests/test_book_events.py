"""Phase 3 tests: book analytics, event detectors, no-lookahead prefix checks, evaluation plumbing."""
import numpy as np
import pandas as pd
import pytest

from orderflow import book as B
from orderflow import events as E
from orderflow.config import load_config
from orderflow.data.synthetic import generate_eval_days
from orderflow.evaluate import run_detectors, score_days, wilson


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def days(cfg):
    return generate_eval_days(cfg, 3, seed_base=500, n_events=6)


def test_wilson_bounds():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert np.isnan(wilson(0, 0)[0])


def test_large_print_hand_built(cfg):
    ts = pd.date_range("2026-02-02 09:20", periods=400, freq="s", tz="Asia/Kolkata")
    qty = np.full(400, 50)
    qty[300] = 50 * 100
    t = pd.DataFrame({"ts": ts, "price": 1000.0, "qty": qty, "side_true": 1})
    hits = E.large_prints(t, cfg)
    assert len(hits) == 1 and hits["time"].iloc[0] == ts[300]


def test_generator_plants_large_prints(days):
    assert all((d.labels["kind"] == "large_print").any() for d in days)


def test_detectors_return_tables(days, cfg):
    hits = run_detectors(days[0], cfg)
    assert hits and all(isinstance(df, pd.DataFrame) for df in hits.values())


def _prefix_equal(full, pref, cutoff):
    a = full[full["time"] < cutoff].reset_index(drop=True)
    b = pref[pref["time"] < cutoff].reset_index(drop=True)
    cols = [c for c in a.columns if c in b.columns and c != "strength"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)


def test_prefix_no_lookahead_event_detectors(days, cfg):
    d = days[0]
    ticks = d.ticks
    cut = ticks["ts"].iloc[int(len(ticks) * 0.6)]
    margin = pd.Timedelta(minutes=3)
    pre = ticks[ticks["ts"] <= cut].reset_index(drop=True)
    ts = d.tick_size
    pairs = {
        "large_prints": (E.large_prints(ticks, cfg), E.large_prints(pre, cfg)),
        "absorption": (E.absorption(ticks, cfg, ts), E.absorption(pre, cfg, ts)),
        "trapped": (E.trapped(ticks, cfg, ts), E.trapped(pre, cfg, ts)),
        "delta_flip": (E.delta_flip_events(ticks, cfg), E.delta_flip_events(pre, cfg)),
    }
    for name, (full, pref) in pairs.items():
        _prefix_equal(full, pref, cut - margin)


def test_prefix_no_lookahead_book(days, cfg):
    d = days[0]
    cut = d.ticks["ts"].iloc[int(len(d.ticks) * 0.6)]
    pre_t = d.ticks[d.ticks["ts"] <= cut].reset_index(drop=True)
    pre_b = d.book[d.book["ts"] <= cut].reset_index(drop=True)
    full = B.track_large_orders(d.book, d.ticks, cfg, d.tick_size)
    pref = B.track_large_orders(pre_b, pre_t, cfg, d.tick_size)
    key = ["side", "price", "first_ts", "confirm_ts"]
    lim = cut - pd.Timedelta(minutes=1)
    a = full[full["confirm_ts"] < lim][key].reset_index(drop=True)
    b = pref[pref["confirm_ts"] < lim][key].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_score_floor_on_calibration_days(days, cfg):
    s = score_days(days, cfg).set_index("detector")
    assert s.loc["large_print", "recall"] >= 0.7
    assert (s["labelled"] > 0).any()
