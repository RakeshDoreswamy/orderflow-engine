import numpy as np
import pandas as pd

from orderflow.config import load_config
from orderflow.delta import (confirmed_swings, cvd_anchored, cvd_divergences, delta_bars, flags,
                             ohlcv_delta_est)

IST = "Asia/Kolkata"


def _ticks(rows):
    return pd.DataFrame(rows, columns=["ts", "price", "qty", "side_true"]).assign(
        ts=lambda d: pd.to_datetime(d["ts"]).dt.tz_localize(IST))


def test_delta_bar_hand_built(cfg):
    t = _ticks([("2026-01-05 09:15:01", 100.00, 10, 1), ("2026-01-05 09:15:02", 100.05, 20, -1),
                ("2026-01-05 09:15:03", 100.10, 5, 1), ("2026-01-05 09:15:04", 100.10, 5, 1),
                ("2026-01-05 09:20:01", 100.00, 8, -1),
                ("2026-01-06 09:15:01", 100.00, 4, 1)])
    b = delta_bars(t, cfg, freq="5min")
    r = b.iloc[0]
    assert (r.ask_vol, r.bid_vol, r.delta, r.volume) == (20, 20, 0, 40)
    assert r.delta_max == 10 and r.delta_min == -10       # running: +10, -10, -5, 0
    assert r.delta_at_high == 10 and r.delta_at_low == 10  # high 100.10 (+5+5), low 100.00 (+10)
    assert r.delta_pct == 0
    assert b.cvd.tolist() == [0, -8, 4]                    # CVD resets each trading day


def test_cvd_anchored(cfg):
    b = pd.DataFrame({"bar": pd.date_range("2026-01-05 09:15", periods=4, freq="5min", tz=IST),
                      "delta": [5, -2, 3, 1]})
    c = cvd_anchored(b, b.bar[1])
    assert np.isnan(c[0]) and c[1:].tolist() == [-2, 1, 2]


def test_ohlcv_delta_est():
    b = pd.DataFrame({"volume": [100, 100, 100, 100], "high": [10, 10, 10, 5], "low": [0, 0, 0, 5],
                      "close": [10, 0, 5, 5]})
    est = ohlcv_delta_est(b)
    assert est[:3].tolist() == [100, -100, 0] and np.isnan(est[3])


def _hand_bars(cvd_second):
    h = np.array([5, 6, 7, 8, 9, 10, 9, 8, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6], float)
    cvd = np.zeros(len(h))
    cvd[5], cvd[12] = 100, cvd_second
    return pd.DataFrame({"bar": pd.date_range("2026-01-05 09:15", periods=len(h), freq="5min", tz=IST),
                         "high": h, "low": h - 1, "cvd": cvd})


def test_swings_and_bearish_divergence_fires_only_when_cvd_lower(cfg):
    c2 = load_config(overrides={"delta": {"swing_left": 2, "swing_right": 2}})
    sw = confirmed_swings(_hand_bars(50).high.to_numpy(), _hand_bars(50).low.to_numpy(), 2, 2)
    assert (sw[sw.kind == "high"].idx.tolist(), sw[sw.kind == "high"].confirm_idx.tolist()) == ([5, 12], [7, 14])
    d = cvd_divergences(_hand_bars(50), c2)
    assert d.kind.tolist() == ["bearish"] and d.confirm_idx.tolist() == [14] and d.first_idx[0] == 5
    assert (cvd_divergences(_hand_bars(150), c2).kind == "bearish").sum() == 0   # CVD confirmed the high


def test_flags(cfg):
    b = pd.DataFrame({"open": [100, 100, 100.0], "close": [100.5, 99.5, 100.0],
                      "delta_pct": [0.4, -0.3, 0.5]})
    f = flags(b, cfg, 0.05)
    assert f.delta_flip.tolist() == [0, -1, 1]
    assert f.delta_no_progress.tolist() == [False, False, True]  # big +delta, close == open


def test_delta_prefix_no_lookahead(cfg, day):
    cut = pd.Timestamp("2026-01-05 12:00", tz=IST)
    full = delta_bars(day.ticks, cfg)
    part = delta_bars(day.ticks[day.ticks.ts < cut], cfg)
    pd.testing.assert_frame_equal(part, full[full.bar < cut].reset_index(drop=True))


def test_divergence_prefix_no_lookahead(cfg, day):
    full_b = delta_bars(day.ticks, cfg)
    full = cvd_divergences(full_b, cfg)
    for hh in (11, 13, 14):
        cut = pd.Timestamp(f"2026-01-05 {hh}:00", tz=IST)
        pb = delta_bars(day.ticks[day.ticks.ts < cut], cfg)
        part = cvd_divergences(pb, cfg)
        exp = full[full.confirm_idx < len(pb)].reset_index(drop=True)
        pd.testing.assert_frame_equal(part, exp)
