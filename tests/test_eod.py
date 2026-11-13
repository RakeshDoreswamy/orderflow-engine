import numpy as np
import pandas as pd
import pytest

from orderflow import swingflow as S
from orderflow.config import load_config
from orderflow.data.providers import CsvProvider, build_series_from_bhavcopies, parse_bhavcopy, validate_eod
from orderflow.data.synthetic_eod import generate_eod, generate_eod_series
from orderflow.evaluate_eod import score_many

cfg = load_config()


@pytest.fixture(scope="module")
def g():
    return generate_eod(cfg)


def test_synthetic_eod_is_valid_and_deterministic(g):
    validate_eod(g.eod)
    again = generate_eod(cfg)
    pd.testing.assert_frame_equal(g.eod, again.eod)
    assert len(g.labels) >= 10 and (g.labels["end"] >= g.labels["start"]).all()


def test_validate_rejects_bad_frames():
    d = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "open": [10, 10], "high": [9, 11], "low": [8, 9], "close": [10, 10], "volume": [1, 1]})
    with pytest.raises(ValueError):
        validate_eod(d)  # high below open/close
    with pytest.raises(ValueError):
        validate_eod(d.drop(columns="volume"))
    dup = pd.DataFrame({"date": ["2024-01-01"] * 2, "open": [10, 10], "high": [11, 11], "low": [9, 9], "close": [10, 10], "volume": [1, 1]})
    with pytest.raises(ValueError):
        validate_eod(dup)


def test_csv_provider_and_missing_file(tmp_path):
    (tmp_path / "ABC.csv").write_text("Date,Open,High,Low,Close,Volume,DELIV_PER\n2024-01-02,10,11,9,10.5,1000,45.5\n2024-01-01,9,10,8.5,10,900,-\n")
    df = CsvProvider(tmp_path).fetch("abc")
    assert list(df["date"].dt.day) == [1, 2] and np.isnan(df["delivery_pct"].iloc[0]) and df["delivery_pct"].iloc[1] == 45.5
    assert len(CsvProvider(tmp_path).fetch("ABC", start="2024-01-02")) == 1
    with pytest.raises(FileNotFoundError):
        CsvProvider(tmp_path).fetch("NOPE")


def test_bhavcopy_parse_hand_built(tmp_path):
    f1 = tmp_path / "a.csv"
    f1.write_text("SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER\n"
                  "ABC, EQ, 01-Jan-2024, 100, 105, 98, 104, 5000, 2500, 50.00\nABC, BE, 01-Jan-2024, 1, 1, 1, 1, 1, 1, 1\nXYZ, EQ, 01-Jan-2024, 5, 6, 4, 5, 10, -, -\n")
    f2 = tmp_path / "b.csv"
    f2.write_text("SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER\n"
                  "ABC, EQ, 02-Jan-2024, 104, 108, 103, 107, 6000, 3600, 60.00\n")
    one = parse_bhavcopy(f1, "ABC")
    assert len(one) == 1 and one["close"].iloc[0] == 104 and one["delivery_pct"].iloc[0] == 50
    s = build_series_from_bhavcopies([f2, f1], "ABC")
    assert list(s["close"]) == [104, 107] and list(s["date"].dt.day) == [1, 2]
    with pytest.raises(ValueError):
        build_series_from_bhavcopies([f1], "NOSUCH")


def test_features_are_causal(g):
    """Row t must be identical whether or not later rows exist."""
    n = 300
    full = S.flow_features(g.eod, cfg).iloc[:n]
    pre = S.flow_features(g.eod.iloc[:n], cfg)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), pre.reset_index(drop=True))


def test_detections_prefix_stable(g):
    n = 300
    full = S.effort_events(S.flow_features(g.eod, cfg), cfg)
    pre = S.effort_events(S.flow_features(g.eod.iloc[:n], cfg), cfg)
    cut = g.eod["date"].iloc[n - 1]
    a, b = full[full["time"] <= cut].reset_index(drop=True), pre.reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_swings_known_only_after_confirmation(g):
    f = S.flow_features(g.eod, cfg)
    sw = S.swing_flow(f, cfg)
    assert (sw["confirm_idx"] == sw["idx"] + cfg.eod.swing_right).all() and (sw["known_on"] > sw["swing_date"]).all()
    n = 300
    sw_pre = S.swing_flow(S.flow_features(g.eod.iloc[:n], cfg), cfg)
    full_known = sw[sw["confirm_idx"] < n].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_known.drop(columns=[]), sw_pre.reset_index(drop=True))


def test_hand_built_effort_day():
    n, i = 60, 50
    rng = np.random.default_rng(1)
    close = np.concatenate([np.linspace(110, 96, i), [95.9] * (n - i)])  # steady slide, then a flat day at the low
    d = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "close": close})
    d["open"] = d["close"].shift(1).fillna(d["close"])
    d["high"] = d[["open", "close"]].max(axis=1) + 0.6
    d["low"] = d[["open", "close"]].min(axis=1) - 0.6
    d["volume"] = 1e6 * np.exp(rng.normal(0, 0.1, n))
    d["delivery_pct"] = 45 + rng.normal(0, 2, n)
    d.loc[i, ["open", "high", "low", "volume", "delivery_pct"]] = [95.95, 96.05, 95.8, 5e6, 75]
    d.loc[i + 1:, ["volume", "delivery_pct"]] = [1e6, 45]
    d = d.iloc[: i + 1]
    ev = S.effort_events(S.flow_features(d, cfg), cfg)
    hit = ev[ev["time"] == d["date"].iloc[i]]
    assert len(hit) == 1 and hit["side"].iloc[0] == "low"
    assert not (ev["time"] < d["date"].iloc[i]).any()


def test_score_floor_on_calibration_series():
    e = cfg.eod_eval
    r = score_many(generate_eod_series(cfg, 3, int(e.calibration_seed_base)), cfg)
    assert r["labelled"] > 20 and r["recall"] >= 0.7 and r["precision"] >= 0.6
