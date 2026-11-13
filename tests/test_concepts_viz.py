"""Smoke tests: each concept visual renders a non-trivial PNG and returns sane facts."""
import pandas as pd
import pytest

from orderflow.data.synthetic import generate_days
from orderflow.viz import concepts as C


@pytest.fixture(scope="module")
def days(cfg):
    return generate_days(cfg, 3)


def _ok(res, tmp_path_file):
    p, facts = res
    assert p.exists() and p.stat().st_size > 8000
    assert isinstance(facts, dict)
    return facts


def test_single_day_visuals(cfg, day, tmp_path):
    for name, fn in {
        "c01": lambda o: C.c01_auction(day, cfg, o), "c04": lambda o: C.c04_aggressor(day, o),
        "c06": lambda o: C.c06_tape(day, o), "c09": lambda o: C.c09_delta_bars(day, cfg, o),
        "c10": lambda o: C.c10_cvd(day, cfg, o), "c11": lambda o: C.c11_flip(day, cfg, o),
        "c13": lambda o: C.c13_poc_va(day, cfg, o), "c14": lambda o: C.c14_imbalance(day, cfg, o),
        "c16": lambda o: C.c16_absorption(day, cfg, o), "c17": lambda o: C.c17_trapped(day, cfg, o),
        "c18": lambda o: C.c18_profile(day, cfg, o), "c20": lambda o: C.c20_tpo(day, cfg, o),
        "c23": lambda o: C.c23_vwap(day, cfg, o),
    }.items():
        _ok(fn(tmp_path / f"{name}.png"), name)


def test_illustrative_and_multiday_visuals(cfg, day, days, tmp_path):
    _ok(C.c02_dom(cfg, tmp_path / "2.png"), 2)
    _ok(C.c03_limit_market(cfg, tmp_path / "3.png"), 3)
    _ok(C.c05_slippage(tmp_path / "5.png"), 5)
    f = _ok(C.c08_ohlcv(day, cfg, tmp_path / "8.png"), 8)
    assert -1 <= f["corr"] <= 1
    f = _ok(C.c19_shapes(cfg, tmp_path / "19.png"), 19)
    assert f["labels"] == {"D (balance)": "D", "P (short-covering)": "P", "b (long-liquidation)": "b",
                           "double distribution": "double", "thin / trend": "thin"}
    _ok(C.c15_finished(days, cfg, tmp_path / "15.png"), 15)
    _ok(C.c21_composite(days, cfg, tmp_path / "21.png"), 21)
    f = _ok(C.c22_open(days, cfg, tmp_path / "22.png"), 22)
    assert f["n_days"] == 3


def test_flat_absorption_has_one_tick_range(cfg, day, tmp_path):
    f = _ok(C.c16_absorption(day, cfg, tmp_path / "a.png"), "a")
    assert f["range_ticks"] <= 2 and f["sold"] > 3 * f["bought"]


def test_multi_day_generator_is_continuous_and_seeded(cfg):
    a, b = generate_days(cfg, 3), generate_days(cfg, 3)
    pd.testing.assert_frame_equal(a[1].ticks, b[1].ticks)
    assert abs(a[1].ticks.price.iloc[0] - a[0].ticks.price.iloc[-1]) < 1.0   # next day starts near prior close
