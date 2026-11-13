import numpy as np
import pandas as pd
import pytest

from orderflow.bridge_eval import compare_outcomes
from orderflow.config import load_config
from orderflow.data.synthetic import generate_eval_days
from orderflow.data.synthetic_eod import generate_eod
from orderflow.ict_bridge import bridge_scan, read_scan_csv, verdict_eod, verdict_intraday

cfg = load_config()


@pytest.fixture(scope="module")
def g():
    return generate_eod(cfg)


def low_episode(g):
    r = g.labels[g.labels["kind"] == "absorption_at_low"].iloc[0]
    return r


def test_read_scan_csv(tmp_path):
    p = tmp_path / "scan.csv"
    p.write_text("symbol,grade,entry,stop,t1,t2,sweep_time,mss_time\nABC,A,100,98,104,108,2026-11-10 10:00,2026-11-10 11:30\n")
    df = read_scan_csv(p)
    assert df["side"].iloc[0] == "long" and pd.notna(df["sweep_time"].iloc[0])
    (tmp_path / "bad.csv").write_text("symbol,entry\nABC,100\n")
    with pytest.raises(ValueError):
        read_scan_csv(tmp_path / "bad.csv")
    (tmp_path / "bad2.csv").write_text("symbol,entry,stop,side\nABC,100,98,sideways\n")
    with pytest.raises(ValueError):
        read_scan_csv(tmp_path / "bad2.csv")


def test_no_data_when_history_short(g):
    v = verdict_eod({"side": "long"}, g.eod.iloc[:20], cfg)
    assert v["verdict"] == "no_data" and v["score"] == 0


def test_eod_long_sees_absorption_at_low_and_short_sees_it_against(g):
    r = low_episode(g)
    as_of = r["end"] + pd.Timedelta(days=1)
    vl = verdict_eod({"side": "long"}, g.eod, cfg, as_of)
    vs = verdict_eod({"side": "short"}, g.eod, cfg, as_of)
    assert "+1 effort without result near a prior low" in vl["evidence"]
    assert "-1 effort without result near a prior low" in vs["evidence"]


def test_eod_verdict_has_no_lookahead(g):
    for k in (150, 260, 400):
        as_of = g.eod["date"].iloc[k]
        e = {"side": "long", "sweep_time": g.eod["date"].iloc[k - 3]}
        full = verdict_eod(e, g.eod, cfg, as_of)
        cut = verdict_eod(e, g.eod.iloc[:k + 1], cfg, as_of)
        assert full == cut


def test_bridge_scan_handles_missing_symbol(g):
    ent = pd.DataFrame({"symbol": ["AAA", "BBB"], "entry": [1, 1], "stop": [0.9, 0.9], "side": ["long", "long"]})
    out = bridge_scan(ent, {"AAA": g.eod}, cfg, as_of=g.eod["date"].iloc[300])
    assert out["verdict"].iloc[1] == "no_data" and out["verdict"].iloc[0] in {"confirm", "neutral", "reject"}


@pytest.fixture(scope="module")
def day():
    return generate_eval_days(cfg, 3, seed_base=500, n_events=6)[1]   # seed 500 day 2 has an absorption_bid window


def test_intraday_confirms_after_absorption_and_not_before(day):
    ev = day.events[day.events["kind"] == "absorption_bid"].iloc[0]
    entry = {"side": "long", "sweep_time": ev["start"]}
    late = verdict_intraday(entry, day.ticks, cfg, day.tick_size, as_of=ev["end"] + pd.Timedelta(minutes=3))
    early = verdict_intraday(entry, day.ticks, cfg, day.tick_size, as_of=ev["start"] - pd.Timedelta(minutes=1))
    assert "absorption (bid)" in late["evidence"] and late["score"] >= 1
    assert "absorption" not in early["evidence"]
    short = verdict_intraday({"side": "short", "sweep_time": ev["start"]}, day.ticks, cfg, day.tick_size, as_of=ev["end"] + pd.Timedelta(minutes=3))
    assert "-1 absorption (bid)" in short["evidence"]


def test_intraday_prefix_equal_to_full(day):
    ev = day.events[day.events["kind"] == "absorption_bid"].iloc[0]
    as_of = ev["end"] + pd.Timedelta(minutes=3)
    e = {"side": "long", "sweep_time": ev["start"]}
    full = verdict_intraday(e, day.ticks, cfg, day.tick_size, as_of=as_of)
    cut = verdict_intraday(e, day.ticks[day.ticks["ts"] <= as_of], cfg, day.tick_size, as_of=as_of)
    assert full == cut


def test_compare_outcomes_null_and_positive_control():
    rng = np.random.default_rng(4)
    v = rng.choice(["confirm", "neutral", "reject"], 600)
    R = rng.normal(0, 1, 600)
    null = compare_outcomes(pd.DataFrame({"verdict": v, "R": R}), cfg)
    assert not null["difference"]["excludes_zero"]
    pos = compare_outcomes(pd.DataFrame({"verdict": v, "R": R + np.where(v == "confirm", 0.6, 0.0)}), cfg)
    assert pos["difference"]["excludes_zero"] and pos["difference"]["confirm_minus_reject"] > 0.3
    assert compare_outcomes(pd.DataFrame({"verdict": ["neutral"], "R": [1.0]}), cfg)["difference"] is None
