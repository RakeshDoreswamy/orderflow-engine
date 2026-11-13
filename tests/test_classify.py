import numpy as np

from orderflow.classify import (accuracy_report, classify_ticks, flip_sides, lee_ready, quote_rule,
                                tick_rule)


def test_tick_rule_hand_built():
    p = [100.0, 100.05, 100.05, 100.0, 100.0, 100.0, 100.10]
    assert tick_rule(p).tolist() == [0, 1, 1, -1, -1, -1, 1]


def test_quote_rule_hand_built():
    bid, ask = [99.95] * 5, [100.05] * 5
    price = [100.05, 99.95, 100.0, 100.05, 99.95]
    # at ask buy, at bid sell, exactly mid unclassified; same price rows repeat
    assert quote_rule(price, bid, ask).tolist() == [1, -1, 0, 1, -1]
    # inside the spread compares with mid (spread of 4 ticks)
    assert quote_rule([100.10], [100.00], [100.20]).tolist() == [0]
    assert quote_rule([100.15], [100.00], [100.20]).tolist() == [1]
    assert quote_rule([100.05], [100.00], [100.20]).tolist() == [-1]
    # missing quote -> unclassified
    assert quote_rule([100.0], [np.nan], [np.nan]).tolist() == [0]


def test_lee_ready_falls_back_to_tick_rule_at_mid():
    price = [100.0, 100.05, 100.0]
    bid, ask = [99.95, 99.95, 99.95], [100.05, 100.05, 100.05]
    # trade 3 is at mid (100.00): quote rule says 0 -> tick rule (down-tick) says -1
    assert lee_ready(price, bid, ask).tolist() == [0, 1, -1]


def test_prefix_does_not_change_earlier_labels(cfg, day):
    t = day.ticks.iloc[:4000]
    full = classify_ticks(t, cfg)
    for cut in (500, 1999, 3000):
        part = classify_ticks(t.iloc[:cut], cfg)
        for c in ("side_tick_est", "side_quote_est", "side_lr_est"):
            assert (part[c].to_numpy() == full[c].to_numpy()[:cut]).all(), (c, cut)


def test_accuracy_report_on_synthetic(cfg, day):
    rep = accuracy_report(classify_ticks(day.ticks, cfg)).set_index("method")
    # Thresholds are for THIS toy generator (stale_quote_prob=0.15), not for real NSE data.
    assert rep.loc["tick_rule", "acc_all"] > 0.85
    assert rep.loc["quote_rule", "acc_all"] > 0.97
    assert rep.loc["lee_ready", "coverage"] == 1.0
    assert rep.loc["lee_ready", "acc_all"] >= rep.loc["tick_rule", "acc_all"]


def test_flip_sides_rate_and_seed():
    s = np.ones(100000, dtype=np.int8)
    a = flip_sides(s, 0.1, np.random.default_rng(1))
    b = flip_sides(s, 0.1, np.random.default_rng(1))
    assert (a == b).all()
    assert abs((a == -1).mean() - 0.1) < 0.005
