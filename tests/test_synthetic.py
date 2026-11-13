import numpy as np
import pandas as pd

from orderflow.data.synthetic import generate_day


def test_deterministic_given_seed(cfg):
    a, b = generate_day(cfg, seed=7), generate_day(cfg, seed=7)
    pd.testing.assert_frame_equal(a.ticks, b.ticks)
    assert not generate_day(cfg, seed=8).ticks.equals(a.ticks)


def test_prices_on_tick_grid_and_inside_session(cfg, day):
    t = day.ticks
    ticks = t["price"] / day.tick_size
    assert np.allclose(ticks, np.rint(ticks), atol=1e-6)
    assert t["ts"].is_monotonic_increasing
    assert t["ts"].min() >= pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")
    assert t["ts"].max() <= pd.Timestamp("2026-01-05 15:30", tz="Asia/Kolkata")
    assert set(t["side_true"].unique()) <= {-1, 1}
    assert (t["qty"] >= 1).all()


def test_true_side_matches_trade_price_vs_true_quote(day):
    # Without stale quotes, buys print at the ask and sells at the bid. Stale quotes may differ,
    # so check the large majority (stale_quote_prob = 0.15 in config).
    t = day.ticks
    buy_at_ask = (t["price"] == t["ask"])[t["side_true"] > 0].mean()
    sell_at_bid = (t["price"] == t["bid"])[t["side_true"] < 0].mean()
    assert buy_at_ask > 0.75 and sell_at_bid > 0.75


def test_events_are_labelled_and_behave(day):
    ev = day.events.set_index("kind")
    assert set(ev.index) == {"burst_up", "absorption_bid", "sweep_reversal", "absorption_ask"}
    t = day.ticks
    up = t[t["event_id"] == ev.loc["burst_up", "event_id"]]
    assert up["side_true"].mean() > 0.6            # one-sided buying
    ab = t[t["event_id"] == ev.loc["absorption_bid", "event_id"]]
    assert ab["side_true"].mean() < -0.4           # heavy aggressive selling ...
    assert ab["price"].max() - ab["price"].min() <= 2 * day.tick_size  # ... with no price progress
    aa = t[t["event_id"] == ev.loc["absorption_ask", "event_id"]]
    assert aa["side_true"].mean() > 0.4
    assert aa["price"].max() - aa["price"].min() <= 2 * day.tick_size
