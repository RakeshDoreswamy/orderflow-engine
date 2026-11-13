import numpy as np
import pandas as pd
import pytest

from orderflow.classify import classify_ticks
from orderflow.config import load_config
from orderflow.data.synthetic import generate_day
from orderflow.recorder import (JsonlFeed, ListFeed, Recorder, load_day, load_instrument_master, normalize_snapshot, quality_report,
                                tick_size_for)

cfg = load_config()
TZ = cfg.market.timezone


def msg(t, price, cum, bid=100.0, ask=100.05, bsz=100, asz=200):
    return {"exchange_timestamp": f"2026-02-02 {t}+05:30", "last_price": price, "volume_traded": cum,
            "depth": {"buy": [{"price": bid, "quantity": bsz}, {"price": bid - 0.05, "quantity": 50}],
                      "sell": [{"price": ask, "quantity": asz}, {"price": ask + 0.05, "quantity": 60}]}}


def rec(tmp_path, **ov):
    return Recorder(tmp_path, "abc", cfg.with_overrides({"recorder": ov}) if ov else cfg, 0.05, source="test")


def test_trade_inferred_from_volume_increase_with_previous_quote(tmp_path):
    r = rec(tmp_path)
    r.on_message(msg("09:15:01", 100.0, 1000, bid=99.95, ask=100.0))
    r.on_message(msg("09:15:02", 100.0, 1040, bid=99.90, ask=100.05))   # post-trade quote moved
    r.on_message(msg("09:15:03", 100.05, 1040))                          # no volume change -> no trade
    st = r.close()
    t, b, m = load_day(tmp_path, "ABC", "2026-02-02")
    assert len(t) == 1 and t["qty"].iloc[0] == 40 and t["cum_volume"].iloc[0] == 1040
    assert (t["bid"].iloc[0], t["ask"].iloc[0]) == (99.95, 100.0)         # previous message's quote, not the later one
    assert st.trades == 1 and m["symbol"] == "ABC" and m["tick_size"] == 0.05


def test_drops_and_gaps_and_resets(tmp_path):
    r = rec(tmp_path, gap_alert_s=30)
    r.on_message(msg("09:10:00", 100, 10))                 # pre-open: out of session
    r.on_message(msg("09:15:10", 100, 100))
    r.on_message(msg("09:15:05", 100, 120))                # out of order
    r.on_message({"bad": 1})                               # malformed
    r.on_message(msg("09:16:10", 100, 150))                # 60 s of silence -> gap
    r.on_message(msg("09:16:11", 100, 20))                 # cumulative volume went backwards
    r.on_message(msg("09:16:12", 100, 60))                 # +40 after the reset -> a trade
    r.on_message(msg("15:30:00", 100, 999))                # at/after close
    st = r.close()
    t, b, m = load_day(tmp_path, "ABC", "2026-02-02")
    assert st.dropped_out_of_session == 2 and st.dropped_out_of_order == 1 and st.dropped_bad_message == 1 and st.volume_resets == 1
    assert len(m["gaps"]) == 1 and list(t["qty"]) == [50, 40]


def test_book_throttle_and_levels(tmp_path):
    r = rec(tmp_path, book_snapshot_sec=1.0)
    for i in range(10):
        r.on_message(msg(f"09:15:0{i}", 100, 100 + i, bsz=100 + i))
    r.close()
    _, b, _ = load_day(tmp_path, "ABC", "2026-02-02")
    assert len(b) == 10 and b["bid_sz_1"].iloc[3] == 103 and "ask_px_5" in b.columns and np.isnan(b["bid_px_3"].iloc[0])
    r2 = Recorder(tmp_path / "x", "ABC", cfg.with_overrides({"recorder": {"book_snapshot_sec": 5.0}}), 0.05)
    for i in range(10):
        r2.on_message(msg(f"09:15:0{i}", 100, 100 + i))
    r2.close()
    assert len(load_day(tmp_path / "x", "ABC", "2026-02-02")[1]) == 2


def test_multi_part_flush_and_atomic_names(tmp_path):
    r = rec(tmp_path, flush_rows=5)
    for i in range(30):
        r.on_message(msg(f"09:15:{i:02d}", 100, 100 + 10 * i))
    r.close()
    d = tmp_path / "ABC" / "2026-02-02"
    assert len(list(d.glob("ticks-*.parquet"))) >= 5 and not list(d.glob(".*.tmp"))
    t, _, m = load_day(tmp_path, "ABC", "2026-02-02")
    assert len(t) == 29 and t["ts"].is_monotonic_increasing and m["rows"]["ticks"] == 29


def test_normalize_requires_fields():
    with pytest.raises(KeyError):
        normalize_snapshot({"last_price": 1, "volume_traded": 1, "depth": {"buy": [], "sell": []}}, TZ)
    s = normalize_snapshot(msg("09:15:01", 100, 5), TZ)
    assert s["bids"][0] == (100.0, 100) and s["ts"].tzinfo is not None


def test_synthetic_day_roundtrip_through_snapshot_messages(tmp_path):
    d = generate_day(cfg)
    n = 3000
    tk = d.ticks.iloc[:n].reset_index(drop=True)
    bk = pd.merge_asof(tk[["ts"]], d.book.sort_values("ts"), on="ts", direction="backward")
    cum = tk["qty"].cumsum()
    msgs = []
    for i in range(n):
        r_ = bk.iloc[i]
        msgs.append({"exchange_timestamp": tk["ts"].iloc[i].isoformat(), "last_price": float(tk["price"].iloc[i]), "volume_traded": int(cum.iloc[i]),
                     "depth": {"buy": [{"price": float(r_[f"bid_px_{k}"]), "quantity": int(r_[f"bid_sz_{k}"])} for k in range(1, 6)],
                               "sell": [{"price": float(r_[f"ask_px_{k}"]), "quantity": int(r_[f"ask_sz_{k}"])} for k in range(1, 6)]}})
    r = Recorder(tmp_path, "SYN", cfg, 0.05)
    r.run(ListFeed(msgs))
    t, b, m = load_day(tmp_path, "SYN", d.date)
    assert len(t) == n - 1 and np.allclose(t["price"], tk["price"].iloc[1:]) and (t["qty"].to_numpy() == tk["qty"].iloc[1:].to_numpy()).all()
    # each trade carries the quote of the message BEFORE it
    assert np.allclose(t["bid"].to_numpy(), bk["bid_px_1"].iloc[:-1].to_numpy()) and np.allclose(t["ask"].to_numpy(), bk["ask_px_1"].iloc[:-1].to_numpy())
    est = classify_ticks(t, cfg)
    assert {"side_tick_est", "side_quote_est", "side_lr_est"} <= set(est.columns)
    q = quality_report(t, b, m, cfg)
    assert q["ts_monotonic"] and q["pct_prices_off_tick"] == 0 and q["trades"] == n - 1


def test_jsonl_feed_and_instrument_master(tmp_path):
    import json
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(msg(f"09:15:0{i}", 100, 100 + 5 * i)) for i in range(5)) + "\n")
    r = Recorder(tmp_path / "s", "ABC", cfg, 0.05)
    assert r.run(JsonlFeed(p)).trades == 4
    im = tmp_path / "inst.csv"
    im.write_text("instrument_token,tradingsymbol,tick_size,lot_size,exchange\n1,ABC,0.05,1,NSE\n2,ABC,0.01,1,BSE\n3,XYZ,0.1,1,NSE\n")
    mst = load_instrument_master(im)
    assert tick_size_for(mst, "abc") == 0.05 and tick_size_for(mst, "XYZ") == 0.1
    with pytest.raises(KeyError):
        tick_size_for(mst, "NOPE")


def test_load_missing_day(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_day(tmp_path, "ABC", "2026-02-02")
