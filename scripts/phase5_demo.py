"""Phase 5 demo: replay a synthetic day through the recorder as broker-style snapshot messages, read it back, and
measure what the snapshot format costs. Writes docs/recorder_demo.json. SYNTHETIC data."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.classify import classify_ticks  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic import generate_day  # noqa: E402
from orderflow.recorder import ListFeed, Recorder, load_day, quality_report  # noqa: E402

cfg = load_config()
d = generate_day(cfg)
tk = d.ticks.reset_index(drop=True)
bk = pd.merge_asof(tk[["ts"]], d.book.sort_values("ts"), on="ts", direction="backward")
cum = tk["qty"].cumsum().to_numpy()
cols = {k: bk[k].to_numpy() for k in bk.columns if k != "ts"}
msgs = [{"exchange_timestamp": tk["ts"].iloc[i].isoformat(), "last_price": float(tk["price"].iloc[i]), "volume_traded": int(cum[i]),
         "depth": {"buy": [{"price": float(cols[f"bid_px_{k}"][i]), "quantity": int(cols[f"bid_sz_{k}"][i])} for k in range(1, 6)],
                   "sell": [{"price": float(cols[f"ask_px_{k}"][i]), "quantity": int(cols[f"ask_sz_{k}"][i])} for k in range(1, 6)]}}
        for i in range(len(tk))]
with tempfile.TemporaryDirectory() as td:
    st = Recorder(td, "SYN", cfg, d.tick_size, source="synthetic replay").run(ListFeed(msgs))
    t, b, m = load_day(td, "SYN", d.date)
q = quality_report(t, b, m, cfg)
est = classify_ticks(t, cfg)
truth = tk["side_true"].iloc[1:].to_numpy()        # the first message only seeds the previous state
vol = t["qty"].to_numpy(float)
res = {}
for name, col in (("tick_rule", "side_tick_est"), ("quote_rule", "side_quote_est"), ("lee_ready", "side_lr_est")):
    s = est[col].to_numpy()
    res[name] = {"coverage": float((s != 0).mean()), "accuracy_all": float((s == truth).mean()), "accuracy_all_vol": float(((s == truth) * vol).sum() / vol.sum())}
orig = classify_ticks(tk, cfg)
res_gen = {n: float((orig[c].to_numpy() == tk["side_true"].to_numpy()).mean()) for n, c in (("tick_rule", "side_tick_est"), ("quote_rule", "side_quote_est"), ("lee_ready", "side_lr_est"))}
out = {"messages": len(msgs), "trades_recorded": int(len(t)), "book_snapshots": int(len(b)), "stats": {k: v for k, v in st.__dict__.items() if k != "gaps"},
       "quality": q, "accuracy_recorded_quotes": res, "accuracy_generator_quotes_all": res_gen}
Path(__file__).resolve().parents[1].joinpath("docs", "recorder_demo.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
