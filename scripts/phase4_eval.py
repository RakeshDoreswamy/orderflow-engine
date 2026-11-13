"""Phase 4: score the EOD effort detector on HELD-OUT synthetic series -> docs/eod_scores.csv / .md and a demo chart."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow import swingflow as S  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic_eod import SyntheticEOD, generate_eod, generate_eod_series  # noqa: E402
from orderflow.evaluate_eod import score_many  # noqa: E402

root = Path(__file__).resolve().parents[1]
cfg = load_config()
e = cfg.eod_eval
held = generate_eod_series(cfg, int(e.n_heldout_series), int(e.heldout_seed_base))
rows = [{"variant": "volume + delivery %", **score_many(held, cfg)},
        {"variant": "volume only (delivery % dropped)", **score_many([SyntheticEOD(s.eod.drop(columns=["delivery_pct", "delivery_qty"]), s.labels) for s in held], cfg)}]
t = pd.DataFrame(rows)
(root / "docs").mkdir(exist_ok=True)
t.to_csv(root / "docs" / "eod_scores.csv", index=False)
fmt = t.copy()
for c in ("recall", "precision"):
    fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")
md = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in fmt.values)
(root / "docs" / "eod_scores.md").write_text(md + "\n")
print(md)

# demo chart on one synthetic series (seed 7)
g = generate_eod(cfg)
f = S.flow_features(g.eod, cfg)
ev = S.effort_events(f, cfg)
sw = S.swing_flow(f, cfg)
sl = slice(0, 260)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
d = f.iloc[sl]
a1.plot(d["date"], d["close"], color="#333", lw=1)
for r in g.labels.itertuples():
    if r.start <= d["date"].iloc[-1]:
        a1.axvspan(r.start, r.end, color="#f1c40f", alpha=0.35)
for side, col, mk in (("low", "#27ae60", "^"), ("high", "#c0392b", "v")):
    x = ev[(ev["side"] == side) & (ev["time"] <= d["date"].iloc[-1])]
    a1.scatter(x["time"], x["price"], c=col, marker=mk, s=30, zorder=3, label=f"effort without result near prior {side}")
h = sw[(sw["heavy"]) & (sw["idx"] < sl.stop)]
for r in h.itertuples():
    a1.annotate(r.read.replace("_", " "), (r.swing_date, r.price), fontsize=6, xytext=(0, 12 if r.kind == "high" else -14), textcoords="offset points", ha="center")
a1.set_title("Phase 4: EOD effort-without-result detector on a TOY series (SYNTHETIC data). Yellow = planted episodes. "
             "Volume/delivery-based estimates only", fontsize=9)
a1.legend(fontsize=8)
a2.bar(d["date"], d["vol_z"], color="#7f8c8d", width=1)
a2.axhline(cfg.eod.effort_vol_z, color="r", lw=0.7, ls=":")
a2.set_ylabel("volume z-score")
(root / "docs" / "img").mkdir(parents=True, exist_ok=True)
fig.savefig(root / "docs" / "img" / "phase4_swingflow.png", dpi=120, bbox_inches="tight")
