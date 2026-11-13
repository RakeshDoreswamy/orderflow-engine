"""Phase 1 demo: generate a synthetic day, print the classifier accuracy report, render a footprint
around the labelled sweep_reversal event.  Run:  python scripts/phase1_demo.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.classify import accuracy_report, classify_ticks  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.bars import footprint  # noqa: E402
from orderflow.data.synthetic import generate_day  # noqa: E402
from orderflow.viz.footprint_chart import plot_footprint  # noqa: E402

cfg = load_config()
day = generate_day(cfg)
ticks = classify_ticks(day.ticks, cfg)
print(f"{len(ticks)} synthetic trades, seed={cfg.synthetic.seed}")
print(day.events.to_string(index=False))
rep = accuracy_report(ticks)
print("\nClassifier accuracy vs known truth (synthetic):")
print(rep.round(4).to_string(index=False))
rep.to_csv("docs/classifier_accuracy.csv", index=False)

ev = day.events[day.events["kind"] == "sweep_reversal"].iloc[0]
pad = pd.Timedelta(minutes=5)
win = ticks[(ticks["ts"] >= ev["start"] - pad) & (ticks["ts"] < ev["end"] + pad)]
rt = cfg.bars.demo_footprint_row_ticks
fp = footprint(win, cfg, tick_size=day.tick_size, row_ticks=rt)
out = plot_footprint(fp, day.tick_size, rt, "docs/img/footprint_synthetic_sweep.png",
                     title="Footprint 5-min: labelled sweep_reversal (down-burst -> absorption -> up-burst)",
                     subtitle=f"synthetic {day.symbol} {day.date}; cell = bid x ask volume")
print("wrote", out)
