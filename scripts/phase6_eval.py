"""Phase 6: backtest the setups (costs on) on HELD-OUT synthetic days, run both controls, and run the 80% rule and
gap-fill harnesses on toy data. Writes docs/phase6_*.csv/md. Everything here is SYNTHETIC: it validates plumbing."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.backtest.controls import control_bars  # noqa: E402
from orderflow.backtest.engine import run_backtest, summarize  # noqa: E402
from orderflow.backtest.setups import build_signals, day_bars  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic import generate_eval_days  # noqa: E402
from orderflow.data.synthetic_eod import generate_eod_series  # noqa: E402
from orderflow.rules import gap_fill, rule80, toy_gapped_days  # noqa: E402

root = Path(__file__).resolve().parents[1]
docs = root / "docs"
cfg = load_config()
sc = cfg.setups


def md(df):
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.values)


def backtest(seed_base, n):
    days = generate_eval_days(cfg, n, seed_base=seed_base, n_events=6)
    bars, sigs = [], []
    for d in days:
        b = day_bars(d.ticks, cfg)
        bars.append(b)
        sigs.append(build_signals(d.ticks, b, cfg, d.tick_size))
    bars, sigs = pd.concat(bars, ignore_index=True), pd.concat(sigs, ignore_index=True)
    tr = run_backtest(bars, sigs, cfg, days[0].tick_size)
    return tr, len(sigs)


def table(tr, label):
    rows = []
    for name, g in [("ALL", tr)] + [(k, v) for k, v in tr.groupby("setup")]:
        s = summarize(g, cfg)
        if s["trades"] == 0:
            rows.append({"set": label, "setup": name, "trades": 0})
            continue
        rows.append({"set": label, "setup": name, "trades": s["trades"], "win_rate": f"{s['win_rate']:.2f}", "mean_gross": f"{s['mean_gross_per_trade']:.1f}",
                     "mean_cost": f"{s['mean_cost_per_trade']:.1f}", "mean_net": f"{s['mean_net_per_trade']:.1f}",
                     "net_ci95": f"{s['mean_net_ci95'][0]:.1f} to {s['mean_net_ci95'][1]:.1f}", "net_total": f"{s['net_total']:.0f}",
                     "max_dd": f"{s['max_drawdown']:.0f}", "note": "small sample" if s["small_sample"] else ""})
    return rows


rows = []
cal, ncs = backtest(int(sc.eval_calibration_seed_base), int(sc.n_calibration_days))
held, nhs = backtest(int(sc.eval_heldout_seed_base), int(sc.n_heldout_days))
rows += table(cal, f"calibration ({sc.n_calibration_days} days, seed {sc.eval_calibration_seed_base})")
rows += table(held, f"held-out ({sc.n_heldout_days} days, seed {sc.eval_heldout_seed_base})")
bt = pd.DataFrame(rows).fillna("")
bt.to_csv(docs / "phase6_backtest.csv", index=False)
(docs / "phase6_backtest.md").write_text(md(bt) + "\n")
held.to_csv(docs / "phase6_heldout_trades.csv", index=False)
print(md(bt))

ctl = []
for label, kw in [("negative control (no edge)", {}), ("positive control (planted drift 3 ticks/min x 15 min)", {"drift_ticks": 3})]:
    b, s = control_bars(cfg, n_days=30, seed=3, stop_ticks=30, target_ticks=45, **kw)
    r = summarize(run_backtest(b, s, cfg, 0.05), cfg)
    ctl.append({"control": label, "trades": r["trades"], "mean_gross": f"{r['mean_gross_per_trade']:.1f}", "mean_cost": f"{r['mean_cost_per_trade']:.1f}",
                "mean_net": f"{r['mean_net_per_trade']:.1f}", "net_ci95": f"{r['mean_net_ci95'][0]:.1f} to {r['mean_net_ci95'][1]:.1f}"})
ct = pd.DataFrame(ctl)
ct.to_csv(docs / "phase6_controls.csv", index=False)
(docs / "phase6_controls.md").write_text(md(ct) + "\n")
print(md(ct))

days80 = toy_gapped_days(cfg, 250)
t80, s80 = rule80(days80, cfg)
s80 = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in s80.items()}
(docs / "phase6_rule80.json").write_text(json.dumps(s80, indent=1, default=str))
print(s80)

eods = generate_eod_series(cfg, 20, int(cfg.eod_eval.heldout_seed_base))
gaps = []
for s in eods:
    gaps.append(gap_fill(s.eod, cfg)[0])
g = pd.concat(gaps)
rows_g = []
import numpy as np  # noqa: E402
from orderflow.evaluate import wilson  # noqa: E402
for (b, di), gg in g.groupby(["bucket", "dir"], observed=True):
    k, n = int(gg["filled"].sum()), len(gg)
    lo, hi = wilson(k, n)
    rows_g.append({"abs_gap_pct": str(b), "direction": di, "gaps": n, "filled": k, "fill_rate": f"{k / n:.2f}", "ci95": f"{lo:.2f}-{hi:.2f}"})
gt = pd.DataFrame(rows_g)
gt.to_csv(docs / "phase6_gapfill.csv", index=False)
(docs / "phase6_gapfill.md").write_text(md(gt) + "\n")
print(md(gt))
