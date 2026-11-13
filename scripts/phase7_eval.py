"""Phase 7: end-to-end null check of the ICT bridge on toy EOD series, plus a demo scan.csv run.

Entries are placed every `step` days on random-walk series (no edge by construction). Outcome R = direction x
(close[t+h] - close[t]) / prior ATR, measured strictly AFTER the as_of day. If the bridge manufactured signal, confirm would beat
reject here. Writes docs/phase7_bridge_null.json/.md."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow import swingflow as S  # noqa: E402
from orderflow.bridge_eval import compare_outcomes  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic_eod import generate_eod_series  # noqa: E402
from orderflow.ict_bridge import verdict_eod  # noqa: E402

root = Path(__file__).resolve().parents[1]
cfg = load_config()
H, STEP, N = 5, 7, 40
rng = np.random.default_rng(21)
rows = []
for si, s in enumerate(generate_eod_series(cfg, N, int(cfg.eod_eval.heldout_seed_base) + 100)):
    f = S.flow_features(s.eod, cfg)
    for k in range(60, len(s.eod) - H - 1, STEP):
        side = "long" if rng.random() < 0.5 else "short"
        sweep = s.eod["date"].iloc[k - 3]
        v = verdict_eod({"side": side, "sweep_time": sweep}, s.eod.iloc[:k + 1], cfg, s.eod["date"].iloc[k])
        d = 1 if side == "long" else -1
        atr = f["atr_prior"].iloc[k]
        if not np.isfinite(atr) or atr <= 0:
            continue
        R = d * (s.eod["close"].iloc[k + H] - s.eod["close"].iloc[k]) / atr
        rows.append({"series": si, "side": side, "verdict": v["verdict"], "score": v["score"], "R": R})
df = pd.DataFrame(rows)
res = compare_outcomes(df, cfg)
bv = res["by_verdict"].copy()
bv["ci95"] = bv["ci95"].map(lambda t: f"{t[0]:.2f} to {t[1]:.2f}")
bv["mean_R"] = bv["mean_R"].map(lambda x: f"{x:.2f}")
md = "| " + " | ".join(bv.columns) + " |\n|" + "---|" * len(bv.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in bv.values)
(root / "docs" / "phase7_bridge_null.md").write_text(md + "\n")
(root / "docs" / "phase7_bridge_null.json").write_text(json.dumps({"entries": len(df), "difference": res["difference"]}, indent=1, default=str))
print(md)
print(res["difference"], len(df))
