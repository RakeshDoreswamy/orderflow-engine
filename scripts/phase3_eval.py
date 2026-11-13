"""Phase 3: score the detectors on HELD-OUT synthetic days and write docs/detector_scores.csv + a markdown table."""
from pathlib import Path
import pandas as pd
from orderflow.config import load_config
from orderflow.data.synthetic import generate_eval_days
from orderflow.evaluate import score_days

cfg = load_config()
days = generate_eval_days(cfg, int(cfg.eval.n_heldout_days), seed_base=int(cfg.eval.heldout_seed_base), n_events=6)
s = score_days(days, cfg)
out = Path("docs"); out.mkdir(exist_ok=True)
s.to_csv(out / "detector_scores.csv", index=False)
cols = [c for c in ["detector", "labelled", "found", "recall", "recall_ci95", "detections", "true_pos", "precision", "precision_ci95", "note"] if c in s.columns]
t = s[cols].copy()
for c in ("recall", "precision"):
    t[c] = t[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in t.values)
(out / "detector_scores.md").write_text(md + "\n")
print(md)
print("days:", len(days), "seed_base:", cfg.eval.heldout_seed_base)
