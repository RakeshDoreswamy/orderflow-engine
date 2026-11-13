"""Score the EOD effort detector against planted labels (a CONSISTENCY test on toy data, not evidence about markets).
Recall = labelled episodes with >=1 detection of the right side in [start, end + grace_days].
Precision = detection episodes (detections of one side closer than gap_days merged) overlapping a labelled window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import swingflow as S
from .config import Config
from .data.synthetic_eod import SyntheticEOD
from .evaluate import wilson


def score_series(s: SyntheticEOD, cfg: Config) -> dict:
    ev = S.effort_events(S.flow_features(s.eod, cfg), cfg)
    grace, gap = pd.Timedelta(days=int(cfg.eod_eval.grace_days)), pd.Timedelta(days=int(cfg.eod_eval.gap_days))
    found = 0
    for r in s.labels.itertuples():
        side = r.kind.split("_")[-1]
        e = ev[(ev["side"] == side) & (ev["time"] >= r.start) & (ev["time"] <= r.end + grace)]
        found += int(len(e) > 0)
    units = tp = 0
    for side, g in ev.groupby("side"):
        t = g["time"].sort_values().reset_index(drop=True)
        start = [0] + [i for i in range(1, len(t)) if t[i] - t[i - 1] > gap]
        ends = start[1:] + [len(t)]
        for a, b in zip(start, ends):
            units += 1
            t0, t1 = t[a], t[b - 1]
            ok = ((s.labels["kind"] == f"absorption_at_{side}") & (s.labels["start"] <= t1) & (s.labels["end"] + grace >= t0)).any()
            tp += int(ok)
    return {"n_labels": len(s.labels), "n_found": found, "n_units": units, "n_tp": tp}


def score_many(series: list[SyntheticEOD], cfg: Config) -> dict:
    parts = pd.DataFrame([score_series(s, cfg) for s in series]).sum()
    rl, rh = wilson(int(parts.n_found), int(parts.n_labels))
    pl, ph = wilson(int(parts.n_tp), int(parts.n_units))
    return {"series": len(series), "labelled": int(parts.n_labels), "found": int(parts.n_found),
            "recall": parts.n_found / parts.n_labels if parts.n_labels else np.nan, "recall_ci95": f"{rl:.2f}-{rh:.2f}",
            "detection_episodes": int(parts.n_units), "true_pos": int(parts.n_tp),
            "precision": parts.n_tp / parts.n_units if parts.n_units else np.nan, "precision_ci95": f"{pl:.2f}-{ph:.2f}"}
