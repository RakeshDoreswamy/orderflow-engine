"""Footprint chart (bid x ask cells). Title says 'estimated' whenever sides are estimated."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def plot_footprint(fp: pd.DataFrame, tick_size: float, row_ticks: int, out_path: str | Path,
                   estimated: bool = False, title: str = "", subtitle: str = "") -> Path:
    """Draw one column per bar, one cell per price row, text ``bid x ask``.
    Cell shade = ask share (green: buyers aggressed, red: sellers aggressed).
    Black box = highest-volume row of the bar (simple POC; the full POC rule comes in Phase 2)."""
    sx = "_est" if estimated else ""
    bars = sorted(fp["bar"].unique())
    row_h = tick_size * row_ticks
    prices = sorted(fp["row_price"].unique())
    fig, ax = plt.subplots(figsize=(1.9 * len(bars) + 1.5, 0.42 * len(prices) + 2.2))
    for i, b in enumerate(bars):
        sub = fp[fp["bar"] == b]
        poc = sub.loc[sub["total_vol"].idxmax(), "row_price"]
        for _, r in sub.iterrows():
            bid, ask = r[f"bid_vol{sx}"], r[f"ask_vol{sx}"]
            share = ask / (ask + bid) if (ask + bid) else 0.5
            color = plt.cm.RdYlGn(0.15 + 0.7 * share)
            ax.add_patch(Rectangle((i - 0.45, r["row_price"]), 0.9, row_h, facecolor=color, alpha=0.55,
                                   edgecolor="white", lw=0.6))
            ax.text(i, r["row_price"] + row_h / 2, f"{int(bid)} x {int(ask)}", ha="center", va="center",
                    fontsize=7.5)
            if r["row_price"] == poc:
                ax.add_patch(Rectangle((i - 0.45, poc), 0.9, row_h, fill=False, edgecolor="black", lw=1.6))
        d = int(sub[f"delta{sx}"].sum())
        ax.text(i, max(prices) + row_h * 1.4, f"{pd.Timestamp(b):%H:%M}\nΔ{'_est' if estimated else ''} {d:+d}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(min(prices) - row_h, max(prices) + row_h * 3.6)
    ax.set_xticks([])
    ax.set_ylabel("price (₹)")
    kind = "ESTIMATED sides" if estimated else "true aggressor side (synthetic data)"
    ax.set_title(f"{title or 'Footprint'}  |  {kind}\n{subtitle}", fontsize=10)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
