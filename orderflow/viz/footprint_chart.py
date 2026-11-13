"""Footprint chart (bid x ask, delta and volume modes). Title says 'estimated' whenever sides are estimated."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

GREEN, RED = "#1e8449", "#c0392b"


def draw_footprint(ax, fp: pd.DataFrame, tick_size: float, row_ticks: int, estimated: bool = False,
                   mode: str = "bidask", analysis: tuple | None = None, show_va: bool = False,
                   show_imb: bool = False, show_status: bool = False, notes: list | None = None) -> None:
    """Draw one column per bar and one cell per price row.
    mode: 'bidask' (text bid x ask, shade = ask share), 'delta' (ask-bid), 'volume' (total).
    analysis: (bars, rows) from footprint.analyze_footprint; enables the true POC box, value-area band
    (show_va), imbalance borders (show_imb: green buy / red sell, thick = stacked) and finished/unfinished
    labels (show_status). notes: [(bar_index, price, text)] annotations."""
    sx = "_est" if estimated else ""
    bars = sorted(fp["bar"].unique())
    row_h = tick_size * row_ticks
    prices = sorted(fp["row_price"].unique())
    vmax = fp["total_vol"].max()
    dmax = max(fp[f"delta{sx}"].abs().max(), 1)
    if analysis:
        abars, arows = analysis
        arows = arows.set_index(["bar", "row_price"])
        abars = abars.set_index("bar")
    for i, b in enumerate(bars):
        sub = fp[fp["bar"] == b]
        poc = abars.loc[b, "poc"] if analysis else sub.loc[sub["total_vol"].idxmax(), "row_price"]
        for _, r in sub.iterrows():
            bid, ask, tot, dl = int(r[f"bid_vol{sx}"]), int(r[f"ask_vol{sx}"]), int(r["total_vol"]), int(r[f"delta{sx}"])
            if mode == "delta":
                color, txt = plt.cm.RdYlGn(0.5 + 0.45 * dl / dmax), f"{dl:+d}"
            elif mode == "volume":
                color, txt = plt.cm.Blues(0.1 + 0.6 * tot / vmax), f"{tot}"
            else:
                share = ask / (ask + bid) if (ask + bid) else 0.5
                color, txt = plt.cm.RdYlGn(0.15 + 0.7 * share), f"{bid} x {ask}"
            fl = arows.loc[(b, r["row_price"])] if analysis else None
            if show_va and analysis and fl.in_va:
                ax.add_patch(Rectangle((i - 0.5, r["row_price"]), 1.0, row_h, facecolor="#6baed6", alpha=0.35, lw=0))
            ax.add_patch(Rectangle((i - 0.45, r["row_price"]), 0.9, row_h, facecolor=color, alpha=0.55,
                                   edgecolor="white", lw=0.6))
            bold = False
            if show_imb and analysis:
                for flag, stacked, col in (("buy_imb", "stacked_buy", GREEN), ("sell_imb", "stacked_sell", RED)):
                    if fl[flag]:
                        bold = True
                        ax.add_patch(Rectangle((i - 0.45, r["row_price"]), 0.9, row_h, fill=False, edgecolor=col,
                                               lw=3.0 if fl[stacked] else 1.2))
            ax.text(i, r["row_price"] + row_h / 2, txt, ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if bold else "normal")
            if r["row_price"] == poc:
                ax.add_patch(Rectangle((i - 0.45, poc), 0.9, row_h, fill=False, edgecolor="black", lw=1.6))
        head = f"{pd.Timestamp(b):%H:%M}\nΔ{sx} {int(sub[f'delta{sx}'].sum()):+d}"
        if show_status and analysis:
            head += f"\nH:{'fin' if abars.loc[b, 'high_finished'] else 'unfin'} L:{'fin' if abars.loc[b, 'low_finished'] else 'unfin'}"
        ax.text(i, max(prices) + row_h * 1.3, head, ha="center", va="bottom", fontsize=8)
    for (bi, price, text) in notes or []:
        ax.annotate(text, xy=(bi, price + row_h / 2), xytext=(len(bars) - 0.35, price + row_h * 2.2), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=1), bbox=dict(boxstyle="round", fc="#fff8dc", ec="#999"))
    ax.set_xlim(-0.6, len(bars) - 0.4 + (2.0 if notes else 0))
    ax.set_ylim(min(prices) - row_h, max(prices) + row_h * 4.6)
    ax.set_xticks([])
    ax.set_ylabel("price (₹)")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


def plot_footprint(fp: pd.DataFrame, tick_size: float, row_ticks: int, out_path: str | Path,
                   estimated: bool = False, title: str = "", subtitle: str = "", **kw) -> Path:
    """One footprint figure. Black box = POC (bar's highest-volume row, or the analysed POC)."""
    n_b, n_p = fp["bar"].nunique(), fp["row_price"].nunique()
    fig, ax = plt.subplots(figsize=(1.9 * n_b + 1.5 + (2.8 if kw.get("notes") else 0), 0.42 * n_p + 2.4))
    draw_footprint(ax, fp, tick_size, row_ticks, estimated, **kw)
    kind = "ESTIMATED sides" if estimated else "true aggressor side (synthetic data)"
    ax.set_title(f"{title or 'Footprint'}  |  {kind}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
