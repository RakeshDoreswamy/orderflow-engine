"""One generated visual per concept (1-23). Everything is drawn from SYNTHETIC data (or from clearly
labelled illustrative numbers) and says so on the chart. Each function returns (path, facts) where
``facts`` are measured numbers the study page quotes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..classify import accuracy_report, classify_ticks  # noqa: E402
from ..data.bars import footprint, time_bars  # noqa: E402
from ..data.synthetic import generate_day  # noqa: E402
from ..delta import cvd_divergences, delta_bars, flags, ohlcv_delta_est  # noqa: E402
from ..footprint import analyze_footprint  # noqa: E402
from ..profile import hvn_lvn, naked_pocs, profile_shape, value_area, volume_profile  # noqa: E402
from ..tpo import build_tpo, open_type  # noqa: E402
from ..vwap import vwap_series  # noqa: E402
from .footprint_chart import GREEN, RED, draw_footprint  # noqa: E402

plt.rcParams["axes.formatter.useoffset"] = False
TZ = "Asia/Kolkata"
SYN = "SYNTHETIC data, for teaching only"
ILL = "ILLUSTRATIVE numbers, not market data"
# Illustrative ladder (tick 0.05) used for concepts 2, 3, 5
ASKS = [(100.20, 1200), (100.15, 800), (100.10, 500), (100.05, 300)]
BIDS = [(100.00, 400), (99.95, 600), (99.90, 900), (99.85, 1300)]


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))


def _event(day, kind):
    return day.events[day.events["kind"] == kind].iloc[0]


def _window(ticks, start, end, pad_min=5):
    p = pd.Timedelta(minutes=pad_min)
    return ticks[(ticks["ts"] >= start - p) & (ticks["ts"] < end + p)]


def c01_auction(day, cfg, out):
    b = delta_bars(day.ticks, cfg)
    m = time_bars(day.ticks, cfg, "1min")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(m["bar"], m["close"], color="#333", lw=1)
    imb = day.events[day.events["kind"].isin(["burst_up", "burst_down", "sweep_reversal"])]
    for _, e in day.events.iterrows():
        one_sided = e["kind"] in ("burst_up", "burst_down")
        a1.axvspan(e["start"], e["end"], color=(RED if one_sided else "#2874a6"), alpha=0.18)
    a1.set_title(f"1. Balance vs imbalance  |  {SYN}\nshaded red = one-sided burst (imbalance), blue = absorption/sweep window; unshaded = rotation (balance)", fontsize=10)
    a2.bar(b["bar"], b["delta_pct"], width=0.0030, color=np.where(b["delta_pct"] >= 0, GREEN, RED))
    a2.axhline(0, color="#777", lw=0.6)
    a2.set_ylabel("delta % of volume")
    _time_axis(a2)
    return _save(fig, out), {"mean_abs_delta_pct": float(b["delta_pct"].abs().mean())}


def _ladder(ax, labels=True):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")
    rows = ASKS + BIDS
    mx = max(s for _, s in rows)
    for i, (p, s) in enumerate(rows):
        y = 7 - i
        col = RED if i < 4 else GREEN
        ax.add_patch(Rectangle((3.2, y - 0.4), 4.0 * s / mx, 0.8, color=col, alpha=0.5))
        ax.text(2.9, y, f"{p:.2f}", ha="right", va="center", fontsize=10)
        ax.text(3.3, y, str(s), ha="left", va="center", fontsize=9)
        if labels:
            ax.text(7.5, y, "ASK: sellers resting" if i == 0 else ("BID: buyers resting" if i == 4 else ""),
                    fontsize=8.5, va="center", color=col)
    ax.annotate("", xy=(1.5, 3.6), xytext=(1.5, 4.4), arrowprops=dict(arrowstyle="<->"))
    ax.text(1.4, 4.0, "spread\n1 tick", ha="right", va="center", fontsize=8)


def c02_dom(cfg, out):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    _ladder(ax)
    ax.set_title(f"2. The order book (DOM): best 4 levels per side\n{ILL}; tick 0.05; queue = price, then time", fontsize=10)
    return _save(fig, out), {}


def c03_limit_market(cfg, out):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _ladder(ax, labels=False)
    ax.annotate("MARKET BUY (aggressive)\nlifts the offer at 100.05:\ntrade happens, liquidity TAKEN", xy=(3.2, 4.0), xytext=(7.7, 5.7),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=GREEN, lw=2), bbox=dict(fc="#eafaf1", ec=GREEN))
    ax.annotate("LIMIT SELL resting (passive)\nwaits in queue, liquidity PROVIDED", xy=(7.0, 2.0), xytext=(7.7, 1.0),
                fontsize=9, arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), bbox=dict(fc="#fdedec", ec=RED))
    ax.set_xlim(-0.5, 13)
    ax.set_title(f"3. Limit vs market orders  |  {ILL}", fontsize=10)
    return _save(fig, out), {}


def c04_aggressor(day, out):
    ev = _event(day, "burst_up")
    t = day.ticks[day.ticks["ts"] >= ev["start"] - pd.Timedelta(seconds=30)].iloc[:60].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(t.index, t["ask"], where="post", color=RED, alpha=0.5, label="best ask (as attached)")
    ax.step(t.index, t["bid"], where="post", color=GREEN, alpha=0.5, label="best bid (as attached)")
    b, s = t[t["side_true"] > 0], t[t["side_true"] < 0]
    ax.scatter(b.index, b["price"], marker="^", s=18 + b["qty"] / 4, color=GREEN, label="buyer-initiated (side +1): at ask")
    ax.scatter(s.index, s["price"], marker="v", s=18 + s["qty"] / 4, color=RED, label="seller-initiated (side -1): at bid")
    ax.set_xlabel("trade number (marker size = quantity)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"4. Aggressor side: buys print at the ask, sells at the bid  |  {SYN}", fontsize=10)
    return _save(fig, out), {}


def c05_slippage(out):
    book = [(p, s) for p, s in ASKS[::-1]] + [(100.25, 1500), (100.30, 2000), (100.35, 2500)]
    sizes = np.arange(100, 6001, 100)
    avg = []
    for q in sizes:
        rem, cost = q, 0.0
        for p, s in book:
            t = min(rem, s)
            cost += t * p
            rem -= t
        avg.append((cost + rem * book[-1][0]) / q if rem == 0 else np.nan)
    slip_ticks = (np.array(avg) - 100.05) / 0.05
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(sizes, slip_ticks, color="#333")
    ax.set_xlabel("market buy size (shares)")
    ax.set_ylabel("slippage vs best ask (ticks)")
    ax.set_title(f"5. Slippage grows as the order walks the book  |  {ILL}", fontsize=10)
    ax.grid(alpha=0.3)
    return _save(fig, out), {}


def c06_tape(day, out):
    ev = _event(day, "burst_up")
    t = day.ticks[day.ticks["ts"] >= ev["start"] + pd.Timedelta(minutes=2)].iloc[:18].copy()
    big = day.ticks["qty"].quantile(0.95)
    cells = [[f"{r.ts:%H:%M:%S.%f}"[:-3], f"{r.price:.2f}", int(r.qty), "BUY" if r.side_true > 0 else "SELL"] for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=["time", "price", "qty", "aggressor"], loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    for i, r in enumerate(t.itertuples(), start=1):
        tb[i, 3].set_facecolor("#d5f5e3" if r.side_true > 0 else "#fadbd8")
        if r.qty >= big:
            tb[i, 2].set_facecolor("#fcf3cf")
    ax.set_title(f"6. The tape (Time & Sales)  |  {SYN}\nyellow qty = large print (top 5% of the day, >= {big:.0f})", fontsize=10)
    return _save(fig, out), {"large_print_threshold": float(big)}


def c07_classify(day, cfg, out):
    t = classify_ticks(day.ticks, cfg)
    rep = accuracy_report(t).set_index("method")
    probs = [0.0, 0.15, 0.30, 0.50]
    sens = {m: [] for m in rep.index}
    for sp in probs:
        d = generate_day(cfg.with_overrides({"synthetic": {"stale_quote_prob": sp}}))
        r = accuracy_report(classify_ticks(d.ticks, cfg)).set_index("method")
        for m in rep.index:
            sens[m].append(float(r.loc[m, "acc_all_vol"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(rep.index, rep["acc_all_vol"] * 100, color=["#7f8c8d", "#2874a6", "#1e8449"])
    for i, v in enumerate(rep["acc_all_vol"]):
        a1.text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=9)
    a1.set_ylim(80, 101)
    a1.set_ylabel("volume-weighted accuracy (%)")
    a1.set_title("at the config default (15% stale quotes)", fontsize=9)
    for m, v in sens.items():
        a2.plot(np.array(probs) * 100, np.array(v) * 100, marker="o", label=m)
    a2.set_xlabel("% of trades carrying a stale quote")
    a2.set_ylabel("accuracy (%)")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    a2.set_title("quote-based rules degrade when quotes are late", fontsize=9)
    fig.suptitle(f"7. Trade-side classifiers vs known truth  |  {SYN}; real NSE error must be measured on real ticks", fontsize=10)
    return _save(fig, out), {"accuracy_vol": {m: float(rep.loc[m, "acc_all_vol"]) for m in rep.index},
                             "stale_sens": {m: dict(zip(map(str, probs), v)) for m, v in sens.items()}}


def c08_ohlcv(day, cfg, out):
    b = delta_bars(day.ticks, cfg)
    est = ohlcv_delta_est(b)
    ok = est.notna()
    corr = float(np.corrcoef(est[ok], b["delta"][ok])[0, 1])
    agree = float((np.sign(est[ok]) == np.sign(b["delta"][ok])).mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(b["delta"][ok], est[ok], s=14, alpha=0.7)
    lim = max(b["delta"].abs().max(), est[ok].abs().max())
    ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.8)
    ax.set_xlabel("true delta (5-min bar)")
    ax.set_ylabel("delta_est from OHLCV")
    ax.set_title(f"8. OHLCV delta estimate vs truth  |  {SYN}\ncorrelation {corr:.2f}; same sign in {agree:.0%} of bars", fontsize=10)
    return _save(fig, out), {"corr": corr, "sign_agreement": agree, "n_bars": int(ok.sum())}


def c09_delta_bars(day, cfg, out):
    b = delta_bars(day.ticks, cfg).iloc[:36]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, b["delta"], color=np.where(b["delta"] >= 0, GREEN, RED), alpha=0.8)
    ax.vlines(x, b["delta_min"], b["delta_max"], color="#333", lw=1)
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_yscale("symlog", linthresh=5000)
    ax.set_ylabel("delta (shares, symlog scale)")
    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"{t:%H:%M}" for t in b["bar"].iloc[::3]])
    ax.set_title(f"9. Delta per bar; black whisker = running-delta max/min inside the bar  |  {SYN}", fontsize=10)
    return _save(fig, out), {}


def _price_cvd(ax1, ax2, b, dv=None):
    ax1.plot(b["bar"], b["close"], color="#333", lw=1.1)
    ax2.plot(b["bar"], b["cvd"], color="#2874a6", lw=1.1)
    ax2.axhline(0, color="#999", lw=0.5)
    ax2.set_ylabel("session CVD")
    if dv is not None:
        for r in dv.itertuples():
            col = RED if r.kind == "bearish" else GREEN
            f, s = b.iloc[r.first_idx], b.iloc[r.second_idx]
            pcol = "high" if r.kind == "bearish" else "low"
            ax1.plot([f["bar"], s["bar"]], [f[pcol], s[pcol]], color=col, lw=1.6)
            ax2.plot([f["bar"], s["bar"]], [f["cvd"], s["cvd"]], color=col, lw=1.6)
            ax1.axvline(b.iloc[r.confirm_idx]["bar"], color=col, ls=":", lw=1)
    _time_axis(ax2)


def c10_cvd(day, cfg, out):
    b = delta_bars(day.ticks, cfg)
    dv = cvd_divergences(b, cfg)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    _price_cvd(a1, a2, b, dv)
    a1.set_title(f"10. Price vs CVD with confirmed-swing divergences ({len(dv)} found)  |  {SYN}\nred = bearish (higher high, lower CVD), green = bullish; dotted line = bar where it became KNOWN", fontsize=10)
    return _save(fig, out), {"divergences": int(len(dv)), "bearish": int((dv.kind == "bearish").sum()),
                             "bullish": int((dv.kind == "bullish").sum())}


def c11_flip(day, cfg, out):
    b = flags(delta_bars(day.ticks, cfg), cfg, day.tick_size)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(b["bar"], b["close"], color="#333", lw=1)
    npg = b[b["delta_no_progress"]]
    a1.scatter(npg["bar"], npg["close"], marker="s", s=40, facecolors="none", edgecolors="#d68910", label="delta without progress")
    a1.legend(fontsize=8)
    a2.bar(b["bar"], b["delta_pct"], width=0.0030, color=np.where(b["delta_pct"] >= 0, GREEN, RED))
    fl = b[b["delta_flip"] != 0]
    a2.scatter(fl["bar"], fl["delta_pct"], marker="o", s=26, color="#8e44ad", label="delta flip", zorder=3)
    a2.legend(fontsize=8)
    a2.set_ylabel("delta %")
    _time_axis(a2)
    a1.set_title(f"11. Delta flips and delta without progress  |  {SYN}\n(exhaustion detector arrives in Phase 3 with the events module)", fontsize=10)
    return _save(fig, out), {"flips": int((b["delta_flip"] != 0).sum()), "no_progress": int(b["delta_no_progress"].sum())}


def _fp_window(day, cfg, kind, row_ticks, pad=5, bars=None):
    e = _event(day, kind)
    w = _window(day.ticks, e["start"], e["end"], pad)
    fp = footprint(w, cfg, tick_size=day.tick_size, row_ticks=row_ticks)
    return e, w, fp


def c12_fp_modes(day, cfg, out):
    e, w, fp = _fp_window(day, cfg, "burst_up", 20, pad=1)
    bars = sorted(fp["bar"].unique())[:3]
    fp = fp[fp["bar"].isin(bars)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    for ax, mode, name in zip(axs, ("bidask", "delta", "volume"), ("bid x ask", "delta", "volume")):
        draw_footprint(ax, fp, day.tick_size, 20, mode=mode)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"12. One footprint, three views (rows of 20 ticks = ₹1.00)  |  {SYN}", fontsize=10)
    return _save(fig, out), {}


def c13_poc_va(day, cfg, out):
    e, w, fp = _fp_window(day, cfg, "burst_up", 20, pad=1)
    fp = fp[fp["bar"].isin(sorted(fp["bar"].unique())[:3])]
    an = analyze_footprint(fp, cfg, day.tick_size, row_ticks=20)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw_footprint(ax, fp, day.tick_size, 20, mode="volume", analysis=an, show_va=True)
    ax.set_title(f"13. POC (black box) and 70% value area (blue band) per bar  |  {SYN}", fontsize=10)
    return _save(fig, out), {}


def c14_imbalance(day, cfg, out):
    e, w, fp = _fp_window(day, cfg, "burst_up", 10, pad=0)
    fp = fp[fp["bar"].isin(sorted(fp["bar"].unique())[:2])]
    an = analyze_footprint(fp, cfg, day.tick_size, row_ticks=10)
    fig, ax = plt.subplots(figsize=(7, 0.42 * fp["row_price"].nunique() + 2.4))
    draw_footprint(ax, fp, day.tick_size, 10, analysis=an, show_imb=True)
    n_stacked = int(an[1]["stacked_buy"].sum() + an[1]["stacked_sell"].sum())
    ax.set_title(f"14. Diagonal imbalances (ratio {cfg.footprint.imbalance_ratio}); thick border = stacked (>= {cfg.footprint.stacked_min_levels})\ngreen buy / red sell  |  {SYN}", fontsize=9)
    return _save(fig, out), {"stacked_rows_shown": n_stacked}


def c15_finished(days, cfg, out, show=5):
    """Only the extreme rows matter for finished/unfinished, so draw the top and bottom ``show`` traded rows
    of three consecutive bars (rows of 1 tick). Window = most finished extremes (capped at 2)."""
    best, pick, hi_all, lo_all = -1, None, [], []
    for d in days:
        fp1 = footprint(d.ticks, cfg, tick_size=d.tick_size, row_ticks=1)
        ab, _ = analyze_footprint(fp1, cfg, d.tick_size, row_ticks=1)
        hi_all += ab.high_finished.tolist()
        lo_all += ab.low_finished.tolist()
        score = (ab.high_finished.astype(int) + ab.low_finished.astype(int)).to_numpy()
        for i in range(len(ab) - 2):
            k = min(score[i:i + 3].sum(), 2)
            if k > best:
                best, pick = k, (d, fp1, ab.iloc[i:i + 3])
    d, fp1, ab = pick
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7.5))
    for ax, side in zip(axs, ("high", "low")):
        for i, r in enumerate(ab.itertuples()):
            g = fp1[fp1["bar"] == r.bar].sort_values("row_price", ascending=(side == "low"))
            g = g[g["total_vol"] > 0].head(show)
            fin = r.high_finished if side == "high" else r.low_finished
            for k, (_, c) in enumerate(g.iterrows()):
                y = -k if side == "high" else k
                extreme = k == 0
                col = "#d5f5e3" if (extreme and fin) else ("#fdebd0" if extreme else "#f4f6f6")
                ax.add_patch(Rectangle((i - 0.45, y - 0.45), 0.9, 0.9, facecolor=col, edgecolor="#1e8449" if (extreme and fin) else "#999", lw=2 if extreme else 0.6))
                ax.text(i, y, f"{int(c['bid_vol'])} x {int(c['ask_vol'])}", ha="center", va="center", fontsize=8.5, fontweight="bold" if extreme else "normal")
                ax.text(i - 0.47, y + 0.38, f"{c['row_price']:.2f}", fontsize=6.5, color="#555", va="top")
            top = 0.7 if side == "high" else show - 0.1
            ax.text(i, 0.75 if side == "high" else -0.75, f"{pd.Timestamp(r.bar):%H:%M}  {side} {'FINISHED' if fin else 'unfinished'}", ha="center",
                    va="bottom" if side == "high" else "top", fontsize=8.5, color="#1e8449" if fin else "#a04000")
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim((-show - 0.3, 1.5) if side == "high" else (-1.5, show + 0.1))
        ax.axis("off")
        ax.set_title("top rows of each bar (extreme row boxed)" if side == "high" else "bottom rows of each bar (extreme row boxed)", fontsize=9)
    fig.suptitle(f"15. Finished vs unfinished extremes  |  {SYN}, {d.date}; cells = bid x ask\nfinished high = ZERO ask volume at the high row; finished low = ZERO bid volume at the low row", fontsize=10)
    fig.tight_layout()
    return _save(fig, out), {"bars": len(hi_all), "finished_high_share": float(np.mean(hi_all)),
                             "finished_low_share": float(np.mean(lo_all)), "days": len(days)}


def c16_absorption(day, cfg, out):
    e, w, fp = _fp_window(day, cfg, "absorption_bid", 1, pad=0)
    fp = fp[fp["bar"].isin(sorted(fp["bar"].unique())[:3])]
    sub = w[(w["ts"] >= e["start"]) & (w["ts"] < e["end"])]
    sold, bought = int(sub[sub.side_true < 0].qty.sum()), int(sub[sub.side_true > 0].qty.sum())
    rng_t = (sub.price.max() - sub.price.min()) / day.tick_size
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    draw_footprint(a1, fp, day.tick_size, 1)
    a1.set_title("footprint, 5-min bars, rows of 1 tick", fontsize=9)
    run = (sub["side_true"] * sub["qty"]).cumsum()
    a2.plot(sub["ts"], sub["price"], color="#333", label="price")
    a2.set_ylabel("price")
    a3 = a2.twinx()
    a3.plot(sub["ts"], run, color=RED, label="cumulative delta")
    a3.set_ylabel("cumulative delta (shares)")
    a2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5), tz=TZ))
    _time_axis(a2)
    a2.set_title(f"sold {sold:,} vs bought {bought:,}; price range {rng_t:.0f} tick(s)", fontsize=9)
    fig.suptitle(f"16. Absorption: heavy aggressive SELLING, price does not fall  |  {SYN}\nlabelled by the generator; the detector is built in Phase 3", fontsize=10, y=1.08)
    return _save(fig, out), {"sold": sold, "bought": bought, "range_ticks": float(rng_t)}


def c17_trapped(day, cfg, out):
    e, w, fp = _fp_window(day, cfg, "sweep_reversal", 20, pad=5)
    ev = day.events[day.events["kind"] == "sweep_reversal"].iloc[0]
    sw = w[(w.ts >= ev["start"]) & (w.ts < ev["end"])]
    low = sw["price"].min()
    lb = fp[fp["row_price"] == fp["row_price"].min()]
    bar_low = lb.loc[lb["total_vol"].idxmax()]
    bars = sorted(fp["bar"].unique())
    bi = bars.index(bar_low["bar"])
    sold_low = int(bar_low["bid_vol"])
    after = sw[sw.ts >= sw.loc[sw["price"].idxmin(), "ts"]]
    ret = (after["price"].max() - low) / day.tick_size
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    draw_footprint(ax, fp, day.tick_size, 20, notes=[(bi, bar_low["row_price"], f"sellers hit the bid for {sold_low:,}\nat the low, then price recovers\n{ret:.0f} ticks: they are TRAPPED")])
    ax.set_title(f"17. Trapped traders: aggression at the extreme, then reversal  |  {SYN}\nlabelled by the generator; detector in Phase 3", fontsize=9)
    return _save(fig, out), {"sold_at_low": sold_low, "recovery_ticks": float(ret)}


def c18_profile(day, cfg, out):
    prof = volume_profile(day.ticks, day.tick_size, cfg.profile.row_ticks)
    va = value_area(prof, cfg.profile.value_area_pct)
    nodes = hvn_lvn(prof, cfg)
    m = time_bars(day.ticks, cfg, "1min")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(m["bar"], m["close"], color="#333", lw=1)
    _time_axis(a1)
    step = cfg.profile.row_ticks * day.tick_size
    inside = (prof.price >= va["val"]) & (prof.price <= va["vah"])
    a2.barh(prof.price + step / 2, prof.vol, height=step * 0.9, color=np.where(inside, "#6baed6", "#c8c8c8"))
    a2.axhline(va["poc"] + step / 2, color="black", lw=1.4)
    for r in nodes.itertuples():
        a2.scatter(r.smoothed_vol, r.price + step / 2, marker="D" if r.kind == "HVN" else "x", color=GREEN if r.kind == "HVN" else RED, zorder=3)
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    a1.set_title(f"18. Volume profile: POC (line), 70% VA (blue), HVN (◆) / LVN (✕)  |  {SYN}", fontsize=10)
    return _save(fig, out), {"poc": float(va["poc"]), "vah": float(va["vah"]), "val": float(va["val"]),
                             "va_share": float(va["share"]), "hvn": int((nodes.kind == "HVN").sum()), "lvn": int((nodes.kind == "LVN").sum())}


def c19_shapes(cfg, out):
    n = 30
    x = np.arange(n)
    g = lambda c, sd: 1000 * np.exp(-0.5 * ((x - c) / sd) ** 2)
    pshape = g(24, 3) + np.where(x < 20, 150 * x / 20, 0)
    prof = {"D (balance)": g(14.5, 5), "P (short-covering)": pshape, "b (long-liquidation)": pshape[::-1],
            "double distribution": g(7, 2.5) + g(23, 2.5), "thin / trend": np.full(n, 100.0)}
    fig, axs = plt.subplots(1, 5, figsize=(14, 4), sharey=True)
    res = {}
    for ax, (name, v) in zip(axs, prof.items()):
        df = pd.DataFrame({"price": x * 0.5, "vol": v + 1.0})
        lab = profile_shape(df, cfg)
        res[name] = lab
        ax.barh(df.price, df.vol, height=0.45, color="#6baed6")
        ax.set_title(f"{name}\nclassifier says: {lab}", fontsize=9)
        ax.set_xticks([])
    fig.suptitle("19. Profile shapes. These are CONSTRUCTED profiles to illustrate and test the classifier, not market data", fontsize=10, y=1.06)
    return _save(fig, out), {"labels": res}


def c20_tpo(day, cfg, out):
    t = build_tpo(day.ticks, cfg, day.tick_size)
    m = t.matrix
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(m) + 2))
    cnt = m.sum(axis=1)
    for i, (p, row) in enumerate(m.iterrows()):
        txt = " ".join(c for c in m.columns if row[c])
        col = "#d35400" if cnt[p] == 1 else "#111"
        ax.text(0.0, p, txt, fontsize=8.5, family="monospace", va="center", color=col)
        ax.text(-0.02, p, f"{p:.2f}", fontsize=8, ha="right", va="center")
    step = cfg.tpo.row_ticks * day.tick_size
    ax.axhspan(t.val, t.vah + step, color="#6baed6", alpha=0.2)
    ax.axhline(t.ibh + step, color="#8e44ad", ls="--", lw=1)
    ax.axhline(t.ibl, color="#8e44ad", ls="--", lw=1)
    ax.text(1.0, t.ibh + step, "IB high", color="#8e44ad", fontsize=8)
    ax.text(1.0, t.ibl, "IB low", color="#8e44ad", fontsize=8)
    ax.set_xlim(-0.35, 1.2)
    ax.set_ylim(m.index.min() - step, m.index.max() + 2 * step)
    ax.axis("off")
    flags_txt = f"poor high: {t.poor_high}, poor low: {t.poor_low}, excess high: {t.excess_high}, excess low: {t.excess_low}"
    ax.set_title(f"20. TPO / market profile (30-min letters, orange = single print)  |  {SYN}\nIB = first hour, extensions {t.extensions}\n{flags_txt}", fontsize=9)
    return _save(fig, out), {"ibh": float(t.ibh), "ibl": float(t.ibl), "extensions": [list(e) for e in t.extensions],
                             "poor_high": t.poor_high, "poor_low": t.poor_low, "excess_high": t.excess_high,
                             "excess_low": t.excess_low, "single_print_runs": len(t.single_prints), "letters": len(m.columns)}


def c21_composite(days, cfg, out):
    sess = []
    for d in days:
        pr = volume_profile(d.ticks, d.tick_size, cfg.profile.row_ticks)
        va = value_area(pr, cfg.profile.value_area_pct)
        sess.append({"date": pd.Timestamp(d.date), "poc": va["poc"], "high": d.ticks.price.max(), "low": d.ticks.price.min()})
    s = pd.DataFrame(sess)
    nk = naked_pocs(s)
    allt = pd.concat([d.ticks for d in days])
    comp = volume_profile(allt, days[0].tick_size, cfg.profile.row_ticks)
    cva = value_area(comp, cfg.profile.value_area_pct)
    step = cfg.profile.row_ticks * days[0].tick_size
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    for i, r in nk.iterrows():
        a1.vlines(i, r.low, r.high, color="#555", lw=6, alpha=0.5)
        end = len(nk) - 0.4 if r.naked else nk.index[nk.date == r.first_tested][0]
        a1.hlines(r.poc + step / 2, i, end, color=RED if r.naked else "#999", lw=1.6, ls="-" if r.naked else ":")
    a1.set_xticks(range(len(nk)))
    a1.set_xticklabels([f"{d:%d-%b}" for d in nk.date], rotation=45, fontsize=8)
    a1.set_title("daily ranges; line = that day's POC (red solid = naked, grey dotted = tested)", fontsize=9)
    a2.barh(comp.price + step / 2, comp.vol, height=step * 0.9, color="#c8c8c8")
    a2.axhline(cva["poc"] + step / 2, color="black")
    a2.set_title("composite profile (all days)", fontsize=9)
    fig.suptitle(f"21. Composite profile and naked POCs  |  {SYN}", fontsize=10)
    return _save(fig, out), {"days": len(days), "naked": int(nk.naked.sum()), "composite_poc": float(cva["poc"])}


def _path(prices, date="2026-01-05"):
    t0 = pd.Timestamp(f"{date} 09:15", tz=TZ)
    return pd.DataFrame({"ts": [t0 + pd.Timedelta(minutes=i * 5) for i in range(len(prices))], "price": prices})


def c22_open(days, cfg, out):
    built = {"constructed: drive": [100, 100.2, 100.4, 100.6, 100.8, 101.0],
             "constructed: test-drive": [100, 99.8, 100.2, 100.6, 100.8, 101.0],
             "constructed: rejection-reverse": [100, 101.5, 100.5, 99.0, 98.0, 98.0],
             "constructed: auction": [100, 100.5, 99.5, 100.4, 99.6, 100.05]}
    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (name, pr) in zip(axs[0], built.items()):
        d = _path(pr)
        ax.plot(d["ts"], d["price"], marker="o", color="#333")
        ax.axhline(pr[0], color="#999", ls=":")
        ax.set_title(f"{name}\nclassifier: {open_type(d, cfg)}", fontsize=9)
        ax.set_xticks([])
    labs = []
    for ax, d in zip(axs[1], days[:4]):
        o = open_type(d.ticks, cfg)
        w = d.ticks[d.ticks.ts < d.ticks.ts.iloc[0].normalize() + pd.Timedelta(hours=9, minutes=15 + cfg.open_type.window_min)]
        ax.plot(w["ts"], w["price"], lw=0.8, color="#333")
        ax.axhline(w["price"].iloc[0], color="#999", ls=":")
        ax.set_title(f"synthetic {d.date}\nclassifier: {o}", fontsize=9)
        ax.set_xticks([])
    all_labs = [open_type(d.ticks, cfg) for d in days]
    fig.suptitle(f"22. Open types: top = CONSTRUCTED paths that show each rule, bottom = synthetic days  |  {SYN}\nheuristic rules, to be TESTED; the toy market is a random walk, so plain 'auction' opens are expected", fontsize=10, y=1.03)
    fig.tight_layout()
    return _save(fig, out), {"synthetic_counts": pd.Series(all_labs).value_counts().to_dict(), "n_days": len(days)}


def c23_vwap(day, cfg, out):
    v = vwap_series(day.ticks, cfg)
    e = _event(day, "sweep_reversal")
    sw = day.ticks[(day.ticks.ts >= e["start"]) & (day.ticks.ts < e["end"])]
    anchor = sw.loc[sw["price"].idxmin(), "ts"]
    av = vwap_series(day.ticks, cfg, anchor_ts=anchor)
    step = max(len(v) // 1500, 1)
    s, a, t = v.iloc[::step], av.iloc[::step], day.ticks.iloc[::step]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t["ts"], t["price"], color="#999", lw=0.7, label="trades")
    ax.plot(s["ts"], s["vwap"], color="#8e44ad", lw=1.5, label="session VWAP")
    for k in cfg.vwap.band_mults:
        ax.fill_between(s["ts"], s[f"lower_{k:g}"], s[f"upper_{k:g}"], color="#8e44ad", alpha=0.10)
    ax.plot(a["ts"], a["vwap"], color="#d35400", lw=1.5, label="anchored VWAP (from the sweep low)")
    ax.axvline(anchor, color="#d35400", ls=":", lw=1)
    ax.legend(fontsize=8)
    _time_axis(ax)
    ax.set_title(f"23. Session VWAP with ±1σ/±2σ bands and an anchored VWAP  |  {SYN}\nanchor chosen with hindsight (lowest print of the labelled sweep) purely to illustrate", fontsize=9)
    return _save(fig, out), {"close_vs_vwap": float(day.ticks.price.iloc[-1] - v.vwap.iloc[-1])}


# ---------------------------------------------------------------- Phase 3: book concepts (24-27)
from .. import book as _B  # noqa: E402
from ..data.synthetic import generate_eval_days  # noqa: E402,F401
from matplotlib.colors import LogNorm  # noqa: E402


def _level_series(book, side, price, tick):
    """Displayed size at one price over time (0 when the price is not within the 5 visible levels)."""
    n = sum(c.startswith(f"{side}_px_") for c in book.columns)
    sz = np.zeros(len(book))
    for i in range(1, n + 1):
        m = np.isclose(book[f"{side}_px_{i}"].to_numpy(), price, atol=tick / 4)
        sz[m] = book[f"{side}_sz_{i}"].to_numpy()[m]
    return pd.Series(sz, index=book["ts"])


def c24_heatmap(day, cfg, out):
    """Liquidity heat map: displayed size at each price over time, with trades on top."""
    e = _event(day, "wall_bid")
    t0, t1 = e["start"] - pd.Timedelta(minutes=1), e["start"] + pd.Timedelta(minutes=5)
    bk = day.book[(day.book.ts >= t0) & (day.book.ts < t1)].iloc[::2]
    tk = day.ticks[(day.ticks.ts >= t0) & (day.ticks.ts < t1)]
    tick = day.tick_size
    lo, hi = tk.price.min() - 8 * tick, tk.price.max() + 8 * tick
    rows = np.arange(lo, hi + tick / 2, tick)
    M = np.full((len(rows), len(bk)), np.nan)
    n = sum(c.startswith("bid_px_") for c in bk.columns)
    for side in ("bid", "ask"):
        for i in range(1, n + 1):
            px, sz = bk[f"{side}_px_{i}"].to_numpy(), bk[f"{side}_sz_{i}"].to_numpy()
            ix = np.round((px - lo) / tick).astype(int)
            ok = (ix >= 0) & (ix < len(rows))
            M[ix[ok], np.arange(len(bk))[ok]] = sz[ok]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = mdates.date2num(bk["ts"].dt.tz_convert(TZ))
    im = ax.pcolormesh(np.append(x, x[-1] + (x[-1] - x[-2])), np.append(rows - tick / 2, rows[-1] + tick / 2), np.ma.masked_invalid(M),
                       norm=LogNorm(vmin=max(np.nanmin(M), 1), vmax=np.nanmax(M)), cmap="YlOrRd", shading="flat")
    s = tk.iloc[::max(len(tk) // 900, 1)]
    ax.scatter(mdates.date2num(s["ts"].dt.tz_convert(TZ)), s["price"], s=3, c=np.where(s["side_true"] > 0, GREEN, RED), alpha=0.8)
    ax.xaxis_date(tz=TZ)
    _time_axis(ax)
    fig.colorbar(im, ax=ax, label="displayed size (log scale)")
    ax.set_title(f"24. Liquidity heat map: displayed size per price over time, trades in green (buy) / red (sell)  |  {SYN}\n"
                 "Darker = more resting size. The planted bid wall holds a row dark while trades approach it.", fontsize=9)
    return _save(fig, out), {"window": f"{t0:%H:%M}-{t1:%H:%M}", "max_displayed": float(np.nanmax(M))}


def c25_iceberg(day, cfg, out):
    """Disclosed-quantity iceberg: displayed size keeps refilling while far more volume executes than was ever displayed."""
    e = _event(day, "iceberg_bid")
    hits = _B.iceberg_refills(day.ticks, day.book, cfg, day.tick_size)
    t0, t1 = e["start"], e["start"] + pd.Timedelta(minutes=3)
    if len(hits):
        h = hits.iloc[0]
        px, t_hit = float(h["price"]), h["time"]
    else:
        px, t_hit = float(day.ticks[(day.ticks.ts >= t0)].price.iloc[0]), None
    bk = day.book[(day.book.ts >= t0) & (day.book.ts < t1)]
    tk = day.ticks[(day.ticks.ts >= t0) & (day.ticks.ts < t1) & np.isclose(day.ticks.price, px, atol=day.tick_size / 4) & (day.ticks.side_true < 0)]
    sz = _level_series(bk, "bid", px, day.tick_size)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.step(sz.index, sz.values, where="post", color="#2980b9", lw=1)
    a1.set_ylabel("displayed bid size")
    a2.step(tk["ts"], tk["qty"].cumsum(), where="post", color=RED, lw=1.2)
    a2.set_ylabel("cumulative sell volume\nexecuted at this price")
    if t_hit is not None:
        for a in (a1, a2):
            a.axvline(t_hit, color="k", ls=":", lw=1)
        a1.text(t_hit, a1.get_ylim()[1] * 0.95, " detector fired", fontsize=8, va="top")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=TZ))
    a1.set_title(f"25. Iceberg (disclosed-quantity) order at {px:.2f}  |  {SYN}\n"
                 "Displayed size stays pinned near the disclosed quantity while executed volume is many times larger. Real icebergs are only ever INFERRED.", fontsize=9)
    f = {"price": px, "detected": bool(len(hits))}
    if len(hits):
        f.update(executed=float(hits.iloc[0]["executed"]), displayed=float(hits.iloc[0]["displayed"]), refills=int(hits.iloc[0]["refills"]))
    return _save(fig, out), f


def c26_spoof(day, cfg, out):
    """Spoof-like pulls vs a genuine wall: size appears, never trades, disappears, versus size that is consumed or sits."""
    ep = _B.track_large_orders(day.book, day.ticks, cfg, day.tick_size)
    sp = ep[(ep.outcome == "pulled") & (ep.side == "ask")].iloc[0]
    w = _event(day, "wall_bid")
    tick = day.tick_size
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    t0, t1 = sp.first_ts - pd.Timedelta(seconds=30), sp.end_ts + pd.Timedelta(seconds=30)
    bk = day.book[(day.book.ts >= t0) & (day.book.ts < t1)]
    s = _level_series(bk, "ask", sp.price, tick)
    a1.step(s.index, s.values, where="post", color="#8e44ad")
    a1.axvspan(sp.first_ts, sp.end_ts, color="#8e44ad", alpha=0.12)
    a1.set_title(f"spoof-like: {sp.max_size:,.0f} lots at {sp.price:.2f}, appeared and was pulled in "
                 f"{(sp.end_ts - sp.first_ts).total_seconds():.0f}s with {sp.traded:,.0f} traded", fontsize=9)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=TZ))
    wep = ep[(ep.side == "bid") & (ep.first_ts >= w["start"]) & (ep.first_ts < w["end"])]
    if len(wep):
        g = wep.sort_values("max_size").iloc[-1]
        t0, t1 = g.first_ts - pd.Timedelta(seconds=30), g.end_ts + pd.Timedelta(seconds=30)
        bk = day.book[(day.book.ts >= t0) & (day.book.ts < t1)]
        s = _level_series(bk, "bid", g.price, tick)
        a2.step(s.index, s.values, where="post", color="#16a085")
        a2.axvspan(g.first_ts, g.end_ts, color="#16a085", alpha=0.12)
        a2.set_title(f"planted wall: {g.max_size:,.0f} lots at {g.price:.2f}, outcome '{g.outcome}', {g.traded:,.0f} traded", fontsize=9)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    fig.suptitle(f"26. Spoofing is RECOGNISED here, never traded  |  {SYN}\n"
                 "A spoof-like pull = large size that vanishes without trading. Intent cannot be proven from the book alone.", fontsize=9, y=1.04)
    return _save(fig, out), {"pulled_episodes": int((ep.outcome == "pulled").sum()), "episodes": int(len(ep))}


def c27_imbalance(day, cfg, out):
    """Depth imbalance = (bid size - ask size) / (bid + ask) over the best N levels, against the mid price."""
    n = int(cfg.book.imbalance_levels)
    imb = _B.depth_imbalance(day.book, n)
    ic = _B.imbalance_forward_ic(day.book, n, 5.0)
    mid = (day.book["bid_px_1"] + day.book["ask_px_1"]) / 2
    sl = slice(len(day.book) // 3, len(day.book) // 3 + 1800)
    ts = day.book["ts"].iloc[sl]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    a1.plot(ts, mid.iloc[sl], color="#333", lw=0.8)
    a1.set_ylabel("mid price")
    a2.fill_between(ts, 0, imb.iloc[sl], color="#2980b9", alpha=0.6, step="post")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylim(-1, 1)
    a2.set_ylabel(f"imbalance ({n} levels)")
    _time_axis(a2)
    a1.set_title(f"27. Depth imbalance  |  {SYN}\nPositive = more resting size on the bid. Rank correlation with the mid change over the next 5 s on this day: {ic:.2f} "
                 "(an artifact of the toy book's own mechanics, NOT evidence about real markets)", fontsize=9)
    return _save(fig, out), {"levels": n, "forward_ic_5s": float(ic)}
