"""Order-book analytics on depth snapshots (best N levels per side).

Snapshot columns (see data.synthetic): ts, trade_idx, bid_px_k / bid_sz_k / ask_px_k / ask_sz_k for k = 1..L.
Everything here is causal: a row is stamped with the first time its rule could be known.

Rules in plain English
* Depth imbalance = (sum bid size - sum ask size) / (sum bid size + sum ask size) over the top N levels.
* Large order ("wall" candidate): a displayed level whose size is >= ``wall_min_mult`` x the typical displayed
  level size (median of the previous ``baseline_minutes``), first seen at level >= ``wall_min_level``.
  It is CONFIRMED once it has survived ``wall_min_s``. It ENDS when its size falls below ``hysteresis`` x the
  threshold. Outcome: 'pulled' if less than ``pull_traded_frac`` of its size traded at its price, else 'consumed'.
* Spoof-like flag (recognition only): a large order first seen at level >= ``spoof_min_start_level`` that was
  PULLED within ``spoof_max_life_s``. A genuine cancel looks the same on the tape: the flag is a pattern, not
  proof of intent. Spoofing is market manipulation under SEBI rules; this repo only teaches recognising it.
* Iceberg (NSE disclosed-quantity) refill: trades keep hitting the same price, the displayed size is
  replenished after each trade, executed volume reaches ``iceberg_min_exec_ratio`` x the displayed size and the
  displayed size after each refill is almost constant (coefficient of variation <= ``iceberg_cv_max``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _levels(book: pd.DataFrame) -> int:
    return sum(c.startswith("bid_sz_") for c in book.columns)


def depth_imbalance(book: pd.DataFrame, n_levels: int) -> pd.Series:
    """(bid - ask) / (bid + ask) over the top ``n_levels``; +1 all bids, -1 all asks, NaN if empty."""
    b = book[[f"bid_sz_{k}" for k in range(1, n_levels + 1)]].sum(axis=1)
    a = book[[f"ask_sz_{k}" for k in range(1, n_levels + 1)]].sum(axis=1)
    return ((b - a) / (b + a).replace(0, np.nan)).rename("depth_imbalance")


def imbalance_forward_ic(book: pd.DataFrame, n_levels: int, horizon_s: float) -> float:
    """EVALUATION ONLY (looks forward by design): Spearman correlation between depth imbalance now and the
    mid-price change over the next ``horizon_s`` seconds. Never use this as a signal. It exists to measure how
    little short-term predictive power imbalance has once real data is available."""
    t = pd.DataFrame({"ts": book["ts"], "imb": depth_imbalance(book, n_levels),
                      "mid": (book["bid_px_1"] + book["ask_px_1"]) / 2}).reset_index(drop=True)
    later = t[["ts", "mid"]].rename(columns={"ts": "ts_f", "mid": "mid_f"})
    target = t[["ts"]].assign(ts_f=t["ts"] + pd.Timedelta(seconds=horizon_s)).reset_index().sort_values("ts_f")
    m = pd.merge_asof(target, later, on="ts_f", direction="backward").set_index("index")
    t["fwd"] = m["mid_f"] - t["mid"]
    ok = t.dropna(subset=["imb", "fwd"])
    return float(ok["imb"].rank().corr(ok["fwd"].rank())) if len(ok) > 10 else float("nan")


def track_large_orders(book: pd.DataFrame, ticks: pd.DataFrame, cfg: Config, tick_size: float) -> pd.DataFrame:
    """Follow every large displayed order through the snapshots. Returns one row per episode with
    side, price, first_ts, confirm_ts (NaT if it never survived wall_min_s), end_ts (NaT if still active),
    outcome ('pulled' | 'consumed' | 'out_of_view' | 'active'), max_size, start_level, end_level
    (last level it was seen at), traded (shares traded at its price while it existed)."""
    c = cfg.book
    L = _levels(book)
    ts = book["ts"].to_numpy("datetime64[ns]")
    sec = (ts - ts[0]).astype("timedelta64[ms]").astype(np.int64) / 1000.0
    bpx = np.rint(book[[f"bid_px_{k}" for k in range(1, L + 1)]].to_numpy() / tick_size).astype(np.int64)
    apx = np.rint(book[[f"ask_px_{k}" for k in range(1, L + 1)]].to_numpy() / tick_size).astype(np.int64)
    bsz = book[[f"bid_sz_{k}" for k in range(1, L + 1)]].to_numpy()
    asz = book[[f"ask_sz_{k}" for k in range(1, L + 1)]].to_numpy()
    tidx = book["trade_idx"].to_numpy()
    tpx = np.rint(ticks["price"].to_numpy() / tick_size).astype(np.int64)
    tq = ticks["qty"].to_numpy()
    # typical level size from the PREVIOUS minutes only
    minute = (sec // 60).astype(int)
    sizes = np.concatenate([bsz, asz], axis=1)
    mm = np.full(minute.max() + 1, np.nan)
    for m_ in np.unique(minute):
        mm[m_] = np.median(sizes[minute == m_])
    base = np.full(minute.max() + 1, np.nan)
    for m in range(int(c.baseline_warmup_min), len(mm)):
        w = mm[max(0, m - int(c.baseline_minutes)):m]
        if np.isfinite(w).any():
            base[m] = np.nanmedian(w)
    active: dict = {}
    out: list[dict] = []

    def close(ep: dict, j: int, outcome: str) -> None:
        ep["end_ts"] = ts[j]
        ep["outcome"] = outcome
        out.append(ep)

    for j in range(len(ts)):
        b = base[minute[j]]
        thr = c.wall_min_mult * b if np.isfinite(b) else np.inf
        if tidx[j] >= 0:
            for key, ep in active.items():
                if key[1] == tpx[tidx[j]]:
                    ep["traded"] += int(tq[tidx[j]])
        present = {}
        for k in range(L):
            present[("bid", bpx[j, k])] = (bsz[j, k], k)
            present[("ask", apx[j, k])] = (asz[j, k], k)
        for key in list(active):
            ep = active[key]
            cur = present.get(key)
            if cur is None:
                close(active.pop(key), j, "out_of_view")
            elif cur[0] >= c.hysteresis * thr:
                ep["max_size"] = max(ep["max_size"], int(cur[0]))
                ep["end_level"] = cur[1]
                if ep["confirm_ts"] is None and sec[j] - ep["_t0"] >= c.wall_min_s:
                    ep["confirm_ts"] = ts[j]
            else:
                close(active.pop(key), j, "pulled" if ep["traded"] < c.pull_traded_frac * ep["max_size"] else "consumed")
        if np.isfinite(thr):
            for key, (size, k) in present.items():
                if key not in active and size >= thr and k >= c.wall_min_level:
                    active[key] = {"side": key[0], "price": round(key[1] * tick_size, 4), "first_ts": ts[j],
                                   "confirm_ts": None, "end_ts": None, "outcome": "active", "max_size": int(size),
                                   "start_level": k, "end_level": k, "traded": 0, "_t0": sec[j]}
    out.extend(active.values())
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["side", "price", "first_ts", "confirm_ts", "end_ts", "outcome", "max_size",
                                     "start_level", "end_level", "traded"])
    df = df.drop(columns="_t0")
    for col in ("first_ts", "confirm_ts", "end_ts"):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize("UTC").dt.tz_convert(book["ts"].dt.tz)
    return df.sort_values("first_ts").reset_index(drop=True)


def walls(episodes: pd.DataFrame) -> pd.DataFrame:
    """Confirmed large orders, stamped at the time they were confirmed (first_ts + wall_min_s)."""
    w = episodes[episodes["confirm_ts"].notna()].copy()
    w["time"] = w["confirm_ts"]
    return w.sort_values("time").reset_index(drop=True)


def pull_stack_events(episodes: pd.DataFrame) -> pd.DataFrame:
    """'stack' = a large order appeared (stamped at first sight); 'pull' = it vanished with (almost) no trade at
    its price (stamped when it vanished)."""
    st = episodes.assign(time=episodes["first_ts"], kind="stack")
    pl = episodes[episodes["outcome"] == "pulled"].assign(time=lambda d: d["end_ts"], kind="pull")
    return pd.concat([st, pl]).sort_values("time")[["time", "kind", "side", "price", "max_size", "start_level", "end_level"]].reset_index(drop=True)


def spoof_flags(episodes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Spoof-LIKE pattern (recognition only): large order, first seen deep in the book, pulled quickly, ~untraded."""
    c = cfg.book
    life = (episodes["end_ts"] - episodes["first_ts"]).dt.total_seconds()
    m = (episodes["outcome"] == "pulled") & (episodes["start_level"] >= c.spoof_min_start_level) & (life <= c.spoof_max_life_s)
    f = episodes[m].assign(time=episodes["end_ts"], life_s=life[m], rule="large order pulled untraded within life limit")
    return f.sort_values("time").reset_index(drop=True)


def iceberg_refills(ticks: pd.DataFrame, book: pd.DataFrame, cfg: Config, tick_size: float) -> pd.DataFrame:
    """Disclosed-quantity (iceberg) detector. Stamped at the trade where the rule first holds for a run."""
    c = cfg.book
    L = _levels(book)
    rows_after = book[book["trade_idx"] >= 0]
    pos = {int(t): i for i, t in zip(rows_after.index, rows_after["trade_idx"])}
    bpx = np.rint(book[[f"bid_px_{k}" for k in range(1, L + 1)]].to_numpy() / tick_size).astype(np.int64)
    apx = np.rint(book[[f"ask_px_{k}" for k in range(1, L + 1)]].to_numpy() / tick_size).astype(np.int64)
    bsz = book[[f"bid_sz_{k}" for k in range(1, L + 1)]].to_numpy()
    asz = book[[f"ask_sz_{k}" for k in range(1, L + 1)]].to_numpy()
    tpx = np.rint(ticks["price"].to_numpy() / tick_size).astype(np.int64)
    tq, tts = ticks["qty"].to_numpy(), ticks["ts"]
    runs: dict = {}
    hits = []
    for i in range(len(ticks)):
        r = pos.get(i)
        if r is None or r == 0:
            continue
        pt, pre_r = tpx[i], r - 1
        if pt == apx[pre_r, 0]:
            side, pre = "ask", asz[pre_r, 0]
            m = np.flatnonzero(apx[r] == pt)
            post = asz[r, m[0]] if len(m) else 0
        elif pt == bpx[pre_r, 0]:
            side, pre = "bid", bsz[pre_r, 0]
            m = np.flatnonzero(bpx[r] == pt)
            post = bsz[r, m[0]] if len(m) else 0
        else:
            continue
        key = (side, pt)
        run = runs.get(key)
        if run is None or (tts.iloc[i] - run["last"]).total_seconds() > c.iceberg_gap_s:
            run = runs[key] = {"last": tts.iloc[i], "exec": 0, "max_disp": 0, "refills": 0, "posts": [], "fired": False}
        run["last"] = tts.iloc[i]
        run["exec"] += int(tq[i])
        run["max_disp"] = max(run["max_disp"], int(pre))
        if post - max(pre - tq[i], 0) >= c.iceberg_refill_min_frac * min(tq[i], pre) and post > 0:
            run["refills"] += 1
            run["posts"].append(float(post))
        p = np.array(run["posts"])
        if (not run["fired"] and run["refills"] >= c.iceberg_min_refills and run["exec"] >= c.iceberg_min_exec_ratio * run["max_disp"]
                and p.std() <= c.iceberg_cv_max * p.mean()):
            run["fired"] = True
            hits.append({"time": tts.iloc[i], "price": round(pt * tick_size, 4), "side": side, "executed": run["exec"],
                         "displayed": float(np.median(p)), "refills": run["refills"],
                         "strength": run["exec"] / max(float(np.median(p)), 1.0),
                         "rule": "same-price refills with near-constant displayed size"})
    return pd.DataFrame(hits, columns=["time", "price", "side", "executed", "displayed", "refills", "strength", "rule"])
