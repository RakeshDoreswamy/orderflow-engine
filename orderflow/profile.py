"""Volume profile: POC, value area, HVN/LVN, shape classifier, naked POCs.

Row labels are the LOWER edge of the price row. Value area uses the standard expansion from the POC:
repeatedly compare the next two rows above with the next two below and add the larger pair (ties go
up) until the target share of volume is inside. The shape classifier is a HEURISTIC to be tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def volume_profile(ticks: pd.DataFrame, tick_size: float, row_ticks: int, start=None, end=None) -> pd.DataFrame:
    """Contiguous price grid (zero-volume rows kept) with ``vol``. ``start``/``end`` give a fixed range
    (end exclusive); pass several days of ticks for a composite profile."""
    t = ticks
    if start is not None:
        t = t[t["ts"] >= start]
    if end is not None:
        t = t[t["ts"] < end]
    if t.empty:
        return pd.DataFrame({"price": [], "vol": []})
    idx = np.rint(t["price"].to_numpy() / tick_size).astype(np.int64) // row_ticks
    lo, hi = idx.min(), idx.max()
    vol = np.bincount(idx - lo, weights=t["qty"].to_numpy(float), minlength=hi - lo + 1)
    return pd.DataFrame({"price": np.round((np.arange(lo, hi + 1)) * row_ticks * tick_size, 6), "vol": vol})


def poc_index(vols: np.ndarray) -> int:
    """Index of the max-volume row; ties go to the row closest to the middle of the range."""
    v = np.asarray(vols, float)
    cand = np.flatnonzero(v == v.max())
    return int(cand[np.argmin(np.abs(cand - (len(v) - 1) / 2))])


def value_area_idx(vols: np.ndarray, pct: float) -> tuple[int, int, int]:
    """(low_idx, high_idx, poc_idx) of the value area by expansion from the POC."""
    v = np.asarray(vols, float)
    n = len(v)
    poc = poc_index(v)
    lo = hi = poc
    acc, target = v[poc], pct * v.sum()
    while acc < target and (lo > 0 or hi < n - 1):
        up = v[hi + 1:hi + 3].sum() if hi < n - 1 else -1.0
        dn = v[max(lo - 2, 0):lo].sum() if lo > 0 else -1.0
        if up >= dn:
            acc += up
            hi = min(hi + 2, n - 1)
        else:
            acc += dn
            lo = max(lo - 2, 0)
    return lo, hi, poc


def value_area(prof: pd.DataFrame, pct: float) -> dict:
    """poc / vah / val as row prices (lower edges) plus the share of volume inside."""
    lo, hi, poc = value_area_idx(prof["vol"].to_numpy(), pct)
    p = prof["price"].to_numpy()
    inside = prof["vol"].to_numpy()[lo:hi + 1].sum() / prof["vol"].sum()
    return {"poc": p[poc], "vah": p[hi], "val": p[lo], "share": inside}


def hvn_lvn(prof: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """High/low volume nodes from a smoothed profile: HVN = local peak >= hvn_min_frac of max;
    LVN = local valley <= lvn_max_frac of max."""
    v = prof["vol"].to_numpy(float)
    w = cfg.profile.smooth_rows
    s = np.convolve(np.pad(v, w, mode="edge"), np.ones(w) / w, mode="same")[w:-w]
    mx, rows = s.max(), []
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] >= cfg.profile.hvn_min_frac * mx:
            rows.append((prof["price"].iloc[i], "HVN", s[i]))
        if s[i] <= s[i - 1] and s[i] < s[i + 1] and s[i] <= cfg.profile.lvn_max_frac * mx:
            rows.append((prof["price"].iloc[i], "LVN", s[i]))
    return pd.DataFrame(rows, columns=["price", "kind", "smoothed_vol"])


def profile_shape(prof: pd.DataFrame, cfg: Config) -> str:
    """Heuristic label: 'double', 'P', 'b', 'D', 'thin' or 'other'. Thresholds in config (profile.*)."""
    v = prof["vol"].to_numpy(float)
    n, total = len(v), v.sum()
    if n < 3 or total == 0:
        return "other"
    pc = cfg.profile
    nodes = hvn_lvn(prof, cfg)
    hv = nodes[nodes["kind"] == "HVN"].reset_index(drop=True)
    pos = {p: i for i, p in enumerate(prof["price"])}
    if len(hv) >= 2:
        for a in range(len(hv) - 1):
            ia, ib = pos[hv.price[a]], pos[hv.price[a + 1]]
            valley = v[ia:ib + 1].min()
            if (ib - ia) >= pc.double_min_sep_frac * n and valley <= pc.double_valley_frac * min(v[ia], v[ib]):
                return "double"
    third = max(n // 3, 1)
    lower, upper = v[:third].sum() / total, v[-third:].sum() / total
    mid = 1 - lower - upper
    p = poc_index(v) / (n - 1)
    if p >= 0.6 and upper >= pc.pb_mass_frac:
        return "P"
    if p <= 0.4 and lower >= pc.pb_mass_frac:
        return "b"
    if mid >= pc.d_mid_frac and abs(upper - lower) <= pc.d_sym:
        return "D"
    if n >= pc.thin_min_rows and v.max() / total <= pc.thin_poc_share:
        return "thin"
    return "other"


def naked_pocs(sessions: pd.DataFrame) -> pd.DataFrame:
    """sessions: columns date, poc, high, low (chronological). A POC is naked if no LATER session's
    range traded through it. The latest session is trivially naked (untested yet)."""
    s = sessions.reset_index(drop=True)
    naked, tested = [], []
    for i in range(len(s)):
        later = s.iloc[i + 1:]
        hit = later[(later["low"] <= s.poc[i]) & (later["high"] >= s.poc[i])]
        naked.append(hit.empty)
        tested.append(hit["date"].iloc[0] if len(hit) else pd.NaT)
    return s.assign(naked=naked, first_tested=tested)
