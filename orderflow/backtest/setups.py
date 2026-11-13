"""Setups = event detector + direction + stop/target rule. All are HYPOTHESES to be tested, not recommendations.

  absorption_reversal : selling absorbed at a low (absorption 'bid') -> long; buying absorbed ('ask') -> short.
  exhaustion_fade     : buying exhausted at a high ('up') -> short; selling exhausted at a low ('down') -> long.
  trapped_reversal    : sellers trapped at a low then rally -> long; buyers trapped at a high then drop -> short.

Signal time = the first moment the detector's rule could be known (events.py is prefix-tested). The stop sits beyond the
extreme of the last ``setups.stop_lookback_min`` COMPLETED bars (+ buffer), with at least ``min_risk_ticks`` of risk; the
target is ``target_r`` times that risk from the last completed close. The engine then fills at the next bar's open.
"""
from __future__ import annotations

import pandas as pd

from .. import events as E
from ..config import Config
from ..data.bars import time_bars

_DIRECTION = {("absorption_reversal", "bid"): 1, ("absorption_reversal", "ask"): -1,
              ("exhaustion_fade", "down"): 1, ("exhaustion_fade", "up"): -1,
              ("trapped_reversal", "sellers_trapped"): 1, ("trapped_reversal", "buyers_trapped"): -1}


def day_bars(ticks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    b = time_bars(ticks, cfg, freq=cfg.backtest.bar_freq)
    return b.reset_index()[["bar", "open", "high", "low", "close", "volume"]] if "bar" not in b.columns else b[["bar", "open", "high", "low", "close", "volume"]]


def raw_hits(ticks: pd.DataFrame, cfg: Config, tick_size: float, side_col: str = "side_true") -> dict[str, pd.DataFrame]:
    return {"absorption_reversal": E.absorption(ticks, cfg, tick_size, side_col),
            "exhaustion_fade": E.exhaustion(ticks, cfg, tick_size, side_col),
            "trapped_reversal": E.trapped(ticks, cfg, tick_size, side_col)}


def build_signals(ticks: pd.DataFrame, bars: pd.DataFrame, cfg: Config, tick_size: float, side_col: str = "side_true",
                  setups: list[str] | None = None) -> pd.DataFrame:
    sc = cfg.setups
    step = pd.Timedelta(cfg.backtest.bar_freq)
    rows = []
    for name, hits in raw_hits(ticks, cfg, tick_size, side_col).items():
        if setups and name not in setups:
            continue
        last_t = None
        for r in hits.sort_values("time").itertuples():
            d = _DIRECTION.get((name, r.side))
            if d is None:
                continue
            if last_t is not None and (r.time - last_t) < pd.Timedelta(minutes=float(sc.cooldown_min)):
                continue
            done = bars[(bars["bar"] + step <= r.time) & (bars["bar"].dt.date == r.time.date())].tail(int(sc.stop_lookback_min))
            if len(done) < 3:
                continue
            ref = float(done["close"].iloc[-1])
            ext = float(done["low"].min()) if d > 0 else float(done["high"].max())
            risk = max(abs(ref - ext) + sc.stop_buffer_ticks * tick_size, sc.min_risk_ticks * tick_size)
            rows.append({"time": r.time, "side": d, "stop": ref - d * risk, "target": ref + d * sc.target_r * risk, "setup": name})
            last_t = r.time
    return pd.DataFrame(rows, columns=["time", "side", "stop", "target", "setup"]).sort_values("time").reset_index(drop=True)
