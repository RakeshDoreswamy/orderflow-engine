"""Live tick + depth recorder (broker-agnostic core).

What it does: turns feed messages into two tables per day, ``ticks`` (one row per trade) and ``book`` (depth
snapshots), written as parquet parts under ``<store_root>/<SYMBOL>/<YYYY-MM-DD>/`` plus ``manifest.json``.
What it does NOT do: connect to a broker (see ``Feed``), classify trades, or fill gaps. Missing data stays missing
and is logged in the manifest.

HONESTY NOTES
* Snapshot feeds (the common retail format) report last price, cumulative volume and depth per message. A trade is
  INFERRED when cumulative volume rises; its qty is the increase since the previous message, so several real trades
  can merge into one row. Order flow built from this is coarser than a true tape.
* The bid/ask stored with a trade are the quote from the PREVIOUS message (the quote the trade met), not the
  post-trade book that arrives with the same message. Using the later one would leak the trade into its own quote.
* Aggressor side is never recorded live. Classify afterwards with ``orderflow.classify`` and keep ``_est`` names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np
import pandas as pd

from .config import Config

TICK_COLS = ["ts", "price", "qty", "cum_volume", "bid", "ask"]


class Feed(Protocol):
    """A source of raw broker messages (dicts). Implement ``messages`` for a real broker websocket."""

    def messages(self) -> Iterator[dict]: ...


class ListFeed:
    def __init__(self, msgs: Iterable[dict]):
        self._m = list(msgs)

    def messages(self) -> Iterator[dict]:
        return iter(self._m)


class JsonlFeed:
    """Replay a file of one JSON message per line (``ts`` as ISO-8601 with offset)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def messages(self) -> Iterator[dict]:
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_instrument_master(path: str | Path, exchange: str = "NSE") -> pd.DataFrame:
    """Read an instruments dump you downloaded (CSV with tradingsymbol, tick_size, lot_size, exchange). Matched by name."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"tradingsymbol", "tick_size"}
    if need - set(df.columns):
        raise ValueError(f"instrument master lacks columns {sorted(need - set(df.columns))}")
    if "exchange" in df.columns:
        df = df[df["exchange"].astype(str).str.upper() == exchange.upper()]
    return df.reset_index(drop=True)


def tick_size_for(master: pd.DataFrame, symbol: str) -> float:
    """Tick size from the instrument master. Raises KeyError rather than guessing (it varies by instrument)."""
    m = master[master["tradingsymbol"].astype(str).str.upper() == symbol.upper()]
    if m.empty:
        raise KeyError(f"{symbol} not in the instrument master")
    return float(m["tick_size"].iloc[0])


def _ts(v, tz: str) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)


def normalize_snapshot(msg: dict, tz: str, levels: int = 5) -> dict:
    """Map a Kite-style full-mode message to ``{ts, price, cum_volume, bids, asks}``.
    Field names assumed (verify against your broker's docs): exchange_timestamp|ts, last_price, volume_traded,
    depth.buy / depth.sell = [{price, quantity}, ...] best first. Raises KeyError if a field is missing."""
    ts = msg.get("exchange_timestamp", msg.get("ts"))
    if ts is None:
        raise KeyError("message has no timestamp")
    d = msg["depth"]
    return {"ts": _ts(ts, tz), "price": float(msg["last_price"]), "cum_volume": int(msg["volume_traded"]),
            "bids": [(float(x["price"]), int(x["quantity"])) for x in d["buy"][:levels]],
            "asks": [(float(x["price"]), int(x["quantity"])) for x in d["sell"][:levels]]}


@dataclass
class Stats:
    messages: int = 0
    trades: int = 0
    books: int = 0
    dropped_out_of_session: int = 0
    dropped_out_of_order: int = 0
    dropped_bad_message: int = 0
    volume_resets: int = 0
    gaps: list = field(default_factory=list)


class Recorder:
    def __init__(self, root: str | Path, symbol: str, cfg: Config, tick_size: float, source: str = "unknown"):
        self.root, self.symbol, self.cfg, self.tick_size, self.source = Path(root), symbol.upper(), cfg, float(tick_size), source
        self.tz = cfg.market.timezone
        self.r = cfg.recorder
        self.levels = int(self.r.book_levels)
        self._ticks: list[dict] = []
        self._book: list[dict] = []
        self._parts = {"ticks": {}, "book": {}}      # date -> part counter
        self._counts: dict = {}                      # date -> dict of row counts
        self._prev: dict | None = None
        self._last_ts: pd.Timestamp | None = None
        self._last_book_ts: pd.Timestamp | None = None
        self.stats = Stats()
        self._gaps_by_day: dict = {}
        self._first_last: dict = {}
        self._open = pd.Timedelta(hours=int(cfg.market.session_open[:2]), minutes=int(cfg.market.session_open[3:]))
        self._close = pd.Timedelta(hours=int(cfg.market.session_close[:2]), minutes=int(cfg.market.session_close[3:]))

    # ---- ingestion -------------------------------------------------------------------------------------
    def in_session(self, ts: pd.Timestamp) -> bool:
        t = ts - ts.normalize()
        return self._open <= t < self._close

    def on_message(self, msg: dict) -> None:
        self.stats.messages += 1
        try:
            s = normalize_snapshot(msg, self.tz, self.levels)
        except (KeyError, TypeError, ValueError):
            self.stats.dropped_bad_message += 1
            return
        self.on_snapshot(s)

    def on_snapshot(self, s: dict) -> None:
        ts = s["ts"]
        if not self.in_session(ts):
            self.stats.dropped_out_of_session += 1
            return
        if self._last_ts is not None and ts < self._last_ts:
            self.stats.dropped_out_of_order += 1
            return
        if self._last_ts is not None and ts.date() == self._last_ts.date() and (ts - self._last_ts).total_seconds() > float(self.r.gap_alert_s):
            self._gaps_by_day.setdefault(ts.date(), []).append([self._last_ts.isoformat(), ts.isoformat()])
            self.stats.gaps.append((self._last_ts, ts))
        prev = self._prev if (self._prev is not None and self._prev["ts"].date() == ts.date()) else None
        if prev is not None:
            dv = s["cum_volume"] - prev["cum_volume"]
            if dv < 0:
                self.stats.volume_resets += 1  # cumulative volume went backwards: do not guess a trade
            elif dv > 0:
                bid = prev["bids"][0][0] if prev["bids"] else np.nan
                ask = prev["asks"][0][0] if prev["asks"] else np.nan
                self.add_trade(ts, s["price"], dv, s["cum_volume"], bid, ask)
        if self._last_book_ts is None or self._last_book_ts.date() != ts.date() or (ts - self._last_book_ts).total_seconds() >= float(self.r.book_snapshot_sec):
            self.add_book(ts, s["bids"], s["asks"])
            self._last_book_ts = ts
        self._prev, self._last_ts = s, ts

    def add_trade(self, ts, price, qty, cum_volume, bid, ask) -> None:
        self._ticks.append({"ts": ts, "price": float(price), "qty": int(qty), "cum_volume": int(cum_volume), "bid": float(bid), "ask": float(ask)})
        self.stats.trades += 1
        self._touch(ts, "ticks")
        if len(self._ticks) >= int(self.r.flush_rows):
            self.flush()

    def add_book(self, ts, bids, asks) -> None:
        row = {"ts": ts}
        for i in range(self.levels):
            bp, bq = bids[i] if i < len(bids) else (np.nan, 0)
            ap, aq = asks[i] if i < len(asks) else (np.nan, 0)
            row.update({f"bid_px_{i + 1}": bp, f"ask_px_{i + 1}": ap, f"bid_sz_{i + 1}": bq, f"ask_sz_{i + 1}": aq})
        self._book.append(row)
        self.stats.books += 1
        self._touch(ts, "book")
        if len(self._book) >= int(self.r.flush_rows):
            self.flush()

    def _touch(self, ts, kind):
        c = self._counts.setdefault(ts.date(), {"ticks": 0, "book": 0})
        c[kind] += 1
        fl = self._first_last.setdefault(ts.date(), [ts, ts])
        fl[0], fl[1] = min(fl[0], ts), max(fl[1], ts)

    # ---- storage ---------------------------------------------------------------------------------------
    def _day_dir(self, date) -> Path:
        return self.root / self.symbol / str(date)

    def _write(self, rows: list[dict], kind: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        for date, g in df.groupby(df["ts"].dt.date):
            d = self._day_dir(date)
            d.mkdir(parents=True, exist_ok=True)
            n = self._parts[kind].get(date, 0)
            final = d / f"{kind}-{n:04d}.parquet"
            tmp = d / f".{final.name}.tmp"
            g.reset_index(drop=True).to_parquet(tmp, index=False)
            os.replace(tmp, final)            # atomic: a crash never leaves a half-written part under the final name
            self._parts[kind][date] = n + 1

    def flush(self) -> None:
        self._write(self._ticks, "ticks")
        self._write(self._book, "book")
        self._ticks, self._book = [], []
        self._write_manifests()

    def _write_manifest(self, date) -> None:
        fl = self._first_last.get(date)
        m = {"schema_version": int(self.r.schema_version), "symbol": self.symbol, "date": str(date), "tick_size": self.tick_size,
             "source": self.source, "rows": self._counts.get(date, {}), "first_ts": fl[0].isoformat() if fl else None,
             "last_ts": fl[1].isoformat() if fl else None, "gaps": self._gaps_by_day.get(date, []),
             "dropped": {"out_of_session": self.stats.dropped_out_of_session, "out_of_order": self.stats.dropped_out_of_order,
                         "bad_message": self.stats.dropped_bad_message, "volume_resets": self.stats.volume_resets},
             "note": "trades inferred from cumulative-volume increases; bid/ask are the previous message's quote; no aggressor side recorded"}
        d = self._day_dir(date)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ".manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=1))
        os.replace(tmp, d / "manifest.json")

    def _write_manifests(self) -> None:
        for date in list(self._counts):
            self._write_manifest(date)

    def close(self) -> Stats:
        self.flush()
        return self.stats

    def run(self, feed: Feed) -> Stats:
        try:
            for m in feed.messages():
                self.on_message(m)
        finally:
            self.close()
        return self.stats


def load_day(root: str | Path, symbol: str, date) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read one recorded day back: (ticks, book, manifest). Raises FileNotFoundError if nothing was recorded."""
    d = Path(root) / symbol.upper() / str(pd.Timestamp(date).date())
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"no recording at {d}")
    manifest = json.loads((d / "manifest.json").read_text())

    def cat(kind):
        parts = sorted(d.glob(f"{kind}-*.parquet"))
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()

    return cat("ticks"), cat("book"), manifest


def quality_report(ticks: pd.DataFrame, book: pd.DataFrame, manifest: dict, cfg: Config) -> dict:
    """Facts about a recording that decide whether it is fit for order-flow work. Nothing here is a trading result."""
    ts = float(manifest["tick_size"])
    out = {"trades": int(len(ticks)), "book_snapshots": int(len(book)), "gaps": len(manifest.get("gaps", [])),
           "gap_seconds_total": float(sum((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() for a, b in manifest.get("gaps", [])))}
    if len(ticks):
        off = (np.abs(ticks["price"] / ts - np.round(ticks["price"] / ts)) > 1e-6)
        q = ticks.dropna(subset=["bid", "ask"])
        mins = ticks["ts"].dt.floor("1min")
        sess_min = int((pd.Timedelta(hours=int(cfg.market.session_close[:2]), minutes=int(cfg.market.session_close[3:]))
                        - pd.Timedelta(hours=int(cfg.market.session_open[:2]), minutes=int(cfg.market.session_open[3:]))).total_seconds() // 60)
        out.update({"pct_prices_off_tick": float(100 * off.mean()), "pct_missing_quote": float(100 * (1 - len(q) / len(ticks))),
                    "pct_crossed_or_locked_quote": float(100 * (q["bid"] >= q["ask"]).mean()) if len(q) else float("nan"),
                    "minutes_with_trades": int(mins.nunique()), "session_minutes": sess_min,
                    "pct_trades_outside_quote": float(100 * ((q["price"] < q["bid"] - 1e-9) | (q["price"] > q["ask"] + 1e-9)).mean()) if len(q) else float("nan"),
                    "ts_monotonic": bool(ticks["ts"].is_monotonic_increasing)})
    return out
