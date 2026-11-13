"""Indian equity cost model. Every rate is in config.yaml (costs.*) and MUST be checked against your broker's current
contract note: the numbers shipped there are from memory. Rates are percentages of turnover (price x qty).

intraday (MIS): brokerage min(pct x turnover, cap) per order, STT on SELL, exchange txn + SEBI both sides,
stamp duty on BUY, GST on (brokerage + exchange txn + SEBI). Slippage is NOT here: the engine applies it to prices.
"""
from __future__ import annotations

from ..config import Config


def leg_cost(side: str, price: float, qty: int, cfg: Config, product: str = "intraday") -> float:
    """Rupee cost of one order leg. side is 'buy' or 'sell'."""
    c = getattr(cfg.costs, product)
    t = price * qty
    brokerage = min(c.brokerage_pct / 100 * t, c.brokerage_cap_per_order) if c.brokerage_cap_per_order else c.brokerage_pct / 100 * t
    txn, sebi = c.exchange_txn_pct / 100 * t, c.sebi_pct / 100 * t
    gst = c.gst_pct / 100 * (brokerage + txn + sebi)
    stt = (c.stt_sell_pct if side == "sell" else getattr(c, "stt_buy_pct", 0.0)) / 100 * t
    stamp = c.stamp_buy_pct / 100 * t if side == "buy" else 0.0
    return float(brokerage + txn + sebi + gst + stt + stamp)


def round_trip_cost(direction: int, entry_px: float, exit_px: float, qty: int, cfg: Config, product: str = "intraday") -> float:
    """direction +1 = long (buy then sell), -1 = short (sell then buy)."""
    a, b = ("buy", "sell") if direction > 0 else ("sell", "buy")
    return leg_cost(a, entry_px, qty, cfg, product) + leg_cost(b, exit_px, qty, cfg, product)
