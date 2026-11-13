"""Split the study page into one page per concept (docs/learn/<id>.html) plus an index (docs/learn.html), and add pages for the
Phase 5-7 topics (costs, backtest controls, rules of thumb, recorder, ICT bridge) whose numbers are read from docs/*.md|json
or computed from config. Run AFTER scripts/build_site.py:   python scripts/build_pages.py"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.backtest.costs import round_trip_cost  # noqa: E402
from orderflow.config import load_config  # noqa: E402

root = Path(__file__).resolve().parents[1]
docs = root / "docs"
cfg = load_config()
src = (docs / "index.html").read_text(encoding="utf-8")
style = re.search(r"<style>.*?</style>", src, re.S).group(0)
script = re.search(r"<script>.*?</script>", src, re.S)
script = script.group(0) if script else ""
secs = {m.group(1): m.group(0) for m in re.finditer(r'<section id="([^"]+)">.*?</section>', src, re.S)}
secs.pop("coming", None)


def md_table(path):
    lines = [l for l in Path(path).read_text().strip().splitlines() if l.strip()]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>"


def page_sec(sid, title, sub, parts):
    return f'<section id="{sid}"><h2>{title}</h2><p class="sub">{sub}</p>' + "".join(parts) + "</section>"


# ---- extra pages (numbers come from files / config, never typed here) ----
rows = []
for px, qty in ((100, 100), (500, 100), (1000, 100), (1000, 500)):
    c = round_trip_cost(1, px, px, qty, cfg)
    tick = float(cfg.instrument.default_tick_size)
    rows.append(f"<tr><td>Rs {px}</td><td>{qty}</td><td>Rs {px * qty:,.0f}</td><td>Rs {c:.0f}</td><td>{c / qty / tick:.1f} ticks</td><td>{c / (px * qty) * 1e4:.1f} bp</td></tr>")
x1 = page_sec("x1", "Costs: the break-even move", "Every backtest here pays these. The rates in config.yaml are from memory; check your contract note.", [
    "<h3>Definition</h3><p>A round trip pays brokerage, STT on the sell side, exchange and SEBI charges, GST on those fees and stamp duty on the buy side. Slippage is separate: the engine fills against you by <code>slippage_ticks</code>.</p>",
    "<h3>Why it matters</h3><p>Costs are almost proportional to turnover, but price moves are measured in ticks. Small targets disappear: the table is the gross move a long needs just to get back to zero (intraday product, flat price, before slippage).</p>",
    "<table><tr><th>price</th><th>shares</th><th>turnover per leg</th><th>round-trip cost</th><th>break-even move</th><th>as basis points</th></tr>" + "".join(rows) + "</table>",
    "<h3>Common mistakes</h3><ul><li>Backtesting a 10-tick target with a Rs 80 cost per trade and no cost line.</li><li>Using a cost model that is out of date: STT and stamp rates have changed before.</li></ul>"])
x2 = page_sec("x2", "Backtest controls: does the engine itself behave?", "Before judging a setup, check that the engine neither invents edge nor misses real edge.", [
    "<h3>Definition</h3><p>A negative control is a driftless random walk with random signals: the result must be about minus costs. A positive control plants a drift after each signal: the engine must see it. Both run in the tests and the table below.</p>",
    md_table(docs / "phase6_controls.md"),
    "<h3>The setups, on held-out synthetic days</h3>" + md_table(docs / "phase6_backtest.md"),
    "<h3>How to read it</h3><ul><li><code>trapped_reversal</code> wins here because the generator's trap event contains a reversal. That is a property of the toy, not a finding.</li><li>Rows marked small sample have wide intervals.</li></ul>"])
r80 = json.loads((docs / "phase6_rule80.json").read_text())
x3 = page_sec("x3", "Rules of thumb: the 80% rule and gap fill", "Harnesses that measure a rule instead of believing it.", [
    "<h3>The 80% rule</h3><p>Open outside the prior day's value area, re-enter and be accepted (two 30-minute closes inside), and price is said to cross to the other side about 80% of the time. The exact definition is fixed in <code>orderflow/rules.py</code> before any data is looked at.</p>",
    f"<div class='note'>Toy result (random-walk days with Gaussian gaps, not market data): {r80['successes']} of {r80['accepted']} accepted days reached the other side, {r80['rate']:.0%} (95% CI {r80['ci95'][0]:.0%} to {r80['ci95'][1]:.0%}) against the claimed {r80['claimed']:.0%}. It needs real recorded days to say anything about NSE, and other acceptance definitions give other numbers.</div>",
    "<h3>Gap fill on toy random-walk series</h3>" + md_table(docs / "phase6_gapfill.md"),
    "<h3>Common mistakes</h3><ul><li>Changing the definition until the rule works.</li><li>Quoting a rate without the sample size and interval.</li></ul>"])
rd = json.loads((docs / "recorder_demo.json").read_text())
q = rd["quality"]
x4 = page_sec("x4", "Recording live data and checking its quality", "What the recorder keeps, what it cannot, and the checks to run on a recording.", [
    "<h3>Definition</h3><p>The recorder stores one row per inferred trade (price, quantity, cumulative volume and the PREVIOUS message's bid/ask) and throttled depth snapshots, as parquet parts plus a manifest. It never fills gaps or records an aggressor side.</p>",
    "<h3>Why it matters</h3><p>Snapshot feeds merge trades, so a recorded row can be several real trades, and a quote taken after the trade would leak the trade into its own quote. The quality report checks gaps, off-tick prices, missing or crossed quotes and minutes with trades.</p>",
    f"<div class='note'>Replay of a synthetic day: {rd['messages']:,} messages gave {rd['trades_recorded']:,} trades and {rd['book_snapshots']:,} depth snapshots; gaps {q['gaps']}, off-tick prices {q['pct_prices_off_tick']:.1f}%, crossed quotes {q['pct_crossed_or_locked_quote']:.1f}%, {q['minutes_with_trades']} of {q['session_minutes']} session minutes with trades. No broker adapter exists yet, and nothing has run on a real feed.</div>"])
nb = json.loads((docs / "phase7_bridge_null.json").read_text())["difference"]
x5 = page_sec("x5", "Bridging to the ICT engine", "Order-flow evidence for an ICT entry: confirm, neutral or reject.", [
    "<h3>Definition</h3><p>For each entry in the ICT scanner's <code>scan.csv</code> the bridge sums transparent +1/-1 pieces of evidence (effort without result, flow around the swing near the sweep, net estimated delta; or absorption, exhaustion and trapped-trader events from ticks) and reports confirm, neutral, reject or no_data. Everything is cut off at the as-of time.</p>",
    "<h3>Does it help?</h3><p>Unknown. On toy random-walk entries with no edge:</p>" + md_table(docs / "phase7_bridge_null.md"),
    f"<div class='note'>Confirm minus reject = {nb['confirm_minus_reject']:+.2f} R, 95% CI {nb['ci95'][0]:+.2f} to {nb['ci95'][1]:+.2f}: not distinguishable from zero, as it should be with no edge. Run <code>compare_outcomes</code> on your real ICT outcomes before trusting a verdict.</div>"])
secs.update({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})

groups = [("Markets and the order book", ["a1", "a2", "a3", "a4", "a5"]), ("The tape and trade classification", ["b6", "b7", "b8"]),
          ("Delta, CVD and the footprint", ["fp", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17"]),
          ("Profile, TPO, VWAP and rules to test", ["c18", "c19", "c20", "c21", "c22", "c23"]),
          ("The book: heat map, icebergs, spoofing, imbalance", ["c24", "c25", "c26", "c27"]),
          ("Detectors and daily flow", ["c28", "c29"]), ("Costs, backtests, rule harnesses, recording, ICT bridge", ["x1", "x2", "x3", "x4", "x5"])]
order = [s for _, g in groups for s in g]
missing = [s for s in order if s not in secs]
assert not missing, missing
title = {s: re.search(r"<h2>(.*?)</h2>", secs[s]).group(1) for s in order}
(docs / "learn").mkdir(exist_ok=True)
for old in (docs / "learn").glob("*.html"):
    old.unlink()
FOOT = '<p class="sub" style="margin-top:28px">Synthetic or illustrative data unless a page says otherwise. Education only, not financial advice.</p>'
for i, sid in enumerate(order):
    body = secs[sid].replace('src="img/', 'src="../img/')
    prev_l = f'<a href="{order[i - 1]}.html">&larr; {title[order[i - 1]]}</a>' if i else "<span></span>"
    next_l = f'<a href="{order[i + 1]}.html">{title[order[i + 1]]} &rarr;</a>' if i < len(order) - 1 else "<span></span>"
    js = script if 'id="lad"' in body else ""
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{re.sub("<[^>]+>", "", title[sid])} | orderflow-engine</title>{style}</head><body><div class="wrap">'
            f'<nav><a href="../learn.html">&larr; All {len(order)} pages</a><a href="../index.html">Single long page</a></nav>{body}'
            f'<div style="display:flex;justify-content:space-between;gap:12px;margin-top:24px">{prev_l}{next_l}</div>{FOOT}</div>{js}</body></html>')
    (docs / "learn" / f"{sid}.html").write_text(html, encoding="utf-8")
idx = "".join(f"<h3>{g}</h3><ul>" + "".join(f'<li><a href="learn/{s}.html">{title[s]}</a></li>' for s in ids) + "</ul>" for g, ids in groups)
(docs / "learn.html").write_text(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learn order flow | orderflow-engine</title>{style}</head><body><div class="wrap">'
    f'<nav><a href="index.html">Single long page</a></nav><h1>Learn order flow: {len(order)} pages</h1>'
    f'<p class="sub">One page per topic, each with the idea, how to read it, common mistakes and a chart or table produced by this repo. Everything runs on synthetic or toy data, so no page claims anything about real NSE behaviour.</p>{idx}{FOOT}</div></body></html>', encoding="utf-8")
print(f"wrote {len(order)} pages + docs/learn.html")
