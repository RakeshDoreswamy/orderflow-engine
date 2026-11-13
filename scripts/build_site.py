"""Build docs/index.html (Phase 1 study page) from measured outputs. Run: python scripts/build_site.py
Numbers in the accuracy table are read from docs/classifier_accuracy.csv (written by phase1_demo.py)."""
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
acc = pd.read_csv(root / "docs" / "classifier_accuracy.csv")
rows = "".join(
    f"<tr><td>{r.method}</td><td>{r.coverage:.2%}</td><td>{r.acc_classified:.2%}</td>"
    f"<td>{r.acc_all_vol:.2%}</td></tr>" for r in acc.itertuples())
n = int(acc.n.iloc[0])

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>orderflow-engine: order flow from first principles (Phase 1)</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#e3e1d8;--acc:#1d6b4f;--buy:#2e8b57;--sell:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe5;--mute:#9a9992;--card:#1f1f1c;--line:#34332e;--acc:#5fc79b;--buy:#5fc79b;--sell:#ef8a7d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{display:grid;grid-template-columns:230px 1fr;max-width:1080px;margin:0 auto;gap:32px;padding:24px}
nav{position:sticky;top:16px;align-self:start;font-size:14px}nav a{display:block;padding:4px 0;color:var(--mute);text-decoration:none}nav a:hover{color:var(--acc)}
nav b{display:block;margin:12px 0 4px;font-size:12px;letter-spacing:.06em;text-transform:uppercase}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:0 0 6px}h3{font-size:14px;margin:16px 0 4px;color:var(--acc);text-transform:uppercase;letter-spacing:.05em}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:18px 0}
.sub{color:var(--mute);margin:0 0 12px}.note{border-left:3px solid var(--acc);padding:6px 12px;background:rgba(29,107,79,.07);margin:12px 0;font-size:14.5px}
.warn{border-left-color:#b9770e;background:rgba(185,119,14,.08)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
code{background:rgba(127,127,127,.15);padding:1px 5px;border-radius:4px;font-size:.9em}pre{background:rgba(127,127,127,.12);padding:10px 12px;border-radius:8px;overflow:auto;font-size:13px}
img{max-width:100%;border:1px solid var(--line);border-radius:8px}ul{margin:4px 0 4px 20px;padding:0}
.ladder{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
.lad td{text-align:center;font-variant-numeric:tabular-nums}.lad .ask{color:var(--sell)}.lad .bid{color:var(--buy)}.lad tr.hit td{background:rgba(192,57,43,.18)}.lad tr.hitb td{background:rgba(46,139,87,.18)}
input,select,button{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}button{background:var(--acc);color:#fff;border:0;cursor:pointer}
@media(max-width:800px){.wrap{grid-template-columns:1fr}nav{position:static}.ladder{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<nav><b>Phase 1</b>
<a href="#a1">1 Auction &amp; balance</a><a href="#a2">2 The order book</a><a href="#a3">3 Limit vs market</a><a href="#a4">4 Aggressor side</a><a href="#a5">5 Slippage (try it)</a>
<a href="#b6">6 The tape</a><a href="#b7">7 Classifying trades</a><a href="#b8">8 OHLCV delta estimate</a><a href="#fp">Footprint demo</a>
<b>Coming</b><a href="#coming">Phases 2 to 8</a></nav>
<main>
<h1>Order flow from first principles</h1>
<p class="sub">orderflow-engine, Phase 1 study notes (concepts A and B). Companion to <code>ict-swing-engine</code>.</p>
<div class="note warn"><b>Honesty note.</b> All example numbers on this page are <b>illustrative or from the synthetic generator</b>, not real NSE data. Nothing here has been validated on real trades yet. Anything estimated is labelled <code>_est</code>.</div>

<section id="a1"><h2>1. Auction market theory</h2><p class="sub">Price advertises, time regulates, volume validates.</p>
<h3>Definition</h3><p>A market is an auction that looks for the price where both sides are willing to trade. <b>Balance</b>: price rotates in a range, both sides trade, volume is two-sided. <b>Imbalance</b>: one side dominates and price moves to find new counterparties. <b>Initiative</b> activity pushes price away from value (buyers lifting offers at highs). <b>Responsive</b> activity pushes it back (buyers stepping in below value).</p>
<h3>Why it matters</h3><p>It tells you what question to ask of every chart: is this move accepted (time and volume build there) or rejected (price returns fast)?</p>
<h3>How to read it</h3><ul><li>Price moves with little volume and quick return: advertised, not validated.</li><li>Price holds and volume builds at a new level: accepted, new value.</li><li>One-sided trade (almost all volume on one aggressor side) is an imbalance; two-sided trade is balance.</li></ul>
<h3>Common mistakes</h3><ul><li>Calling every breakout "initiative" without checking volume or time spent beyond the old range.</li><li>Treating balance as dead: most sessions are balance.</li></ul></section>

<section id="a2"><h2>2. The order book (DOM, Level 2)</h2>
<h3>Definition</h3><p>The list of resting limit orders: <b>bids</b> (buyers waiting) below, <b>asks</b> (sellers waiting) above. The <b>spread</b> is best ask minus best bid. <b>Depth</b> is the size resting at each level (NSE feeds typically give the best 5 levels; some give 20). Orders at one price fill in <b>price-time priority</b>: earlier orders first.</p>
<h3>Why it matters</h3><p>Resting size is the liquidity price must chew through. Queue position decides whether your limit order gets filled when price only touches your level.</p>
<h3>Common mistakes</h3><ul><li>Trusting displayed size: NSE <b>disclosed quantity</b> orders show only part of the order, so true size can be larger.</li><li>Assuming the book is the whole story: orders can be cancelled in microseconds.</li></ul>
<div class="note">Tick size varies by instrument and price band on NSE. This repo never hard-codes it for real runs; the <code>0.05</code> in <code>config.yaml</code> is a synthetic/test fallback only.</div></section>

<section id="a3"><h2>3. Limit orders vs market orders</h2>
<p><b>Limit orders</b> are passive: they rest in the book and <i>provide</i> liquidity. <b>Market orders</b> are aggressive: they trade immediately and <i>take</i> liquidity. A trade happens only when an aggressor meets a resting order.</p>
<h3>How to read it</h3><p>Every print has one aggressor and one passive side. "Volume" alone cannot say who was in a hurry; the aggressor side can.</p>
<h3>Common mistakes</h3><ul><li>"Buyers and sellers are always equal, so volume is neutral." True in count of shares, false about <i>urgency</i>: that is exactly what delta measures (Phase 2).</li></ul></section>

<section id="a4"><h2>4. Aggressor side</h2>
<p>A market buy <b>lifts the offer</b> (trade at the ask, side = +1). A market sell <b>hits the bid</b> (trade at the bid, side = -1). This one label is the base of delta, CVD, footprint and imbalance.</p>
<pre>side = +1  ->  counted in ask_vol   (buyer was aggressive)
side = -1  ->  counted in bid_vol   (seller was aggressive)
delta = ask_vol - bid_vol</pre>
<div class="note">NSE's public feeds do not give the aggressor flag for every trade in a way we can rely on, so most of this repo works with <b>estimated</b> sides (section 7). Estimated outputs carry the <code>_est</code> suffix.</div></section>

<section id="a5"><h2>5. Slippage and market impact (try it)</h2>
<p class="sub">Send a market order into an illustrative ladder (tick = ₹0.05). Bigger orders walk the book and pay more.</p>
<div class="ladder"><div><table class="lad" id="lad"><thead><tr><th>Price</th><th>Size</th><th>Side</th></tr></thead><tbody></tbody></table></div>
<div><p><select id="side"><option value="buy">Market BUY</option><option value="sell">Market SELL</option></select>
<input id="qty" type="number" value="1500" min="1" step="100" style="width:100px"> shares <button id="go">Send</button></p><div id="res"></div></div></div>
<h3>Why it matters</h3><p>Average fill price = Σ(price × size) ÷ Σ size. Slippage = average fill − the price you saw. Costs in <code>config.yaml</code> include <code>slippage_ticks</code> for exactly this.</p>
<h3>Common mistakes</h3><ul><li>Backtesting at the last price: a market order never fills at the last price for free.</li><li>Ignoring that the book refills or thins as you trade; this ladder is static.</li></ul></section>

<section id="b6"><h2>6. Time &amp; Sales (the tape)</h2>
<p>The tape is the list of trades: time, price, size (and aggressor side when known). Read for <b>large prints</b>, <b>bursts</b> (many trades in seconds) and <b>speed</b>.</p>
<pre>ts                    price    qty  side_true  bid      ask
2026-01-05 09:15:03   1000.05   47     +1      1000.00  1000.05   <- illustrative row</pre>
<h3>Common mistakes</h3><ul><li>Reading one big print as "an institution". It may be a block crossed between two parties or one order split on both sides.</li></ul></section>

<section id="b7"><h2>7. Classifying trades without exchange flags</h2>
<h3>The three rules (all causal: only past and present data)</h3>
<ul><li><b>Tick rule:</b> price up vs previous trade = buy, down = sell, unchanged = copy the last direction.</li><li><b>Quote rule:</b> at or above the ask = buy, at or below the bid = sell, inside the spread compare to the midpoint, exactly mid = unclassified.</li><li><b>Lee–Ready:</b> quote rule first, tick rule for what it cannot decide.</li></ul>
<h3>Measured error on the synthetic day (__N__ trades, seed 42)</h3>
<table><thead><tr><th>method</th><th>coverage</th><th>accuracy (labelled)</th><th>accuracy (volume-weighted)</th></tr></thead><tbody>__ROWS__</tbody></table>
<div class="note warn">This is a toy generator: trades always print exactly at bid or ask, and 15% of attached quotes are stale. Real NSE error rates must be measured on real recorded ticks (Phase 5). The ordering (quote rule beats tick rule) is expected; the exact figures are not transferable.</div>
<h3>Common mistakes</h3><ul><li>Treating estimated delta as exact. A 1% label error is small on one trade but can flip the sign of a quiet bar's delta.</li><li>Using the quote of <i>after</i> the trade: that is lookahead. The tests check that classifying a prefix gives the same labels as classifying the whole day.</li></ul></section>

<section id="b8"><h2>8. Bar-level estimate from OHLCV</h2>
<p>When only candles exist (e.g. Yahoo): <code>delta_est = volume × (2·close − high − low) / (high − low)</code>. It assumes volume splits by where the close sits in the bar's range.</p>
<h3>Why it is crude</h3><ul><li>A bar that dips then closes at the high gets "all buying" even if sellers dominated the dip.</li><li>It cannot see inside the bar, so absorption (heavy volume, no progress) is invisible: range is small, so the ratio is unstable.</li><li>Zero-range bars divide by zero and must be skipped.</li></ul>
<div class="note">Use it only as a labelled <code>delta_est</code>. This is the free-data fallback, not the real thing.</div></section>

<section id="fp"><h2>Footprint demo (synthetic, labelled sweep)</h2>
<p>Each cell is <code>bid × ask</code> volume at that price, 5-minute bars, true aggressor side from the generator. Black box = highest-volume row.</p>
<img src="img/footprint_synthetic_sweep.png" alt="Footprint chart of a synthetic sweep reversal">
<h3>How to read it</h3><ul><li><b>12:45 bar:</b> sellers aggress at every price on the way down (red cells, Δ strongly negative).</li><li><b>12:50 bar:</b> at the low, a huge bid-side volume and no further price progress: <b>absorption</b>, passive buyers soak up the selling.</li><li><b>12:55 bar:</b> buyers take over and the price returns up: the reversal.</li></ul>
<p>This is a planted event in synthetic data, so detecting it proves the machinery, not that real markets behave this way.</p></section>

<section id="coming"><h2>Coming next</h2><p>Phase 2: delta, CVD and divergence, footprint analytics (POC, value area, imbalances), volume profile, TPO, VWAP. Then book analytics, EOD swing flow, live recorder, setups and backtests with honest results, ICT bridge, Pine indicator.</p></section>
</main></div>
<script>
const book=[[100.20,1200,'a'],[100.15,800,'a'],[100.10,500,'a'],[100.05,300,'a'],[100.00,400,'b'],[99.95,600,'b'],[99.90,900,'b'],[99.85,1300,'b']];
const tb=document.querySelector('#lad tbody');
function draw(hit){tb.innerHTML=book.map((r,i)=>`<tr class="${hit&&hit.has(i)?(r[2]=='a'?'hit':'hitb'):''}"><td class="${r[2]=='a'?'ask':'bid'}">${r[0].toFixed(2)}</td><td>${r[1]}</td><td>${r[2]=='a'?'ask':'bid'}</td></tr>`).join('')}
draw();
document.getElementById('go').onclick=()=>{
  const side=document.getElementById('side').value,q0=+document.getElementById('qty').value;let q=q0,cost=0,filled=0;const hit=new Set();
  const lv=side=='buy'?[3,2,1,0]:[4,5,6,7];
  for(const i of lv){if(q<=0)break;const t=Math.min(q,book[i][1]);cost+=t*book[i][0];filled+=t;q-=t;hit.add(i)}
  draw(hit);const best=side=='buy'?100.05:100.00,avg=cost/filled,slip=Math.abs(avg-best);
  document.getElementById('res').innerHTML=`Filled <b>${filled}</b> of ${q0}${q>0?` (<span style="color:var(--sell)">${q} unfilled: book exhausted</span>)`:''}<br>Average price <b>${avg.toFixed(4)}</b><br>Slippage vs best ${side=='buy'?'ask':'bid'} (${best.toFixed(2)}): <b>₹${slip.toFixed(4)}</b> = ${(slip/0.05).toFixed(2)} ticks<br>Levels used: ${hit.size}`};
</script></body></html>"""

import json

F = json.loads((root / "docs" / "concept_facts.json").read_text())
IMG = {"a1": "01_auction", "a2": "02_order_book", "a3": "03_limit_vs_market", "a4": "04_aggressor_side",
       "a5": "05_slippage", "b6": "06_tape", "b7": "07_classifiers", "b8": "08_ohlcv_estimate"}
html = TEMPLATE.replace("__ROWS__", rows).replace("__N__", f"{n:,}")
html = html.replace("Phase 1 study notes (concepts A and B)", "study notes for concepts A to H (Phases 1 to 4)")
html = html.replace("orderflow-engine: order flow from first principles (Phase 1)", "orderflow-engine: order flow from first principles (Phases 1-4)")
for sid, name in IMG.items():
    a = html.index(f'<section id="{sid}">')
    b = html.index("</section>", a)
    html = html[:b] + f'<h3>Generated visual</h3><img src="img/concepts/{name}.png" alt="{name}" loading="lazy">' + html[b:]
f7, f8 = F["07_classifiers"], F["08_ohlcv_estimate"]
html = html.replace("<h3>Common mistakes</h3><ul><li>Treating estimated delta as exact.",
    f"<p>Sensitivity (same generator, different share of stale quotes): the tick rule does not use quotes so it stays at "
    f"{f7['accuracy_vol']['tick_rule']:.1%}; the quote rule falls from 100% with fresh quotes to "
    f"{f7['stale_sens']['quote_rule']['0.5']:.1%} when half the quotes are stale.</p>"
    "<h3>Common mistakes</h3><ul><li>Treating estimated delta as exact.")
html = html.replace("<h3>Why it is crude</h3>", f"<p><b>Measured on the synthetic day ({f8['n_bars']} five-minute bars):</b> correlation with true delta {f8['corr']:.2f}, "
    f"but the sign matches in only {f8['sign_agreement']:.0%} of bars. The correlation is inflated by a few huge one-sided bars; the sign agreement on ordinary bars is the more honest number.</p><h3>Why it is crude</h3>")

def sec(sid, title, sub, defin, why, read, mist, fact="", name=None, extra=""):
    img = f'<img src="img/concepts/{name}.png" alt="{title}" loading="lazy">' if name else ""
    fct = f'<div class="note">{fact}</div>' if fact else ""
    return (f'<section id="{sid}"><h2>{title}</h2><p class="sub">{sub}</p>{img}<h3>Definition</h3><p>{defin}</p>'
            f'<h3>Why it matters</h3><p>{why}</p><h3>How to read it</h3><ul>{"".join(f"<li>{x}</li>" for x in read)}</ul>'
            f'<h3>Common mistakes</h3><ul>{"".join(f"<li>{x}</li>" for x in mist)}</ul>{fct}{extra}</section>')

d10, d11, d15, d16, d17 = F["10_cvd_divergence"], F["11_flip_no_progress"], F["15_finished_unfinished"], F["16_absorption"], F["17_trapped"]
d18, d20, d21, d22 = F["18_volume_profile"], F["20_tpo"], F["21_composite_naked"], F["22_open_types"]
S = []
S.append(sec("c9", "9. Delta per bar", "Who was more aggressive inside the bar?",
  "delta = ask volume (buyer-initiated) - bid volume (seller-initiated). delta % = delta / volume. We also keep the running-delta maximum and minimum inside the bar and the delta of trades printed exactly at the bar's high and low.",
  "Two bars with the same close can hide very different fights. A bar that closes up with negative delta was pushed up by passive buyers while sellers hit the bid.",
  ["Look at sign and size of delta % first, then compare with the bar's price move.", "Whiskers (running max/min) show how much the bar swung before settling.", "Delta at the high negative = sellers were hitting the bid up there."],
  ["Reading delta without volume context: 500 shares of delta can be huge or nothing.", "Treating estimated delta (`delta_est`) as exact."],
  "This synthetic day has a few giant one-sided bars (the planted bursts); the chart uses a symmetric-log axis so ordinary bars stay visible.", "09_delta_bars"))
S.append(sec("c10", "10. Cumulative volume delta (CVD) and divergence", "Running total of delta, reset each session.",
  "CVD is the running sum of bar delta. A <b>bearish divergence</b>: price makes a higher swing high but CVD makes a lower high. <b>Bullish</b>: lower swing low, higher CVD low. Swings are only confirmed `swing_right` bars later, so a divergence is stamped at the confirming bar, never at the swing itself.",
  "It asks whether new price extremes were driven by aggressive flow or just drifted there.",
  ["Find two consecutive confirmed swings of the same kind in the same session.", "Compare price and CVD at those swings.", "Act only from the confirmation bar onward (dotted line on the chart)."],
  ["Marking swings with hindsight (the most common backtest error).", "Assuming a divergence means reversal: it often just means a pause."],
  f"On the synthetic day the rule found <b>{d10['divergences']}</b> divergences ({d10['bearish']} bearish, {d10['bullish']} bullish). This says nothing about whether they predict anything; that is tested in Phase 6 on real data.", "10_cvd_divergence"))
S.append(sec("c11", "11. Delta flip, exhaustion and delta without progress", "Aggression that changes sign, or that goes nowhere.",
  "<b>Delta flip</b>: delta % changes sign from one bar to the next with both bars above a minimum size. <b>Delta without progress</b>: large one-sided delta but price barely moves in that direction (close - open no more than a few ticks). Exhaustion (aggression dries up at an extreme) needs the events module in Phase 3.",
  "Progress without delta is a thin move; delta without progress hints that someone is absorbing the aggression.",
  ["Orange squares: big delta, price stuck.", "Purple dots: delta changed side.", "Judge both against what price did next, only on confirmed bars."],
  ["Calling every flip a reversal; in noise, delta flips constantly.", "Ignoring that thresholds (in `config.yaml`) decide how many you see."],
  f"Synthetic day: {d11['flips']} flips, {d11['no_progress']} no-progress bars with the default thresholds.", "11_flip_no_progress"))
S.append(sec("c12", "12. Reading a footprint cell", "Bid x ask at every price, in three views.",
  "A footprint splits each bar's volume by price. Each cell shows bid volume (sellers hit the bid) x ask volume (buyers lifted the offer). The delta view shows ask - bid per cell; the volume view shows the total.",
  "It is the same trades as the candle, but now you can see where inside the bar the fight happened.",
  ["Bid x ask view: green-ish cell = buyers more aggressive at that price.", "Delta view: find where the bar's delta was earned.", "Volume view: find the heavy prices."],
  ["Using too-fine rows so every cell looks empty (choose a row size that matches the bar's range)."], "", "12_footprint_modes"))
S.append(sec("c13", "13. POC and value area per bar", "Where most of the bar's volume traded.",
  "POC = the price row with the most volume (ties go to the row nearest the middle). Value area = the rows around the POC holding 70% of the volume, built by expansion: compare the next two rows above with the next two below, add the larger pair, repeat.",
  "The bar's value tells you which prices were accepted; trades far outside were rejected.",
  ["Black box = POC.", "Blue band = value area.", "A bar closing outside its own value area is worth a look."],
  ["Confusing the value-area share (about 70%) with a fixed price distance."], "", "13_poc_value_area"))
S.append(sec("c14", "14. Diagonal and stacked imbalances", "Aggression that the other side could not match.",
  "<b>Buy imbalance</b> at row P: ask[P] >= ratio x bid[P - 1 row] and ask[P] >= a minimum volume. <b>Sell imbalance</b> at P: bid[P] >= ratio x ask[P + 1 row] and bid[P] >= the minimum. Default ratio 3.0. The comparison is diagonal because a buyer at P is lifting orders that were resting one row above the bid at P-1. <b>Stacked</b> = at least 3 consecutive rows with the same-side imbalance.",
  "Stacked imbalances mark zones of one-sided aggression that may later act as support or resistance. That is a hypothesis to test, not a fact.",
  ["Thin border = single imbalance, thick = stacked.", "Zero on the opposite side counts if the minimum volume is met.", "The lowest row has no buy test and the highest has no sell test."],
  ["Reading single imbalances as signals: they are common.", "Forgetting the minimum-volume filter."],
  "The chart uses a planted one-sided burst, so nearly every row is a buy imbalance. Real data will be far less clean.", "14_imbalances"))
S.append(sec("c15", "15. Finished vs unfinished auctions", "Did the auction end at the extreme, or is business left there?",
  "Finished high = zero ask volume at the high row (nobody was still lifting offers up there). Finished low = zero bid volume at the low row. If volume traded on that side at the extreme, the auction is <b>unfinished</b> and the level may be revisited. In footprint terms an unfinished extreme is a poor high or low (the TPO version is separate, concept 20).",
  "It is a rule of thumb for which extremes might be retested. The retest idea is a hypothesis for Phase 6.",
  ["Look only at the extreme rows (this page shows just those).", "Header of each bar: FINISHED or unfinished.", "Row size changes the answer: coarser rows are less likely to be finished."],
  ["Treating 'unfinished' as a guaranteed retest."],
  f"In this toy model finished extremes are rare: across {d15['bars']} bars (8 days) finished highs occurred in {d15['finished_high_share']:.1%} and finished lows in {d15['finished_low_share']:.1%}, only inside one-sided bursts. The real NSE frequency is unknown until measured.", "15_finished_unfinished"))
S.append(sec("c16", "16. Absorption", "Heavy aggressive volume, no price progress.",
  "Passive orders soak up aggressive flow at a level, so price does not move even though plenty trades. Exhaustion is the opposite: aggressive volume dries up at an extreme.",
  "Absorption at a known level is the core of the reversal setups (Phase 6).",
  ["Big one-sided delta, tiny price range.", "Look at who is passive: sellers absorbed here would show as bid volume repeatedly refilled."],
  ["Calling every quiet high-volume bar absorption: you need the aggressive side to be one-sided."],
  f"The generator planted absorption at the bid: during the window {d16['sold']:,} shares were sold aggressively vs {d16['bought']:,} bought, and price moved {d16['range_ticks']:.0f} tick. A detector is built in Phase 3 and will be scored against these labels.", "16_absorption"))
S.append(sec("c17", "17. Trapped traders", "Aggression at an extreme, then reversal.",
  "Traders who aggressed at the extreme are now on the wrong side: sellers who hit the bid at the low when price reverses up are trapped; their stops/covering fuel the move.",
  "It explains why sweeps can reverse sharply.",
  ["Find the extreme row with heavy aggressive volume.", "Check price leaving the level and not returning."],
  ["Inferring 'trapped' from price alone: you need the aggressive volume at that level."],
  f"In the planted sweep, {d17['sold_at_low']:,} shares were sold at the bid at the low; price then recovered {d17['recovery_ticks']:.0f} ticks. Labelled by the generator; detector in Phase 3.", "17_trapped"))
S.append(sec("c18", "18. Volume profile", "Volume by price instead of by time.",
  "POC = most-volume row; value area = 70% of volume built by expansion from the POC; VAH/VAL = its upper/lower rows (we label rows by their lower edge); HVN/LVN = peaks and valleys of a smoothed profile.",
  "HVNs act like magnets (acceptance), LVNs like air pockets (price moves through them fast): hypotheses to test.",
  ["Where is the POC relative to the range?", "Is price inside or outside the value area?", "LVN between two HVNs: expect quick travel."],
  ["Mixing profiles of different time ranges without saying so."],
  f"Synthetic day: POC {d18['poc']:.2f}, value area {d18['val']:.2f} to {d18['vah']:.2f} ({d18['va_share']:.1%} of volume), {d18['hvn']} HVN and {d18['lvn']} LVN with the default smoothing.", "18_volume_profile"))
S.append(sec("c19", "19. Profile shapes", "P, b, D, thin and double distribution.",
  "P: volume concentrated high (short covering / buying). b: concentrated low (long liquidation / selling). D: symmetrical balance. Thin/trend: flat, no clear POC. Double: two separated peaks.",
  "Shape is a quick description of the day's auction.",
  ["Check where the POC sits and where the volume mass is.", "Two peaks with a thin valley = two value areas."],
  ["Over-trusting a label: the classifier is a heuristic with thresholds in config."],
  "These five profiles are <b>constructed</b> to illustrate and unit-test the classifier. It correctly labels all five; that proves the code, not that real days fall into these boxes.", "19_profile_shapes"))
S.append(sec("c20", "20. Market profile (TPO)", "Price by time, in 30-minute letters.",
  "Each 30-minute period gets a letter (A, B, C...). A row's letters show which periods traded there. Initial balance (IB) = the first hour's range. Range extension = price trading outside the IB. Single prints = rows with one letter. Poor high/low = the extreme row has two or more letters (retested). Excess/tail = at least two single-print rows at the extreme.",
  "TPO shows time spent at each price, which volume profile hides.",
  ["Orange = single print (fast travel).", "Compare the day to the IB: which side extended and when.", "Poor extremes may be revisited; excess suggests rejection (hypotheses)."],
  ["Using letters from an unfinished period for decisions."],
  f"Synthetic day: {d20['letters']} periods, IB {d20['ibl']:.2f} to {d20['ibh']:.2f}, extension(s) {d20['extensions']}, poor high {d20['poor_high']}, poor low {d20['poor_low']}, excess high {d20['excess_high']}, excess low {d20['excess_low']}.", "20_tpo"))
S.append(sec("c21", "21. Session, composite and naked POCs", "Profiles over longer ranges.",
  "Session profile = one day. Composite = several days combined. Fixed range = any start/end you choose. A <b>naked POC</b> is a past session's POC that no later session has traded through.",
  "Untested POCs are watch levels; whether price is drawn to them is a hypothesis for Phase 6.",
  ["Grey dotted = a POC that was later tested; red solid = still naked.", "Composite POC shows where most volume traded across days."],
  ["Counting the most recent day's POC as 'naked' evidence: it simply has not had time to be tested."],
  f"8 synthetic days: {d21['naked']} of {d21['days']} session POCs are naked (the last one always counts); composite POC {d21['composite_poc']:.2f}.", "21_composite_naked"))
S.append(sec("c22", "22. Rules of thumb to TEST, not believe", "80% value-area rule, open types, gap fills.",
  "Open types (open-drive, open-test-drive, open-rejection-reverse, open-auction) are classified from the first 30 minutes with explicit rules in `config.yaml`. The 80% rule (a day that opens outside value, re-enters it and is accepted tends to rotate across it) and gap-fill tendencies are listed here as hypotheses; their test harness belongs to Phase 6.",
  "Traders repeat these as facts. This repo makes them measurable.",
  ["Top row: constructed paths that the rules classify correctly (unit-tested).", "Bottom row: synthetic days."],
  ["Believing a rule because it sounds right; the sample size and out-of-sample result decide."],
  f"On {d22['n_days']} synthetic days the classifier output was {d22['synthetic_counts']}. The synthetic market is a random walk with no opening structure, so plain auction opens are expected. <b>No result about real NSE opens exists yet.</b>", "22_open_types"))
S.append(sec("c23", "23. VWAP, anchored VWAP and bands", "The volume-weighted average price.",
  "Session VWAP = cumulative (price x volume) / cumulative volume since the open; it restarts every day. Anchored VWAP starts at a chosen event (swing, gap, news). Bands are +/- k volume-weighted standard deviations of price around VWAP.",
  "Institutions are often benchmarked to VWAP, so it is a widely watched reference. Why price reacts to it is a hypothesis.",
  ["Price above VWAP and rising VWAP: buyers in control of the average.", "Anchored VWAP from a swing: the average price of everyone who traded since."],
  ["Anchoring with hindsight (the chart does exactly that, only to illustrate; real signals must anchor at an already-confirmed event).", "Expecting bands to behave like a fixed percentage."],
  "", "23_vwap"))
d25, d26, d27 = F["25_iceberg"], F["26_spoofing_recognition"], F["27_depth_imbalance"]
S.append(sec("c24", "24. Liquidity heat map", "Resting size per price, over time.",
  "Each column is a book snapshot; colour is the displayed size at each price (log scale). Trades are drawn on top. Only the best 5 levels per side are recorded, so anything deeper is invisible.",
  "It shows where resting liquidity sat when trades arrived: a dark row that trades keep touching is a wall being tested.",
  ["Dark horizontal band = large resting order.", "If trades eat into the band and it thins, it is being consumed; if it vanishes with no trades, it was pulled."],
  ["Reading displayed size as intent. Orders can be cancelled at any time.", "Forgetting that hidden (iceberg) size does not show."],
  "Synthetic data with a planted bid wall. No claim about real NSE books.", "24_liquidity_heatmap"))
S.append(sec("c25", "25. Iceberg (disclosed-quantity) orders", "Small displayed size, large executed size.",
  "NSE allows a disclosed quantity: only part of an order is shown and the rest refills after each fill. The detector flags a price where, within a run, refills repeat, executed volume is at least 5x the largest displayed size and the refilled sizes are nearly constant.",
  "A level that keeps absorbing sells without its displayed size shrinking suggests a larger hidden buyer. This is an inference, never a certainty.",
  ["Top: displayed size stays near the disclosed quantity.", "Bottom: cumulative executed volume at that price keeps climbing far beyond it."],
  ["Calling every busy level an iceberg: stale snapshots and ordinary liquidity can look similar.", "Trading it as a signal without testing it."],
  f"Here the detector fired at {d25['price']:.2f}: executed {d25.get('executed', 0):,.0f} against a displayed {d25.get('displayed', 0):,.0f} after {d25.get('refills', 0)} refills (synthetic). Held-out recall is only about 0.58 (see scores below).", "25_iceberg"))
S.append(sec("c26", "26. Spoofing: recognition only", "Large size that appears and disappears without trading.",
  "A spoof-like pull is a large resting order that is cancelled within a short life with almost nothing traded against it. The right panel shows a genuine wall for contrast: it sits, is consumed, and trades against it.",
  "Knowing the pattern prevents mistaking a vanishing wall for real support or resistance. Spoofing is illegal; this repo recognises the pattern and never trades it.",
  ["Large size, short life, near-zero traded = spoof-like.", "Large size, long life, traded against = a real wall being tested."],
  ["Claiming intent: a cancelled order is not proof of manipulation.", "Using the flag as a signal."],
  f"On this synthetic day the tracker found {d26['episodes']} large-order episodes, {d26['pulled_episodes']} of them pulled. Spoof-like precision on held-out days is only about 0.46: the generator's ordinary deep orders and wall removals also look like pulls.", "26_spoofing_recognition"))
S.append(sec("c27", "27. Depth imbalance", "Which side has more resting size near the touch.",
  "(bid size - ask size) / (bid size + ask size), summed over the best N levels (N = 3 in config.yaml). It ranges from -1 (all ask) to +1 (all bid).",
  "A common idea is that a heavy bid supports price. Whether that predicts anything is an empirical question, and the displayed book can be pulled.",
  ["Positive = more bid size; negative = more ask size.", "Check it against what happens next on real data before believing it."],
  ["Treating it as a prediction. On this toy day its 5-second rank correlation with the next mid move is " + f"{d27['forward_ic_5s']:.2f}" + ", which comes from the toy book's own mechanics.", "Ignoring that spoofed size inflates it."],
  "", "27_depth_imbalance"))
_sc = (root / "docs" / "detector_scores.md").read_text().strip().splitlines()
_hd = [c.strip() for c in _sc[0].strip("|").split("|")]
_tb = "<table><tr>" + "".join(f"<th>{c}</th>" for c in _hd) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in _sc[2:]) + "</table>"
S.append(sec("c28", "Event detectors and how well they were scored", "Absorption, exhaustion, trapped traders, delta flip, large prints, walls, icebergs.",
  "Each detector is a plain rule in <code>orderflow/events.py</code> or <code>orderflow/book.py</code> with thresholds in <code>config.yaml</code>. They only use data up to the moment they fire (prefix-tested).",
  "A rule that sounds right is not a rule that works. Scoring against planted labels at least shows whether each rule finds what it claims to find.",
  ["Recall: labelled events found. Precision: detections that matched a label.", "Intervals are Wilson 95%; small samples are flagged."],
  ["Reading these as market performance. The generator and the detectors share one author's assumptions, so this is a consistency test only.", "Ignoring the weak ones: delta flip precision is about 0.17."],
  "Thresholds tuned on 12 calibration days, scored once on 20 held-out synthetic days.", None, _tb))
_e = (root / "docs" / "eod_scores.md").read_text().strip().splitlines()
_eh = [c.strip() for c in _e[0].strip("|").split("|")]
_et = "<table><tr>" + "".join(f"<th>{c}</th>" for c in _eh) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in _e[2:]) + "</table>"
S.append(sec("c29", "Daily (EOD) order-flow proxies and swing flow", "What daily bars and delivery % can and cannot say.",
  "Daily bars have no aggressor side, so flow is estimated: volume and delivery % z-scores against the previous 20 days, close-location value, and an estimated delta (<code>delta_est</code>). <i>Effort without result</i> = heavy volume and delivery with a small range, placed near a prior low or high.",
  "End-of-day data is what most traders can get for free. NSE publishes delivered quantity, which separates shares that changed hands to stay from intraday round trips.",
  ["Yellow bands are planted episodes in a toy series; triangles are detections.", "A swing is known only 3 days after it forms; its flow summary uses days up to then."],
  ["Reading delivery % as institutional buying: it is not split by participant.", "Using the swing flow label before its confirmation date."],
  "Toy data; the generator plants the pattern, so scores only show the rule does what it says. Held-out table below.", None,
  '<img src="img/phase4_swingflow.png" alt="phase 4 swingflow" loading="lazy">' + _et))
extra_nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in [("c9", "9 Delta per bar"), ("c10", "10 CVD & divergence"), ("c11", "11 Flip / no progress"), ("c12", "12 Footprint cell"), ("c13", "13 POC & value area"), ("c14", "14 Imbalances"), ("c15", "15 Finished / unfinished"), ("c16", "16 Absorption"), ("c17", "17 Trapped traders"), ("c18", "18 Volume profile"), ("c19", "19 Profile shapes"), ("c20", "20 TPO"), ("c21", "21 Composite & naked POC"), ("c22", "22 Rules to test"), ("c23", "23 VWAP"), ("c24", "24 Liquidity heat map"), ("c25", "25 Iceberg"), ("c26", "26 Spoofing"), ("c27", "27 Depth imbalance"), ("c28", "Detector scores"), ("c29", "EOD swing flow")])
html = html.replace('<a href="#fp">Footprint demo</a>', '<a href="#fp">Footprint demo</a>' + extra_nav)
html = html.replace('<section id="coming">', "".join(S) + '<section id="coming">')
html = html.replace("<h2>Coming next</h2><p>Phase 2: delta, CVD and divergence, footprint analytics (POC, value area, imbalances), volume profile, TPO, VWAP. Then book analytics",
                    "<h2>Coming next</h2><p>Phase 5: live recorder, live recorder, setups and backtests with honest results (including the rules of thumb above), ICT bridge, Pine indicator. Book analytics")
(root / "docs" / "index.html").write_text(html, encoding="utf-8")
print("wrote docs/index.html")
