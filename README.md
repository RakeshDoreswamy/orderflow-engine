# orderflow-engine

Order flow taught from first principles and applied to real NSE India trades. Sits beside
[`ict-swing-engine`](https://github.com/RakeshDoreswamy/ict-swing-engine) and is meant to confirm or reject its entries.

> **Status: Phases 1 and 2 of 8 done.** Everything below is what exists today. Nothing here has been
> tested on real NSE data yet, and no trading results exist yet.

## Ground rules
- No lookahead; every detector gets a prefix test.
- Anything estimated from coarse data is named `*_est` and charts say "estimated".
- All thresholds, session times, tick sizes and costs live in `config.yaml`.
- Tick size must come from the broker instrument master in real runs (the config value is a synthetic/test fallback).
- Spoofing/layering is market manipulation under SEBI rules: this repo teaches recognition only.

## Phase 1 (done): skeleton, synthetic data, bars/footprint, trade-side classification
| module | what it does |
|---|---|
| `orderflow/config.py` + `config.yaml` | loads all parameters |
| `orderflow/data/synthetic.py` | seeded tick stream with known aggressor side and labelled events: `burst_up`, `absorption_bid`, `absorption_ask`, `sweep_reversal` |
| `orderflow/data/bars.py` | time / tick / volume bars; footprint (price row x bid_vol, ask_vol, trades) |
| `orderflow/classify.py` | tick rule, quote rule, Lee-Ready, accuracy report, label-flip helper for the later stress test |
| `orderflow/viz/footprint_chart.py` | footprint chart |

Run it: `python scripts/phase1_demo.py` (writes `docs/img/footprint_synthetic_sweep.png` and `docs/classifier_accuracy.csv`). Tests: `pytest`.

![footprint](docs/img/footprint_synthetic_sweep.png)

### Measured: classifier accuracy on synthetic data (17,014 trades, seed 42)
| method | coverage | acc (labelled) | acc (all) | acc (volume-weighted) |
|---|---|---|---|---|
| tick_rule | 0.9999 | 0.9228 | 0.9227 | 0.9361 |
| quote_rule | 0.9993 | 0.9934 | 0.9927 | 0.9947 |
| lee_ready | 1.0000 | 0.9933 | 0.9933 | 0.9953 |

**Caveat:** this is a toy generator where a trade always prints exactly at the bid or ask and the
attached quote is stale 15% of the time (`stale_quote_prob`). Real NSE accuracy will differ and
must be measured on real recorded data (Phase 5). These numbers only show the machinery works.

### Not generated yet
Iceberg refills, exhaustion and pulled spoofs need an order book (Phase 3).

## Phase 2 (done): delta/CVD, footprint analytics, profile/TPO, VWAP
| module | what it does |
|---|---|
| `orderflow/delta.py` | delta, delta %, intrabar max/min, delta at high/low, session and anchored CVD, confirmed swings, CVD divergences (stamped at the confirming bar), delta flips, delta without progress, OHLCV delta estimate |
| `orderflow/footprint.py` | per-bar POC and value area, diagonal and stacked imbalances, finished/unfinished extremes, zero rows |
| `orderflow/profile.py` | volume profile (session / composite / fixed range), value area by expansion, HVN/LVN, shape classifier (P, b, D, thin, double), naked POCs |
| `orderflow/tpo.py` | TPO letters, initial balance, extensions, single prints, poor/excess extremes, open-type heuristic |
| `orderflow/vwap.py` | session and anchored VWAP with sigma bands |
| `orderflow/viz/concepts.py` | one generated visual per concept 1-23 (`python scripts/build_gallery.py`) |

Open `docs/index.html` for the study page (concepts 1-27, each with definition, why it matters, how to read it, common mistakes and a generated visual). Rebuild with `python scripts/build_gallery.py && python scripts/build_site.py`.

### Measured on synthetic data (toy generator; nothing here is evidence about real NSE behaviour)
- CVD divergences found on the seed-42 day: 6 (4 bearish, 2 bullish). Predictive value is untested.
- OHLCV delta estimate vs true bar delta: correlation 0.98 (inflated by a few huge bars) but same sign in only 75% of 75 bars.
- Quote rule accuracy falls from 100% to 98.0% as the share of stale quotes goes 0% to 50%; the tick rule stays at 93.6%.
- Finished highs/lows are rare in the toy model (1.0% / 1.7% of 600 bars, only inside one-sided bursts).
- Open types on 8 synthetic days: {'open_auction': 8}. The toy market has no opening structure; the rules are verified on hand-built paths in the unit tests.

### Known limits / not done
- Concepts 16 (absorption) and 17 (trapped traders) are visualised on generator-labelled events; the detectors are Phase 3.
- The 80% value-area rule and gap-fill tendencies are listed as hypotheses; their test harness is Phase 6.
- Golden-image tests for charts are not written yet (smoke tests check each chart renders).
- One study page with 23 concept sections; the planned 41 separate pages come in Phase 8.

## Roadmap
3 book + events (done) | 4 EOD providers + swingflow (done) |
5 live recorder core (done; broker adapter not written) | 6 setups + backtest (done, synthetic only) | 7 ICT bridge, Pine, Actions | 8 learning site.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow matplotlib pyyaml pytest
pytest
```
Targets Python 3.11; developed and tested on 3.10 so far. Copy `.env.example` to `.env` for broker credentials (Phase 5; git-ignored).

## Phase 3 results: book analytics and event detectors (synthetic data)

Detectors: `orderflow/book.py` (depth imbalance, walls, pull/stack, spoof-like pulls, iceberg refills) and
`orderflow/events.py` (large prints, absorption, exhaustion, trapped traders, delta flip). Scored by
`scripts/phase3_eval.py` against the labels the generator planted.

**Protocol.** Thresholds were tuned only on 12 calibration days (seed base 500). The table below is 20 held-out
days (seed base 9000), run once, no tuning afterwards. Recall = labelled events with a detection inside
[start, end + 2 min]. Precision = detection episodes overlapping a labelled window. Intervals are Wilson 95%.

| detector | labelled | found | recall | recall_ci95 | detections | true_pos | precision | precision_ci95 | note |
|---|---|---|---|---|---|---|---|---|---|
| large_print | 100 | 89 | 0.89 | 0.81-0.94 | 154 | 89 | 0.58 | 0.50-0.65 |  |
| absorption | 21 | 21 | 1.00 | 0.85-1.00 | 53 | 53 | 1.00 | 0.93-1.00 |  |
| iceberg | 19 | 11 | 0.58 | 0.36-0.77 | 11 | 11 | 1.00 | 0.74-1.00 | small sample |
| exhaustion | 16 | 16 | 1.00 | 0.81-1.00 | 23 | 20 | 0.87 | 0.68-0.95 | small sample |
| trapped | 11 | 5 | 0.45 | 0.21-0.72 | 8 | 6 | 0.75 | 0.41-0.93 | small sample |
| delta_flip | 11 | 7 | 0.64 | 0.35-0.85 | 40 | 7 | 0.17 | 0.09-0.32 | small sample |
| wall | 18 | 18 | 1.00 | 0.82-1.00 | 351 | 192 | 0.55 | 0.49-0.60 | small sample |
| spoof_like | 174 | 158 | 0.91 | 0.86-0.94 | 344 | 158 | 0.46 | 0.41-0.51 |  |

Raw numbers: `docs/detector_scores.csv`.

**How to read this honestly.**
- The generator and the detectors share one author's mental model. This is a *consistency test* (does each rule find
  what it describes?), not evidence that these rules work on NSE data.
- Large print: the natural size distribution has a heavy tail (99th percentile ~15x the median), so a lower
  threshold fires on ordinary trades. The 30x threshold was picked on calibration days as a recall/precision trade-off.
- Wall precision is ~0.55 and spoof-like ~0.46: the tracker also flags the generator's ordinary deep orders, and a
  wall removed at the end of its window looks like a pull. Spoofing is recognition only, never a trade signal.
- Iceberg, trapped and delta-flip recall are modest, and delta-flip precision is low (~0.17). Treat delta flip as a
  context flag, not a signal.
- Tiny samples are marked "small sample"; their intervals are wide.

## Phase 4 results: EOD providers and swingflow

**Providers** (`orderflow/data/providers/`): `CsvProvider` (one CSV per symbol) and `parse_bhavcopy` /
`build_series_from_bhavcopies` for NSE security-wise delivery files you download yourself. They are offline: no
network calls, no credentials, no gap filling, and a broken frame raises. The bhavcopy parser matches columns by name and
was tested only on hand-built files shaped like the NSE layout, so check it against a real file before relying on it.

**Swingflow** (`orderflow/swingflow.py`): causal daily features (volume z-score, delivery % z-score, close-location value,
estimated delta `delta_est`, range vs prior ATR, position in the prior 20-day range), an *effort without result* detector
(heavy volume and delivery, small range, near a prior low or high), and a summary of the flow around each confirmed swing
(known only `swing_right` days after the swing). Every baseline uses previous days only; tests prove rows and detections
do not change when later data is removed.

**Scoring.** Thresholds were set from the rule's definition and checked on 6 calibration series (seed base 300; recall
0.85, precision 0.80); no tuning followed. The table is 20 held-out toy series (seed base 7000), 600 days each:

| variant | series | labelled | found | recall | recall_ci95 | detection_episodes | true_pos | precision | precision_ci95 |
|---|---|---|---|---|---|---|---|---|---|
| volume + delivery % | 20 | 240 | 217 | 0.90 | 0.86-0.94 | 399 | 323 | 0.81 | 0.77-0.85 |
| volume only (delivery % dropped) | 20 | 240 | 217 | 0.90 | 0.86-0.94 | 587 | 329 | 0.56 | 0.52-0.60 |

- This is a consistency test on a toy generator that plants the very pattern the rule looks for. It says nothing about whether
  the pattern means anything on NSE data, and the generator makes price memoryless afterwards, so there is no profit to read.
- Delivery % is what separates the two variants: with volume alone the detector fires on many more unlabelled days (precision 0.56).
- Chart: `docs/img/phase4_swingflow.png`. Raw numbers: `docs/eod_scores.csv`.

## Phase 5 results: live recorder (core only; no broker connection)

`orderflow/recorder.py` turns feed messages into per-day parquet parts (`ticks`, `book`) plus a `manifest.json`
under `data/store/<SYMBOL>/<date>/` (git-ignored). It drops out-of-session, out-of-order and malformed messages,
logs in-session silences longer than `recorder.gap_alert_s` as gaps, never guesses a trade when cumulative volume
goes backwards, and writes parts atomically (temp file then rename). `load_day` reads a day back and `quality_report`
lists facts that decide whether a recording is usable (gaps, off-tick prices, missing or crossed quotes, minutes with
trades). Tick size comes from `load_instrument_master` (an instruments CSV you download) and raises if the symbol is
missing; the config value is a fallback for synthetic data only. `scripts/record.py replay` records a captured raw feed
from a JSONL file.

**Verified (tests, 8 of them, plus `scripts/phase5_demo.py`).** A full synthetic day replayed as 17,014 broker-style
snapshot messages produced 17,013 trades (the first message only seeds the state) whose prices and quantities equal the
originals, with zero off-tick prices, zero crossed quotes and all 375 session minutes covered. The recorded bid/ask are
tested to be the previous message's quote.

**Not done, on purpose or by necessity**
- There is **no live broker adapter**. This sandbox has no market-data access and I will not ship an untested websocket
  client. Implement `orderflow.recorder.Feed` for your broker; the field names `normalize_snapshot` expects
  (`exchange_timestamp`, `last_price`, `volume_traded`, `depth.buy/sell[{price, quantity}]`) follow the common Kite-style
  full-mode layout from memory, so check them against your broker's documentation.
- Snapshot feeds merge trades: a recorded row is the volume increase since the previous message, which can be several
  real trades. Order flow from it is coarser than a true tape, and the aggressor side is never recorded (classify later,
  `_est` columns).
- The demo's classifier accuracy on recorded quotes (quote rule 99.99%) is an artifact of the toy: its book snapshots are never
  stale, whereas the generator's own tick quotes are stale 15% of the time (99.3%). Do not read it as evidence about real feeds.
- Never run against real data yet, so real-feed behaviour (reconnects, bad timestamps, pre-open ticks) is untested.

## Phase 6 results: setups, backtest with costs, controls, rule harnesses (all SYNTHETIC)

**Engine** (`orderflow/backtest/`). Bar-level, one position at a time, 100 shares, no compounding. A signal carries the
time it became known and fills at the OPEN of the next 1-minute bar plus 1 tick of adverse slippage; if stop and target
sit inside one bar the stop is assumed first; time exits are forced before the close. Costs (brokerage, STT, exchange,
SEBI, GST, stamp) hit every trade. **The cost rates in `config.yaml` are from memory; check them against your contract
note.** At 100 shares of a Rs 1000 stock a round trip costs about Rs 82, which is roughly 16 ticks of price movement
(Rs 5 per tick), so a setup must make that much gross per trade just to break even.

**Controls (does the engine itself behave?).** A driftless random walk with random signals must lose about the costs, and
a planted drift must show up:

| control | trades | mean_gross | mean_cost | mean_net | net_ci95 |
|---|---|---|---|---|---|
| negative control (no edge) | 188 | -12.5 | 82.4 | -94.9 | -118.8 to -72.3 |
| positive control (planted drift 3 ticks/min x 15 min) | 211 | 176.8 | 82.6 | 94.2 | 80.0 to 106.3 |

**Setups** (`orderflow/backtest/setups.py`): `absorption_reversal`, `exhaustion_fade`, `trapped_reversal`. Parameters
were fixed from the rules' definitions before any run (nothing was tuned on the calibration days, which are shown only as
a sanity check). Held-out = 30 synthetic days, run once:

| set | setup | trades | win_rate | mean_gross | mean_cost | mean_net | net_ci95 | net_total | max_dd | note |
|---|---|---|---|---|---|---|---|---|---|---|
| calibration (6 days, seed 800) | ALL | 28 | 0.11 | 33.6 | 82.4 | -48.8 | -88.1 to -0.3 | -1366 | 1509 | small sample |
| calibration (6 days, seed 800) | absorption_reversal | 17 | 0.00 | -2.4 | 82.5 | -84.8 | -107.5 to -60.1 | -1442 | 1310 | small sample |
| calibration (6 days, seed 800) | exhaustion_fade | 8 | 0.00 | -0.6 | 82.0 | -82.6 | -120.7 to -44.3 | -661 | 528 | small sample |
| calibration (6 days, seed 800) | trapped_reversal | 3 | 1.00 | 328.3 | 82.6 | 245.7 | 62.5 to 402.0 | 737 | 0 | small sample |
| held-out (30 days, seed 9500) | ALL | 136 | 0.13 | 43.1 | 82.5 | -39.3 | -57.1 to -18.6 | -5351 | 5594 |  |
| held-out (30 days, seed 9500) | absorption_reversal | 91 | 0.03 | 17.1 | 82.5 | -65.4 | -75.7 to -55.6 | -5952 | 5833 |  |
| held-out (30 days, seed 9500) | exhaustion_fade | 25 | 0.00 | -13.6 | 82.2 | -95.8 | -113.3 to -78.1 | -2396 | 2264 | small sample |
| held-out (30 days, seed 9500) | trapped_reversal | 20 | 0.75 | 232.5 | 82.7 | 149.8 | 65.1 to 219.3 | 2997 | 324 | small sample |

**How to read this honestly.**
- The generator and the setups share one author's assumptions. `trapped_reversal` is profitable here because the generator's
  planted trap event contains a reversal, so it wins by construction. `absorption_reversal` and `exhaustion_fade` lose because
  the generator pins price during absorption and then walks randomly, so there is nothing to follow. None of this is evidence
  about NSE.
- 91 / 25 / 20 trades per setup is a small sample; anything marked "small sample" has a wide interval. No profit claim is made.
- Across all setups the held-out net is negative with a confidence interval below zero: costs beat the gross edge.

**80% rule harness** (`orderflow/rules.py`, exact definition in its docstring). On 250 toy days with random overnight gaps
(the base generator has none, so a Gaussian gap is added), the market opened outside the prior value area on
168 days, was accepted back inside on 41, and traversed to the other side on 8:
**20% (95% CI 10% to 34%)** against the claimed 80%. A random-walk toy day is not expected to
obey the rule; this shows the harness reports what it measures instead of confirming the folklore. It needs real recorded
days (Phase 5) to say anything about NSE, and results depend on how acceptance is defined.

**Gap fill harness** on 20 toy EOD series (random-walk gaps, so small gaps fill often and large gaps rarely):

| abs_gap_pct | direction | gaps | filled | fill_rate | ci95 |
|---|---|---|---|---|---|
| [0.0, 0.2) | down | 3220 | 2967 | 0.92 | 0.91-0.93 |
| [0.0, 0.2) | up | 3118 | 2886 | 0.93 | 0.92-0.93 |
| [0.2, 0.5) | down | 2239 | 1722 | 0.77 | 0.75-0.79 |
| [0.2, 0.5) | up | 2312 | 1779 | 0.77 | 0.75-0.79 |
| [0.5, 1.0) | down | 517 | 290 | 0.56 | 0.52-0.60 |
| [0.5, 1.0) | up | 549 | 315 | 0.57 | 0.53-0.61 |
| [1.0, inf) | down | 5 | 0 | 0.00 | -0.00-0.43 |
| [1.0, inf) | up | 3 | 0 | 0.00 | 0.00-0.56 |

Raw numbers: `docs/phase6_*.csv`, held-out trades in `docs/phase6_heldout_trades.csv`.
