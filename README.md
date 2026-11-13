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

Open `docs/index.html` for the study page (concepts 1-23, each with definition, why it matters, how to read it, common mistakes and a generated visual). Rebuild with `python scripts/build_gallery.py && python scripts/build_site.py`.

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
3 book + events | 4 EOD providers + swingflow |
5 live recorder | 6 setups + backtest | 7 ICT bridge, Pine, Actions | 8 learning site.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow matplotlib pyyaml pytest
pytest
```
Targets Python 3.11; developed and tested on 3.10 so far. Copy `.env.example` to `.env` for broker credentials (Phase 5; git-ignored).
