# orderflow-engine

Order flow taught from first principles and applied to real NSE India trades. Sits beside
[`ict-swing-engine`](https://github.com/RakeshDoreswamy/ict-swing-engine) and is meant to confirm or reject its entries.

> **Status: Phase 1 of 8 done.** Everything below is what exists today. Nothing here has been
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

## Roadmap
2 delta/CVD, footprint analytics, profile/TPO, VWAP | 3 book + events | 4 EOD providers + swingflow |
5 live recorder | 6 setups + backtest | 7 ICT bridge, Pine, Actions | 8 learning site.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy pyarrow matplotlib pyyaml pytest
pytest
```
Targets Python 3.11; developed and tested on 3.10 so far. Copy `.env.example` to `.env` for broker credentials (Phase 5; git-ignored).
