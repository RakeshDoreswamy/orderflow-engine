"""Recorder CLI.

  python scripts/record.py replay --jsonl raw.jsonl --symbol ABC --tick-size 0.05     # record a captured raw feed
  python scripts/record.py replay --jsonl raw.jsonl --symbol ABC --instruments instruments.csv
  python scripts/record.py live ...   # NOT IMPLEMENTED: needs a broker adapter that implements orderflow.recorder.Feed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.config import load_config  # noqa: E402
from orderflow.recorder import JsonlFeed, Recorder, load_instrument_master, tick_size_for  # noqa: E402

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("replay")
r.add_argument("--jsonl", required=True)
r.add_argument("--symbol", required=True)
r.add_argument("--tick-size", type=float)
r.add_argument("--instruments")
r.add_argument("--store")
sub.add_parser("live")
a = ap.parse_args()
if a.cmd == "live":
    sys.exit("live mode is not implemented here: write a Feed for your broker (see orderflow/recorder.py) and call Recorder.run(feed).")
cfg = load_config()
if a.instruments:
    tick = tick_size_for(load_instrument_master(a.instruments), a.symbol)
elif a.tick_size:
    tick = a.tick_size
else:
    sys.exit("give --instruments (preferred: tick size comes from the instrument master) or --tick-size")
root = Path(a.store or cfg.recorder.store_root)
st = Recorder(root, a.symbol, cfg, tick, source=f"jsonl:{a.jsonl}").run(JsonlFeed(a.jsonl))
print(st)
