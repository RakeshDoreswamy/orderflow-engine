"""Add order-flow verdicts to an ict-swing-engine scan.

  python scripts/bridge_scan.py --scan ../ict-swing-engine/reports/scans/2026-11-12/scan.csv --daily-dir ../ict-swing-engine/data/raw [--as-of 2026-11-12]

--daily-dir holds one CSV per symbol (date, open, high, low, close, volume[, delivery_pct]). Output: <scan>_orderflow.csv.
Verdicts are evidence, not predictions: see orderflow/ict_bridge.py and docs/phase7_bridge_null.md."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.config import load_config  # noqa: E402
from orderflow.data.providers import CsvProvider  # noqa: E402
from orderflow.ict_bridge import bridge_scan, read_scan_csv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scan", required=True)
ap.add_argument("--daily-dir", required=True)
ap.add_argument("--as-of")
a = ap.parse_args()
cfg = load_config()
ent = read_scan_csv(a.scan)
prov = CsvProvider(a.daily_dir)
data = {}
for s in ent["symbol"].unique():
    try:
        data[s] = prov.fetch(s)
    except (FileNotFoundError, ValueError) as e:
        print(f"{s}: {e}")
out = bridge_scan(ent, data, cfg, a.as_of)
p = Path(a.scan).with_name(Path(a.scan).stem + "_orderflow.csv")
out.to_csv(p, index=False)
print(out[["symbol", "verdict", "score", "evidence"]].to_string(index=False))
print("wrote", p)
