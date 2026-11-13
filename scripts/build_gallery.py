"""Generate one visual per concept (1-27) into docs/img/concepts/ and docs/concept_facts.json.
Run: python scripts/build_gallery.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic import generate_day, generate_days, generate_eval_days  # noqa: E402
from orderflow.viz import concepts as C  # noqa: E402

root = Path(__file__).resolve().parents[1]
img = root / "docs" / "img" / "concepts"
cfg = load_config()
day = generate_day(cfg)
days = generate_days(cfg, 8)
eds = generate_eval_days(cfg, 2, seed_base=500, n_events=6)
jobs = {
    "01_auction": lambda o: C.c01_auction(day, cfg, o), "02_order_book": lambda o: C.c02_dom(cfg, o),
    "03_limit_vs_market": lambda o: C.c03_limit_market(cfg, o), "04_aggressor_side": lambda o: C.c04_aggressor(day, o),
    "05_slippage": lambda o: C.c05_slippage(o), "06_tape": lambda o: C.c06_tape(day, o),
    "07_classifiers": lambda o: C.c07_classify(day, cfg, o), "08_ohlcv_estimate": lambda o: C.c08_ohlcv(day, cfg, o),
    "09_delta_bars": lambda o: C.c09_delta_bars(day, cfg, o), "10_cvd_divergence": lambda o: C.c10_cvd(day, cfg, o),
    "11_flip_no_progress": lambda o: C.c11_flip(day, cfg, o), "12_footprint_modes": lambda o: C.c12_fp_modes(day, cfg, o),
    "13_poc_value_area": lambda o: C.c13_poc_va(day, cfg, o), "14_imbalances": lambda o: C.c14_imbalance(day, cfg, o),
    "15_finished_unfinished": lambda o: C.c15_finished(days, cfg, o), "16_absorption": lambda o: C.c16_absorption(day, cfg, o),
    "17_trapped": lambda o: C.c17_trapped(day, cfg, o), "18_volume_profile": lambda o: C.c18_profile(day, cfg, o),
    "19_profile_shapes": lambda o: C.c19_shapes(cfg, o), "20_tpo": lambda o: C.c20_tpo(day, cfg, o),
    "21_composite_naked": lambda o: C.c21_composite(days, cfg, o), "22_open_types": lambda o: C.c22_open(days, cfg, o),
    "23_vwap": lambda o: C.c23_vwap(day, cfg, o),
    "24_liquidity_heatmap": lambda o: C.c24_heatmap(eds[0], cfg, o), "25_iceberg": lambda o: C.c25_iceberg(eds[1], cfg, o),
    "26_spoofing_recognition": lambda o: C.c26_spoof(eds[0], cfg, o), "27_depth_imbalance": lambda o: C.c27_imbalance(eds[0], cfg, o),
}
facts = {}
for name, fn in jobs.items():
    p, f = fn(img / f"{name}.png")
    facts[name] = f
    print("ok", name)
(root / "docs" / "concept_facts.json").write_text(json.dumps(facts, indent=1, default=str))
