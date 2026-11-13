"""Regenerate tests/golden/thumbs.npz. Run only when a chart was CHANGED ON PURPOSE, and look at the new chart first."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from tests.golden_util import GOLDEN, thumb  # noqa: E402
from orderflow.config import load_config  # noqa: E402
from orderflow.data.synthetic import generate_day  # noqa: E402
from orderflow.viz import concepts as C  # noqa: E402
from tests.test_golden_images import FRESH, PUBLISHED  # noqa: E402

cfg = load_config()
day = generate_day(cfg)
out = {}
with tempfile.TemporaryDirectory() as td:
    for name, fn in FRESH.items():
        p = fn(cfg, day, Path(td) / f"{name}.png")[0]
        t, ar = thumb(p)
        out[f"fresh_{name}"], out[f"fresh_{name}_ar"] = t, np.array(ar)
root = Path(__file__).resolve().parents[1]
for f in sorted((root / "docs" / "img" / "concepts").glob("*.png")):
    t, ar = thumb(f)
    out[f"pub_{f.stem}"], out[f"pub_{f.stem}_ar"] = t, np.array(ar)
GOLDEN.parent.mkdir(exist_ok=True)
np.savez_compressed(GOLDEN, **out)
print("wrote", GOLDEN, len(out) // 2, "images")
