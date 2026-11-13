"""Golden-image tests.
* FRESH: charts re-rendered by the current code must stay close to their golden thumbnail (catches code regressions).
* PUBLISHED: every committed docs/img/concepts/*.png must match its golden thumbnail (catches an accidentally changed or
  overwritten chart). Both are deliberately coarse; a deliberate chart change means running scripts/update_golden.py."""
from pathlib import Path

import pytest

from tests.golden_util import distance, load_golden, thumb
from orderflow.config import load_config
from orderflow.data.synthetic import generate_day
from orderflow.viz import concepts as C

ROOT = Path(__file__).resolve().parents[1]
TOL, AR_TOL = 0.02, 0.02

FRESH = {
    "c02": lambda cfg, day, o: C.c02_dom(cfg, o), "c03": lambda cfg, day, o: C.c03_limit_market(cfg, o),
    "c05": lambda cfg, day, o: C.c05_slippage(o), "c19": lambda cfg, day, o: C.c19_shapes(cfg, o),
    "c09": lambda cfg, day, o: C.c09_delta_bars(day, cfg, o), "c13": lambda cfg, day, o: C.c13_poc_va(day, cfg, o),
    "c16": lambda cfg, day, o: C.c16_absorption(day, cfg, o), "c23": lambda cfg, day, o: C.c23_vwap(day, cfg, o),
}
PUBLISHED = sorted(p.stem for p in (ROOT / "docs" / "img" / "concepts").glob("*.png"))


@pytest.fixture(scope="module")
def golden():
    return load_golden()


@pytest.fixture(scope="module")
def env():
    cfg = load_config()
    return cfg, generate_day(cfg)


@pytest.mark.parametrize("name", sorted(FRESH))
def test_fresh_chart_matches_golden(name, golden, env, tmp_path):
    cfg, day = env
    p = FRESH[name](cfg, day, tmp_path / f"{name}.png")[0]
    t, ar = thumb(p)
    assert abs(ar / float(golden[f"fresh_{name}_ar"]) - 1) < AR_TOL, "chart shape changed"
    assert distance(t, golden[f"fresh_{name}"]) < TOL, "chart looks different from its golden image"


@pytest.mark.parametrize("stem", PUBLISHED)
def test_published_chart_matches_golden(stem, golden):
    assert f"pub_{stem}" in golden, "new chart without a golden image: run scripts/update_golden.py"
    t, ar = thumb(ROOT / "docs" / "img" / "concepts" / f"{stem}.png")
    assert abs(ar / float(golden[f"pub_{stem}_ar"]) - 1) < AR_TOL
    assert distance(t, golden[f"pub_{stem}"]) < TOL


def test_golden_detects_a_changed_chart(golden, tmp_path):
    """The check must be able to fail: a flipped copy of a chart must exceed the tolerance."""
    import matplotlib.image as mpimg
    import numpy as np
    f = ROOT / "docs" / "img" / "concepts" / "09_delta_bars.png"
    im = mpimg.imread(str(f))
    mpimg.imsave(str(tmp_path / "flip.png"), np.flipud(im))
    t, _ = thumb(tmp_path / "flip.png")
    assert distance(t, golden["pub_09_delta_bars"]) > TOL
