from orderflow.data.bars import footprint
from orderflow.viz.footprint_chart import plot_footprint


def test_footprint_chart_renders(cfg, day, tmp_path):
    w = day.ticks.iloc[:1500]
    fp = footprint(w, cfg, tick_size=day.tick_size, row_ticks=10)
    out = plot_footprint(fp, day.tick_size, 10, tmp_path / "fp.png")
    assert out.exists() and out.stat().st_size > 5000
