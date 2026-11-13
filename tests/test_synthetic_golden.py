"""Regression guard: the default synthetic day and the 3-day set must never change silently
(earlier charts, README numbers and tests depend on them). If you change the generator ON PURPOSE,
update these fingerprints and regenerate docs."""
import pandas as pd

from orderflow.data.synthetic import generate_day, generate_days


def _h(df):
    return sum(map(int, pd.util.hash_pandas_object(df, index=False).to_numpy())) % (2**63)


def test_default_day_fingerprint(cfg):
    d = generate_day(cfg)
    assert len(d.ticks) == 17014 and _h(d.ticks) == 276836567801474357


def test_three_day_set_fingerprint(cfg):
    ds = generate_days(cfg, 3)
    assert sum(_h(x.ticks) for x in ds) % (2**63) == 7113352183213557814
