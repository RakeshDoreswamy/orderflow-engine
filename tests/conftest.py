import pytest

from orderflow.config import load_config
from orderflow.data.synthetic import generate_day


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def day(cfg):
    return generate_day(cfg)
