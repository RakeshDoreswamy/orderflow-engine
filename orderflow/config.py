"""Config loader. Everything tunable lives in config.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Config:
    """Read-only attribute view over a nested dict: ``cfg.synthetic.seed``."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, key: str) -> Any:
        try:
            v = self._d[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e
        return Config(v) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def with_overrides(self, overrides: dict) -> "Config":
        """New Config with ``overrides`` deep-merged (the original is untouched)."""
        return Config(_merge(self._d, overrides))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._d)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load config.yaml (or ``path``) and deep-merge optional ``overrides`` (used in tests)."""
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if overrides:
        data = _merge(data, overrides)
    return Config(data)
