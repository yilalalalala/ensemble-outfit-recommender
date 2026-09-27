"""Configuration loading. All tunables come from configs/*.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict with attribute access and repo-relative path resolution."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        return ROOT / self["paths"][key]


def load_config(name: str = "default") -> Config:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return Config(yaml.safe_load(f))
