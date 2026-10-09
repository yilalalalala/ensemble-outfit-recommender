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
        # Cloud deployments mount data independently from the application image.  A
        # per-path override keeps the checked-in configuration useful locally while
        # letting Modal point the same code at its persistent Volume.
        import os
        override = os.environ.get(f"ENSEMBLE_PATH_{key.upper()}")
        return Path(override) if override else ROOT / self["paths"][key]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(name: str | None = None) -> Config:
    """Load configs/<name>.yaml (default: $ENSEMBLE_CONFIG or "default"). An experiment config may
    set ``extends: default`` and override only the keys it changes."""
    import os
    name = name or os.environ.get("ENSEMBLE_CONFIG", "default")
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        data = yaml.safe_load(f)
    parent = data.pop("extends", None)
    if parent:
        data = _merge(dict(load_config(parent)), data)
    return Config(data)
