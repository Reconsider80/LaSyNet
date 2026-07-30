"""Configuration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Save a configuration dict to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
