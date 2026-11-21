"""Configuration loading. Every tunable value lives in config/*.yaml."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT: Path = Path(os.environ.get("PROCURE_ROOT", Path(__file__).resolve().parents[2]))


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """Load config/settings.yaml (cached). Override the location with PROCURE_SETTINGS."""
    path = Path(os.environ.get("PROCURE_SETTINGS", ROOT / "config" / "settings.yaml"))
    return _read_yaml(path)


def load_rules_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the rule definitions (config/rules.yaml by default)."""
    return _read_yaml(Path(path) if path else resolve(load_settings()["paths"]["rules_file"]))


@lru_cache(maxsize=1)
def load_geo() -> dict[str, Any]:
    """Load LGA coordinates and name aliases (config/geo.yaml)."""
    return _read_yaml(resolve(load_settings()["paths"]["geo_file"]))


def resolve(relative: str | Path) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def path(key: str) -> Path:
    """Return an absolute path for a key under settings['paths'] and make sure dirs exist."""
    p = resolve(load_settings()["paths"][key])
    if key.endswith("_dir"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int | None = None) -> int:
    """Seed python, numpy and hashing for reproducibility. Returns the seed used."""
    s = int(load_settings()["seed"] if seed is None else seed)
    random.seed(s)
    np.random.seed(s)
    os.environ.setdefault("PYTHONHASHSEED", str(s))
    return s


def quick_mode() -> bool:
    """True when the quick-mode env var (smoke runs only) is set."""
    return os.environ.get(load_settings()["models"]["quick_mode_env"], "0") == "1"
