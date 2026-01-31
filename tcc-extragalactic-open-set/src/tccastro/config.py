from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    survey: str = "legacy"
    bands: str = "grz"
    size: int = 64
    rate_limit_rps: float = 2.0
    max_workers: int = 4
    timeout_s: int = 20
    retries: int = 3
    backoff_s: float = 1.0


def load_config(path: Path) -> AppConfig:
    data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    return AppConfig(**data)
