from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CatalogEntry:
    object_id: str
    ra_deg: float
    dec_deg: float
    label: str
    split: str | None = None


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"object_id", "ra_deg", "dec_deg", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "split" not in df.columns:
        df["split"] = "train"
    return df
