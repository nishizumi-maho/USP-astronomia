from pathlib import Path

import pandas as pd

from tccastro.io.catalog import load_catalog


def test_load_catalog_adds_split(tmp_path: Path) -> None:
    csv_path = tmp_path / "catalog.csv"
    df = pd.DataFrame(
        {
            "object_id": ["a"],
            "ra_deg": [1.0],
            "dec_deg": [2.0],
            "label": ["spiral"],
        }
    )
    df.to_csv(csv_path, index=False)
    loaded = load_catalog(csv_path)
    assert "split" in loaded.columns
    assert loaded.loc[0, "split"] == "train"
