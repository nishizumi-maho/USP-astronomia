from pathlib import Path

import numpy as np
import pandas as pd

from tccastro.cli import build_dataset, eval_cmd, rank_unknowns, train_rf_cmd
from tccastro.utils.cache import CacheManager


def test_end_to_end_mvp(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    data = {
        "object_id": [f"obj{i}" for i in range(6)],
        "ra_deg": np.linspace(10.0, 10.5, 6),
        "dec_deg": np.linspace(-2.0, -1.5, 6),
        "label": ["spiral", "elliptical", "spiral", "elliptical", "spiral", "elliptical"],
        "split": ["train"] * 6,
    }
    pd.DataFrame(data).to_csv(catalog_path, index=False)

    cache_dir = tmp_path / "cache"
    cache = CacheManager(cache_dir)
    rng = np.random.default_rng(42)
    for ra, dec in zip(data["ra_deg"], data["dec_deg"]):
        path = cache.path_for("legacy", float(ra), float(dec), "grz", 64)
        cache.ensure_dir(path)
        patch = rng.random((3, 64, 64), dtype=np.float32)
        np.savez_compressed(path, array=patch, meta={"ra": ra, "dec": dec})

    dataset_path = tmp_path / "dataset.npz"
    build_dataset(
        catalog=catalog_path,
        survey="legacy",
        size=64,
        out=dataset_path,
        bands="grz",
        cache_dir=cache_dir,
    )

    model_path = tmp_path / "rf_model.joblib"
    train_rf_cmd(dataset=dataset_path, out=model_path)
    assert model_path.exists()

    eval_dir = tmp_path / "eval"
    eval_cmd(dataset=dataset_path, model=model_path, out=eval_dir)
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "confusion.png").exists()

    unknown_dir = tmp_path / "unknowns"
    rank_unknowns(
        dataset=dataset_path,
        model=model_path,
        method="entropy",
        topk=3,
        out=unknown_dir,
    )
    assert (unknown_dir / "unknowns_topk.csv").exists()
