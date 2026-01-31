from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import typer

from tccastro.config import AppConfig, load_config
from tccastro.eval.metrics import compute_metrics
from tccastro.eval.plots import save_confusion_matrix
from tccastro.features.classical import batch_extract_features
from tccastro.io.catalog import load_catalog
from tccastro.logging_utils import setup_logging
from tccastro.models.rf import load_model, save_model, train_rf
from tccastro.openset.scores import entropy_score, energy_score
from tccastro.preprocess.patches import stack_patches
from tccastro.surveys import get_client
from tccastro.utils.cache import CacheManager
from tccastro.utils.http import HttpClient

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _save_thumbnail(patch: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if patch.shape[0] == 3:
        img = np.transpose(patch, (1, 2, 0))
    else:
        img = patch[0]
    plt.imsave(path, img)


def _iter_catalog_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _build_http_client(config: AppConfig) -> HttpClient:
    return HttpClient(
        timeout_s=config.timeout_s,
        retries=config.retries,
        backoff_s=config.backoff_s,
        rate_limit_rps=config.rate_limit_rps,
    )


@app.command()
def download(
    catalog: Path = typer.Option(..., exists=True),
    survey: str = typer.Option("legacy"),
    bands: str = typer.Option("grz"),
    size: int = typer.Option(64),
    out: Path = typer.Option(Path("data/cache/cutouts")),
    config: Path | None = typer.Option(None),
    max_workers: int = typer.Option(4),
) -> None:
    """Download cutouts with cache and retry."""
    setup_logging()
    app_config = load_config(config) if config else AppConfig()
    cache = CacheManager(out)
    http_client = _build_http_client(app_config)
    client = get_client(survey, http_client=http_client)

    failures: list[dict[str, str]] = []

    def _process_row(row: dict[str, str]) -> None:
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        cache_path = cache.path_for(survey, ra, dec, bands, size)
        if cache.exists(cache_path):
            return
        try:
            result = client.fetch_cutout(ra, dec, size, bands)
            cache.ensure_dir(cache_path)
            np.savez_compressed(cache_path, array=result.array, meta=result.meta)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", row.get("object_id"), exc)
            failures.append({"object_id": row.get("object_id", ""), "error": str(exc)})

    rows = list(_iter_catalog_rows(catalog))
    if max_workers <= 1:
        for row in rows:
            _process_row(row)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_process_row, rows))

    if failures:
        fail_path = out / "failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["object_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
        logger.warning("Saved failures to %s", fail_path)


@app.command("build-dataset")
def build_dataset(
    catalog: Path = typer.Option(..., exists=True),
    survey: str = typer.Option("legacy"),
    size: int = typer.Option(64),
    out: Path = typer.Option(Path("data/outputs/dataset_legacy.npz")),
    bands: str = typer.Option("grz"),
    cache_dir: Path = typer.Option(Path("data/cache/cutouts")),
) -> None:
    """Build dataset NPZ from cached cutouts."""
    setup_logging()
    cache = CacheManager(cache_dir)
    df = load_catalog(catalog)
    labels = sorted(df["label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    patches: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        cache_path = cache.path_for(survey, ra, dec, bands, size)
        if not cache.exists(cache_path):
            raise FileNotFoundError(f"Missing cutout cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        patch = data["array"]
        patches.append(patch)
        meta_rows.append(
            {
                "object_id": row["object_id"],
                "ra_deg": ra,
                "dec_deg": dec,
                "label": row["label"],
                "split": row["split"],
            }
        )

    X = stack_patches(patches)
    y = np.array([label_to_id[row["label"]] for row in meta_rows], dtype=np.int64)
    meta = {
        "object_id": np.array([row["object_id"] for row in meta_rows], dtype=object),
        "ra_deg": np.array([row["ra_deg"] for row in meta_rows], dtype=np.float32),
        "dec_deg": np.array([row["dec_deg"] for row in meta_rows], dtype=np.float32),
        "label": np.array([row["label"] for row in meta_rows], dtype=object),
        "split": np.array([row["split"] for row in meta_rows], dtype=object),
        "survey": np.array([survey] * len(meta_rows), dtype=object),
        "labels": np.array(labels, dtype=object),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, **meta)
    logger.info("Saved dataset to %s", out)


@app.command("train-rf")
def train_rf_cmd(
    dataset: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(Path("data/outputs/rf_model.joblib")),
) -> None:
    setup_logging()
    data = _load_dataset(dataset)
    X = data["X"]
    y = data["y"]
    result = train_rf(X, y)
    save_model(result.model, out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    logger.info("Saved model to %s", out)


@app.command("eval")
def eval_cmd(
    dataset: Path = typer.Option(..., exists=True),
    model: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(Path("data/outputs/eval")),
) -> None:
    setup_logging()
    data = _load_dataset(dataset)
    X = data["X"]
    y_true = data["y"]
    labels = data["labels"].tolist()
    model_rf = load_model(model)
    feats = batch_extract_features(X)
    y_pred = model_rf.predict(feats)
    result = compute_metrics(y_true, y_pred)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "macro_f1": result.macro_f1,
                "balanced_accuracy": result.balanced_accuracy,
                "report": result.report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_confusion_matrix(result.confusion, labels, out / "confusion.png")


@app.command("rank-unknowns")
def rank_unknowns(
    dataset: Path = typer.Option(..., exists=True),
    model: Path = typer.Option(..., exists=True),
    method: str = typer.Option("entropy"),
    topk: int = typer.Option(100),
    out: Path = typer.Option(Path("data/outputs/unknowns")),
) -> None:
    setup_logging()
    data = _load_dataset(dataset)
    X = data["X"]
    labels = data["label"]
    object_ids = data["object_id"]
    model_rf = load_model(model)
    feats = batch_extract_features(X)
    probs = model_rf.predict_proba(feats)

    if method == "entropy":
        scores = entropy_score(probs)
    elif method == "energy":
        logits = np.log(probs + 1e-12)
        scores = energy_score(logits)
    else:
        raise ValueError(f"Unknown method: {method}")

    order = np.argsort(scores)[::-1]
    topk = min(topk, len(order))
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "unknowns_topk.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "object_id", "label", "score", "thumbnail"])
        for rank, idx in enumerate(order[:topk], start=1):
            thumb_path = out / "thumbnails" / f"{object_ids[idx]}.png"
            _save_thumbnail(X[idx], thumb_path)
            writer.writerow([rank, object_ids[idx], labels[idx], scores[idx], thumb_path])


if __name__ == "__main__":
    app()
