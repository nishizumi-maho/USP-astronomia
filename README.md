# USP Astronomia

## Overview
This repository contains the **TCC Extragalactic Open-Set** project: a reproducible Python pipeline for building astronomical cutout datasets, training classic baselines, and evaluating open-set (unknown) detection in both in-domain and cross-survey scenarios.

## Repository layout
```
USP-astronomia/
  README.md
  tcc-extragalactic-open-set/
```

The application lives inside the `tcc-extragalactic-open-set/` folder. That directory is a standalone Python project managed with Poetry.

## Application structure (`tcc-extragalactic-open-set/`)
```
tcc-extragalactic-open-set/
  configs/
    # Configuration files for experiments and runs.
  data/
    catalogs/  # Input catalogs (RA/Dec, labels, etc.).
    cache/     # Cached survey cutouts.
    outputs/   # Generated datasets, models, and evaluation outputs.
  src/tccastro/
    cli.py       # CLI entrypoint for the pipeline.
    io/          # I/O utilities for datasets and metadata.
    surveys/     # Survey clients and cutout retrieval.
    preprocess/  # Normalization and preprocessing steps.
    features/    # Feature extraction for classic baselines.
    models/      # Training and model persistence.
    openset/     # Open-set scoring (entropy/energy, etc.).
    eval/        # Evaluation and reporting utilities.
    utils/       # Shared helpers.
  tests/
    # Automated tests (offline-friendly; mocks for networked steps).
```

## What the app does
- Downloads survey cutouts from catalogs (RA/Dec).
- Builds normalized image patch datasets.
- Trains a classic, explainable baseline (features + RandomForest).
- Computes open-set scores (entropy/energy) and ranks top-K anomalies.

## Getting started
```bash
cd tcc-extragalactic-open-set
poetry install
poetry run tccastro --help
```

## Typical workflows
### Download cutouts (online)
```bash
poetry run tccastro download \
  --catalog data/catalogs/sample_catalog.csv \
  --survey legacy \
  --bands grz \
  --size 64 \
  --out data/cache/cutouts
```

### Build dataset, train model, evaluate (online)
```bash
poetry run tccastro build-dataset \
  --catalog data/catalogs/sample_catalog.csv \
  --survey legacy \
  --size 64 \
  --out data/outputs/dataset_legacy.npz

poetry run tccastro train-rf \
  --dataset data/outputs/dataset_legacy.npz \
  --out data/outputs/rf_model.joblib

poetry run tccastro eval \
  --dataset data/outputs/dataset_legacy.npz \
  --model data/outputs/rf_model.joblib \
  --out data/outputs/eval
```

### Rank unknowns (open-set)
```bash
poetry run tccastro rank-unknowns \
  --dataset data/outputs/dataset_legacy.npz \
  --model data/outputs/rf_model.joblib \
  --method entropy \
  --topk 100 \
  --out data/outputs/unknowns
```

### Run tests (offline)
```bash
poetry run pytest
```

## Extending to new surveys
Implement a new client under `src/tccastro/surveys/` following the `SurveyClient` interface:
- `build_url(ra, dec, size, band)`
- `fetch_cutout(...)`
- `parse_to_array(...)`

Then register it in `surveys/__init__.py`.

## License
MIT.
