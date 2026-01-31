# TCC Extragalactic Open-Set

Pipeline reprodutível em Python para construir datasets de cutouts astronômicos, treinar baselines clássicos e avaliar detecção open-set (desconhecidos) em cenários in-domain e cross-survey.

## Objetivos do MVP
- Baixar cutouts (Legacy Survey) a partir de catálogos RA/Dec.
- Construir dataset de patches normalizados e rotulados.
- Treinar baseline clássico explicável (features + RandomForest).
- Calcular scores open-set (entropia/energy) e gerar top-K anomalias.

## Instalação
```bash
cd tcc-extragalactic-open-set
poetry install
```

## Uso rápido (CLI)
```bash
poetry run tccastro --help
```

### Download real (online)
```bash
poetry run tccastro download \
  --catalog data/catalogs/sample_catalog.csv \
  --survey legacy \
  --bands grz \
  --size 64 \
  --out data/cache/cutouts
```

### Pipeline MVP (offline com mocks no teste)
Para rodar testes sem internet:
```bash
poetry run pytest
```

### Pipeline MVP (online)
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

poetry run tccastro rank-unknowns \
  --dataset data/outputs/dataset_legacy.npz \
  --model data/outputs/rf_model.joblib \
  --method entropy \
  --topk 100 \
  --out data/outputs/unknowns
```

## Estrutura de pastas
```
tcc-extragalactic-open-set/
  configs/
  data/
    catalogs/
    cache/
    outputs/
  src/tccastro/
    cli.py
    io/
    surveys/
    preprocess/
    features/
    models/
    openset/
    eval/
    utils/
  tests/
```

## Como adicionar um novo survey
Implemente um client em `src/tccastro/surveys/` seguindo a interface `SurveyClient`:
- `build_url(ra, dec, size, band)`
- `fetch_cutout(...)`
- `parse_to_array(...)`

Depois registre no `get_client` em `surveys/__init__.py`.

## Licença
MIT.
