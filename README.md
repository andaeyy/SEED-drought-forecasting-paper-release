# SEED drought forecasting paper release

This repository is the public, paper-aligned implementation for *Integrating the E3SM Land Model (ELM) and Deep Learning for Drought Forecasting*. SEED uses seven historical NLDAS forcing fields to predict ELM-derived evapotranspiration (ET) or soil moisture (SM) at 7-, 30-, and 90-day endpoint leads. It contains preprocessing, three supervised ConvLSTM families, frozen target-specific checkpoints, independent 2020 evaluation, joint ET-SM dryness translation, runtime benchmarking, the FastAPI/Next.js application, and paper figure resources.

## Supported versions

- Python 3.10-3.13; Python 3.11 is recommended.
- Node.js 20 LTS and npm 10 for the map application.
- TensorFlow 2.16-2.20 for checkpoint inference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,figures,ml,api,test]'
cd apps/web && npm ci && cd ../..
git lfs pull
```

The six `.keras` checkpoints are tracked with Git LFS.

## External data

Raw NLDAS and full ELM archives are not committed. See [docs/data.md](docs/data.md) for acquisition, expected filenames, variables, grids, and validation. Set the paths in `apps/api/.env.example` or pass explicit paths to commands.

## Preprocessing

```bash
python scripts/prepare_data/prepare_nldas.py data/raw/clmforc.nldas.2020-*.nc \
  --output data/processed/clmforc.nldas.2020.daily.nc
python scripts/prepare_data/prepare_elm_targets.py data/raw/elm_history.nc \
  --et-output data/processed/elm_et.nc --sm-output data/processed/elm_sm.nc
```

Precipitation is the daily sum of hourly `PRECTmms`; `TBOT`, `WIND`, `QBOT`, `PSRF`, `FSDS`, and `FLDS` use daily means. ET is `(QSOIL + QVEGE + QVEGT) * 86400`, clipped below zero. SM is `H2OSOI` layer index 2.

## Training and selection

Each selected target-and-lead configuration is under `configs/models/{et,sm}`. To inspect the exact model graph:

```bash
python scripts/train/train_selected.py configs/models/et/7day.json --summary-only
```

The data-driven training procedure, 2015-2018 split, 2019 selection, training-only normalization, masked RMSE, early stopping, and checkpoint rule are specified in [docs/reproducibility.md](docs/reproducibility.md). Frozen selected checkpoints are in `checkpoints/selected_2019`.

## Evaluation

```bash
python scripts/evaluate/evaluate_predictions.py outputs/et_7d_predictions.npz \
  --output outputs/et_7d_metrics.json
```

The library implements pooled, grid-cell temporal, and daily spatial RMSE/KGE, training climatology, raw persistence, anomaly persistence, and signed gains. The 141-date comparison is exactly 13 August through 31 December 2020. Positive gain favors SEED.

## Figures

```bash
python figures/scripts/generate_figures_05_09.py
python scripts/evaluate/verify_figure_resolution.py
```

Outputs are written to `figures/final`. Figures 5-9 are saved as lossless 900-DPI PNGs with PDF companions. Figure 9 keeps the 28 September through 27 October 2020 shaded interval and places the complete three-entry legend below the lower row. The command for every manuscript figure is mapped in [docs/paper-code-map.md](docs/paper-code-map.md).

## Runtime benchmarking

```bash
python scripts/benchmark/extract_elm_endpoint.py --help
python scripts/benchmark/benchmark_warm_inference.py --help
```

The measurement boundary, warm-up rule, ET-then-SM order, hardware, repetitions, and median/IQR/p95 summaries are in `configs/evaluation/runtime.json`. Published timing tables are under `figures/data/runtime`.

## Application

```bash
cd apps/api
PYTHONPATH=. uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd apps/web
npm run dev
```

The web interface provides ET, SM, and joint dryness layers, model provenance, coordinate inspection, and GeoJSON export. Configure `NEXT_PUBLIC_API_BASE_URL` if the API is not at `http://127.0.0.1:8000`.

## Tests

```bash
python -m unittest discover -s tests/unit -p 'test_*.py'
python -m unittest discover -s tests/integration -p 'test_*.py'
python scripts/evaluate/evaluate_predictions.py tests/fixtures/tiny_predictions.npz \
  --output outputs/tiny_metrics.json
ruff check src scripts figures apps/api tests
python scripts/evaluate/verify_figure_resolution.py
cd apps/web && npm run lint && npm run build
npx playwright test
```

Expected outputs are written to `data/processed`, `outputs`, and `figures/final`.

## Citation and history

Citation metadata are in [CITATION.cff](CITATION.cff). This release begins with fresh Git history because the source workspace was not a Git repository.
