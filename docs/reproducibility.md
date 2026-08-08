# Reproducibility

## Scientific contract

SEED maps seven historical forcing fields to one future ET or SM endpoint map. Weekly, monthly, and seasonal configurations use 10, 45, and 135 input days and 7, 30, and 90 lead days. ET and SM are separate networks. Every architecture terminates in a linear 1x1 convolution.

Training uses 2015-2018 target dates. Feature and target normalization statistics are fit only on those records. 2019 validation determines the target-specific architecture and checkpoint. Early stopping monitors physical-unit masked validation RMSE and restores the best checkpoint. The six selected model identities are frozen before the independent 2020 test.

## Evaluation

`src/seed/evaluation/metrics.py` is the shared implementation. RMSE and KGE use paired finite observations and predictions. The evaluation includes:

- spatiotemporally pooled metrics;
- grid-cell temporal metrics with at least 300 dates;
- daily spatial metrics with at least 100 cells;
- training-only no-leap climatology;
- raw and anomaly persistence;
- signed gains where positive values favor SEED.

The persistence sensitivity period has 141 endpoint dates from 2020-08-13 through 2020-12-31. RMSE gain is persistence RMSE minus SEED RMSE. KGE gain is SEED KGE minus persistence KGE.

## Joint dryness

`src/seed/dryness/msdi.py` fits circular 31-day ET and SM climatologies on 2015-2018, forms standardized anomalies, transforms the two marginals with the standard normal CDF, applies a bivariate Clayton copula with prescribed Kendall tau 0.4, applies the analytic Kendall distribution, and maps the standardized joint index to bounded low-ET/low-SM dryness. Application categories use thresholds 0.70, 0.80, 0.90, 0.95, and 0.98.

## Figures

Figures 5-9 can be regenerated without raw predictions because `figures/data/*_analysis.npz` contains the final derived fields. These archives preserve the scientific values, coordinates, model identifiers, dates, and 141-date sensitivity gains while omitting full daily target/prediction cubes. Run:

```bash
python figures/scripts/generate_figures_05_09.py
python scripts/evaluate/verify_figure_resolution.py
python figures/scripts/generate_figure10.py
```

Map rows use shared target-specific color scales. Figures 5-8 use compact two-by-three layouts with explicit shared row colorbars. Figure 9 places the three-entry legend below the lower panels. Raster outputs are rendered at 900 DPI and PDF companions are produced when supported.

## Benchmark boundary

ELM timing starts immediately before `srun` and stops after validated endpoint writing. Each case has one warm-up and three timed repetitions. SEED timing excludes input preparation, model loading, and first-call tracing, and records 200 warm ET-then-SM repetitions at batch size one. Summaries report median, IQR, and p95. `configs/evaluation/runtime.json` records hardware and boundaries.

## Clean-room validation

```bash
python -m venv /tmp/seed-release-venv
source /tmp/seed-release-venv/bin/activate
python -m pip install -e '.[data,figures,ml,api,test]'
python -m unittest discover -s tests/unit -p 'test_*.py'
python -m unittest discover -s tests/integration -p 'test_*.py'
python scripts/evaluate/evaluate_predictions.py tests/fixtures/tiny_predictions.npz \
  --output /tmp/seed-tiny-metrics.json
ruff check src scripts figures apps/api tests
python figures/scripts/generate_figures_05_09.py
python scripts/evaluate/verify_figure_resolution.py
cd apps/web && npm ci && npm run lint && npm run build
```

Raw-data and GPU-dependent steps require the external archives and hardware described above.
