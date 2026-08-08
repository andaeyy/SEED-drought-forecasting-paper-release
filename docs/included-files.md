# Included material

This release contains only the supervised ConvLSTM workflow and resources needed to trace the distributed paper results.

The release commit tracks 148 files: 44 application files, 18 checkpoint and grid files, 9 configurations, 5 documentation files, 37 figure resources, 7 command scripts, 12 library files, 9 tests or fixtures, and 7 repository-level metadata files. `git ls-files` is the authoritative complete path manifest.

| Area | Included paths | Contents |
|---|---|---|
| Data preparation | `src/seed/data`, `scripts/prepare_data` | Daily NLDAS aggregation, ELM ET and layer-2 SM targets, masks, missing values, and grid alignment |
| Models and training | `src/seed/models`, `src/seed/training`, `scripts/train`, `configs/models` | Sequence-to-map, encoder-decoder, and autoregressive ConvLSTM implementations and six selected configurations |
| Evaluation | `src/seed/evaluation`, `scripts/evaluate`, `configs/evaluation` | RMSE, KGE, climatology, persistence, gains, and spatial/temporal diagnostics |
| Dryness | `src/seed/dryness` | Circular climatology, standardized anomalies, Clayton copula, Kendall transform, and application categories |
| Benchmarking | `src/seed/benchmarking`, `scripts/benchmark`, `figures/data/runtime` | ELM endpoint and warm sequential SEED timing resources |
| Application | `apps/api`, `apps/web`, `configs/application` | FastAPI service, Next.js maps, provenance, coordinate inspection, and GeoJSON export |
| Figures | `figures/scripts`, `figures/data`, `figures/final` | Regeneration code, compact derived data, final Figures 5-11, and PDF companions where supported |
| Tests | `tests/unit`, `tests/integration`, `tests/browser`, `tests/fixtures` | Scientific, API, checkpoint-contract, fixture, and responsive browser checks |
| Documentation | `README.md`, `CITATION.cff`, `docs`, `paper` | Setup, data acquisition, reproducibility, paper mapping, and release boundaries |

## Selected inference artifacts

Six target-specific Keras checkpoints and their training-period normalization statistics are under `checkpoints/selected_2019`. `checkpoints/selection_manifest.json` records every SHA-256 digest. The `.keras` files are tracked with Git LFS; all are below GitHub's 100 MB per-file limit.

## History boundary

The source workspace was not a Git repository. This release therefore begins with one fresh, attributed release-preparation commit. No previous authorship or commit history is claimed or reconstructed.
