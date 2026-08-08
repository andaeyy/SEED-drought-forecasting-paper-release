# Paper-to-code map

| Paper component | Files | Command |
|---|---|---|
| NLDAS preprocessing | `src/seed/data/nldas.py`, `scripts/prepare_data/prepare_nldas.py` | `python scripts/prepare_data/prepare_nldas.py ...` |
| ELM ET and SM targets | `src/seed/data/elm.py`, `scripts/prepare_data/prepare_elm_targets.py` | `python scripts/prepare_data/prepare_elm_targets.py ...` |
| Sequence-to-map ConvLSTM | `src/seed/models/convlstm.py` | `python scripts/train/train_selected.py configs/models/et/30day.json --summary-only` |
| Encoder-decoder ConvLSTM | `src/seed/models/convlstm.py` | `python scripts/train/train_selected.py configs/models/sm/30day.json --summary-only` |
| Autoregressive ConvLSTM | `src/seed/models/convlstm.py` | `python scripts/train/train_selected.py configs/models/et/7day.json --summary-only` |
| Six selected models | `configs/models/et/*.json`, `configs/models/sm/*.json`, `checkpoints/selected_2019` | `git lfs pull` |
| RMSE and KGE | `src/seed/evaluation/metrics.py` | `python scripts/evaluate/evaluate_predictions.py ...` |
| Climatology and persistence | `src/seed/evaluation/metrics.py` | evaluated by the same command |
| Spatial and temporal diagnostics | `src/seed/evaluation/metrics.py` | evaluated by the same command |
| MSDI-Kendall calculation | `src/seed/dryness/msdi.py` | imported by `apps/api/app/adapt/inference.py` |
| Runtime benchmark | `scripts/benchmark`, `configs/evaluation/runtime.json` | `python scripts/benchmark/benchmark_warm_inference.py --help` |
| Web application | `apps/api`, `apps/web` | `uvicorn app.main:app`; `npm run dev` |

## Figure map

| Figure | Public status | Source and command |
|---|---|---|
| 1 | excluded by release boundary | no file or command distributed |
| 2 | excluded by release boundary | no file or command distributed |
| 3 | excluded by release boundary | no file or command distributed |
| 4 | excluded by release boundary | no file or command distributed |
| 5, grid-cell temporal RMSE | included | `figures/scripts/generate_figures_05_09.py`; `python figures/scripts/generate_figures_05_09.py` |
| 6, grid-cell temporal KGE | included | same command |
| 7, RMSE gain over persistence | included | same command |
| 8, KGE gain over persistence | included | same command |
| 9, daily spatial metrics | included | same command |
| 10, joint ET-SM dryness | included | `figures/scripts/generate_figure10.py`; `python figures/scripts/generate_figure10.py` |
| 11, application | included | `apps/web`, `tests/browser`; `cd apps/web && npx playwright test` |

Figures 1-4 are not distributed because their manuscript versions describe model families outside this supervised ConvLSTM release. Their omission is intentional, not a missing dependency.
