# Data requirements

## NLDAS forcing

Acquire hourly NLDAS forcing through NASA Earthdata or an institutional archive authorized for your use. The required variables, in training order, are `PRECTmms`, `TBOT`, `WIND`, `QBOT`, `PSRF`, `FSDS`, and `FLDS`. Expected monthly names are `clmforc.nldas.YYYY-MM.nc`. A no-leap year has 8,760 hourly records.

`scripts/prepare_data/prepare_nldas.py` applies a float32 sum across each 24-hour block for `PRECTmms` and a float32 mean for the other six fields. Missing values remain missing, except that NumPy's historical `nansum` behavior makes an all-missing precipitation block zero. Coordinates are checked before model-grid alignment.

Checksums are acquisition-specific and are not asserted for raw NLDAS files in this release. Record `sha256sum` values immediately after download and retain the provider manifest with the data.

## ELM targets

The required daily ELM history variables are `QSOIL`, `QVEGE`, `QVEGT`, and `H2OSOI`, with `lat`, `lon`, `time`, and `levgrnd`. ET is the signed component sum converted from mm/s to mm/day, then clipped below zero. SM uses zero-based `H2OSOI` index 2 at approximately 0.06225858 m and remains in volumetric units.

Expected processed names are:

- `ELM_EVAPOTRANSPIRATION_2000_2020.nc`
- `ELM_SM_2000_2020.nc`
- `clmforc.nldas.YYYY.nc`

The full ELM and NLDAS archives are intentionally absent. Small figure metric archives and the model grid are committed under `figures/data` and `checkpoints/grid`.

## Splits and alignment

Training targets span 2015-2018, model selection uses 2019, and the independent test uses 2020. Leap day is removed. Input and target grids must share the same date order after bilinear alignment. Interpolation does not fill a cell when any required source corner is missing. Evaluation always uses the exact paired finite intersection.

## Checkpoint integrity

The six model JSON files contain checkpoint SHA-256 values. Verify them with:

```bash
sha256sum checkpoints/selected_2019/*/target_specific/*.keras
```

Normalizer hashes are listed in `checkpoints/selection_manifest.json`.
