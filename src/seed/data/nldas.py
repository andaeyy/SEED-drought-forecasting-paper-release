"""NLDAS daily aggregation and grid-alignment utilities."""

from __future__ import annotations

import numpy as np

FORCING_VARIABLES = (
    "PRECTmms",
    "TBOT",
    "WIND",
    "QBOT",
    "PSRF",
    "FSDS",
    "FLDS",
)


def aggregate_daily(values: np.ndarray, variable: str) -> np.ndarray:
    """Aggregate hourly forcing to complete daily records."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] % 24:
        raise ValueError("hourly record count must be divisible by 24")
    if variable not in FORCING_VARIABLES:
        raise ValueError(f"unsupported NLDAS variable: {variable}")
    daily = array.reshape(array.shape[0] // 24, 24, *array.shape[1:])
    with np.errstate(invalid="ignore"):
        if variable == "PRECTmms":
            result = np.nansum(daily, axis=1, dtype=np.float32)
        else:
            result = np.nanmean(daily, axis=1, dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def finite_land_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return the exact finite intersection of aligned arrays."""
    if not arrays:
        raise ValueError("at least one array is required")
    shape = np.asarray(arrays[0]).shape
    if any(np.asarray(array).shape != shape for array in arrays[1:]):
        raise ValueError("land-mask arrays must share a shape")
    mask = np.ones(shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def bilinear_regrid(
    values: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """Regrid time-lat-lon values without filling missing corners."""
    source = np.asarray(values, dtype=np.float32)
    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (lat.size, lon.size):
        raise ValueError("source grid shape mismatch")
    if target_lat.shape != target_lon.shape:
        raise ValueError("target coordinates must share a shape")
    i1 = np.searchsorted(lat, target_lat, side="right")
    j1 = np.searchsorted(lon, target_lon, side="right")
    i0 = np.clip(i1 - 1, 0, lat.size - 2)
    j0 = np.clip(j1 - 1, 0, lon.size - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    y0, y1 = lat[i0], lat[i1]
    x0, x1 = lon[j0], lon[j1]
    wy = np.divide(target_lat - y0, y1 - y0, out=np.zeros_like(target_lat), where=y1 != y0)
    wx = np.divide(target_lon - x0, x1 - x0, out=np.zeros_like(target_lon), where=x1 != x0)
    v00 = source[:, i0, j0]
    v10 = source[:, i1, j0]
    v01 = source[:, i0, j1]
    v11 = source[:, i1, j1]
    result = v00 * (1 - wy) * (1 - wx) + v10 * wy * (1 - wx) + v01 * (1 - wy) * wx + v11 * wy * wx
    corners = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
    return np.where(corners, result, np.nan).astype(np.float32)
