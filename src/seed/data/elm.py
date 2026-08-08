"""ELM evapotranspiration and soil-moisture target preparation."""

from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86_400.0
SOIL_LAYER_INDEX = 2
SOIL_LAYER_DEPTH_METERS = 0.06225858


def et_from_components(
    qsoil: np.ndarray,
    qvege: np.ndarray,
    qvegt: np.ndarray,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Convert QSOIL + QVEGE + QVEGT from mm/s to mm/day."""
    components = [np.asarray(value, dtype=np.float32) for value in (qsoil, qvege, qvegt)]
    if components[0].shape != components[1].shape or components[0].shape != components[2].shape:
        raise ValueError("ET component shapes differ")
    et = (components[0] + components[1] + components[2]) * np.float32(SECONDS_PER_DAY)
    if clip_negative:
        et = np.maximum(et, np.float32(0.0))
    return et.astype(np.float32)


def soil_moisture_layer(h2osoi: np.ndarray, layer_axis: int = 1) -> np.ndarray:
    """Select H2OSOI layer index 2 without unit conversion."""
    values = np.asarray(h2osoi, dtype=np.float32)
    if values.shape[layer_axis] <= SOIL_LAYER_INDEX:
        raise ValueError("H2OSOI lacks layer index 2")
    return np.take(values, SOIL_LAYER_INDEX, axis=layer_axis).astype(np.float32)
