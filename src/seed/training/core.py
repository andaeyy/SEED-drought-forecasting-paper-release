"""Training-only normalization and masked loss calculations."""

from __future__ import annotations

import numpy as np


def training_normalization(values: np.ndarray, split_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit means and scales on training records only."""
    training = np.asarray(values, dtype=np.float32)[:split_index]
    mean = np.nanmean(training, axis=tuple(range(training.ndim - 1)), keepdims=True).astype(np.float32)
    scale = np.nanstd(training, axis=tuple(range(training.ndim - 1)), keepdims=True).astype(np.float32)
    return mean, np.maximum(scale, np.float32(1e-8))


def masked_rmse(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute RMSE on the exact finite masked support."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted shapes differ")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((predicted[valid] - observed[valid]) ** 2)))
