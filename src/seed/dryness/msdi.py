"""Paper MSDI-Kendall dryness translation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

COPULA_TAU = 0.4
CLIMATOLOGY_HALF_WINDOW = 15
MIN_CLIMATOLOGY_SAMPLES = 100
EPSILON = 1e-6
VARIANCE_TOLERANCE_MULTIPLIER = 32.0


def circular_climatology(training_targets):
    """Fit circular 31-day means and scales."""
    targets = np.asarray(training_targets, dtype=np.float64)
    if targets.shape[0] != 4 * 365:
        raise ValueError("expected four no-leap training years")
    years = targets.reshape(4, 365, *targets.shape[1:])
    finite = np.isfinite(years)
    by_day_sum = np.where(finite, years, 0.0).sum(axis=0)
    by_day_squares = np.where(finite, years**2, 0.0).sum(axis=0)
    by_day_count = finite.sum(axis=0).astype(np.float64)

    def circular_sum(values):
        half = CLIMATOLOGY_HALF_WINDOW
        padded = np.concatenate([values[-half:], values, values[:half]], axis=0)
        cumulative = np.concatenate(
            [np.zeros((1, *values.shape[1:]), dtype=np.float64), np.cumsum(padded, axis=0)],
            axis=0,
        )
        window = 2 * half + 1
        return cumulative[window:] - cumulative[:-window]

    total = circular_sum(by_day_sum)
    squares = circular_sum(by_day_squares)
    count = circular_sum(by_day_count)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    second = np.divide(squares, count, out=np.full_like(squares, np.nan), where=count > 0)
    scale = np.sqrt(np.maximum(second - mean**2, 0.0))
    return mean.astype(np.float32), scale.astype(np.float32), count.astype(np.int16)


def standardize(values, dayofyear, mean, scale, sample_count):
    """Standardize fields against the circular climatology."""
    indices = np.asarray(dayofyear, dtype=np.int64) - 1
    values = np.asarray(values, dtype=np.float32)
    local_mean = mean[indices]
    local_scale = scale[indices]
    local_count = sample_count[indices]
    floor = VARIANCE_TOLERANCE_MULTIPLIER * np.finfo(np.float32).eps * np.maximum(np.abs(local_mean), 1.0)
    valid = (
        np.isfinite(values)
        & np.isfinite(local_mean)
        & np.isfinite(local_scale)
        & (local_count >= MIN_CLIMATOLOGY_SAMPLES)
        & (local_scale > floor)
    )
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[valid] = ((values[valid] - local_mean[valid]) / local_scale[valid]).astype(np.float32)
    return result


def _clayton_theta(tau=COPULA_TAU):
    tau = float(np.clip(tau, 1e-6, 0.999))
    return 2.0 * tau / (1.0 - tau)


def drought_score(z_et, z_sm, tau=COPULA_TAU):
    """Return bounded joint low-ET and low-SM dryness."""
    z_et = np.asarray(z_et)
    z_sm = np.asarray(z_sm)
    if z_et.shape != z_sm.shape:
        raise ValueError("ET and SM z-score shapes differ")
    valid = np.isfinite(z_et) & np.isfinite(z_sm)
    result = np.full(z_et.shape, np.nan, dtype=np.float32)
    if not valid.any():
        return result
    u = np.clip(norm.cdf(z_et[valid]), EPSILON, 1.0 - EPSILON)
    v = np.clip(norm.cdf(z_sm[valid]), EPSILON, 1.0 - EPSILON)
    theta = _clayton_theta(tau)
    joint = np.power(np.maximum(u ** (-theta) + v ** (-theta) - 1.0, 1e-12), -1.0 / theta)
    joint = np.clip(joint, EPSILON, 1.0 - EPSILON)
    kendall = joint * (1.0 + 1.0 / theta) - joint ** (theta + 1.0) / theta
    standardized = norm.ppf(np.clip(kendall, EPSILON, 1.0 - EPSILON))
    result[valid] = norm.cdf(-standardized).astype(np.float32)
    return result


def dryness_category(score):
    """Translate bounded dryness to application categories 0-5."""
    score = np.asarray(score, dtype=np.float32)
    categories = np.zeros_like(score, dtype=np.int8)
    for threshold, category in ((0.70, 1), (0.80, 2), (0.90, 3), (0.95, 4), (0.98, 5)):
        categories[score >= threshold] = category
    return categories
