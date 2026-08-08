"""RMSE, KGE, baseline, and diagnostic calculations."""

from __future__ import annotations

import numpy as np


def rmse(observed, predicted, *, axis=None, minimum_count=1):
    """Calculate RMSE and paired finite counts."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted shapes differ")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    count = np.sum(valid, axis=axis)
    squared = np.sum(np.where(valid, (predicted - observed) ** 2, 0.0), axis=axis)
    value = np.sqrt(squared / np.maximum(count, 1))
    return np.where(count >= minimum_count, value, np.nan), count


def kge(observed, predicted, *, axis=None, minimum_count=2):
    """Calculate KGE components and paired finite counts."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted shapes differ")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    count = np.sum(valid, axis=axis)
    denominator = np.maximum(count, 1)
    obs_mean = np.sum(np.where(valid, observed, 0.0), axis=axis) / denominator
    sim_mean = np.sum(np.where(valid, predicted, 0.0), axis=axis) / denominator
    if axis is None:
        obs_center = observed - obs_mean
        sim_center = predicted - sim_mean
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        expanded_obs = obs_mean
        expanded_sim = sim_mean
        for current in sorted(item % observed.ndim for item in axes):
            expanded_obs = np.expand_dims(expanded_obs, current)
            expanded_sim = np.expand_dims(expanded_sim, current)
        obs_center = observed - expanded_obs
        sim_center = predicted - expanded_sim
    obs_var = np.sum(np.where(valid, obs_center**2, 0.0), axis=axis) / denominator
    sim_var = np.sum(np.where(valid, sim_center**2, 0.0), axis=axis) / denominator
    covariance = np.sum(np.where(valid, obs_center * sim_center, 0.0), axis=axis) / denominator
    obs_std = np.sqrt(obs_var)
    sim_std = np.sqrt(sim_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / (obs_std * sim_std)
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean
        value = 1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    defined = (count >= minimum_count) & np.isfinite(value) & (obs_std > 0) & (sim_std > 0) & (np.abs(obs_mean) > 1e-12)
    fields = [value, correlation, alpha, beta]
    return tuple(np.where(defined, field, np.nan) for field in fields) + (count,)


def temporal_gridcell_metrics(observed, predicted, minimum_count=300):
    """Compute grid-cell temporal RMSE and KGE."""
    rmse_value, count = rmse(observed, predicted, axis=0, minimum_count=minimum_count)
    kge_value, correlation, alpha, beta, kge_count = kge(
        observed,
        predicted,
        axis=0,
        minimum_count=minimum_count,
    )
    if not np.array_equal(count, kge_count):
        raise ValueError("RMSE and KGE support differs")
    return {
        "rmse": rmse_value,
        "kge": kge_value,
        "correlation": correlation,
        "alpha": alpha,
        "beta": beta,
        "count": count,
    }


def daily_spatial_metrics(observed, predicted, minimum_count=100):
    """Compute daily spatial RMSE and KGE."""
    axes = tuple(range(1, np.asarray(observed).ndim))
    rmse_value, count = rmse(observed, predicted, axis=axes, minimum_count=minimum_count)
    kge_value, correlation, alpha, beta, kge_count = kge(
        observed,
        predicted,
        axis=axes,
        minimum_count=minimum_count,
    )
    if not np.array_equal(count, kge_count):
        raise ValueError("RMSE and KGE support differs")
    return {
        "rmse": rmse_value,
        "kge": kge_value,
        "correlation": correlation,
        "alpha": alpha,
        "beta": beta,
        "count": count,
    }


def raw_persistence(targets, target_indices, horizon):
    """Repeat the final finite value at each forecast origin."""
    targets = np.asarray(targets, dtype=np.float32)
    origins = np.asarray(target_indices, dtype=np.int32) - int(horizon)
    if np.any(origins < 0):
        raise ValueError("persistence origin precedes available targets")
    requested = {int(index): position for position, index in enumerate(origins)}
    result = np.full((origins.size, *targets.shape[1:]), np.nan, dtype=np.float32)
    last_finite = np.full(targets.shape[1:], np.nan, dtype=np.float32)
    for index in range(int(origins.max()) + 1):
        field = targets[index]
        last_finite = np.where(np.isfinite(field), field, last_finite)
        if index in requested:
            result[requested[index]] = last_finite
    return result


def anomaly_persistence(current, current_climatology, target_climatology):
    """Add the current anomaly to target-date climatology."""
    return np.asarray(target_climatology) + np.asarray(current) - np.asarray(current_climatology)


def training_climatology(targets, dayofyear, split_index):
    """Fit no-leap day-of-year climatology on training only."""
    training = np.asarray(targets, dtype=np.float32)[:split_index]
    days = np.asarray(dayofyear, dtype=np.int16)[:split_index]
    climatology = np.full((365, *training.shape[1:]), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        fallback = np.nanmean(training, axis=0)
        for day in range(1, 366):
            climatology[day - 1] = np.nanmean(training[days == day], axis=0)
    return np.where(np.isfinite(climatology), climatology, fallback).astype(np.float32)


def rmse_gain(seed_rmse, persistence_rmse):
    """Positive RMSE gain favors SEED."""
    return np.asarray(persistence_rmse) - np.asarray(seed_rmse)


def kge_gain(seed_kge, persistence_kge):
    """Positive KGE gain favors SEED."""
    return np.asarray(seed_kge) - np.asarray(persistence_kge)
