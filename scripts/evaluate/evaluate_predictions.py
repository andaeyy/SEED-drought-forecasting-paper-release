#!/usr/bin/env python3
"""Evaluate a prediction archive with paper metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seed.evaluation.metrics import daily_spatial_metrics, kge, rmse, temporal_gridcell_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="NPZ with observed and predicted arrays")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.archive, allow_pickle=False) as data:
        observed = data["observed"]
        predicted = data["predicted"]
    pooled_rmse, count = rmse(observed, predicted, axis=None, minimum_count=2)
    pooled_kge, correlation, alpha, beta, _ = kge(observed, predicted, axis=None, minimum_count=2)
    temporal = temporal_gridcell_metrics(observed, predicted)
    spatial = daily_spatial_metrics(observed, predicted)
    payload = {
        "pooled": {
            "rmse": float(pooled_rmse),
            "kge": float(pooled_kge),
            "correlation": float(correlation),
            "alpha": float(alpha),
            "beta": float(beta),
            "count": int(count),
        },
        "defined_temporal_rmse_cells": int(np.isfinite(temporal["rmse"]).sum()),
        "defined_temporal_kge_cells": int(np.isfinite(temporal["kge"]).sum()),
        "defined_daily_rmse_dates": int(np.isfinite(spatial["rmse"]).sum()),
        "defined_daily_kge_dates": int(np.isfinite(spatial["kge"]).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
