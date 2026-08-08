#!/usr/bin/env python3
"""Benchmark warm ET-then-SM endpoint inference."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def quantiles(values):
    """Return median, IQR, and p95 seconds."""
    values = np.asarray(values, dtype=np.float64)
    q25, median, q75, p95 = np.quantile(values, [0.25, 0.5, 0.75, 0.95])
    return float(median), float(q75 - q25), float(p95)


def load_model(path):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--et-model", required=True, type=Path)
    parser.add_argument("--sm-model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing x")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        inputs = np.asarray(archive["x"], dtype=np.float32)
    et_model = load_model(args.et_model)
    sm_model = load_model(args.sm_model)
    et_model.predict(inputs, verbose=0)
    sm_model.predict(inputs, verbose=0)
    rows = []
    for repetition in range(args.repetitions):
        start = time.perf_counter()
        et_model.predict(inputs, verbose=0)
        after_et = time.perf_counter()
        sm_model.predict(inputs, verbose=0)
        stop = time.perf_counter()
        rows.append(
            {
                "repetition": repetition,
                "et_seconds": after_et - start,
                "sm_seconds": stop - after_et,
                "pair_seconds": stop - start,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for field in ("et_seconds", "sm_seconds", "pair_seconds"):
        median, iqr, p95 = quantiles([row[field] for row in rows])
        print(f"{field}: median={median:.6f}, IQR={iqr:.6f}, p95={p95:.6f}")


if __name__ == "__main__":
    main()
