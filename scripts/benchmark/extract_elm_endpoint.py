#!/usr/bin/env python3
"""Extract and validate the final daily ET/SM endpoint from ELM history."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from cftime import DatetimeNoLeap
from netCDF4 import Dataset


def _filled(variable, key) -> np.ndarray:
    return np.asarray(np.ma.filled(variable[key], np.nan), dtype=np.float32)


def extract_endpoint(
    history_files: list[Path],
    output: Path,
    validation_path: Path,
    initialization_date: str,
    horizon_days: int,
) -> dict[str, object]:
    if not history_files:
        raise FileNotFoundError("no ELM history files were supplied")

    chosen_path: Path | None = None
    chosen_index = -1
    chosen_time = -np.inf
    for path in history_files:
        with Dataset(path) as ds:
            values = np.asarray(ds.variables["time"][:], dtype=np.float64)
            if values.size and float(np.nanmax(values)) > chosen_time:
                chosen_time = float(np.nanmax(values))
                chosen_index = int(np.nanargmax(values))
                chosen_path = path

    if chosen_path is None:
        raise ValueError("all ELM history files had empty time axes")
    if not np.isclose(chosen_time, float(horizon_days), rtol=0.0, atol=1.0e-5):
        raise ValueError(f"final ELM time {chosen_time} does not equal horizon {horizon_days}")

    with Dataset(chosen_path) as source:
        required = {"QSOIL", "QVEGE", "QVEGT", "H2OSOI", "lat", "lon", "landmask", "time_bounds"}
        missing = sorted(required.difference(source.variables))
        if missing:
            raise ValueError(f"history is missing required variables: {missing}")

        bounds = np.asarray(source.variables["time_bounds"][chosen_index], dtype=np.float64)
        expected_bounds = np.array([horizon_days - 1, horizon_days], dtype=np.float64)
        if not np.allclose(bounds, expected_bounds, rtol=0.0, atol=1.0e-5):
            raise ValueError(f"final averaging bounds {bounds.tolist()} != {expected_bounds.tolist()}")

        qsoil = _filled(source.variables["QSOIL"], chosen_index)
        qvege = _filled(source.variables["QVEGE"], chosen_index)
        qvegt = _filled(source.variables["QVEGT"], chosen_index)
        sm = _filled(source.variables["H2OSOI"], (chosen_index, 2, slice(None), slice(None)))
        et = qsoil + qvege + qvegt
        lat = np.asarray(source.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(source.variables["lon"][:], dtype=np.float32)
        landmask = np.asarray(source.variables["landmask"][:], dtype=np.int32)
        levgrnd = float(source.variables["levgrnd"][2])

    if et.shape != (140, 93) or sm.shape != (140, 93) or landmask.shape != (140, 93):
        raise ValueError(f"unexpected endpoint shapes ET={et.shape}, SM={sm.shape}, mask={landmask.shape}")
    active = landmask == 1
    active_count = int(np.count_nonzero(active))
    if active_count != 12741:
        raise ValueError(f"expected 12,741 active cells, found {active_count}")
    # preserves the established boundary mask
    et_finite = np.isfinite(et)
    sm_finite = np.isfinite(sm)
    analysis_valid = active & et_finite & sm_finite
    et_nonfinite_active = int(np.count_nonzero(active & ~et_finite))
    sm_nonfinite_active = int(np.count_nonzero(active & ~sm_finite))
    if et_nonfinite_active not in {0, 19} or sm_nonfinite_active != 0:
        raise ValueError(
            "unexpected native endpoint mask: "
            f"ET nonfinite active={et_nonfinite_active}, SM nonfinite active={sm_nonfinite_active}"
        )
    expected_valid = active_count - et_nonfinite_active
    if int(np.count_nonzero(analysis_valid)) != expected_valid:
        raise ValueError(f"expected {expected_valid:,} paired-valid ET/SM cells")

    init = date.fromisoformat(initialization_date)
    init_noleap = DatetimeNoLeap(init.year, init.month, init.day)
    endpoint = init_noleap + timedelta(days=horizon_days - 1)
    endpoint_text = endpoint.strftime("%Y-%m-%d")
    output.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(output, "w", format="NETCDF4") as target:
        target.createDimension("time", 1)
        target.createDimension("lat", 140)
        target.createDimension("lon", 93)

        time_var = target.createVariable("time", "i4", ("time",))
        time_var[:] = [horizon_days]
        time_var.units = f"days since {initialization_date} 00:00:00"
        time_var.calendar = "noleap"
        time_var.long_name = "end of final daily averaging interval"

        lat_var = target.createVariable("lat", "f4", ("lat",))
        lon_var = target.createVariable("lon", "f4", ("lon",))
        mask_var = target.createVariable("landmask", "i4", ("lat", "lon"), zlib=True, complevel=1)
        valid_var = target.createVariable("valid_analysis_mask", "i1", ("lat", "lon"), zlib=True, complevel=1)
        lat_var[:] = lat
        lon_var[:] = lon
        mask_var[:] = landmask
        valid_var[:] = analysis_valid.astype(np.int8)
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"

        et_var = target.createVariable(
            "EVAPOTRANSPIRATION", "f4", ("time", "lat", "lon"), fill_value=np.float32(1.0e36), zlib=True, complevel=1
        )
        sm_var = target.createVariable(
            "H2OSOI", "f4", ("time", "lat", "lon"), fill_value=np.float32(1.0e36), zlib=True, complevel=1
        )
        et_var[0] = np.where(analysis_valid, et, np.float32(1.0e36))
        sm_var[0] = np.where(active, sm, np.float32(1.0e36))
        et_var.long_name = "Total evapotranspiration (QSOIL + QVEGE + QVEGT)"
        et_var.units = "mm/s"
        et_var.cell_methods = "time: mean"
        sm_var.long_name = "Volumetric Soil Water"
        sm_var.units = "mm3/mm3"
        sm_var.depth = f"levgrnd index 2 ({levgrnd:.8f} m)"
        sm_var.cell_methods = "time: mean"

        target.title = "ELM Great Plains final-lead daily ET and SM endpoint"
        target.initialization_date = initialization_date
        target.endpoint_date = endpoint_text
        target.horizon_days = horizon_days
        target.source_history_file = str(chosen_path.resolve())
        target.source_time_index = chosen_index
        target.source_time_value = chosen_time
        target.source_time_bounds = ",".join(str(float(value)) for value in bounds)
        target.native_grid = "140 x 93; 12,741 active cells"
        target.analysis_mask = (
            f"{int(np.count_nonzero(analysis_valid)):,} paired-valid cells; "
            f"{et_nonfinite_active} native ET-masked active cells"
        )

    result: dict[str, object] = {
        "status": "pass",
        "output": str(output.resolve()),
        "output_file_bytes": output.stat().st_size,
        "source_history_file": str(chosen_path.resolve()),
        "source_time_index": chosen_index,
        "source_time_value": chosen_time,
        "source_time_bounds": bounds.tolist(),
        "initialization_date": initialization_date,
        "endpoint_date": endpoint_text,
        "horizon_days": horizon_days,
        "soil_depth_m": levgrnd,
        "active_cells": active_count,
        "paired_valid_cells": int(np.count_nonzero(analysis_valid)),
        "et_nonfinite_active_cells": et_nonfinite_active,
        "sm_nonfinite_active_cells": sm_nonfinite_active,
        "et_active_min": float(np.min(et[analysis_valid])),
        "et_active_max": float(np.max(et[analysis_valid])),
        "sm_active_min": float(np.min(sm[active])),
        "sm_active_max": float(np.max(sm[active])),
        "et_paired_valid_all_finite": True,
        "sm_active_all_finite": True,
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--initialization-date", required=True)
    parser.add_argument("--horizon-days", type=int, required=True)
    args = parser.parse_args()
    result = extract_endpoint(
        args.history,
        args.output,
        args.validation,
        args.initialization_date,
        args.horizon_days,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
