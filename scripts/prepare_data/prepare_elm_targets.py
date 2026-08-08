#!/usr/bin/env python3
"""Create daily ET and layer-2 SM targets from ELM output."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from seed.data.elm import et_from_components, soil_moisture_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--et-output", required=True, type=Path)
    parser.add_argument("--sm-output", required=True, type=Path)
    args = parser.parse_args()
    with xr.open_dataset(args.source) as source:
        et = et_from_components(source["QSOIL"].values, source["QVEGE"].values, source["QVEGT"].values)
        sm = soil_moisture_layer(source["H2OSOI"].values)
        coordinates = {name: source[name] for name in ("time", "lat", "lon")}
        et_dataset = xr.Dataset({"EVAPOTRANSPIRATION": (("time", "lat", "lon"), et)}, coords=coordinates)
        sm_dataset = xr.Dataset({"H2OSOI": (("time", "lat", "lon"), sm)}, coords=coordinates)
    args.et_output.parent.mkdir(parents=True, exist_ok=True)
    args.sm_output.parent.mkdir(parents=True, exist_ok=True)
    et_dataset.to_netcdf(args.et_output, engine="h5netcdf")
    sm_dataset.to_netcdf(args.sm_output, engine="h5netcdf")


if __name__ == "__main__":
    main()
