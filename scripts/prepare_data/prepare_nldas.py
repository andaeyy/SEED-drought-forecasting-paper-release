#!/usr/bin/env python3
"""Aggregate monthly hourly NLDAS files to one daily dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from seed.data.nldas import FORCING_VARIABLES, aggregate_daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monthly_files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    daily = []
    for path in args.monthly_files:
        with xr.open_dataset(path, decode_times=True) as source:
            variables = {}
            for name in FORCING_VARIABLES:
                if name not in source:
                    raise KeyError(f"{path} lacks {name}")
                values = aggregate_daily(source[name].values, name)
                variables[name] = (("time", *source[name].dims[1:]), values)
            coordinates = {name: source[name] for name in source.coords if name != "time"}
            daily.append(xr.Dataset(variables, coords=coordinates))
    output = xr.concat(daily, dim="time")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(args.output, engine="h5netcdf")


if __name__ == "__main__":
    main()
