#!/usr/bin/env python3
"""Regenerate the joint ET-SM dryness figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seed-paper-matplotlib")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

EXTENT = (-107.56, -94.32, 25.32, 44.43)
HORIZONS = (7, 30, 90)


def map_axis(axis, show_left, show_bottom):
    axis.set_extent(EXTENT, crs=ccrs.PlateCarree())
    axis.set_facecolor("#D2D5D8")
    axis.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#545A60", facecolor="none", linewidth=0.52)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1D2731", linewidth=0.68)
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1D2731", linewidth=0.72)
    grid = axis.gridlines(draw_labels=True, linewidth=0.28, color="#66717C", alpha=0.38, linestyle="--")
    grid.top_labels = False
    grid.right_labels = False
    grid.left_labels = show_left
    grid.bottom_labels = show_bottom
    grid.xlabel_style = {"size": 6.6}
    grid.ylabel_style = {"size": 6.6}
    grid.xlocator = ticker.FixedLocator([-106, -101, -96])
    grid.ylocator = ticker.FixedLocator([30, 35, 40])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "figure10_joint_dryness_fields.npz")
    parser.add_argument("--output", type=Path, default=root / "final" / "figure10_joint_et_sm_dryness.png")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.25), subplot_kw={"projection": ccrs.PlateCarree()})
    figure.subplots_adjust(left=0.095, right=0.90, bottom=0.08, top=0.93, wspace=0.08, hspace=0.14)
    columns = (("ELM-derived dryness", "target"),) + tuple(
        (f"{horizon}-day lead", f"{horizon}d") for horizon in HORIZONS
    )
    handle = None
    for row, (case, label) in enumerate((("dry", "Driest target case"), ("wet", "Wettest target case"))):
        for column, (title, suffix) in enumerate(columns):
            field = data[f"{case}_{suffix}"]
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#D2D5D8")
            handle = axes[row, column].pcolormesh(
                ((data["lon"] + 180.0) % 360.0) - 180.0,
                data["lat"],
                field,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                shading="nearest",
                transform=ccrs.PlateCarree(),
            )
            map_axis(axes[row, column], column == 0, row == 1)
            if row == 0:
                axes[row, column].set_title(title, fontsize=7.5)
            axes[row, column].text(
                -0.13,
                1.015,
                f"({'abcdefgh'[row * 4 + column]})",
                transform=axes[row, column].transAxes,
                fontsize=7.2,
                fontweight="bold",
            )
        figure.text(
            0.022,
            0.70 if row == 0 else 0.29,
            f"{label}\n{data[f'{case}_date']}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.5,
        )
    color_axis = figure.add_axes([0.895, 0.15, 0.015, 0.70])
    colorbar = figure.colorbar(handle, cax=color_axis)
    colorbar.set_label("Joint low-ET/low-SM dryness score\n0 = wetter; 1 = drier", fontsize=6.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.03}
    figure.savefig(args.output, dpi=900, **options)
    figure.savefig(args.output.with_suffix(".pdf"), dpi=900, **options)
    plt.close(figure)


if __name__ == "__main__":
    main()
