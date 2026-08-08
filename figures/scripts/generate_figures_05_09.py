#!/usr/bin/env python3
"""Regenerate paper Figures 5-9 from compact derived metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seed-paper-matplotlib")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

HORIZONS = (7, 30, 90)
TARGETS = ("ET", "SM")
UNITS = {"ET": r"mm day$^{-1}$", "SM": r"m$^3$ m$^{-3}$"}
EXTENT = (-107.56, -94.32, 25.32, 44.43)
MASK_COLOR = "#D2D5D8"
PUBLICATION_DPI = 900
PANEL_LABELS = "abcdef"


def load_data(directory: Path):
    data = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            path = directory / f"{target.lower()}_{horizon}d_analysis.npz"
            with np.load(path, allow_pickle=False) as archive:
                data[target, horizon] = {key: archive[key] for key in archive.files}
    return data


def normalized_lon(values):
    return ((np.asarray(values) + 180.0) % 360.0) - 180.0


def map_axis(axis, show_left, show_bottom):
    axis.set_facecolor(MASK_COLOR)
    axis.set_extent(EXTENT, crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#545A60", facecolor="none", linewidth=0.52)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1D2731", linewidth=0.68)
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1D2731", linewidth=0.72)
    grid = axis.gridlines(
        draw_labels=True,
        linewidth=0.28,
        color="#66717C",
        alpha=0.38,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    grid.top_labels = False
    grid.right_labels = False
    grid.left_labels = show_left
    grid.bottom_labels = show_bottom
    grid.xlabel_style = {"size": 6.6}
    grid.ylabel_style = {"size": 6.6}
    grid.xlocator = ticker.FixedLocator([-106, -101, -96])
    grid.ylocator = ticker.FixedLocator([30, 35, 40])


def map_layout(title=None):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.15),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.190,
        top=0.915 if title else 0.955,
        wspace=-0.34,
        hspace=0.36,
    )
    if title:
        figure.suptitle(title, y=0.985, fontsize=7.3)
    return figure, axes


def row_colorbar(figure, axes, handle, row, label):
    boxes = [axis.get_position() for axis in axes[row]]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    width = min(0.46, (right - left) * 0.58)
    x0 = (left + right - width) / 2.0
    y0 = min(box.y0 for box in boxes) - (0.032 if row == 0 else 0.090)
    bar_axis = figure.add_axes([x0, y0, width, 0.018])
    bar = figure.colorbar(handle, cax=bar_axis, orientation="horizontal", extend="both")
    bar.set_label(label, labelpad=2.0)
    bar.ax.tick_params(labelsize=6.7, pad=1.5)
    return bar


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.03}
    figure.savefig(path, dpi=PUBLICATION_DPI, **options)
    figure.savefig(path.with_suffix(".pdf"), dpi=PUBLICATION_DPI, **options)
    plt.close(figure)


def metric_map(data, metric, output):
    field = f"gridcell_model_{metric.lower()}"
    figure, axes = map_layout()
    limits = {}
    for target in TARGETS:
        finite = np.concatenate(
            [data[target, horizon][field][np.isfinite(data[target, horizon][field])] for horizon in HORIZONS]
        )
        limits[target] = (
            (0.0, float(np.quantile(finite, 0.98)))
            if metric == "RMSE"
            else (min(-1.0, float(np.quantile(finite, 0.02))), 1.0)
        )
    handles = {}
    for panel, (row, target, column, horizon) in enumerate(
        (row, target, column, horizon) for row, target in enumerate(TARGETS) for column, horizon in enumerate(HORIZONS)
    ):
        values = data[target, horizon]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(MASK_COLOR)
        handle = axes[row, column].pcolormesh(
            normalized_lon(values["lon"]),
            values["lat"],
            values[field],
            cmap=cmap,
            vmin=limits[target][0],
            vmax=limits[target][1],
            shading="nearest",
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        handles[target] = handle
        map_axis(axes[row, column], column == 0, row == 1)
        if row == 0:
            axes[row, column].set_title(f"{horizon}-day lead", fontsize=8.2)
        axes[row, column].text(
            -0.08,
            1.015 if row == 0 else 0.88,
            f"({PANEL_LABELS[panel]})",
            transform=axes[row, column].transAxes,
            fontsize=7.2,
            fontweight="bold",
        )
    for row, target in enumerate(TARGETS):
        unit = UNITS[target] if metric == "RMSE" else "dimensionless"
        row_colorbar(figure, axes, handles[target], row, f"{target} {metric} ({unit})")
    save(figure, output)


def gain_map(data, metric, output):
    field = f"sensitivity_{metric.lower()}_gain"
    figure, axes = map_layout("Standalone-2020-context sensitivity: 13 Aug to 31 Dec 2020 (141 endpoint dates)")
    limits = {}
    for target in TARGETS:
        finite = np.concatenate(
            [np.abs(data[target, horizon][field][np.isfinite(data[target, horizon][field])]) for horizon in HORIZONS]
        )
        limits[target] = float(np.quantile(finite, 0.98))
    handles = {}
    for panel, (row, target, column, horizon) in enumerate(
        (row, target, column, horizon) for row, target in enumerate(TARGETS) for column, horizon in enumerate(HORIZONS)
    ):
        values = data[target, horizon]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(MASK_COLOR)
        handle = axes[row, column].pcolormesh(
            normalized_lon(values["lon"]),
            values["lat"],
            values[field],
            cmap=cmap,
            norm=colors.TwoSlopeNorm(vmin=-limits[target], vcenter=0.0, vmax=limits[target]),
            shading="nearest",
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )
        handles[target] = handle
        map_axis(axes[row, column], column == 0, row == 1)
        if row == 0:
            axes[row, column].set_title(f"{horizon}-day lead", fontsize=8.2)
        axes[row, column].text(
            -0.08,
            1.015 if row == 0 else 0.88,
            f"({PANEL_LABELS[panel]})",
            transform=axes[row, column].transAxes,
            fontsize=7.2,
            fontweight="bold",
        )
    for row, target in enumerate(TARGETS):
        formula = "persistence - SEED" if metric == "RMSE" else "SEED - persistence"
        unit = UNITS[target] if metric == "RMSE" else "dimensionless"
        bar = row_colorbar(figure, axes, handles[target], row, f"{target} {metric} gain ({formula}; {unit})")
        bar.ax.axvline(0.0, color="#1D2731", linewidth=0.72, clip_on=False)
    save(figure, output)


def daily_metrics(data, output):
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex=True)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.195, top=0.84, wspace=0.32, hspace=0.18)
    line_colors = {7: "#0072B2", 30: "#B66A00", 90: "#009E73"}
    styles = {7: "-", 30: "--", 90: ":"}
    panels = (("ET", "rmse", "RMSE"), ("ET", "kge", "KGE"), ("SM", "rmse", "RMSE"), ("SM", "kge", "KGE"))
    for panel, (axis, (target, field, label)) in enumerate(zip(axes.flat, panels, strict=True)):
        series_values = []
        for horizon in HORIZONS:
            values = data[target, horizon]
            dates = values["dates"].astype("datetime64[D]")
            series = values[f"spatial_model_{field}"]
            series_values.append(series[np.isfinite(series)])
            axis.plot(
                dates,
                series,
                color=line_colors[horizon],
                linestyle=styles[horizon],
                linewidth=0.85,
                label=f"{horizon}-day lead",
            )
        axis.axvspan(np.datetime64("2020-09-28"), np.datetime64("2020-10-27"), color="#B7B7B7", alpha=0.24, linewidth=0)
        axis.axvline(np.datetime64("2020-09-28"), color="#66717C", linewidth=0.45, alpha=0.7)
        axis.axvline(np.datetime64("2020-10-27"), color="#66717C", linewidth=0.45, alpha=0.7)
        axis.grid(True, color="#CBD0D5", linewidth=0.45, alpha=0.65)
        unit = UNITS[target] if label == "RMSE" else "dimensionless"
        axis.set_ylabel(f"{target} spatial {label} ({unit})", fontsize=7.5, labelpad=3.0)
        if label == "RMSE":
            axis.set_ylim(bottom=0.0)
        else:
            finite = np.concatenate(series_values)
            axis.set_ylim(min(float(np.min(finite)), 1.0) - 0.03, max(float(np.max(finite)), 1.0) + 0.03)
            axis.axhline(1.0, color="#66717C", linewidth=0.5, alpha=0.5)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axis.text(
            0.015,
            0.97,
            f"({PANEL_LABELS[panel]})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            zorder=10,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False)
    figure.text(
        0.5,
        0.965,
        "Per-date spatial metrics; shaded: lowest 30-day mean domain ELM-SM anomaly (28 Sep to 27 Oct 2020; predictions not used)",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#66717C",
    )
    save(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "final")
    args = parser.parse_args()
    data = load_data(args.data_dir)
    metric_map(data, "RMSE", args.output_dir / "figure05_gridcell_temporal_rmse.png")
    metric_map(data, "KGE", args.output_dir / "figure06_gridcell_temporal_kge.png")
    gain_map(data, "RMSE", args.output_dir / "figure07_rmse_gain_vs_persistence.png")
    gain_map(data, "KGE", args.output_dir / "figure08_kge_gain_vs_persistence.png")
    daily_metrics(data, args.output_dir / "figure09_daily_spatial_metrics.png")


if __name__ == "__main__":
    main()
