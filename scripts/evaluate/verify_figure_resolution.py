#!/usr/bin/env python3
"""Verify effective raster resolution at manuscript display size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

FIGURES = {
    5: "figure05_gridcell_temporal_rmse.png",
    6: "figure06_gridcell_temporal_kge.png",
    7: "figure07_rmse_gain_vs_persistence.png",
    8: "figure08_kge_gain_vs_persistence.png",
    9: "figure09_daily_spatial_metrics.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--figure-dir", type=Path, default=root / "figures" / "final")
    parser.add_argument("--display-width-in", type=float, default=5.04)
    parser.add_argument("--minimum-dpi", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=root / "figures" / "final" / "figure_verification.json")
    args = parser.parse_args()

    records = []
    for number, filename in FIGURES.items():
        path = args.figure_dir / filename
        with Image.open(path) as image:
            width_px, height_px = image.size
            metadata_dpi = image.info.get("dpi")
        display_height_in = args.display_width_in * height_px / width_px
        effective_dpi = width_px / args.display_width_in
        record = {
            "figure": number,
            "file": filename,
            "pixel_dimensions": [width_px, height_px],
            "display_dimensions_in": [round(args.display_width_in, 3), round(display_height_in, 3)],
            "effective_dpi": round(effective_dpi, 1),
            "png_metadata_dpi": [round(value, 1) for value in metadata_dpi] if metadata_dpi else None,
            "pdf_companion": path.with_suffix(".pdf").exists(),
        }
        if effective_dpi < args.minimum_dpi:
            raise SystemExit(f"Figure {number} is only {effective_dpi:.1f} effective DPI")
        records.append(record)

    payload = {
        "manuscript_display_width_in": args.display_width_in,
        "minimum_effective_dpi": args.minimum_dpi,
        "all_pass": True,
        "figures": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
