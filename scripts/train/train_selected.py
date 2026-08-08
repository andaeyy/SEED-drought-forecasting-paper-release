#!/usr/bin/env python3
"""Build a target-specific paper model from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed.models.convlstm import build_autoregressive, build_encoder_decoder, build_sequence_to_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--height", type=int, default=142)
    parser.add_argument("--width", type=int, default=95)
    parser.add_argument("--channels", type=int, default=7)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shape = (config["input_days"], args.height, args.width, args.channels)
    architecture = config["architecture"]
    widths = tuple(config["hidden_channels"])
    if architecture == "sequence_to_map":
        model = build_sequence_to_map(shape, filters=widths)
    elif architecture == "encoder_decoder":
        model = build_encoder_decoder(shape, config["lead_days"], encoder_filters=widths, decoder_filters=widths)
    elif architecture == "autoregressive":
        model = build_autoregressive(shape, config["lead_days"], filters=widths)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    model.summary()
    if not args.summary_only:
        raise SystemExit("data-driven fitting is documented in docs/reproducibility.md")


if __name__ == "__main__":
    main()
