"""Utility script to compute ERA5 normalization statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

from weatherflow.data.era5 import ERA5Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute normalization stats for ERA5")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ERA5 zarr store (defaults to WeatherBench2).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="t,z,u,v",
        help="Comma-separated variable short names (e.g., t,z,u,v).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="500",
        help="Comma-separated pressure levels (hPa).",
    )
    parser.add_argument(
        "--time-start",
        type=str,
        default="2015",
        help="Start time (e.g., 2015 or 2015-01-01).",
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default="2016",
        help="End time (exclusive) (e.g., 2016 or 2016-01-01).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stats/era5_stats.json"),
        help="Where to write the computed stats.",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        default=None,
        help="Optional directory for simplecache to store downloaded chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    levels = [int(lvl) for lvl in args.levels.split(",") if lvl]

    ds = ERA5Dataset(
        variables=variables,
        pressure_levels=levels,
        data_path=args.data_path,
        time_slice=(args.time_start, args.time_end),
        normalize=False,
        stats_path=args.output,
        auto_compute_stats=True,
        local_cache_dir=args.local_cache,
        verbose=True,
    )

    print(f"Wrote stats to {args.output.resolve()}")
    for var, stats in ds.normalize_stats.items():
        print(f"{var}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
