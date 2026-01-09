#!/usr/bin/env python3
"""
Command-line interface for managing the WeatherFlow dataset cache.

Usage:
    python -m weatherflow.data.cache_cli info
    python -m weatherflow.data.cache_cli list
    python -m weatherflow.data.cache_cli clear [dataset_name]
    python -m weatherflow.data.cache_cli config --cache-dir /path/to/cache
"""

import argparse
import sys

from .cache import (
    DatasetCache,
    configure_cache,
    get_default_cache,
    print_cache_info,
)


def cmd_info(args: argparse.Namespace) -> None:
    """Show detailed cache information."""
    cache = get_default_cache()
    print(cache.info(detailed=args.detailed))


def cmd_list(args: argparse.Namespace) -> None:
    """List all cached datasets."""
    cache = get_default_cache()
    datasets = cache.list_datasets()

    if not datasets:
        print("No datasets cached.")
        print(f"Cache location: {cache.cache_dir}")
        return

    print(f"Cached datasets in {cache.cache_dir}:")
    print("-" * 60)

    for ds in datasets:
        print(f"  {ds.name}")
        print(f"    Size: {ds.size_human()}")
        print(f"    Path: {ds.path}")
        print(f"    Last accessed: {ds.last_accessed}")
        if ds.metadata:
            for key, value in ds.metadata.items():
                print(f"    {key}: {value}")
        print()


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear cached data."""
    cache = get_default_cache()

    if args.dataset:
        if not cache.has_dataset(args.dataset):
            print(f"Dataset '{args.dataset}' not found in cache.")
            sys.exit(1)

        if not args.yes:
            response = input(f"Clear dataset '{args.dataset}'? [y/N] ")
            if response.lower() != "y":
                print("Cancelled.")
                return

        cache.clear(args.dataset)
        print(f"Cleared dataset '{args.dataset}'.")
    else:
        datasets = cache.list_datasets()
        if not datasets:
            print("Cache is already empty.")
            return

        total_size = sum(ds.size_bytes for ds in datasets)
        print(f"This will clear {len(datasets)} dataset(s), {cache._format_size(total_size)} total.")

        if not args.yes:
            response = input("Clear ALL cached datasets? [y/N] ")
            if response.lower() != "y":
                print("Cancelled.")
                return

        cache.clear()
        print("Cleared all cached datasets.")


def cmd_config(args: argparse.Namespace) -> None:
    """Configure cache settings."""
    if args.cache_dir:
        configure_cache(args.cache_dir)
        print(f"Cache directory configured: {args.cache_dir}")
        print("This setting will be used for future sessions.")
    else:
        cache = get_default_cache()
        print(f"Current cache directory: {cache.cache_dir}")


def cmd_years(args: argparse.Namespace) -> None:
    """Show which ERA5 years are cached."""
    cache = get_default_cache()
    years = cache.get_cached_era5_years()

    if not years:
        print("No ERA5 data cached.")
        return

    print("Cached ERA5 years:")
    print(f"  {', '.join(str(y) for y in years)}")
    print(f"  Total: {len(years)} year(s)")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="WeatherFlow Dataset Cache Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info                    Show cache information
  %(prog)s info --detailed         Show detailed cache information
  %(prog)s list                    List all cached datasets
  %(prog)s years                   Show cached ERA5 years
  %(prog)s clear                   Clear all cached data (with confirmation)
  %(prog)s clear era5              Clear only ERA5 data
  %(prog)s clear --yes             Clear without confirmation
  %(prog)s config --cache-dir /data/cache  Set cache directory
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show cache information")
    info_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed information"
    )
    info_parser.set_defaults(func=cmd_info)

    # list command
    list_parser = subparsers.add_parser("list", help="List cached datasets")
    list_parser.set_defaults(func=cmd_list)

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cached data")
    clear_parser.add_argument(
        "dataset", nargs="?", help="Dataset to clear (omit for all)"
    )
    clear_parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    clear_parser.set_defaults(func=cmd_clear)

    # config command
    config_parser = subparsers.add_parser("config", help="Configure cache settings")
    config_parser.add_argument(
        "--cache-dir", help="Set the cache directory"
    )
    config_parser.set_defaults(func=cmd_config)

    # years command
    years_parser = subparsers.add_parser("years", help="Show cached ERA5 years")
    years_parser.set_defaults(func=cmd_years)

    args = parser.parse_args()

    if not args.command:
        # Default to showing info
        args.detailed = False
        cmd_info(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
