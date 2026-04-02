#!/usr/bin/env python3
"""Download datasets for gs-sim2real.

Usage:
    python scripts/download_datasets.py --dataset ggrt --dest data/
    python scripts/download_datasets.py --dataset covla --dest data/
    python scripts/download_datasets.py --dataset mcd --dest data/
    python scripts/download_datasets.py --list

This script reads dataset metadata from configs/datasets.yaml and
downloads the specified dataset to the destination directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    """Download a dataset based on CLI arguments."""
    parser = argparse.ArgumentParser(description="Download datasets for gs-sim2real")
    parser.add_argument("--dataset", help="Dataset to download (e.g. ggrt, covla, mcd)")
    parser.add_argument("--dest", default=None, help="Destination directory (default: data/)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--sample-images", action="store_true", help="Download sample images for demo")
    parser.add_argument("--num-images", type=int, default=10, help="Number of sample images (default: 10)")
    args = parser.parse_args()

    if args.list:
        from gs_sim2real.common.config import load_datasets_config

        datasets = load_datasets_config()
        print("Available datasets:")
        print("-" * 60)
        for name, config in datasets.items():
            desc = config.get("description", "No description").strip().split("\n")[0]
            print(f"  {name:10s} - {desc}")
        return

    if args.sample_images:
        from gs_sim2real.common.download import download_sample_images

        dest = Path(args.dest) if args.dest else project_root / "data" / "sample"
        download_sample_images(dest, num_images=args.num_images)
        return

    if not args.dataset:
        parser.print_help()
        print("\nError: --dataset is required (or use --list to see available datasets)")
        sys.exit(1)

    from gs_sim2real.common.download import download_dataset

    dest = Path(args.dest) if args.dest else None

    print(f"Downloading dataset '{args.dataset}'...")
    output = download_dataset(
        name=args.dataset,
        output_dir=dest,
        max_samples=args.max_samples,
    )
    print(f"Done. Dataset at: {output}")


if __name__ == "__main__":
    main()
