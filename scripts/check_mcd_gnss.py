#!/usr/bin/env python3
"""Preflight MCD NavSatFix data before running GNSS-seeded preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gs_sim2real.datasets.mcd_gnss_preflight import (  # noqa: E402
    GnssPreflightSummary,
    resolve_image_timestamps,
    scan_mcd_gnss,
)


def _fmt_sec(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def print_summary(summary: GnssPreflightSummary) -> None:
    print("=== MCD GNSS Preflight ===")
    print(f"  data_dir: {summary.data_dir}")
    print(f"  bag_count: {summary.bag_count}")
    print(f"  topic: {summary.topic or 'n/a'}")

    print("\n=== GNSS Samples ===")
    print(f"  total: {summary.total_samples}")
    print(f"  valid: {summary.valid_samples}")
    print(f"  zero placeholders: {summary.zero_placeholder_samples}")
    print(f"  invalid status: {summary.invalid_status_samples}")
    print(f"  non-finite: {summary.nonfinite_samples}")
    print(f"  valid time range: {_fmt_sec(summary.first_valid_sec)} .. {_fmt_sec(summary.last_valid_sec)}")

    print("\n=== Trajectory ===")
    if summary.reference_wgs84 is None:
        print("  reference WGS84: n/a")
    else:
        lat, lon, alt = summary.reference_wgs84
        print(f"  reference WGS84: lat={lat:.9f} lon={lon:.9f} alt={alt:.3f}")
    print(f"  translation extent: {summary.translation_extent_m:.3f} m")
    print(f"  horizontal extent: {summary.horizontal_extent_m:.3f} m")
    print(f"  vertical extent: {summary.vertical_extent_m:.3f} m")
    print(f"  path length: {summary.path_length_m:.3f} m")
    print(f"  horizontal path length: {summary.horizontal_path_length_m:.3f} m")
    if summary.altitude_min_m is not None and summary.altitude_max_m is not None:
        print(
            "  raw altitude range: "
            f"{summary.altitude_min_m:.3f} .. {summary.altitude_max_m:.3f} m "
            f"(span {summary.altitude_span_m:.3f} m)"
        )
    if summary.horizontal_speed_p95_mps is not None and summary.horizontal_speed_max_mps is not None:
        print(
            "  horizontal speed: "
            f"p95={summary.horizontal_speed_p95_mps:.3f} m/s "
            f"max={summary.horizontal_speed_max_mps:.3f} m/s"
        )

    if summary.image_timestamps is not None:
        img = summary.image_timestamps
        print("\n=== Image Timestamp Overlap ===")
        print(f"  path: {img.path}")
        print(f"  images: {img.count}")
        print(f"  image time range: {_fmt_sec(img.start_sec)} .. {_fmt_sec(img.end_sec)}")
        print(f"  overlap count: {img.overlap_count} (tolerance {img.overlap_tolerance_sec:.3f}s)")

    if summary.warnings:
        print("\n=== Warnings ===")
        for warning in summary.warnings:
            print(f"  [WARN] {warning}")

    print("\n=== Summary ===")
    if summary.failures:
        for failure in summary.failures:
            print(f"  [MISS] {failure}")
        print("  Result: not suitable for GNSS-seeded MCD preprocessing")
    else:
        print("  [OK] GNSS looks suitable for pose-seeded MCD preprocessing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcd_session", type=Path, help="MCD session directory or bag path")
    parser.add_argument("--gnss-topic", default=None, help="NavSatFix topic, e.g. /vn200/GPS")
    parser.add_argument(
        "--image-timestamps",
        default=None,
        help="Optional image_timestamps.csv path or directory containing it",
    )
    parser.add_argument("--min-valid-fixes", type=int, default=2, help="Minimum valid non-zero fixes")
    parser.add_argument("--min-translation-m", type=float, default=1.0, help="Minimum horizontal ENU trajectory extent")
    parser.add_argument(
        "--max-vertical-extent-m",
        type=float,
        default=250.0,
        help="Maximum allowed ENU vertical extent; use a negative value to disable",
    )
    parser.add_argument(
        "--max-horizontal-speed-p95-mps",
        type=float,
        default=50.0,
        help="Maximum allowed p95 horizontal speed; use a negative value to disable",
    )
    parser.add_argument(
        "--flatten-altitude",
        action="store_true",
        help="Project all NavSatFix samples to the median valid altitude before trajectory checks",
    )
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Skip the first N seconds of the selected GNSS topic before checking",
    )
    parser.add_argument("--min-image-overlap", type=int, default=2, help="Minimum image timestamps inside GNSS range")
    parser.add_argument(
        "--overlap-tolerance-sec",
        type=float,
        default=0.5,
        help="Tolerance when checking image timestamps against GNSS range",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_ts = resolve_image_timestamps(args.image_timestamps)
    max_vertical_extent_m = None if args.max_vertical_extent_m < 0 else args.max_vertical_extent_m
    max_horizontal_speed_p95_mps = None if args.max_horizontal_speed_p95_mps < 0 else args.max_horizontal_speed_p95_mps
    summary = scan_mcd_gnss(
        args.mcd_session,
        gnss_topic=args.gnss_topic,
        image_timestamps=image_ts,
        min_valid_fixes=args.min_valid_fixes,
        min_translation_m=args.min_translation_m,
        max_vertical_extent_m=max_vertical_extent_m,
        max_horizontal_speed_p95_mps=max_horizontal_speed_p95_mps,
        flatten_altitude=args.flatten_altitude,
        start_offset_sec=max(0.0, args.start_offset_sec),
        min_image_overlap=args.min_image_overlap,
        overlap_tolerance_sec=args.overlap_tolerance_sec,
    )
    if args.json:
        print(json.dumps(asdict(summary), indent=2))
    else:
        print_summary(summary)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
