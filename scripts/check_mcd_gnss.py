#!/usr/bin/env python3
"""Preflight MCD NavSatFix data before running GNSS-seeded preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gs_sim2real.datasets.mcd import MCDLoader  # noqa: E402
from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor  # noqa: E402


@dataclass
class ImageTimestampSummary:
    path: str
    count: int
    start_sec: float | None
    end_sec: float | None
    overlap_count: int
    overlap_tolerance_sec: float


@dataclass
class GnssPreflightSummary:
    data_dir: str
    bag_count: int
    topic: str
    total_samples: int = 0
    valid_samples: int = 0
    zero_placeholder_samples: int = 0
    invalid_status_samples: int = 0
    nonfinite_samples: int = 0
    first_valid_sec: float | None = None
    last_valid_sec: float | None = None
    reference_wgs84: tuple[float, float, float] | None = None
    translation_extent_m: float = 0.0
    horizontal_extent_m: float = 0.0
    vertical_extent_m: float = 0.0
    path_length_m: float = 0.0
    horizontal_path_length_m: float = 0.0
    altitude_min_m: float | None = None
    altitude_max_m: float | None = None
    altitude_span_m: float = 0.0
    horizontal_speed_p95_mps: float | None = None
    horizontal_speed_max_mps: float | None = None
    image_timestamps: ImageTimestampSummary | None = None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _status_is_invalid(status_obj: Any) -> bool:
    if status_obj is None:
        return False
    code = getattr(status_obj, "status", status_obj)
    try:
        return int(code) < 0
    except (TypeError, ValueError):
        return False


def _resolve_image_timestamps(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "image_timestamps.csv"
    return p


def _read_image_timestamps(path: Path) -> list[float]:
    rows: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("timestamp_ns")
            if not value:
                continue
            rows.append(float(value) * 1e-9)
    return rows


def _summarize_image_overlap(
    path: Path,
    valid_start: float | None,
    valid_end: float | None,
    tolerance_sec: float,
) -> ImageTimestampSummary:
    stamps = _read_image_timestamps(path)
    if not stamps:
        return ImageTimestampSummary(
            path=str(path),
            count=0,
            start_sec=None,
            end_sec=None,
            overlap_count=0,
            overlap_tolerance_sec=tolerance_sec,
        )

    start = min(stamps)
    end = max(stamps)
    if valid_start is None or valid_end is None:
        overlap_count = 0
    else:
        lo = valid_start - tolerance_sec
        hi = valid_end + tolerance_sec
        overlap_count = sum(1 for ts in stamps if lo <= ts <= hi)
    return ImageTimestampSummary(
        path=str(path),
        count=len(stamps),
        start_sec=start,
        end_sec=end,
        overlap_count=overlap_count,
        overlap_tolerance_sec=tolerance_sec,
    )


def scan_mcd_gnss(
    data_dir: Path,
    *,
    gnss_topic: str | None = None,
    image_timestamps: Path | None = None,
    min_valid_fixes: int = 2,
    min_translation_m: float = 1.0,
    max_vertical_extent_m: float | None = 250.0,
    max_horizontal_speed_p95_mps: float | None = 50.0,
    flatten_altitude: bool = False,
    start_offset_sec: float = 0.0,
    min_image_overlap: int = 2,
    overlap_tolerance_sec: float = 0.5,
) -> GnssPreflightSummary:
    loader = MCDLoader(data_dir)
    bag_paths = loader._find_bag_paths(loader.data_dir)
    if not bag_paths:
        summary = GnssPreflightSummary(data_dir=str(data_dir), bag_count=0, topic=gnss_topic or "")
        summary.failures.append(f"no rosbag files or rosbag2 directories found under {data_dir}")
        return summary

    reader_cls = loader._get_anyreader()
    selected_topic = gnss_topic or ""
    valid_rows: list[tuple[float, float, float, float]] = []
    summary = GnssPreflightSummary(data_dir=str(data_dir), bag_count=len(bag_paths), topic=selected_topic)

    with loader._create_reader(reader_cls, bag_paths) as reader:
        connection = loader._select_connection(
            reader.topics,
            requested_topic=gnss_topic,
            preferred_topics=loader.DEFAULT_GNSS_TOPICS,
            allowed_msgtypes=loader.NAVSAT_MSGTYPES,
        )
        if connection is None:
            summary.failures.append("no sensor_msgs/NavSatFix topic found")
            return summary
        summary.topic = str(connection.topic)

        first_ts: float | None = None
        for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
            ts = float(timestamp_ns) * 1e-9
            if first_ts is None:
                first_ts = ts
            if start_offset_sec > 0.0 and ts - first_ts < start_offset_sec:
                continue
            summary.total_samples += 1
            msg = reader.deserialize(rawdata, connection.msgtype)
            lat = float(getattr(msg, "latitude", float("nan")))
            lon = float(getattr(msg, "longitude", float("nan")))
            alt = float(getattr(msg, "altitude", 0.0))
            if not (np.isfinite(lat) and np.isfinite(lon)):
                summary.nonfinite_samples += 1
                continue
            if _status_is_invalid(getattr(msg, "status", None)):
                summary.invalid_status_samples += 1
                continue
            if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                summary.zero_placeholder_samples += 1
                continue
            valid_rows.append((ts, lat, lon, alt))

    summary.valid_samples = len(valid_rows)
    if valid_rows:
        summary.first_valid_sec = valid_rows[0][0]
        summary.last_valid_sec = valid_rows[-1][0]
        raw_altitudes = np.array([row[3] for row in valid_rows], dtype=np.float64)
        summary.altitude_min_m = float(np.min(raw_altitudes))
        summary.altitude_max_m = float(np.max(raw_altitudes))
        summary.altitude_span_m = float(summary.altitude_max_m - summary.altitude_min_m)
        ref_lat, ref_lon = valid_rows[0][1], valid_rows[0][2]
        ref_alt = float(np.median(raw_altitudes)) if flatten_altitude else valid_rows[0][3]
        summary.reference_wgs84 = (ref_lat, ref_lon, ref_alt)
        if flatten_altitude:
            summary.warnings.append(
                f"flattened altitude to median {ref_alt:.3f} m; raw span was {summary.altitude_span_m:.3f} m"
            )
        enu = np.array(
            [
                LiDARSLAMProcessor._wgs84_to_enu(
                    lat,
                    lon,
                    ref_alt if flatten_altitude else alt,
                    ref_lat,
                    ref_lon,
                    ref_alt,
                )
                for _, lat, lon, alt in valid_rows
            ],
            dtype=np.float64,
        )
        if len(enu) >= 2:
            summary.translation_extent_m = float(np.linalg.norm(enu.max(axis=0) - enu.min(axis=0)))
            summary.horizontal_extent_m = float(np.linalg.norm(enu[:, :2].max(axis=0) - enu[:, :2].min(axis=0)))
            summary.vertical_extent_m = float(np.ptp(enu[:, 2]))
            summary.path_length_m = float(np.linalg.norm(np.diff(enu, axis=0), axis=1).sum())
            diff_xy = np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)
            summary.horizontal_path_length_m = float(diff_xy.sum())
            ts = np.array([row[0] for row in valid_rows], dtype=np.float64)
            dt = np.diff(ts)
            positive_dt = dt > 0.0
            if np.any(positive_dt):
                horizontal_speed = diff_xy[positive_dt] / dt[positive_dt]
                summary.horizontal_speed_p95_mps = float(np.percentile(horizontal_speed, 95))
                summary.horizontal_speed_max_mps = float(np.max(horizontal_speed))

    if summary.valid_samples < min_valid_fixes:
        summary.failures.append(f"valid fixes {summary.valid_samples} < required {min_valid_fixes}")
    if summary.valid_samples >= min_valid_fixes and summary.horizontal_extent_m < min_translation_m:
        summary.failures.append(
            f"horizontal translation extent {summary.horizontal_extent_m:.3f} m < required {min_translation_m:.3f} m"
        )
    if (
        summary.valid_samples >= min_valid_fixes
        and max_vertical_extent_m is not None
        and summary.vertical_extent_m > max_vertical_extent_m
    ):
        summary.failures.append(
            f"vertical extent {summary.vertical_extent_m:.3f} m > allowed {max_vertical_extent_m:.3f} m"
        )
    if (
        summary.horizontal_speed_p95_mps is not None
        and max_horizontal_speed_p95_mps is not None
        and summary.horizontal_speed_p95_mps > max_horizontal_speed_p95_mps
    ):
        summary.failures.append(
            f"horizontal p95 speed {summary.horizontal_speed_p95_mps:.3f} m/s > allowed "
            f"{max_horizontal_speed_p95_mps:.3f} m/s"
        )
    if summary.zero_placeholder_samples and summary.valid_samples == 0:
        summary.failures.append("all finite GNSS fixes are zero placeholders")
    elif summary.zero_placeholder_samples:
        summary.warnings.append(f"ignored {summary.zero_placeholder_samples} zero-placeholder fixes")

    if image_timestamps is not None:
        if image_timestamps.is_file():
            img_summary = _summarize_image_overlap(
                image_timestamps,
                summary.first_valid_sec,
                summary.last_valid_sec,
                overlap_tolerance_sec,
            )
            summary.image_timestamps = img_summary
            if img_summary.count == 0:
                summary.failures.append(f"no image timestamps found in {image_timestamps}")
            elif img_summary.overlap_count < min_image_overlap:
                summary.failures.append(
                    f"image/GNSS overlap {img_summary.overlap_count} < required {min_image_overlap}"
                )
        else:
            summary.failures.append(f"image timestamp file not found: {image_timestamps}")
    else:
        summary.warnings.append("no image_timestamps.csv provided; skipped image/GNSS overlap check")

    return summary


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
    image_ts = _resolve_image_timestamps(args.image_timestamps)
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
