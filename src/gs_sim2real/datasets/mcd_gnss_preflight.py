"""GNSS suitability checks for MCD NavSatFix-based pose seeding."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gs_sim2real.datasets.mcd import MCDLoader
from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor


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
    nonmonotonic_valid_timestamps: int = 0
    duplicate_valid_timestamps: int = 0
    image_timestamps: ImageTimestampSummary | None = None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def status_is_invalid(status_obj: Any) -> bool:
    if status_obj is None:
        return False
    code = getattr(status_obj, "status", status_obj)
    try:
        return int(code) < 0
    except (TypeError, ValueError):
        return False


def resolve_image_timestamps(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "image_timestamps.csv"
    return p


def read_image_timestamps(path: Path) -> list[float]:
    rows: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row.get("timestamp_ns")
            if not value:
                continue
            rows.append(float(value) * 1e-9)
    return rows


def summarize_image_overlap(
    path: Path,
    valid_start: float | None,
    valid_end: float | None,
    tolerance_sec: float,
) -> ImageTimestampSummary:
    stamps = read_image_timestamps(path)
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


@dataclass(slots=True)
class GnssPreflightConfig:
    gnss_topic: str | None = None
    image_timestamps: Path | None = None
    min_valid_fixes: int = 2
    min_translation_m: float = 1.0
    max_vertical_extent_m: float | None = 250.0
    max_horizontal_speed_p95_mps: float | None = 50.0
    flatten_altitude: bool = False
    start_offset_sec: float = 0.0
    min_image_overlap: int = 2
    overlap_tolerance_sec: float = 0.5


class MCDGnssPreflight:
    """Scan MCD rosbags and decide whether GNSS pose seeding is viable."""

    def __init__(self, data_dir: str | Path, config: GnssPreflightConfig | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or GnssPreflightConfig()
        self.loader = MCDLoader(self.data_dir)

    def scan(self) -> GnssPreflightSummary:
        bag_paths = self.loader._find_bag_paths(self.loader.data_dir)
        if not bag_paths:
            summary = GnssPreflightSummary(
                data_dir=str(self.data_dir),
                bag_count=0,
                topic=self.config.gnss_topic or "",
            )
            summary.failures.append(f"no rosbag files or rosbag2 directories found under {self.data_dir}")
            return summary

        reader_cls = self.loader._get_anyreader()
        summary = GnssPreflightSummary(
            data_dir=str(self.data_dir),
            bag_count=len(bag_paths),
            topic=self.config.gnss_topic or "",
        )
        valid_rows = self._read_valid_rows(reader_cls, bag_paths, summary)
        self._summarize_trajectory(valid_rows, summary)
        self._apply_thresholds(summary)
        self._check_image_overlap(summary)
        return summary

    def _read_valid_rows(
        self,
        reader_cls,
        bag_paths: list[Path],
        summary: GnssPreflightSummary,
    ) -> list[tuple[float, float, float, float]]:
        valid_rows: list[tuple[float, float, float, float]] = []
        with self.loader._create_reader(reader_cls, bag_paths) as reader:
            connection = self.loader._select_connection(
                reader.topics,
                requested_topic=self.config.gnss_topic,
                preferred_topics=self.loader.DEFAULT_GNSS_TOPICS,
                allowed_msgtypes=self.loader.NAVSAT_MSGTYPES,
            )
            if connection is None:
                summary.failures.append("no sensor_msgs/NavSatFix topic found")
                return valid_rows
            summary.topic = str(connection.topic)

            first_ts: float | None = None
            for _, timestamp_ns, rawdata in reader.messages(connections=[connection]):
                ts = float(timestamp_ns) * 1e-9
                if first_ts is None:
                    first_ts = ts
                if self.config.start_offset_sec > 0.0 and ts - first_ts < self.config.start_offset_sec:
                    continue
                summary.total_samples += 1
                msg = reader.deserialize(rawdata, connection.msgtype)
                lat = float(getattr(msg, "latitude", float("nan")))
                lon = float(getattr(msg, "longitude", float("nan")))
                alt = float(getattr(msg, "altitude", 0.0))
                if not (np.isfinite(lat) and np.isfinite(lon)):
                    summary.nonfinite_samples += 1
                    continue
                if status_is_invalid(getattr(msg, "status", None)):
                    summary.invalid_status_samples += 1
                    continue
                if abs(lat) < 1e-12 and abs(lon) < 1e-12:
                    summary.zero_placeholder_samples += 1
                    continue
                valid_rows.append((ts, lat, lon, alt))
        summary.valid_samples = len(valid_rows)
        return valid_rows

    def _summarize_trajectory(
        self,
        valid_rows: list[tuple[float, float, float, float]],
        summary: GnssPreflightSummary,
    ) -> None:
        if not valid_rows:
            return
        summary.first_valid_sec = valid_rows[0][0]
        summary.last_valid_sec = valid_rows[-1][0]
        raw_altitudes = np.array([row[3] for row in valid_rows], dtype=np.float64)
        summary.altitude_min_m = float(np.min(raw_altitudes))
        summary.altitude_max_m = float(np.max(raw_altitudes))
        summary.altitude_span_m = float(summary.altitude_max_m - summary.altitude_min_m)
        ref_lat, ref_lon = valid_rows[0][1], valid_rows[0][2]
        ref_alt = float(np.median(raw_altitudes)) if self.config.flatten_altitude else valid_rows[0][3]
        summary.reference_wgs84 = (ref_lat, ref_lon, ref_alt)
        if self.config.flatten_altitude:
            summary.warnings.append(
                f"flattened altitude to median {ref_alt:.3f} m; raw span was {summary.altitude_span_m:.3f} m"
            )
        enu = np.array(
            [
                LiDARSLAMProcessor._wgs84_to_enu(
                    lat,
                    lon,
                    ref_alt if self.config.flatten_altitude else alt,
                    ref_lat,
                    ref_lon,
                    ref_alt,
                )
                for _, lat, lon, alt in valid_rows
            ],
            dtype=np.float64,
        )
        if len(enu) < 2:
            return
        summary.translation_extent_m = float(np.linalg.norm(enu.max(axis=0) - enu.min(axis=0)))
        summary.horizontal_extent_m = float(np.linalg.norm(enu[:, :2].max(axis=0) - enu[:, :2].min(axis=0)))
        summary.vertical_extent_m = float(np.ptp(enu[:, 2]))
        summary.path_length_m = float(np.linalg.norm(np.diff(enu, axis=0), axis=1).sum())
        diff_xy = np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)
        summary.horizontal_path_length_m = float(diff_xy.sum())
        ts = np.array([row[0] for row in valid_rows], dtype=np.float64)
        dt = np.diff(ts)
        summary.nonmonotonic_valid_timestamps = int(np.count_nonzero(dt < 0.0))
        summary.duplicate_valid_timestamps = int(np.count_nonzero(dt == 0.0))
        positive_dt = dt > 0.0
        if np.any(positive_dt):
            horizontal_speed = diff_xy[positive_dt] / dt[positive_dt]
            summary.horizontal_speed_p95_mps = float(np.percentile(horizontal_speed, 95))
            summary.horizontal_speed_max_mps = float(np.max(horizontal_speed))

    def _apply_thresholds(self, summary: GnssPreflightSummary) -> None:
        cfg = self.config
        if summary.valid_samples < cfg.min_valid_fixes:
            summary.failures.append(f"valid fixes {summary.valid_samples} < required {cfg.min_valid_fixes}")
        if summary.valid_samples >= cfg.min_valid_fixes and summary.horizontal_extent_m < cfg.min_translation_m:
            summary.failures.append(
                f"horizontal translation extent {summary.horizontal_extent_m:.3f} m < "
                f"required {cfg.min_translation_m:.3f} m"
            )
        if (
            summary.valid_samples >= cfg.min_valid_fixes
            and cfg.max_vertical_extent_m is not None
            and summary.vertical_extent_m > cfg.max_vertical_extent_m
        ):
            summary.failures.append(
                f"vertical extent {summary.vertical_extent_m:.3f} m > allowed {cfg.max_vertical_extent_m:.3f} m"
            )
        if (
            summary.horizontal_speed_p95_mps is not None
            and cfg.max_horizontal_speed_p95_mps is not None
            and summary.horizontal_speed_p95_mps > cfg.max_horizontal_speed_p95_mps
        ):
            summary.failures.append(
                f"horizontal p95 speed {summary.horizontal_speed_p95_mps:.3f} m/s > allowed "
                f"{cfg.max_horizontal_speed_p95_mps:.3f} m/s"
            )
        if summary.nonmonotonic_valid_timestamps:
            summary.failures.append(f"valid GNSS timestamps regressed {summary.nonmonotonic_valid_timestamps} times")
        if summary.duplicate_valid_timestamps:
            summary.warnings.append(f"valid GNSS timestamps repeated {summary.duplicate_valid_timestamps} times")
        if summary.zero_placeholder_samples and summary.valid_samples == 0:
            summary.failures.append("all finite GNSS fixes are zero placeholders")
        elif summary.zero_placeholder_samples:
            summary.warnings.append(f"ignored {summary.zero_placeholder_samples} zero-placeholder fixes")

    def _check_image_overlap(self, summary: GnssPreflightSummary) -> None:
        image_timestamps = self.config.image_timestamps
        if image_timestamps is None:
            summary.warnings.append("no image_timestamps.csv provided; skipped image/GNSS overlap check")
            return
        if not image_timestamps.is_file():
            summary.failures.append(f"image timestamp file not found: {image_timestamps}")
            return
        img_summary = summarize_image_overlap(
            image_timestamps,
            summary.first_valid_sec,
            summary.last_valid_sec,
            self.config.overlap_tolerance_sec,
        )
        summary.image_timestamps = img_summary
        if img_summary.count == 0:
            summary.failures.append(f"no image timestamps found in {image_timestamps}")
        elif img_summary.overlap_count < self.config.min_image_overlap:
            summary.failures.append(
                f"image/GNSS overlap {img_summary.overlap_count} < required {self.config.min_image_overlap}"
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
    config = GnssPreflightConfig(
        gnss_topic=gnss_topic,
        image_timestamps=image_timestamps,
        min_valid_fixes=min_valid_fixes,
        min_translation_m=min_translation_m,
        max_vertical_extent_m=max_vertical_extent_m,
        max_horizontal_speed_p95_mps=max_horizontal_speed_p95_mps,
        flatten_altitude=flatten_altitude,
        start_offset_sec=start_offset_sec,
        min_image_overlap=min_image_overlap,
        overlap_tolerance_sec=overlap_tolerance_sec,
    )
    return MCDGnssPreflight(data_dir, config).scan()
