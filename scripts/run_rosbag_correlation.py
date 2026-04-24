#!/usr/bin/env python3
"""Correlate a sim trajectory JSONL against a rosbag2 GNSS pose stream.

The bag is read with the same ``rosbags`` machinery the dataset
loaders rely on, so zstd-compressed sqlite3 bags work without
decompression. The first valid NavSatFix anchors a local ENU origin
unless ``--reference-origin`` is supplied. The sim trajectory is a
JSONL of ``{timestampSeconds, position, orientationXyzw}`` records —
typically the output of a headless rollout. The script writes the
correlation report as JSON and optionally as a Markdown summary.

Example:

    python3 scripts/run_rosbag_correlation.py \\
        --bag data/autoware_leo_drive_bag1 \\
        --sim-rollout artifacts/rollout/bag1.jsonl \\
        --output artifacts/correlation/bag1.json \\
        --markdown artifacts/correlation/bag1.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.robotics import (  # noqa: E402
    correlate_against_sim_trajectory,
    load_sim_pose_samples_jsonl,
    merge_navsat_with_imu_orientation,
    read_imu_orientation_stream,
    read_navsat_pose_stream,
    render_real_vs_sim_correlation_markdown,
    write_real_vs_sim_correlation_report_json,
)


def _parse_reference_origin(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("reference-origin must be 'lat,lon,alt' (three comma-separated floats)")
    try:
        return tuple(float(item) for item in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"reference-origin parse error: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        required=True,
        help="Path to a rosbag2 directory (containing metadata.yaml).",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="NavSatFix topic to read; defaults to the first NavSatFix-typed connection.",
    )
    parser.add_argument(
        "--sim-rollout",
        type=Path,
        required=True,
        help="JSONL file with one {timestampSeconds, position, orientationXyzw} record per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the correlation report JSON.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to write a Markdown summary alongside the JSON.",
    )
    parser.add_argument(
        "--max-match-dt-seconds",
        type=float,
        default=0.05,
        help="Reject pose pairs whose clock skew exceeds this many seconds (default: 0.05).",
    )
    parser.add_argument(
        "--reference-origin",
        type=_parse_reference_origin,
        default=None,
        help="Optional 'lat,lon,alt' anchor for the local ENU frame; otherwise the first valid bag fix anchors it.",
    )
    parser.add_argument(
        "--max-pairs-kept",
        type=int,
        default=1024,
        help="Cap on the number of CorrelatedPosePair entries embedded in the JSON report (default: 1024).",
    )
    parser.add_argument(
        "--no-pairs",
        action="store_true",
        help="Drop the per-pair entries from the JSON report (keeps only aggregate statistics).",
    )
    parser.add_argument(
        "--imu-topic",
        default=None,
        help=(
            "Optional sensor_msgs/Imu topic; when provided, its orientation is merged "
            "onto the NavSatFix stream so the correlator can compute heading errors."
        ),
    )
    parser.add_argument(
        "--imu-pair-dt-seconds",
        type=float,
        default=0.05,
        help="Reject IMU/NavSatFix pairings whose clock skew exceeds this many seconds (default: 0.05).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sim_samples = load_sim_pose_samples_jsonl(args.sim_rollout)
    bag_stream = read_navsat_pose_stream(
        [args.bag],
        topic=args.topic,
        reference_origin_wgs84=args.reference_origin,
    )
    if args.imu_topic is not None:
        imu_orientations = read_imu_orientation_stream([args.bag], topic=args.imu_topic)
        bag_stream = merge_navsat_with_imu_orientation(
            bag_stream,
            imu_orientations,
            max_pair_dt_seconds=args.imu_pair_dt_seconds,
        )
    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=args.max_match_dt_seconds,
        keep_pairs=not args.no_pairs,
        max_pairs_kept=args.max_pairs_kept,
    )
    write_real_vs_sim_correlation_report_json(args.output, report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_real_vs_sim_correlation_markdown(report), encoding="utf-8")
    print(
        f"matched {report.matched_pair_count} / {report.sim_sample_count} sim samples "
        f"(translation mean={report.translation_error_mean_meters:.4f} m, "
        f"p95={report.translation_error_p95_meters:.4f} m)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
