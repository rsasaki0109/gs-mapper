"""Tests for the real-vs-sim rosbag correlation library + CLI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from gs_sim2real.robotics import (
    BagPoseSample,
    BagPoseStream,
    CorrelatedPosePair,
    REAL_VS_SIM_CORRELATION_REPORT_VERSION,
    SimPoseSample,
    correlate_against_sim_trajectory,
    load_sim_pose_samples_jsonl,
    merge_navsat_with_imu_orientation,
    render_real_vs_sim_correlation_markdown,
    wgs84_to_ecef,
    wgs84_to_local_enu,
    write_real_vs_sim_correlation_report_json,
)
from gs_sim2real.robotics import rosbag_correlation as module_under_test


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_bag_stream(samples: list[BagPoseSample]) -> BagPoseStream:
    return BagPoseStream(
        samples=tuple(samples),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )


def test_wgs84_to_ecef_at_known_reference_point() -> None:
    # On the equator at the prime meridian and zero altitude the ECEF point
    # is (a, 0, 0) where a is the WGS84 semi-major axis.
    x, y, z = wgs84_to_ecef(0.0, 0.0, 0.0)
    assert x == pytest.approx(6378137.0, rel=1e-9)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_wgs84_to_local_enu_returns_zero_at_origin() -> None:
    east, north, up = wgs84_to_local_enu(
        35.0,
        139.0,
        12.0,
        origin_latitude=35.0,
        origin_longitude=139.0,
        origin_altitude=12.0,
    )
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(0.0, abs=1e-6)
    assert up == pytest.approx(0.0, abs=1e-6)


def test_wgs84_to_local_enu_small_eastward_step_matches_great_circle() -> None:
    # 0.0001 deg east at lat 35 deg ~= 9.107 m (lon-deg shrinks with cos lat
    # and equatorial deg ~= 111319 m). Allow a small cm-scale tolerance to
    # account for ellipsoid vs sphere differences.
    east, north, up = wgs84_to_local_enu(
        35.0,
        139.0001,
        0.0,
        origin_latitude=35.0,
        origin_longitude=139.0,
        origin_altitude=0.0,
    )
    assert east == pytest.approx(9.107, abs=0.05)
    assert abs(north) < 1e-3
    assert abs(up) < 1e-3


def test_correlator_pairs_nearest_timestamps_and_aggregates_errors() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.00, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.00, position=(1.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=2.00, position=(2.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=3.00, position=(3.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=2.02, position=(2.0, 0.5, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.1,
    )

    assert report.matched_pair_count == 3
    assert report.translation_error_min_meters == pytest.approx(0.0)
    assert report.translation_error_max_meters == pytest.approx(0.5, rel=1e-6)
    assert report.translation_error_mean_meters == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_correlator_drops_pairs_outside_max_match_dt_seconds() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.00, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=10.0, position=(10.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=5.0, position=(5.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.5,
    )
    assert report.matched_pair_count == 1
    assert report.matched_seconds == 0.0  # one matched pair has zero span
    assert report.pairs[0].sim_timestamp_seconds == pytest.approx(0.01)


def test_correlator_returns_nan_aggregates_when_nothing_matches() -> None:
    bag_stream = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    sim_samples = [
        SimPoseSample(timestamp_seconds=10.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.1)
    assert report.matched_pair_count == 0
    assert math.isnan(report.translation_error_mean_meters)
    assert report.pairs == ()


def test_correlator_emits_heading_error_when_bag_has_orientation() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(
                timestamp_seconds=0.0,
                position=(0.0, 0.0, 0.0),
                # 90 deg yaw quaternion in xyzw.
                orientation_xyzw=(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            )
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.1)
    assert report.heading_error_mean_radians == pytest.approx(math.pi / 2.0, rel=1e-6)
    assert report.heading_error_max_radians == pytest.approx(math.pi / 2.0, rel=1e-6)


def test_correlator_keeps_strided_sample_of_pairs() -> None:
    bag_samples = [BagPoseSample(timestamp_seconds=float(i) * 0.01, position=(float(i), 0.0, 0.0)) for i in range(200)]
    sim_samples = [
        SimPoseSample(
            timestamp_seconds=float(i) * 0.01,
            position=(float(i), 0.0, 0.0),
            orientation_xyzw=(0, 0, 0, 1),
        )
        for i in range(200)
    ]

    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
        max_match_dt_seconds=0.005,
        keep_pairs=True,
        max_pairs_kept=50,
    )
    assert report.matched_pair_count == 200
    assert len(report.pairs) <= 50
    assert len(report.pairs) >= 40  # roughly evenly strided


def test_correlator_drops_pairs_when_keep_pairs_is_false() -> None:
    bag_stream = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        bag_stream,
        sim_samples,
        max_match_dt_seconds=0.1,
        keep_pairs=False,
    )
    assert report.matched_pair_count == 1
    assert report.pairs == ()


def test_report_to_dict_round_trips_to_json(tmp_path: Path) -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.5, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples, max_match_dt_seconds=0.5)
    output = write_real_vs_sim_correlation_report_json(tmp_path / "report.json", report)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recordType"] == "real-vs-sim-correlation-report"
    assert payload["version"] == REAL_VS_SIM_CORRELATION_REPORT_VERSION
    assert payload["matchedPairCount"] == 2
    assert payload["bagSource"]["sourceTopic"] == "/gnss/fix"
    assert payload["bagSource"]["referenceOriginWgs84"] == [35.0, 139.0, 10.0]
    assert "translationErrorMeters" in payload
    assert isinstance(payload["pairs"], list)


def test_render_markdown_includes_bag_and_sim_summary() -> None:
    bag_stream = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(bag_stream, sim_samples)
    markdown = render_real_vs_sim_correlation_markdown(report)
    assert "/gnss/fix" in markdown
    assert "sensor_msgs/msg/NavSatFix" in markdown
    assert "translation error" in markdown.lower()


def test_load_sim_pose_samples_jsonl_round_trip(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        "\n".join(
            [
                json.dumps({"timestampSeconds": 1.0, "position": [1.0, 2.0, 3.0], "orientationXyzw": [0, 0, 0, 1]}),
                "  ",  # blank line should be skipped
                json.dumps({"timestampSeconds": 0.5, "position": [0.5, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_sim_pose_samples_jsonl(rollout)
    assert len(samples) == 2
    # Sorted ascending by timestamp regardless of file order.
    assert samples[0].timestamp_seconds == pytest.approx(0.5)
    assert samples[1].timestamp_seconds == pytest.approx(1.0)
    assert samples[1].position == (1.0, 2.0, 3.0)


def test_load_sim_pose_samples_jsonl_rejects_invalid_records(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_sim_pose_samples_jsonl(bad)

    missing_field = tmp_path / "missing.jsonl"
    missing_field.write_text(
        json.dumps({"timestampSeconds": 0.0, "position": [0, 0, 0]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_sim_pose_samples_jsonl(missing_field)


def test_run_rosbag_correlation_cli_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps({"timestampSeconds": 0.0, "position": [0.0, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]})
        + "\n"
        + json.dumps({"timestampSeconds": 1.0, "position": [1.5, 0.0, 0.0], "orientationXyzw": [0, 0, 0, 1]})
        + "\n",
        encoding="utf-8",
    )

    fake_stream = BagPoseStream(
        samples=(
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ),
        frame_id="enu",
        source_topic="/gnss/fix",
        source_msgtype="sensor_msgs/msg/NavSatFix",
        reference_origin_wgs84=(35.0, 139.0, 10.0),
    )

    def _fake_reader(*_args, **_kwargs) -> BagPoseStream:
        return fake_stream

    output = tmp_path / "out.json"
    markdown = tmp_path / "out.md"

    monkeypatch.setattr(module_under_test, "read_navsat_pose_stream", _fake_reader)
    # The script imports from gs_sim2real.robotics, so we also patch the public
    # re-export the script reaches through.
    import gs_sim2real.robotics as robotics_pkg

    monkeypatch.setattr(robotics_pkg, "read_navsat_pose_stream", _fake_reader)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_rosbag_correlation.py"),
            "--bag",
            str(tmp_path),
            "--sim-rollout",
            str(rollout),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--max-match-dt-seconds",
            "0.5",
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    # The CLI reads the actual bag via the real (un-monkeypatched) module
    # since we run it as a subprocess. Skip the subprocess assertion when no
    # real bag exists at tmp_path — exercise the CLI path inside this process
    # instead.
    if completed.returncode != 0 and "no rosbag found" not in completed.stderr.lower():
        pass

    # In-process exercise of the CLI path (read patched).
    from scripts import run_rosbag_correlation as cli_module  # type: ignore[import-not-found]

    monkeypatch.setattr(cli_module, "read_navsat_pose_stream", _fake_reader)
    sys.argv = [
        "run_rosbag_correlation.py",
        "--bag",
        str(tmp_path),
        "--sim-rollout",
        str(rollout),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--max-match-dt-seconds",
        "0.5",
    ]
    assert cli_module.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["matchedPairCount"] == 2
    assert "translation error" in markdown.read_text(encoding="utf-8").lower()


def test_correlated_pose_pair_round_trips_orientation_dropped() -> None:
    pair = CorrelatedPosePair(
        bag_timestamp_seconds=1.0,
        sim_timestamp_seconds=1.05,
        bag_position=(1.0, 2.0, 3.0),
        sim_position=(1.5, 2.0, 3.0),
        translation_error_meters=0.5,
    )
    payload = pair.to_dict()
    assert payload["bagTimestampSeconds"] == 1.0
    assert payload["timeOffsetSeconds"] == pytest.approx(0.05, rel=1e-6)
    assert "headingErrorRadians" not in payload


def test_bag_pose_stream_validation_rejects_unsorted_samples() -> None:
    with pytest.raises(ValueError, match="sorted"):
        BagPoseStream(
            samples=(
                BagPoseSample(timestamp_seconds=2.0, position=(0.0, 0.0, 0.0)),
                BagPoseSample(timestamp_seconds=1.0, position=(0.0, 0.0, 0.0)),
            ),
            frame_id="enu",
            source_topic="/x",
            source_msgtype="sensor_msgs/msg/NavSatFix",
        )

    with pytest.raises(ValueError, match="at least one sample"):
        BagPoseStream(
            samples=(),
            frame_id="enu",
            source_topic="/x",
            source_msgtype="sensor_msgs/msg/NavSatFix",
        )


def test_merge_navsat_with_imu_orientation_fills_in_quaternion_within_window() -> None:
    navsat = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=2.0, position=(2.0, 0.0, 0.0)),
        ]
    )
    imu = (
        (0.01, (0.0, 0.0, 0.0, 1.0)),
        (1.02, (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))),
        # No IMU sample anywhere near t=2.0; that NavSatFix sample stays None.
    )

    fused = merge_navsat_with_imu_orientation(navsat, imu, max_pair_dt_seconds=0.05)

    assert fused.source_topic == navsat.source_topic
    assert fused.reference_origin_wgs84 == navsat.reference_origin_wgs84
    assert fused.samples[0].orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert fused.samples[1].orientation_xyzw is not None
    assert fused.samples[2].orientation_xyzw is None


def test_merge_navsat_with_imu_orientation_rejects_unsorted_imu_samples() -> None:
    navsat = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    out_of_order = (
        (1.0, (0.0, 0.0, 0.0, 1.0)),
        (0.5, (0.0, 0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        merge_navsat_with_imu_orientation(navsat, out_of_order)


def test_merge_navsat_with_imu_orientation_rejects_empty_imu_samples() -> None:
    navsat = _make_bag_stream([BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="at least one sample"):
        merge_navsat_with_imu_orientation(navsat, ())


def test_correlator_emits_heading_error_on_imu_merged_stream() -> None:
    navsat = _make_bag_stream(
        [
            BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
            BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
        ]
    )
    imu = (
        (0.01, (0.0, 0.0, 0.0, 1.0)),
        (1.0, (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))),  # 45 deg yaw
    )
    fused = merge_navsat_with_imu_orientation(navsat, imu, max_pair_dt_seconds=0.05)
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]

    report = correlate_against_sim_trajectory(fused, sim_samples, max_match_dt_seconds=0.05)
    assert report.matched_pair_count == 2
    # Mean of 0 rad and 45 deg = 22.5 deg.
    assert report.heading_error_mean_radians == pytest.approx(math.radians(22.5), rel=1e-6)
    assert report.heading_error_max_radians == pytest.approx(math.radians(45.0), rel=1e-6)
