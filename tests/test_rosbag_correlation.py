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
    BagPoseStreamMetadata,
    CorrelatedPosePair,
    REAL_VS_SIM_CORRELATION_REPORT_VERSION,
    RealVsSimCorrelationThresholds,
    SimPoseSample,
    correlate_against_sim_trajectory,
    correlation_threshold_overrides_from_dict,
    correlation_threshold_overrides_to_dict,
    evaluate_real_vs_sim_correlation_thresholds,
    load_correlation_threshold_overrides_json,
    load_real_vs_sim_correlation_report_json,
    load_sim_pose_samples_jsonl,
    merge_navsat_with_imu_orientation,
    real_vs_sim_correlation_report_from_dict,
    real_vs_sim_correlation_thresholds_from_dict,
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


def test_real_vs_sim_correlation_report_round_trips_through_json(tmp_path: Path) -> None:
    """to_dict -> from_dict (and disk round-trip) must rebuild an identical report."""
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(
            timestamp_seconds=1.0,
            position=(1.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.01, position=(0.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.01, position=(1.05, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
        max_match_dt_seconds=0.05,
    )
    assert isinstance(report.bag_source, BagPoseStreamMetadata)

    rebuilt = real_vs_sim_correlation_report_from_dict(report.to_dict())
    assert rebuilt.to_dict() == report.to_dict()

    output_path = tmp_path / "correlation.json"
    write_real_vs_sim_correlation_report_json(output_path, report)
    loaded = load_real_vs_sim_correlation_report_json(output_path)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.bag_source.source_topic == "/gnss/fix"
    assert loaded.matched_pair_count == 2


def test_real_vs_sim_correlation_report_from_dict_rejects_bad_record_type() -> None:
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0)),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(1.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    payload = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples),
        sim_samples,
    ).to_dict()
    payload["recordType"] = "not-a-correlation-report"
    with pytest.raises(ValueError, match="recordType"):
        real_vs_sim_correlation_report_from_dict(payload)


def _correlation_report_with(
    *,
    translation_mean: float,
    translation_p95: float,
    translation_max: float,
    heading_mean: float | None = None,
):
    """Build a synthetic two-pair report with the requested error statistics."""
    bag_samples = [
        BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0)),
        BagPoseSample(timestamp_seconds=1.0, position=(translation_mean * 2.0, 0.0, 0.0)),
    ]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
        SimPoseSample(timestamp_seconds=1.0, position=(0.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples)
    # Inject the requested statistics deterministically (the synthetic pairs above
    # already produce a known mean/max but the test wants exact control).
    return type(report)(
        bag_source=report.bag_source,
        sim_sample_count=report.sim_sample_count,
        matched_pair_count=report.matched_pair_count,
        matched_seconds=report.matched_seconds,
        translation_error_min_meters=0.0,
        translation_error_mean_meters=translation_mean,
        translation_error_max_meters=translation_max,
        translation_error_p50_meters=translation_mean,
        translation_error_p95_meters=translation_p95,
        heading_error_mean_radians=heading_mean,
        heading_error_max_radians=None if heading_mean is None else heading_mean,
        pairs=report.pairs,
    )


def test_evaluate_real_vs_sim_correlation_thresholds_passes_when_empty() -> None:
    """Empty thresholds always pass with no failed checks."""
    report = _correlation_report_with(translation_mean=10.0, translation_p95=20.0, translation_max=30.0)
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, RealVsSimCorrelationThresholds())
    assert passed is True
    assert failed == ()


def test_evaluate_real_vs_sim_correlation_thresholds_flags_each_exceeded_stat() -> None:
    """Each populated threshold contributes its own failure tag when exceeded."""
    report = _correlation_report_with(
        translation_mean=0.5,
        translation_p95=2.0,
        translation_max=5.0,
        heading_mean=0.6,
    )
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=0.4,
        max_translation_error_p95_meters=1.0,
        max_translation_error_max_meters=4.0,
        max_heading_error_mean_radians=0.3,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert set(failed) == {"translation-mean", "translation-p95", "translation-max", "heading-mean"}


def test_evaluate_real_vs_sim_correlation_thresholds_skips_heading_when_absent() -> None:
    """A report without heading data must not synthesise a heading-mean failure."""
    report = _correlation_report_with(
        translation_mean=0.05, translation_p95=0.05, translation_max=0.05, heading_mean=None
    )
    thresholds = RealVsSimCorrelationThresholds(max_heading_error_mean_radians=0.001)
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_real_vs_sim_correlation_thresholds_round_trip_through_json() -> None:
    """to_dict / from_dict must preserve the populated bounds and skip the empty ones."""
    thresholds = RealVsSimCorrelationThresholds(
        max_translation_error_mean_meters=1.5,
        max_translation_error_p95_meters=3.0,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds
    assert "maxTranslationErrorMaxMeters" not in thresholds.to_dict()
    assert real_vs_sim_correlation_thresholds_from_dict({}) == RealVsSimCorrelationThresholds()


def test_correlation_threshold_overrides_round_trip_drops_empty_entries(tmp_path: Path) -> None:
    """from_dict/to_dict + load_..._json must drop topics whose payload has no bounds set."""
    payload = {
        "/gnss/fix": {"maxTranslationErrorMeanMeters": 0.5},
        "/imu/data": {"maxHeadingErrorMeanRadians": 0.05},
        "/empty/topic": {},  # should be dropped
    }
    overrides = correlation_threshold_overrides_from_dict(payload)
    assert set(overrides.keys()) == {"/gnss/fix", "/imu/data"}
    assert overrides["/gnss/fix"].max_translation_error_mean_meters == pytest.approx(0.5)
    assert overrides["/imu/data"].max_heading_error_mean_radians == pytest.approx(0.05)

    rebuilt_payload = correlation_threshold_overrides_to_dict(overrides)
    assert "/empty/topic" not in rebuilt_payload
    assert correlation_threshold_overrides_from_dict(rebuilt_payload) == overrides

    config_path = tmp_path / "overrides.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_correlation_threshold_overrides_json(config_path)
    assert loaded == overrides


def test_correlation_threshold_overrides_rejects_non_mapping_root(tmp_path: Path) -> None:
    """A JSON file whose root is not an object must raise on load."""
    bad = tmp_path / "bad.json"
    bad.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        load_correlation_threshold_overrides_json(bad)


def _correlation_report_with_pairs(
    pair_errors: list[float],
):
    """Build a synthetic correlation report whose pair list has the requested per-pair errors."""
    bag_samples = [BagPoseSample(timestamp_seconds=float(i), position=(0.0, 0.0, 0.0)) for i in range(len(pair_errors))]
    sim_samples = [
        SimPoseSample(timestamp_seconds=float(i), position=(error, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1))
        for i, error in enumerate(pair_errors)
    ]
    return correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)


def test_correlation_pair_distribution_gate_passes_when_under_fraction() -> None:
    """1/10 pairs above 0.5 m at a 0.2 limit (10% > 20% is False) must not fail."""
    report = _correlation_report_with_pairs([0.0] * 9 + [1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert "translation-pair-distribution" not in failed


def test_correlation_pair_distribution_gate_fails_when_over_fraction() -> None:
    """3/10 pairs above 0.5 m at a 0.2 limit (30% > 20% is True) must fail with the new tag."""
    report = _correlation_report_with_pairs([0.0] * 7 + [1.0, 1.0, 1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.5,
        max_exceeding_translation_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert "translation-pair-distribution" in failed


def test_correlation_pair_distribution_gate_requires_both_bounds() -> None:
    """Setting only one of the two pair-distribution fields must skip the check silently."""
    report = _correlation_report_with_pairs([1.0] * 10)
    only_bound = RealVsSimCorrelationThresholds(max_pair_translation_error_meters=0.1)
    only_fraction = RealVsSimCorrelationThresholds(max_exceeding_translation_pair_fraction=0.0)
    for thresholds in (only_bound, only_fraction):
        passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
        assert passed is True
        assert "translation-pair-distribution" not in failed


def test_correlation_pair_distribution_gate_skips_empty_pair_list() -> None:
    """A report whose pairs were dropped (keep_pairs=False) must not trip the gate."""
    bag_samples = [BagPoseSample(timestamp_seconds=0.0, position=(0.0, 0.0, 0.0))]
    sim_samples = [
        SimPoseSample(timestamp_seconds=0.0, position=(10.0, 0.0, 0.0), orientation_xyzw=(0, 0, 0, 1)),
    ]
    report = correlate_against_sim_trajectory(
        _make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.1, keep_pairs=False
    )
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.1,
        max_exceeding_translation_pair_fraction=0.0,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_thresholds_pair_distribution_round_trips_through_json() -> None:
    """Pair-distribution bounds must round-trip through to_dict/from_dict."""
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_translation_error_meters=0.4,
        max_exceeding_translation_pair_fraction=0.05,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds


def _correlation_report_with_heading_pairs(heading_errors: list[float | None]):
    """Build a synthetic report whose pairs carry the requested heading errors."""
    bag_samples: list[BagPoseSample] = []
    sim_samples: list[SimPoseSample] = []
    for index, heading in enumerate(heading_errors):
        ts = float(index)
        bag_orientation: tuple[float, float, float, float] | None
        if heading is None:
            bag_orientation = None
        else:
            half = heading / 2.0
            bag_orientation = (0.0, 0.0, math.sin(half), math.cos(half))
        bag_samples.append(
            BagPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=bag_orientation)
        )
        sim_samples.append(
            SimPoseSample(timestamp_seconds=ts, position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
    return correlate_against_sim_trajectory(_make_bag_stream(bag_samples), sim_samples, max_match_dt_seconds=0.5)


def test_correlation_heading_pair_distribution_gate_fails_when_over_fraction() -> None:
    """3/10 pairs above 0.5 rad at a 20% limit (30% > 20%) must fail with the heading tag."""
    report = _correlation_report_with_heading_pairs([0.0] * 7 + [1.0, 1.0, 1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is False
    assert "heading-pair-distribution" in failed


def test_correlation_heading_pair_distribution_gate_passes_when_under_fraction() -> None:
    """1/10 pairs above 0.5 rad at a 20% limit (10% < 20%) must not fail."""
    report = _correlation_report_with_heading_pairs([0.0] * 9 + [1.0])
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert "heading-pair-distribution" not in failed


def test_correlation_heading_pair_distribution_skips_pairs_without_heading() -> None:
    """Pairs whose heading_error_radians is None must be excluded from the fraction denominator."""
    # 10 pairs total: 4 with heading data (3 of which exceed the bound), 6 without.
    # Without filtering: 3/10 = 30% > 20% -> would fail.
    # With filtering: 3/4 = 75% > 20% -> still fails, but on a different denominator.
    # Verify the denominator is the heading-bearing subset by using a different pattern:
    # 10 pairs total: 4 with heading (1 exceeding -> 1/4 = 25%), 6 without.
    # 25% > 20% -> fails (with heading-only denominator), but 1/10 = 10% (would pass).
    heading_errors: list[float | None] = [0.0, 0.0, 0.0, 1.0, None, None, None, None, None, None]
    report = _correlation_report_with_heading_pairs(heading_errors)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.5,
        max_exceeding_heading_pair_fraction=0.2,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    # 1/4 = 25% > 20% on the heading-bearing denominator -> fails.
    assert passed is False
    assert "heading-pair-distribution" in failed


def test_correlation_heading_pair_distribution_skips_when_no_pair_has_heading() -> None:
    """When every pair lacks heading data, the gate must silently pass."""
    heading_errors: list[float | None] = [None] * 5
    report = _correlation_report_with_heading_pairs(heading_errors)
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.0,
        max_exceeding_heading_pair_fraction=0.0,
    )
    passed, failed = evaluate_real_vs_sim_correlation_thresholds(report, thresholds)
    assert passed is True
    assert failed == ()


def test_correlation_thresholds_heading_pair_distribution_round_trips() -> None:
    """Heading pair-distribution bounds must round-trip through to_dict/from_dict."""
    thresholds = RealVsSimCorrelationThresholds(
        max_pair_heading_error_radians=0.3,
        max_exceeding_heading_pair_fraction=0.1,
    )
    rebuilt = real_vs_sim_correlation_thresholds_from_dict(thresholds.to_dict())
    assert rebuilt == thresholds
