"""Unit tests for IMU quaternion fusion in extract_navsat_trajectory."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.datasets.mcd import (
    _interp_imu_quaternion,
    _load_imu_orientation_csv,
    _quat_to_rotmat,
)


def _write_imu_csv(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_ns",
                "timestamp_sec",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ]
        )
        for ts, qx, qy, qz, qw in rows:
            writer.writerow([int(ts * 1e9), ts, qx, qy, qz, qw, 0, 0, 0, 0, 0, 0])


def test_load_imu_orientation_csv_returns_sorted_samples(tmp_path: Path) -> None:
    p = tmp_path / "imu.csv"
    _write_imu_csv(p, [(2.0, 0.1, 0.2, 0.3, 0.9), (1.0, 0.0, 0.0, 0.0, 1.0)])
    arr = _load_imu_orientation_csv(p)
    assert arr is not None and arr.shape == (2, 5)
    assert arr[0, 0] == 1.0
    assert arr[1, 0] == 2.0


def test_load_imu_orientation_csv_rejects_all_identity(tmp_path: Path) -> None:
    p = tmp_path / "imu.csv"
    _write_imu_csv(p, [(1.0, 0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0, 1.0)])
    assert _load_imu_orientation_csv(p) is None


def test_load_imu_orientation_csv_returns_none_when_missing(tmp_path: Path) -> None:
    assert _load_imu_orientation_csv(tmp_path / "does_not_exist.csv") is None


def test_interp_imu_quaternion_clamps_outside_range() -> None:
    samples = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 1.0],
            [20.0, 0.0, 0.0, 0.7071068, 0.7071068],
        ]
    )
    q_low = _interp_imu_quaternion(samples, 5.0)
    q_high = _interp_imu_quaternion(samples, 30.0)
    assert q_low == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-6)
    assert q_high == pytest.approx((0.0, 0.0, 0.7071068, 0.7071068), abs=1e-6)


def test_interp_imu_quaternion_blends_and_normalises() -> None:
    samples = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    quat = _interp_imu_quaternion(samples, 1.0)
    assert quat is not None
    norm = sum(q * q for q in quat) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)
    # Midway blend should lie between the two poles on the great circle.
    qx, qy, qz, qw = quat
    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    assert qz > 0.5 and qw > 0.5


def test_quat_to_rotmat_identity() -> None:
    R = _quat_to_rotmat((0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_quat_to_rotmat_90deg_z() -> None:
    # 90 degree rotation around z axis: quat = (0, 0, sin(45), cos(45))
    s = 0.7071067811865475
    R = _quat_to_rotmat((0.0, 0.0, s, s))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R, expected, atol=1e-6)
