"""Sanity checks for the bundled sensor-noise profile fixtures.

Both JSONs under ``docs/fixtures/sensor-noise/`` and
``docs/fixtures/raw-noise/`` back the partial-information benchmark recipe
in ``docs/physical-ai-sim.md``. These tests pin the field values so the
recipe's "copy this JSON" promise stays honest when the profile dataclasses
or serialization shift.
"""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.sim import (
    load_raw_sensor_noise_profile_json,
    load_route_policy_sensor_noise_profile_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POSE_FIXTURE = REPO_ROOT / "docs" / "fixtures" / "sensor-noise" / "outdoor-gnss.json"
RAW_FIXTURE = REPO_ROOT / "docs" / "fixtures" / "raw-noise" / "outdoor-sensor.json"


def test_bundled_pose_sensor_noise_profile_loads_with_recipe_values() -> None:
    assert POSE_FIXTURE.is_file(), f"missing pose-facing fixture: {POSE_FIXTURE}"
    profile = load_route_policy_sensor_noise_profile_json(POSE_FIXTURE)

    assert profile.profile_id == "outdoor-gnss"
    assert profile.pose_position_std_meters == 0.25
    assert profile.pose_heading_std_radians == 0.02
    assert profile.goal_position_std_meters == 0.15
    assert profile.is_noise_free is False
    assert profile.metadata.get("source", "").startswith("docs/physical-ai-sim.md")


def test_bundled_raw_sensor_noise_profile_loads_with_recipe_values() -> None:
    assert RAW_FIXTURE.is_file(), f"missing raw-sensor fixture: {RAW_FIXTURE}"
    profile = load_raw_sensor_noise_profile_json(RAW_FIXTURE)

    assert profile.profile_id == "outdoor-sensor"
    assert profile.depth_range_std_meters == 0.10
    assert profile.lidar_range_std_meters == 0.05
    # RGB / IMU σ stay at 0 in the recipe example so the profile only perturbs depth + LiDAR.
    assert profile.rgb_intensity_std == 0.0
    assert profile.imu_angular_velocity_std_rad_per_sec == 0.0
    assert profile.imu_linear_acceleration_std_m_per_sec_sq == 0.0
    assert profile.is_noise_free is False
    assert profile.metadata.get("source", "").startswith("docs/physical-ai-sim.md")
