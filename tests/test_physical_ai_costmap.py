"""Tests for Physical AI costmap-style trajectory summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.sim import (
    CollisionQuery,
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RobotFootprint,
    VoxelOccupancyGrid,
    build_simulation_catalog,
    summarize_collision_queries,
)


def test_collision_summary_reports_rates_clearance_and_reasons() -> None:
    pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    summary = summarize_collision_queries(
        (
            CollisionQuery(pose=pose, collides=False, reason="free-footprint:unit-grid", clearance_meters=0.75),
            CollisionQuery(pose=pose, collides=True, reason="occupied-footprint-voxel:unit-grid", clearance_meters=0.0),
            CollisionQuery(pose=pose, collides=False, reason="free-footprint:unit-grid", clearance_meters=1.25),
        )
    )

    assert summary.pose_count == 3
    assert summary.collision_count == 1
    assert summary.collision_rate == pytest.approx(1 / 3)
    assert summary.metric_payload()["collision-rate"] == pytest.approx(1 / 3)
    assert summary.metric_payload()["collision-count"] == 1.0
    assert summary.metric_payload()["minimum-clearance-meters"] == 0.0
    assert summary.metric_payload()["mean-clearance-meters"] == pytest.approx(2.0 / 3.0)
    assert summary.notes() == (
        "collision-reason:free-footprint:unit-grid=2",
        "collision-reason:occupied-footprint-voxel:unit-grid=1",
    )


def test_headless_score_trajectory_uses_occupancy_footprint_collision_summary() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog(), robot_footprint=RobotFootprint(radius_meters=0.5))
    env.set_occupancy_grid(
        VoxelOccupancyGrid.from_points(
            np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            voxel_size_meters=0.25,
            source="unit-grid",
        )
    )

    colliding = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    clear = Pose3D(position=(-0.75, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
    score = env.score_trajectory("unit-scene", [colliding, clear])

    assert score.passed is False
    assert score.metrics["inside-bounds-rate"] == 1.0
    assert score.metrics["collision-rate"] == 0.5
    assert score.metrics["collision-count"] == 1.0
    assert score.metrics["minimum-clearance-meters"] == 0.0
    assert "collision-reason:occupied-footprint-voxel:unit-grid=1" in score.notes
    assert "collision-reason:free-footprint:unit-grid=1" in score.notes


def build_unit_catalog():
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )
