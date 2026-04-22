"""Tests for cached Physical AI planning context helpers."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
import pytest

from gs_sim2real.sim import (
    Observation,
    OccupancyPlanningContext,
    PlanningViewpointKey,
    Pose3D,
    VoxelOccupancyGrid,
)


def test_occupancy_planning_context_caches_by_quantized_viewpoint() -> None:
    env = FakeLidarEnvironment(scene_id="unit-scene")
    context = OccupancyPlanningContext(voxel_size_meters=0.5, pose_resolution_meters=0.5)
    pose = unit_pose(position=(0.1, 0.0, 0.0))
    nearby_pose = unit_pose(position=(0.2, 0.0, 0.0))

    first = context.get_or_render(env, scene_id="unit-scene", pose=pose)
    second = context.get_or_render(env, scene_id="unit-scene", pose=nearby_pose)

    assert first is second
    assert env.render_count == 1
    assert len(context) == 1
    assert context.cache_info() == {"entryCount": 1, "hitCount": 1, "missCount": 1}
    assert first.source == "lidar-ray-proxy:unit-scene"


def test_occupancy_planning_context_separates_viewpoints() -> None:
    env = FakeLidarEnvironment(scene_id="unit-scene")
    context = OccupancyPlanningContext(voxel_size_meters=0.5, pose_resolution_meters=0.25)

    first = context.get_or_render(env, scene_id="unit-scene", pose=unit_pose(position=(0.0, 0.0, 0.0)))
    second = context.get_or_render(env, scene_id="unit-scene", pose=unit_pose(position=(1.0, 0.0, 0.0)))

    assert first is not second
    assert env.render_count == 2
    assert context.cache_info() == {"entryCount": 2, "hitCount": 0, "missCount": 2}


def test_occupancy_planning_context_injects_environment_occupancy() -> None:
    env = FakeLidarEnvironment(scene_id="unit-scene")
    context = OccupancyPlanningContext(voxel_size_meters=0.25)

    grid = context.set_environment_occupancy(env, scene_id="unit-scene", pose=unit_pose())

    assert isinstance(grid, VoxelOccupancyGrid)
    assert env.occupancy_grid is grid
    assert env.set_count == 1
    assert grid.cell_count == 1


def test_occupancy_planning_context_rejects_scene_mismatch() -> None:
    env = FakeLidarEnvironment(scene_id="other-scene")
    context = OccupancyPlanningContext(voxel_size_meters=0.25)

    with pytest.raises(ValueError, match="does not match"):
        context.get_or_render(env, scene_id="unit-scene", pose=unit_pose())


def test_planning_viewpoint_key_is_public_serializable_contract() -> None:
    context = OccupancyPlanningContext(voxel_size_meters=0.25)
    key = context.key_for("unit-scene", unit_pose())

    assert isinstance(key, PlanningViewpointKey)
    assert key.to_dict()["sceneId"] == "unit-scene"
    assert key.to_dict()["sensorId"] == "lidar-ray-proxy"
    assert key.to_dict()["outputs"] == ["ranges", "points"]


@dataclass
class FakeLidarEnvironment:
    scene_id: str
    render_count: int = 0
    set_count: int = 0
    occupancy_grid: VoxelOccupancyGrid | None = None

    def render_observation(self, request) -> Observation:
        self.render_count += 1
        assert request.sensor_id == "lidar-ray-proxy"
        assert request.outputs == ("ranges", "points")
        points = np.array([[0.25, 0.0, 0.0]], dtype="<f4")
        return Observation(
            sensor_id=request.sensor_id,
            pose=request.pose,
            outputs={
                "sceneId": self.scene_id,
                "points": {
                    "encoding": "float32-le-xyz",
                    "pointsBase64": base64.b64encode(points.tobytes()).decode("ascii"),
                },
            },
        )

    def set_occupancy_grid(self, grid: VoxelOccupancyGrid) -> None:
        self.set_count += 1
        self.occupancy_grid = grid


def unit_pose(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
