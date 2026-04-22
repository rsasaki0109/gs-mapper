"""Tests for route-level Physical AI planning helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Observation,
    OccupancyPlanningContext,
    Pose3D,
    RobotFootprint,
    RouteCandidate,
    TrajectoryScore,
    build_simulation_catalog,
    select_best_route,
)


def test_select_best_route_ranks_passed_clear_short_route() -> None:
    env = StaticRouteScoreEnvironment()
    risky = RouteCandidate("risky", (unit_pose(position=(0.0, 0.0, 0.0)),))
    clear_long = RouteCandidate("clear-long", (unit_pose(position=(1.0, 0.0, 0.0)),))
    clear_short = RouteCandidate("clear-short", (unit_pose(position=(2.0, 0.0, 0.0)),))

    plan = select_best_route(
        env,
        scene_id="unit-scene",
        candidates=(risky, clear_long, clear_short),
    )

    assert plan.selected.candidate.route_id == "clear-short"
    assert [evaluation.candidate.route_id for evaluation in plan.evaluations] == [
        "clear-short",
        "clear-long",
        "risky",
    ]
    assert [evaluation.rank for evaluation in plan.evaluations] == [1, 2, 3]
    assert plan.to_dict()["selectedRouteId"] == "clear-short"


def test_select_best_route_rejects_empty_candidate_set() -> None:
    with pytest.raises(ValueError, match="at least one route"):
        select_best_route(StaticRouteScoreEnvironment(), scene_id="unit-scene", candidates=())


def test_select_best_route_uses_cached_occupancy_context_with_headless_environment() -> None:
    renderer = UnitLidarRenderer()
    env = HeadlessPhysicalAIEnvironment(
        build_unit_catalog(),
        observation_renderer=renderer,
        robot_footprint=RobotFootprint(radius_meters=0.5),
    )
    env.reset("unit-scene")
    context = OccupancyPlanningContext(voxel_size_meters=0.25, pose_resolution_meters=2.0)
    colliding = RouteCandidate(
        "blocked",
        (unit_pose(position=(0.0, 0.0, 0.0)),),
    )
    clear = RouteCandidate(
        "clear",
        (unit_pose(position=(-0.75, 0.0, 0.0)),),
    )

    plan = select_best_route(
        env,
        scene_id="unit-scene",
        candidates=(colliding, clear),
        planning_context=context,
    )

    assert plan.selected.candidate.route_id == "clear"
    assert plan.evaluations[0].score.metrics["collision-rate"] == 0.0
    assert plan.evaluations[1].score.metrics["collision-rate"] == 1.0
    assert renderer.render_count == 1
    assert context.cache_info() == {"entryCount": 1, "hitCount": 1, "missCount": 1}


class StaticRouteScoreEnvironment:
    def score_trajectory(self, scene_id: str, trajectory) -> TrajectoryScore:
        x = trajectory[0].position[0]
        if x == 0.0:
            return TrajectoryScore(
                metrics={
                    "collision-rate": 1.0,
                    "collision-count": 1.0,
                    "minimum-clearance-meters": 0.0,
                    "path-length": 1.0,
                },
                passed=False,
            )
        if x == 1.0:
            return TrajectoryScore(
                metrics={
                    "collision-rate": 0.0,
                    "collision-count": 0.0,
                    "minimum-clearance-meters": 0.5,
                    "path-length": 5.0,
                },
                passed=True,
            )
        return TrajectoryScore(
            metrics={
                "collision-rate": 0.0,
                "collision-count": 0.0,
                "minimum-clearance-meters": 0.5,
                "path-length": 2.0,
            },
            passed=True,
        )


class UnitLidarRenderer:
    def __init__(self) -> None:
        self.render_count = 0

    def can_render(self, scene, request) -> bool:
        return request.sensor_id == "lidar-ray-proxy"

    def render_observation(self, scene, request) -> Observation:
        self.render_count += 1
        points = np.array([[0.5, 0.0, 0.0]], dtype="<f4")
        return Observation(
            sensor_id=request.sensor_id,
            pose=request.pose,
            outputs={
                "sceneId": scene.scene_id,
                "points": {
                    "encoding": "float32-le-xyz",
                    "pointsBase64": base64.b64encode(points.tobytes()).decode("ascii"),
                },
            },
        )


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


def unit_pose(position: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")
