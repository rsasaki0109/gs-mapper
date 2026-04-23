"""Tests for route execution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RouteCandidate,
    RouteEvaluation,
    RoutePlan,
    TrajectoryScore,
    build_route_actions,
    build_simulation_catalog,
    rollout_route,
)


def test_build_route_actions_accepts_selected_plan_and_serializes_teleports() -> None:
    start = unit_pose(position=(0.0, 0.0, 0.0))
    waypoint = unit_pose(position=(0.25, 0.0, 0.0))
    goal = unit_pose(position=(0.25, 0.0, 0.5), orientation_xyzw=(0.0, 0.0, 0.5, 0.5))
    candidate = RouteCandidate("selected-route", (start, waypoint, goal))
    evaluation = RouteEvaluation(
        candidate=candidate,
        score=TrajectoryScore(metrics={"collision-rate": 0.0}, passed=True),
        rank=1,
        original_index=0,
    )
    plan = RoutePlan(scene_id="unit-scene", evaluations=(evaluation,), selected=evaluation)

    steps = build_route_actions(plan, action_type="teleport", segment_duration_seconds=0.25)

    assert len(steps) == 2
    assert steps[0].route_id == "selected-route"
    assert steps[0].action.action_type == "teleport"
    assert steps[0].action.values["x"] == 0.25
    assert steps[0].action.duration_seconds == 0.25
    assert steps[1].action.values["z"] == 0.5
    assert steps[1].action.values["qz"] == 0.5
    assert steps[1].to_dict()["targetPose"]["position"] == [0.25, 0.0, 0.5]


def test_build_route_actions_can_emit_twist_segments() -> None:
    route = RouteCandidate(
        "twist-route",
        (
            unit_pose(position=(0.0, 0.0, 0.0)),
            unit_pose(position=(2.0, 0.0, -1.0)),
        ),
    )

    (step,) = build_route_actions(route, action_type="twist", segment_duration_seconds=2.0)

    assert step.action.action_type == "twist"
    assert step.action.values["linearX"] == pytest.approx(1.0)
    assert step.action.values["linearY"] == pytest.approx(0.0)
    assert step.action.values["linearZ"] == pytest.approx(-0.5)
    assert step.distance_meters == pytest.approx(5**0.5)


def test_build_route_actions_rejects_empty_route() -> None:
    with pytest.raises(ValueError, match="at least one pose"):
        build_route_actions(RouteCandidate("empty", ()))


def test_rollout_route_executes_clear_route_and_reports_metrics() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    waypoint = pose_like(start, position=(0.25, 0.0, 0.0))
    goal = pose_like(start, position=(0.25, 0.0, 0.25))
    route = RouteCandidate("clear", (start, waypoint, goal))

    rollout = rollout_route(env, route, action_type="teleport")

    assert rollout.passed is True
    assert rollout.blocked_step_index is None
    assert len(rollout.outcomes) == 2
    assert all(outcome.applied for outcome in rollout.outcomes)
    assert env.state.pose.position == goal.position
    assert rollout.metrics()["completion-rate"] == 1.0
    assert rollout.metrics()["applied-step-count"] == 2.0
    assert rollout.to_dict()["passed"] is True


def test_rollout_route_records_blocked_step_and_stops() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    fallback = pose_like(start, position=(0.5, 0.0, 0.0))
    route = RouteCandidate("blocked", (start, outside, fallback))

    rollout = rollout_route(env, route, action_type="teleport", stop_on_collision=True)

    assert rollout.passed is False
    assert rollout.blocked_step_index == 0
    assert len(rollout.outcomes) == 1
    assert rollout.outcomes[0].applied is False
    assert rollout.outcomes[0].collides is True
    assert rollout.outcomes[0].transition["collision"]["reason"] == "outside-bounds"
    assert env.state.pose.position == start.position
    assert rollout.metrics()["completion-rate"] == 0.5
    assert rollout.metrics()["collision-count"] == 1.0


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


def unit_pose(
    *,
    position: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=orientation_xyzw, frame_id="generic_world")


def pose_like(template: Pose3D, *, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=template.orientation_xyzw, frame_id=template.frame_id)
