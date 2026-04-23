"""Tests for closed-loop route replanning helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RouteCandidate,
    build_simulation_catalog,
    last_applied_route_pose,
    reanchor_route_candidates,
    replan_after_blocked_rollout,
    rollout_route,
    rollout_route_with_replanning,
)


def test_last_applied_route_pose_uses_transition_state_before_blocked_step() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    waypoint = pose_like(start, position=(0.5, 0.0, 0.0))
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    rollout = rollout_route(
        env,
        RouteCandidate("blocked-after-progress", (start, waypoint, outside)),
        action_type="teleport",
    )

    assert rollout.blocked_step_index == 1
    assert last_applied_route_pose(rollout).position == waypoint.position


def test_reanchor_route_candidates_prepends_replan_start_pose_once() -> None:
    start = unit_pose(position=(0.5, 0.0, 0.0))
    goal = unit_pose(position=(0.75, 0.0, 0.0))
    already_anchored = RouteCandidate("anchored", (start, goal))
    continuation = RouteCandidate("continuation", (goal,))

    anchored = reanchor_route_candidates(start, (already_anchored, continuation))

    assert anchored[0].trajectory == (start, goal)
    assert anchored[1].trajectory == (start, goal)


def test_replan_after_blocked_rollout_reanchors_and_selects_clear_candidate() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("blocked", (start, outside)), action_type="teleport")
    bad = RouteCandidate("bad", (outside,))
    clear = RouteCandidate("clear", (pose_like(start, position=(0.5, 0.0, 0.0)),))

    replan = replan_after_blocked_rollout(
        env,
        scene_id="unit-scene",
        rollout=rollout,
        candidates=(bad, clear),
    )

    assert replan.start_pose.position == start.position
    assert replan.trigger.step.action_index == 0
    assert replan.selected_route.route_id == "clear"
    assert replan.selected_route.trajectory[0] == start
    assert replan.plan.selected.score.metrics["collision-rate"] == 0.0
    assert replan.to_dict()["plan"]["selectedRouteId"] == "clear"


def test_replan_after_blocked_rollout_can_execute_selected_route() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    goal = pose_like(start, position=(0.5, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("blocked", (start, outside)), action_type="teleport")

    replan = replan_after_blocked_rollout(
        env,
        scene_id="unit-scene",
        rollout=rollout,
        candidates=(RouteCandidate("clear", (goal,)),),
        execute=True,
        action_type="teleport",
    )

    assert replan.passed is True
    assert replan.rollout is not None
    assert replan.rollout.passed is True
    assert env.state.pose.position == goal.position


def test_replan_after_blocked_rollout_rejects_unblocked_rollout() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    goal = pose_like(start, position=(0.5, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("clear", (start, goal)), action_type="teleport")

    with pytest.raises(ValueError, match="blocked step"):
        replan_after_blocked_rollout(
            env,
            scene_id="unit-scene",
            rollout=rollout,
            candidates=(RouteCandidate("unused", (goal,)),),
        )


def test_rollout_route_with_replanning_recovers_from_blocked_initial_route() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    goal = pose_like(start, position=(0.5, 0.0, 0.0))

    closed_loop = rollout_route_with_replanning(
        env,
        scene_id="unit-scene",
        initial_route=RouteCandidate("initial-blocked", (start, outside)),
        replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
        action_type="teleport",
    )

    assert closed_loop.passed is True
    assert len(closed_loop.rollouts) == 2
    assert len(closed_loop.replans) == 1
    assert closed_loop.final_rollout.route_id == "recovery"
    assert closed_loop.metrics()["replan-count"] == 1.0
    assert closed_loop.metrics()["collision-count"] == 1.0
    assert closed_loop.to_dict()["passed"] is True
    assert env.state.pose.position == goal.position


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


def unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="generic_world")


def pose_like(template: Pose3D, *, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(position=position, orientation_xyzw=template.orientation_xyzw, frame_id=template.frame_id)
