"""Tests for route policy feedback records."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RouteCandidate,
    RouteEvaluation,
    RoutePlan,
    RouteRewardWeights,
    TrajectoryScore,
    build_route_policy_batch,
    build_route_policy_observation,
    build_route_policy_sample,
    build_simulation_catalog,
    rollout_route,
    rollout_route_with_replanning,
    score_route_policy_reward,
)


def test_route_plan_policy_observation_is_compact_and_ranked() -> None:
    clear = RouteCandidate("clear", (unit_pose((0.0, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0))))
    risky = RouteCandidate("risky", (unit_pose((0.0, 0.0, 0.0)), unit_pose((1.5, 0.0, 0.0))))
    evaluations = (
        RouteEvaluation(
            candidate=clear,
            score=TrajectoryScore(
                metrics={
                    "collision-rate": 0.0,
                    "collision-count": 0.0,
                    "minimum-clearance-meters": 0.5,
                    "path-length": 0.5,
                },
                passed=True,
                notes=("free",),
            ),
            rank=1,
            original_index=0,
        ),
        RouteEvaluation(
            candidate=risky,
            score=TrajectoryScore(
                metrics={
                    "collision-rate": 1.0,
                    "collision-count": 1.0,
                    "minimum-clearance-meters": 0.0,
                    "path-length": 1.5,
                },
                passed=False,
            ),
            rank=2,
            original_index=1,
        ),
    )
    plan = RoutePlan(
        scene_id="unit-scene",
        evaluations=evaluations,
        selected=evaluations[0],
    )

    observation = build_route_policy_observation(plan)
    payload = observation.to_dict()

    assert observation.source_type == "route-plan"
    assert observation.source_id == "unit-scene:clear"
    assert observation.features["candidate-count"] == 2.0
    assert observation.features["passed-candidate-count"] == 1.0
    assert observation.features["selected-collision-rate"] == 0.0
    assert observation.features["selection-collision-rate-margin"] == 1.0
    assert "selected-route:clear" in observation.events
    assert "trajectory" not in str(payload)


def test_blocked_rollout_policy_reward_penalizes_collision_without_completion_reward() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = pose_like(start, position=(2.0, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("blocked", (start, outside)), action_type="teleport")

    sample = build_route_policy_sample(
        rollout,
        weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
    )

    assert sample.observation.source_type == "route-rollout"
    assert sample.observation.events == ("blocked",)
    assert sample.observation.features["blocked-step-index"] == 0.0
    assert sample.reward.terminal is True
    assert sample.reward.passed is False
    assert sample.reward.components["completion"] == 0.0
    assert sample.reward.components["collision"] == -1.0
    assert sample.reward.reward == pytest.approx(-1.0)


def test_closed_loop_policy_sample_reports_replan_recovery_reward() -> None:
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

    sample = build_route_policy_sample(
        closed_loop,
        weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
    )

    assert sample.observation.source_type == "closed-loop-route-rollout"
    assert sample.observation.features["replan-count"] == 1.0
    assert sample.observation.features["collision-count"] == 1.0
    assert sample.observation.features["final-passed"] == 1.0
    assert sample.reward.terminal is True
    assert sample.reward.passed is True
    assert sample.reward.components == {
        "success": 1.0,
        "completion": 0.5,
        "collision": -1.0,
        "distance": 0.0,
        "step": 0.0,
        "replan": -0.1,
    }
    assert sample.reward.reward == pytest.approx(0.4)


def test_route_policy_batch_preserves_sample_order() -> None:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    goal = pose_like(start, position=(0.5, 0.0, 0.0))
    rollout = rollout_route(env, RouteCandidate("clear", (start, goal)), action_type="teleport")

    batch = build_route_policy_batch((rollout, rollout))
    reward = score_route_policy_reward(rollout)

    assert len(batch) == 2
    assert [sample.observation.source_id for sample in batch] == ["clear", "clear"]
    assert batch[0].reward.reward == reward.reward


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
