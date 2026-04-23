"""Tests for route policy rollout quality gates and baseline reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyQualityThresholds,
    RouteRewardWeights,
    build_simulation_catalog,
    collect_route_policy_dataset,
    evaluate_route_policy_baselines,
    evaluate_route_policy_dataset_quality,
    render_route_policy_quality_markdown,
)


def test_route_policy_dataset_quality_passes_direct_rollouts() -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="direct-rollouts",
        goals=(unit_pose((0.25, 0.0, 0.0)),),
    )

    report = evaluate_route_policy_dataset_quality(
        dataset,
        thresholds=RoutePolicyQualityThresholds(
            min_success_rate=1.0,
            max_collision_rate=0.0,
            max_truncation_rate=0.0,
            min_episode_count=2,
            min_transition_count=2,
        ),
        metadata={"split": "qa"},
    )

    assert report.passed is True
    assert report.failed_checks == ()
    assert report.metrics["success-rate"] == pytest.approx(1.0)
    assert report.metrics["collision-rate"] == pytest.approx(0.0)
    assert report.metrics["mean-reward"] > 0.0
    assert report.scene_coverage == {"unit-scene": 2}
    assert report.termination_counts == {"goal-reached": 2}

    payload = report.to_dict()
    assert payload["recordType"] == "route-policy-quality-report"
    assert payload["passed"] is True
    assert payload["metadata"]["split"] == "qa"
    assert payload["rewardDistribution"]["count"] == 2.0


def test_route_policy_dataset_quality_flags_blocked_rollouts() -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        blocked_policy,
        episode_count=1,
        dataset_id="blocked-rollouts",
        goals=(unit_pose((0.5, 0.0, 0.0)),),
    )

    report = evaluate_route_policy_dataset_quality(
        dataset,
        thresholds=RoutePolicyQualityThresholds(
            min_success_rate=1.0,
            max_collision_rate=0.0,
            max_truncation_rate=0.0,
        ),
    )

    assert report.passed is False
    assert "min-success-rate" in report.failed_checks
    assert "max-collision-rate" in report.failed_checks
    assert report.metrics["blocked-rate"] == pytest.approx(1.0)
    assert report.metrics["collision-rate"] == pytest.approx(1.0)
    assert report.termination_counts == {"blocked-route": 1}
    assert "FAIL" in render_route_policy_quality_markdown(report)


def test_route_policy_dataset_quality_flags_collector_truncation() -> None:
    dataset = collect_route_policy_dataset(
        (build_adapter(max_steps=8),),
        partial_goal_policy,
        episode_count=1,
        dataset_id="partial-rollouts",
        goals=(unit_pose((0.75, 0.0, 0.0)),),
        max_steps=1,
    )

    report = evaluate_route_policy_dataset_quality(
        dataset,
        thresholds=RoutePolicyQualityThresholds(
            min_success_rate=0.0,
            max_collision_rate=0.0,
            max_truncation_rate=0.0,
        ),
    )

    assert report.passed is False
    assert report.failed_checks == ("max-truncation-rate",)
    assert report.metrics["truncation-rate"] == pytest.approx(1.0)
    assert report.termination_counts == {"collector-max-steps": 1}


def test_route_policy_baseline_evaluation_ranks_direct_policy() -> None:
    evaluation = evaluate_route_policy_baselines(
        (build_adapter(),),
        {
            "blocked": blocked_policy,
            "direct": direct_goal_policy,
        },
        episode_count=1,
        evaluation_id="unit-baseline",
        goals=(unit_pose((0.25, 0.0, 0.0)),),
        thresholds=RoutePolicyQualityThresholds(
            min_success_rate=1.0,
            max_collision_rate=0.0,
            max_truncation_rate=0.0,
        ),
        metadata={"sceneSet": "unit"},
    )

    assert evaluation.best_policy_name == "direct"
    assert [result.policy_name for result in evaluation.results] == ["blocked", "direct"]
    assert [result.dataset.episodes[0].episode_index for result in evaluation.results] == [0, 0]
    assert evaluation.results[0].passed is False
    assert evaluation.results[1].passed is True

    payload = evaluation.to_dict()
    assert payload["recordType"] == "route-policy-baseline-evaluation"
    assert payload["bestPolicyName"] == "direct"
    assert payload["metadata"]["sceneSet"] == "unit"
    assert payload["results"][1]["quality"]["passed"] is True


def build_adapter(*, max_steps: int = 4) -> RoutePolicyGymAdapter:
    env = HeadlessPhysicalAIEnvironment(build_unit_catalog())
    return RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(
            scene_id="unit-scene",
            max_steps=max_steps,
            goal_reward=2.0,
            reward_weights=RouteRewardWeights(distance_penalty_per_meter=0.0, step_penalty=0.0),
        ),
    )


def direct_goal_policy(observation, info):
    del observation
    return {
        "routeId": f"direct-{info['stepIndex']}",
        "target": info["goal"],
    }


def partial_goal_policy(observation, info):
    del observation
    pose = info["pose"]["position"]
    goal = info["goal"]["position"]
    return {
        "routeId": f"partial-{info['stepIndex']}",
        "target": {
            "x": pose[0] + (goal[0] - pose[0]) * 0.25,
            "y": pose[1] + (goal[1] - pose[1]) * 0.25,
            "z": pose[2] + (goal[2] - pose[2]) * 0.25,
        },
    }


def blocked_policy(observation, info):
    del observation
    return {
        "routeId": f"blocked-{info['stepIndex']}",
        "target": {"x": 2.0, "y": 0.0, "z": 0.0},
    }


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
