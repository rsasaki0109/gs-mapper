"""Tests for dependency-free route policy imitation baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    Pose3D,
    RoutePolicyActionDecoderConfig,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyImitationFitConfig,
    RoutePolicyQualityThresholds,
    RoutePolicyReplayBatch,
    RoutePolicyReplayFeatureSchema,
    RoutePolicyReplaySample,
    RouteRewardWeights,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    build_simulation_catalog,
    collect_route_policy_dataset,
    decode_route_policy_action_vector,
    evaluate_route_policy_imitation_model,
    fit_route_policy_imitation_model,
)


def test_fit_route_policy_imitation_model_predicts_nearest_replay_action() -> None:
    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-imitation",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    batch = build_route_policy_replay_batch(dataset, schema=schema)

    model = fit_route_policy_imitation_model(batch, metadata={"split": "train"})
    action = model.predict_action(
        dataset.episodes[0].transitions[0].observation,
        {"episodeIndex": 2, "stepIndex": 3, "goal": goals[0].to_dict()},
    )

    assert model.sample_count == 2
    assert model.to_dict()["metadata"]["split"] == "train"
    assert model.predict_action_vector(dataset.episodes[0].transitions[0].observation) == pytest.approx(
        (0.25, 0.0, 0.0)
    )
    assert action["routeId"] == "imitation-route-2-3"
    assert action["target"]["position"] == pytest.approx([0.25, 0.0, 0.0])
    assert action["target"]["orientationXyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert action["target"]["frameId"] == "generic_world"


def test_route_policy_imitation_model_runs_through_baseline_evaluator() -> None:
    goals = (unit_pose((0.25, 0.0, 0.0)), unit_pose((0.5, 0.0, 0.0)))
    dataset = collect_route_policy_dataset(
        (build_adapter(),),
        direct_goal_policy,
        episode_count=2,
        dataset_id="unit-imitation-eval-train",
        goals=goals,
    )
    schema = build_route_policy_replay_schema(dataset, action_keys=target_position_keys())
    model = fit_route_policy_imitation_model(build_route_policy_replay_batch(dataset, schema=schema))

    evaluation = evaluate_route_policy_imitation_model(
        (build_adapter(),),
        model,
        episode_count=2,
        goals=goals,
        thresholds=RoutePolicyQualityThresholds(
            min_success_rate=1.0,
            max_collision_rate=0.0,
            max_truncation_rate=0.0,
            min_episode_count=2,
            min_transition_count=2,
        ),
    )

    assert evaluation.best_policy_name == "imitation"
    assert evaluation.results[0].passed
    assert evaluation.results[0].quality.metrics["success-rate"] == pytest.approx(1.0)
    assert evaluation.results[0].quality.metrics["truncation-rate"] == pytest.approx(0.0)


def test_weighted_knn_imitation_averages_neighbor_actions_and_decodes_xyz_keys() -> None:
    schema = RoutePolicyReplayFeatureSchema(
        observation_keys=("goal-distance-meters",),
        action_keys=("target.x", "target.y", "target.z"),
        next_observation_keys=(),
    )
    batch = RoutePolicyReplayBatch(
        schema=schema,
        samples=(
            replay_sample(schema, observation=(0.0,), action=(0.0, 0.0, 0.0), step_index=0),
            replay_sample(schema, observation=(1.0,), action=(2.0, 0.0, 0.0), step_index=1),
        ),
    )
    model = fit_route_policy_imitation_model(batch, config=RoutePolicyImitationFitConfig(neighbor_count=2))

    vector = model.predict_action_vector({"goal-distance-meters": 0.5})
    action = model.predict_action({"goal-distance-meters": 0.5}, {"stepIndex": 4})

    assert vector == pytest.approx((1.0, 0.0, 0.0))
    assert action["routeId"] == "imitation-route-4"
    assert action["target"]["position"] == pytest.approx([1.0, 0.0, 0.0])


def test_decode_route_policy_action_vector_accepts_explicit_target_keys() -> None:
    schema = RoutePolicyReplayFeatureSchema(
        observation_keys=("feature",),
        action_keys=("agent.target_x", "agent.target_y", "agent.target_z"),
        next_observation_keys=(),
    )

    action = decode_route_policy_action_vector(
        (1.0, 2.0, 3.0),
        schema,
        {"goal": {"frameId": "map", "orientationXyzw": [0.0, 0.0, 0.0, 1.0]}},
        config=RoutePolicyActionDecoderConfig(
            target_keys=("agent.target_x", "agent.target_y", "agent.target_z"),
            route_id_prefix="custom-imitation",
        ),
    )

    assert action == {
        "routeId": "custom-imitation",
        "target": {
            "position": [1.0, 2.0, 3.0],
            "orientationXyzw": [0.0, 0.0, 0.0, 1.0],
            "frameId": "map",
        },
    }


def test_fit_route_policy_imitation_model_rejects_empty_batches() -> None:
    schema = RoutePolicyReplayFeatureSchema(
        observation_keys=("feature",),
        action_keys=("target.x", "target.y", "target.z"),
        next_observation_keys=(),
    )

    with pytest.raises(ValueError, match="at least one replay sample"):
        fit_route_policy_imitation_model(RoutePolicyReplayBatch(schema=schema, samples=()))


def replay_sample(
    schema: RoutePolicyReplayFeatureSchema,
    *,
    observation: tuple[float, ...],
    action: tuple[float, ...],
    step_index: int,
) -> RoutePolicyReplaySample:
    assert len(observation) == schema.observation_feature_count
    assert len(action) == schema.action_feature_count
    return RoutePolicyReplaySample(
        dataset_id="manual-imitation",
        episode_id=f"manual-episode-{step_index}",
        scene_id="unit-scene",
        episode_index=0,
        step_index=step_index,
        observation_vector=observation,
        action_vector=action,
        reward=0.0,
        next_observation_vector=(),
        terminated=False,
        truncated=False,
    )


def target_position_keys() -> tuple[str, str, str]:
    return ("payload.target.position.0", "payload.target.position.1", "payload.target.position.2")


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
