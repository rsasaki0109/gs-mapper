"""Tests for route policy dynamic obstacles."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gs_sim2real.sim import (
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
    HeadlessPhysicalAIEnvironment,
    RoutePolicyEnvConfig,
    RoutePolicyEnvState,
    RoutePolicyGymAdapter,
    build_simulation_catalog,
    load_route_policy_dynamic_obstacle_timeline_json,
    render_route_policy_dynamic_obstacle_timeline_markdown,
    route_policy_dynamic_obstacle_timeline_from_dict,
    write_route_policy_dynamic_obstacle_timeline_json,
)


def _unit_catalog():
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


def _linear_obstacle(
    obstacle_id: str,
    *,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_step: int = 0,
    end_step: int = 10,
    radius_meters: float = 0.1,
) -> DynamicObstacle:
    return DynamicObstacle(
        obstacle_id=obstacle_id,
        waypoints=(
            DynamicObstacleWaypoint(step_index=start_step, position=start),
            DynamicObstacleWaypoint(step_index=end_step, position=end),
        ),
        radius_meters=radius_meters,
    )


def test_dynamic_obstacle_position_clamps_and_interpolates() -> None:
    obstacle = _linear_obstacle("a", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0))

    assert obstacle.position_at_step(-5) == (0.0, 0.0, 0.0)
    assert obstacle.position_at_step(0) == (0.0, 0.0, 0.0)
    assert obstacle.position_at_step(5) == (5.0, 0.0, 0.0)
    assert obstacle.position_at_step(10) == (10.0, 0.0, 0.0)
    assert obstacle.position_at_step(42) == (10.0, 0.0, 0.0)


def test_dynamic_obstacle_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        DynamicObstacle(obstacle_id="no-waypoints", waypoints=(), radius_meters=0.1)
    with pytest.raises(ValueError):
        DynamicObstacle(
            obstacle_id="zero-radius",
            waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),),
            radius_meters=0.0,
        )
    with pytest.raises(ValueError):
        DynamicObstacle(
            obstacle_id="duplicate-step",
            waypoints=(
                DynamicObstacleWaypoint(step_index=0, position=(0.0, 0.0, 0.0)),
                DynamicObstacleWaypoint(step_index=0, position=(1.0, 0.0, 0.0)),
            ),
            radius_meters=0.1,
        )
    with pytest.raises(ValueError):
        DynamicObstacleWaypoint(step_index=-1, position=(0.0, 0.0, 0.0))


def test_dynamic_obstacle_timeline_round_trips(tmp_path: Path) -> None:
    timeline = DynamicObstacleTimeline(
        timeline_id="unit-timeline",
        obstacles=(
            _linear_obstacle("a", start=(0.0, 0.0, 0.0), end=(2.0, 0.0, 0.0)),
            _linear_obstacle(
                "b",
                start=(-1.0, 0.0, 1.0),
                end=(-1.0, 0.0, -1.0),
                radius_meters=0.25,
            ),
        ),
        metadata={"source": "unit-test"},
    )
    path = write_route_policy_dynamic_obstacle_timeline_json(tmp_path / "timeline.json", timeline)
    loaded = load_route_policy_dynamic_obstacle_timeline_json(path)

    assert loaded == timeline
    rebuilt = route_policy_dynamic_obstacle_timeline_from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert rebuilt == timeline
    markdown = render_route_policy_dynamic_obstacle_timeline_markdown(loaded)
    assert "Route Policy Dynamic Obstacle Timeline: unit-timeline" in markdown
    assert "| a |" in markdown and "| b |" in markdown


def test_dynamic_obstacle_timeline_reports_blocking_obstacle() -> None:
    timeline = DynamicObstacleTimeline(
        timeline_id="blocking",
        obstacles=(_linear_obstacle("a", start=(0.0, 0.0, 0.0), end=(10.0, 0.0, 0.0), radius_meters=0.5),),
    )

    # At step 0 the obstacle is at (0,0,0); a probe near (0.3,0,0) is inside.
    assert timeline.blocking_obstacle((0.3, 0.0, 0.0), step_index=0) is not None
    # The same probe at step 10 (obstacle has moved to (10,0,0)) is far away.
    assert timeline.blocking_obstacle((0.3, 0.0, 0.0), step_index=10) is None


def test_headless_environment_flags_dynamic_obstacle_collision() -> None:
    env = HeadlessPhysicalAIEnvironment(
        _unit_catalog(),
        dynamic_obstacles=DynamicObstacleTimeline(
            timeline_id="unit",
            obstacles=(
                _linear_obstacle(
                    "runner",
                    start=(0.25, 0.0, 0.0),
                    end=(0.25, 0.0, 0.0),
                    radius_meters=0.1,
                ),
            ),
        ),
    )
    env.reset("unit-scene")

    from gs_sim2real.sim import Pose3D

    blocked = env.query_collision(Pose3D(position=(0.25, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert blocked.collides is True
    assert blocked.reason == "dynamic-obstacle:runner"

    clear = env.query_collision(Pose3D(position=(0.5, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert clear.collides is False


def test_gym_adapter_rollout_respects_dynamic_obstacle() -> None:
    obstacle = _linear_obstacle(
        "door",
        start=(0.25, 0.0, 0.0),
        end=(0.25, 0.0, 0.0),
        radius_meters=0.05,
    )
    timeline = DynamicObstacleTimeline(timeline_id="door-gate", obstacles=(obstacle,))
    env = HeadlessPhysicalAIEnvironment(_unit_catalog(), dynamic_obstacles=timeline)
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4, goal_tolerance_meters=0.02),
    )
    _, _ = adapter.reset(seed=7, goal=(0.5, 0.0, 0.0))

    # Teleport straight into the obstacle. Headless env refuses to update the
    # pose, and the adapter reports a blocked termination reason. Removing the
    # obstacle should let the same route complete normally on a fresh adapter.
    start_pose = adapter.state.pose.position
    _, _, terminated, _, info = adapter.step({"x": 0.25, "y": 0.0, "z": 0.0})

    assert terminated is True
    assert info["termination_reason"] == "blocked-route"
    assert info["blocked"] is True
    # Blocked rollout: adapter stays at its starting pose (no progress into the
    # obstacle) because stop_on_collision=True short-circuits the route.
    assert math.dist(adapter.state.pose.position, start_pose) < 1e-6

    # Same route without the obstacle completes cleanly (goal reached or
    # truncated, not blocked).
    env_clear = HeadlessPhysicalAIEnvironment(_unit_catalog())
    adapter_clear = RoutePolicyGymAdapter(
        env_clear,
        RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4, goal_tolerance_meters=0.02),
    )
    adapter_clear.reset(seed=7, goal=(0.5, 0.0, 0.0))
    _, _, _, _, info_clear = adapter_clear.step({"x": 0.25, "y": 0.0, "z": 0.0})
    assert info_clear["blocked"] is False


def test_gym_adapter_omits_obstacle_features_when_no_timeline() -> None:
    env = HeadlessPhysicalAIEnvironment(_unit_catalog())
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4, goal_tolerance_meters=0.02),
    )
    observation, _ = adapter.reset(seed=1, goal=(0.3, 0.0, 0.0))

    # Feature block is silent by default — back-compat for existing fixtures.
    assert "nearest-dynamic-obstacle-distance-meters" not in observation
    assert "dynamic-obstacle-count" not in observation


def test_gym_adapter_emits_obstacle_features_and_tracks_movement() -> None:
    # Single obstacle that starts 0.5m to the +x of an always-at-origin
    # trajectory and then drifts further away. The distance feature should
    # grow with the adapter's step index.
    timeline = DynamicObstacleTimeline(
        timeline_id="tracker",
        obstacles=(
            DynamicObstacle(
                obstacle_id="mover",
                waypoints=(
                    DynamicObstacleWaypoint(step_index=0, position=(0.5, 0.0, 0.0)),
                    DynamicObstacleWaypoint(step_index=2, position=(1.5, 0.0, 0.0)),
                ),
                radius_meters=0.1,
            ),
        ),
    )
    env = HeadlessPhysicalAIEnvironment(_unit_catalog(), dynamic_obstacles=timeline)
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4, goal_tolerance_meters=0.02),
    )

    from gs_sim2real.sim import Pose3D

    observation, _ = adapter.reset(seed=11, goal=(0.3, 0.0, 0.0))
    # Force the observed pose to the origin so bearings are deterministic.
    adapter._state = RoutePolicyEnvState(  # type: ignore[attr-defined]
        scene_id=adapter.state.scene_id,
        episode_index=adapter.state.episode_index,
        step_index=0,
        pose=Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        goal=adapter.state.goal,
        done=False,
    )
    obs_step_0 = adapter._observation_features(adapter.state)

    assert math.isclose(obs_step_0["dynamic-obstacle-count"], 1.0)
    assert math.isclose(obs_step_0["nearest-dynamic-obstacle-distance-meters"], 0.4, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(obs_step_0["nearest-dynamic-obstacle-bearing-radians"], 0.0, abs_tol=1e-9)
    assert math.isclose(obs_step_0["nearest-dynamic-obstacle-bearing-x"], 1.0, abs_tol=1e-9)
    assert math.isclose(obs_step_0["nearest-dynamic-obstacle-bearing-y"], 0.0, abs_tol=1e-9)

    adapter._state = RoutePolicyEnvState(  # type: ignore[attr-defined]
        scene_id=adapter.state.scene_id,
        episode_index=adapter.state.episode_index,
        step_index=2,
        pose=Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        goal=adapter.state.goal,
        done=False,
    )
    obs_step_2 = adapter._observation_features(adapter.state)
    # Obstacle drifted from 0.5m to 1.5m, clearance grew from 0.4 to 1.4.
    assert (
        obs_step_2["nearest-dynamic-obstacle-distance-meters"] > obs_step_0["nearest-dynamic-obstacle-distance-meters"]
    )
    assert math.isclose(obs_step_2["nearest-dynamic-obstacle-distance-meters"], 1.4, rel_tol=0, abs_tol=1e-9)


def test_gym_adapter_obstacle_features_respect_bearing_signs() -> None:
    timeline = DynamicObstacleTimeline(
        timeline_id="bearings",
        obstacles=(
            DynamicObstacle(
                obstacle_id="south-west",
                waypoints=(DynamicObstacleWaypoint(step_index=0, position=(-1.0, 0.0, -1.0)),),
                radius_meters=0.1,
            ),
        ),
    )
    env = HeadlessPhysicalAIEnvironment(_unit_catalog(), dynamic_obstacles=timeline)
    adapter = RoutePolicyGymAdapter(
        env,
        RoutePolicyEnvConfig(scene_id="unit-scene", max_steps=4, goal_tolerance_meters=0.02),
    )

    from gs_sim2real.sim import Pose3D

    adapter.reset(seed=3, goal=(0.2, 0.0, 0.0))
    adapter._state = RoutePolicyEnvState(  # type: ignore[attr-defined]
        scene_id=adapter.state.scene_id,
        episode_index=adapter.state.episode_index,
        step_index=0,
        pose=Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        goal=adapter.state.goal,
        done=False,
    )
    observation = adapter._observation_features(adapter.state)

    # Obstacle is at (-1, 0, -1) in scene coords. The adapter only uses XY for
    # bearing, so this obstacle should read as the -x, -y quadrant.
    assert observation["nearest-dynamic-obstacle-bearing-x"] < 0.0
    # The XY direction from origin to (-1, 0) is pure -x; -y component is zero.
    # (We set position.y to 0 on both sides.)
    assert math.isclose(observation["nearest-dynamic-obstacle-bearing-y"], 0.0, abs_tol=1e-9)
    assert math.isclose(observation["nearest-dynamic-obstacle-bearing-radians"], math.pi, abs_tol=1e-9)


def test_trajectory_score_uses_per_step_obstacle_positions() -> None:
    # Obstacle moves from (0.3, 0, 0) at step 0 to (-10, 0, 0) at step 1,
    # so only the first pose of a two-pose trajectory is blocked.
    obstacle = DynamicObstacle(
        obstacle_id="fleeting",
        waypoints=(
            DynamicObstacleWaypoint(step_index=0, position=(0.3, 0.0, 0.0)),
            DynamicObstacleWaypoint(step_index=1, position=(-10.0, 0.0, 0.0)),
        ),
        radius_meters=0.05,
    )
    timeline = DynamicObstacleTimeline(timeline_id="fleeting", obstacles=(obstacle,))
    env = HeadlessPhysicalAIEnvironment(_unit_catalog(), dynamic_obstacles=timeline)
    env.reset("unit-scene")

    from gs_sim2real.sim import Pose3D

    score = env.score_trajectory(
        "unit-scene",
        (
            Pose3D(position=(0.3, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            Pose3D(position=(0.3, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        ),
    )

    # Collision count tracks dynamic obstacle presence per step.
    assert score.metrics["collision-count"] == 1.0
