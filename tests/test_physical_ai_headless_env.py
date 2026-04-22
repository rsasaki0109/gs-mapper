"""Tests for the bounds-based Physical AI headless environment."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    PhysicalAIEnvironment,
    Pose3D,
    Vec3,
    load_simulation_catalog_from_scene_picker,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_env() -> HeadlessPhysicalAIEnvironment:
    catalog = load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json")
    return HeadlessPhysicalAIEnvironment(catalog)


def test_headless_environment_matches_protocol_and_resets_to_scene_center() -> None:
    env = build_env()

    assert isinstance(env, PhysicalAIEnvironment)
    payload = env.reset("outdoor-demo")

    scene = payload["scene"]
    state = payload["state"]
    assert payload["backend"] == "headless-bounds"
    assert scene["sceneId"] == "outdoor-demo"
    assert state["sceneId"] == "outdoor-demo"
    assert state["stepIndex"] == 0
    assert state["pose"]["frameId"] == "autoware_metric_world"
    assert state["pose"]["position"][0] == pytest.approx(scene["bounds"]["center"][0])


def test_sample_goal_is_deterministic_and_inside_bounds() -> None:
    env = build_env()
    first = env.sample_goal("bag6-mast3r", seed=7)
    second = env.sample_goal("bag6-mast3r", seed=7)
    scene = env.catalog.scene_by_id("bag6-mast3r")

    assert first == second
    assert first.frame_id == "autoware_metric_world"
    assert scene.bounds.contains(tuple_to_vec3(first.position))


def test_query_collision_detects_bounds_exit_after_reset() -> None:
    env = build_env()
    env.reset("outdoor-demo")
    scene = env.catalog.scene_by_id("outdoor-demo")

    inside = env.query_collision(env.state.pose)
    outside_pose = Pose3D(
        position=(scene.bounds.maximum.x + 10.0, env.state.pose.position[1], env.state.pose.position[2]),
        orientation_xyzw=env.state.pose.orientation_xyzw,
        frame_id=env.state.pose.frame_id,
    )
    outside = env.query_collision(outside_pose)

    assert inside.collides is False
    assert inside.reason == "inside-bounds"
    assert outside.collides is True
    assert outside.reason == "outside-bounds"


def test_twist_step_updates_pose_and_blocks_out_of_bounds_teleport() -> None:
    env = build_env()
    env.reset("outdoor-demo")
    original = env.state.pose

    moved = env.step(AgentAction("twist", {"linearX": 2.0}, duration_seconds=0.5))
    assert moved["applied"] is True
    assert env.state.step_index == 1
    assert env.state.pose.position[0] == pytest.approx(original.position[0] + 1.0)

    scene = env.catalog.scene_by_id("outdoor-demo")
    blocked = env.step(
        AgentAction(
            "teleport",
            {
                "x": scene.bounds.maximum.x + 100.0,
                "y": env.state.pose.position[1],
                "z": env.state.pose.position[2],
            },
        )
    )
    assert blocked["applied"] is False
    assert blocked["collision"]["reason"] == "outside-bounds"
    assert env.state.step_index == 2
    assert env.state.pose.position[0] == pytest.approx(original.position[0] + 1.0)


def test_metadata_observation_and_error_paths() -> None:
    env = build_env()
    env.reset("outdoor-demo")

    observation = env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward"))
    assert observation.outputs["mode"] == "metadata-only"
    assert observation.outputs["sceneId"] == "outdoor-demo"
    assert observation.outputs["viewerUrl"].endswith("splat.html?url=assets/outdoor-demo/outdoor-demo.splat")

    with pytest.raises(ValueError, match="unsupported sensor"):
        env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="thermal"))

    with pytest.raises(ValueError, match="unsupported outputs"):
        env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward", outputs=("depth",)))

    with pytest.raises(ValueError, match="unsupported headless action"):
        env.step(AgentAction("jump", {}))


def test_score_trajectory_reports_bounds_rate_and_path_length() -> None:
    env = build_env()
    env.reset("outdoor-demo")
    scene = env.catalog.scene_by_id("outdoor-demo")
    start = env.state.pose
    end = Pose3D(
        position=(start.position[0] + 1.0, start.position[1], start.position[2]),
        orientation_xyzw=start.orientation_xyzw,
        frame_id=start.frame_id,
    )
    outside = Pose3D(
        position=(scene.bounds.maximum.x + 1.0, start.position[1], start.position[2]),
        orientation_xyzw=start.orientation_xyzw,
        frame_id=start.frame_id,
    )

    inside_score = env.score_trajectory("outdoor-demo", [start, end])
    mixed_score = env.score_trajectory("outdoor-demo", [start, outside])

    assert inside_score.passed is True
    assert inside_score.metrics["inside-bounds-rate"] == 1.0
    assert inside_score.metrics["path-length"] == pytest.approx(1.0)
    assert mixed_score.passed is False
    assert mixed_score.metrics["inside-bounds-rate"] == 0.5
    assert math.isfinite(mixed_score.metrics["path-length"])


def tuple_to_vec3(position: tuple[float, float, float]) -> Vec3:
    return Vec3(*position)
