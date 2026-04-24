"""Tests for the bounds-based Physical AI headless environment."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    Observation,
    ObservationRequest,
    PhysicalAIEnvironment,
    Pose3D,
    RawSensorNoiseProfile,
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
    assert inside_score.metrics["collision-rate"] == 0.0
    assert mixed_score.passed is False
    assert mixed_score.metrics["inside-bounds-rate"] == 0.5
    assert mixed_score.metrics["collision-rate"] == 0.5
    assert mixed_score.metrics["collision-count"] == 1.0
    assert math.isfinite(mixed_score.metrics["path-length"])


def test_score_trajectory_reports_empty_path_as_failed() -> None:
    env = build_env()

    score = env.score_trajectory("outdoor-demo", [])

    assert score.passed is False
    assert score.metrics["inside-bounds-rate"] == 0.0
    assert score.metrics["collision-rate"] == 0.0
    assert score.metrics["collision-count"] == 0.0
    assert score.notes == ("empty-trajectory",)


def tuple_to_vec3(position: tuple[float, float, float]) -> Vec3:
    return Vec3(*position)


class _ConstantDepthRenderer:
    """Stub ``ObservationRenderer`` that returns a deterministic depth or ranges block."""

    def __init__(self, depth_value: float = 12.5, range_values: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)) -> None:
        self._depth = np.full((4, 6), fill_value=float(depth_value), dtype=np.float32)
        self._ranges = np.asarray(range_values, dtype=np.float32)

    def can_render(self, scene, request: ObservationRequest) -> bool:
        if request.sensor_id == "depth-proxy":
            return set(request.outputs).issubset({"depth", "validity-mask"})
        if request.sensor_id == "lidar-ray-proxy":
            return set(request.outputs).issubset({"ranges", "points"})
        return False

    def render_observation(self, scene, request: ObservationRequest) -> Observation:
        if request.sensor_id == "depth-proxy":
            depth_bytes = self._depth.tobytes()
            outputs = {
                "mode": "stub-depth",
                "depth": {
                    "encoding": "float32-le",
                    "unit": "meter",
                    "width": 6,
                    "height": 4,
                    "depthBase64": base64.b64encode(depth_bytes).decode("ascii"),
                    "byteLength": len(depth_bytes),
                    "farClipMeters": 80.0,
                },
            }
        else:
            range_bytes = self._ranges.tobytes()
            outputs = {
                "mode": "stub-lidar",
                "ranges": {
                    "encoding": "float32-le",
                    "unit": "meter",
                    "count": int(self._ranges.size),
                    "rangesBase64": base64.b64encode(range_bytes).decode("ascii"),
                    "byteLength": len(range_bytes),
                },
            }
        return Observation(sensor_id=request.sensor_id, pose=request.pose, outputs=outputs)


def _decode_depth(observation: Observation) -> np.ndarray:
    return np.frombuffer(base64.b64decode(observation.outputs["depth"]["depthBase64"]), dtype="<f4")


def _decode_ranges(observation: Observation) -> np.ndarray:
    return np.frombuffer(base64.b64decode(observation.outputs["ranges"]["rangesBase64"]), dtype="<f4")


def _depth_request(env: HeadlessPhysicalAIEnvironment) -> ObservationRequest:
    return ObservationRequest(pose=env.state.pose, sensor_id="depth-proxy", outputs=("depth",))


def _ranges_request(env: HeadlessPhysicalAIEnvironment) -> ObservationRequest:
    return ObservationRequest(pose=env.state.pose, sensor_id="lidar-ray-proxy", outputs=("ranges",))


def test_headless_env_leaves_observations_untouched_when_no_raw_sensor_profile_set() -> None:
    env = HeadlessPhysicalAIEnvironment(
        load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json"),
        observation_renderer=_ConstantDepthRenderer(),
    )
    env.reset("outdoor-demo")

    depth_obs = env.render_observation(_depth_request(env))
    ranges_obs = env.render_observation(_ranges_request(env))

    assert np.allclose(_decode_depth(depth_obs), 12.5)
    assert np.allclose(_decode_ranges(ranges_obs), (5.0, 10.0, 15.0, 20.0))


def test_headless_env_applies_raw_sensor_noise_when_profile_and_renderer_are_set() -> None:
    profile = RawSensorNoiseProfile(
        profile_id="env-integration",
        depth_range_std_meters=0.3,
        lidar_range_std_meters=0.2,
    )
    env = HeadlessPhysicalAIEnvironment(
        load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json"),
        observation_renderer=_ConstantDepthRenderer(),
        raw_sensor_noise_profile=profile,
    )
    env.reset("outdoor-demo", seed=11)

    depth = _decode_depth(env.render_observation(_depth_request(env)))
    ranges = _decode_ranges(env.render_observation(_ranges_request(env)))

    assert not np.allclose(depth, 12.5)
    assert float(depth.min()) >= 0.0
    assert float(depth.max()) <= 80.0
    assert not np.allclose(ranges, (5.0, 10.0, 15.0, 20.0))
    assert float(ranges.min()) >= 0.0


def test_headless_env_raw_sensor_noise_is_deterministic_per_reset_seed() -> None:
    profile = RawSensorNoiseProfile(profile_id="env-determinism", depth_range_std_meters=0.4)

    def fresh_env() -> HeadlessPhysicalAIEnvironment:
        return HeadlessPhysicalAIEnvironment(
            load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json"),
            observation_renderer=_ConstantDepthRenderer(),
            raw_sensor_noise_profile=profile,
        )

    env_a = fresh_env()
    env_b = fresh_env()
    env_a.reset("outdoor-demo", seed=17)
    env_b.reset("outdoor-demo", seed=17)
    first = env_a.render_observation(_depth_request(env_a))
    second = env_b.render_observation(_depth_request(env_b))
    assert first.outputs["depth"]["depthBase64"] == second.outputs["depth"]["depthBase64"]

    env_c = fresh_env()
    env_c.reset("outdoor-demo", seed=23)
    diff = env_c.render_observation(_depth_request(env_c))
    assert diff.outputs["depth"]["depthBase64"] != first.outputs["depth"]["depthBase64"]


def test_headless_env_raw_sensor_noise_advances_request_index_across_calls() -> None:
    profile = RawSensorNoiseProfile(profile_id="env-counter", depth_range_std_meters=0.5)
    env = HeadlessPhysicalAIEnvironment(
        load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json"),
        observation_renderer=_ConstantDepthRenderer(),
        raw_sensor_noise_profile=profile,
    )
    env.reset("outdoor-demo", seed=5)

    first = env.render_observation(_depth_request(env))
    second = env.render_observation(_depth_request(env))

    assert first.outputs["depth"]["depthBase64"] != second.outputs["depth"]["depthBase64"]


def test_headless_env_set_raw_sensor_noise_profile_toggles_behavior() -> None:
    env = HeadlessPhysicalAIEnvironment(
        load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json"),
        observation_renderer=_ConstantDepthRenderer(),
    )
    env.reset("outdoor-demo", seed=2)

    clean = env.render_observation(_depth_request(env))
    assert np.allclose(_decode_depth(clean), 12.5)

    env.set_raw_sensor_noise_profile(RawSensorNoiseProfile(profile_id="switch-on", depth_range_std_meters=0.3))
    noisy = env.render_observation(_depth_request(env))
    assert not np.allclose(_decode_depth(noisy), 12.5)

    env.set_raw_sensor_noise_profile(None)
    restored = env.render_observation(_depth_request(env))
    assert np.allclose(_decode_depth(restored), 12.5)
