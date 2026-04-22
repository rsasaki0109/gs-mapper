"""Bounds-based headless Physical AI environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from .contract import AxisAlignedBounds, SceneEnvironment, SimulationCatalog, Vec3
from .interfaces import (
    AgentAction,
    CollisionQuery,
    Observation,
    ObservationRequest,
    PhysicalAIEnvironment,
    Pose3D,
    TrajectoryScore,
)


DEFAULT_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class HeadlessEnvironmentState:
    """Current state for the headless environment."""

    scene_id: str
    pose: Pose3D
    step_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "pose": self.pose.to_dict(),
            "stepIndex": self.step_index,
        }


class HeadlessPhysicalAIEnvironment(PhysicalAIEnvironment):
    """Minimal environment that executes the simulation contract without a renderer."""

    def __init__(self, catalog: SimulationCatalog):
        self.catalog = catalog
        self._state: HeadlessEnvironmentState | None = None

    @property
    def state(self) -> HeadlessEnvironmentState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    def reset(self, scene_id: str, *, seed: int | None = None) -> Mapping[str, Any]:
        scene = self.catalog.scene_by_id(scene_id)
        pose = self._initial_pose(scene, seed=seed)
        self._state = HeadlessEnvironmentState(scene_id=scene.scene_id, pose=pose, step_index=0)
        return {
            "scene": scene.to_dict(),
            "state": self._state.to_dict(),
            "backend": "headless-bounds",
        }

    def step(self, action: AgentAction) -> Mapping[str, Any]:
        state = self.state
        scene = self.catalog.scene_by_id(state.scene_id)
        next_pose = self._apply_action(state.pose, action)
        collision = self.query_collision(next_pose)
        if collision.collides:
            next_pose = state.pose
        self._state = HeadlessEnvironmentState(scene_id=scene.scene_id, pose=next_pose, step_index=state.step_index + 1)
        return {
            "sceneId": scene.scene_id,
            "state": self._state.to_dict(),
            "collision": collision.to_dict(),
            "applied": not collision.collides,
        }

    def render_observation(self, request: ObservationRequest) -> Observation:
        scene = self.catalog.scene_by_id(self.state.scene_id)
        if request.sensor_id not in scene.sensor_rig.sensor_ids():
            raise ValueError(f"unsupported sensor for scene {scene.scene_id}: {request.sensor_id}")
        unsupported = tuple(
            output for output in request.outputs if output not in _supported_outputs(scene, request.sensor_id)
        )
        if unsupported:
            raise ValueError(f"unsupported outputs for sensor {request.sensor_id}: {', '.join(unsupported)}")
        return Observation(
            sensor_id=request.sensor_id,
            pose=request.pose,
            outputs={
                "mode": "metadata-only",
                "sceneId": scene.scene_id,
                "viewerUrl": scene.viewer_url,
                "requestedOutputs": list(request.outputs),
            },
        )

    def query_collision(self, pose: Pose3D) -> CollisionQuery:
        scene = self.catalog.scene_by_id(self.state.scene_id)
        point = Vec3.from_sequence(pose.position)
        if scene.bounds.contains(point):
            return CollisionQuery(
                pose=pose,
                collides=False,
                reason="inside-bounds",
                clearance_meters=_bounds_clearance(scene.bounds, point),
            )
        return CollisionQuery(
            pose=pose,
            collides=True,
            reason="outside-bounds",
            clearance_meters=0.0,
        )

    def sample_goal(self, scene_id: str, *, seed: int | None = None) -> Pose3D:
        scene = self.catalog.scene_by_id(scene_id)
        rng = random.Random(seed)
        bounds = scene.bounds
        return Pose3D(
            position=(
                rng.uniform(bounds.minimum.x, bounds.maximum.x),
                _navigation_height(bounds),
                rng.uniform(bounds.minimum.z, bounds.maximum.z),
            ),
            orientation_xyzw=DEFAULT_ORIENTATION_XYZW,
            frame_id=scene.coordinate_frame.frame_id,
        )

    def score_trajectory(self, scene_id: str, trajectory: Sequence[Pose3D]) -> TrajectoryScore:
        scene = self.catalog.scene_by_id(scene_id)
        if not trajectory:
            return TrajectoryScore(metrics={"inside-bounds-rate": 0.0, "path-length": 0.0}, passed=False)

        inside = sum(scene.bounds.contains(Vec3.from_sequence(pose.position)) for pose in trajectory)
        inside_rate = inside / len(trajectory)
        path_length = _path_length(trajectory)
        return TrajectoryScore(
            metrics={
                "inside-bounds-rate": inside_rate,
                "path-length": path_length,
            },
            passed=inside_rate == 1.0,
            notes=("bounds-only score",),
        )

    def _initial_pose(self, scene: SceneEnvironment, *, seed: int | None) -> Pose3D:
        if seed is None:
            center = scene.bounds.center
            position = (center.x, _navigation_height(scene.bounds), center.z)
        else:
            position = self.sample_goal(scene.scene_id, seed=seed).position
        return Pose3D(
            position=position,
            orientation_xyzw=DEFAULT_ORIENTATION_XYZW,
            frame_id=scene.coordinate_frame.frame_id,
        )

    def _apply_action(self, pose: Pose3D, action: AgentAction) -> Pose3D:
        action_type = str(action.action_type).strip().lower()
        if action_type == "teleport":
            position = _action_position(action)
            orientation = tuple(
                action.values.get(name, pose.orientation_xyzw[index]) for index, name in enumerate(_quat_keys())
            )
            return Pose3D(
                position=position,
                orientation_xyzw=_as_quaternion(orientation),
                frame_id=pose.frame_id,
            )

        if action_type == "twist":
            dt = max(0.0, float(action.duration_seconds))
            linear_x = float(action.values.get("linearX", action.values.get("vx", 0.0)))
            linear_y = float(action.values.get("linearY", action.values.get("vy", 0.0)))
            linear_z = float(action.values.get("linearZ", action.values.get("vz", 0.0)))
            x, y, z = pose.position
            return Pose3D(
                position=(x + linear_x * dt, y + linear_y * dt, z + linear_z * dt),
                orientation_xyzw=pose.orientation_xyzw,
                frame_id=pose.frame_id,
            )

        raise ValueError(f"unsupported headless action: {action.action_type}")


def _supported_outputs(scene: SceneEnvironment, sensor_id: str) -> tuple[str, ...]:
    for sensor in scene.sensor_rig.sensors:
        if sensor.sensor_id == sensor_id:
            return sensor.outputs
    return ()


def _navigation_height(bounds: AxisAlignedBounds) -> float:
    return min(max(0.0, bounds.minimum.y), bounds.maximum.y)


def _bounds_clearance(bounds: AxisAlignedBounds, point: Vec3) -> float:
    return min(
        point.x - bounds.minimum.x,
        bounds.maximum.x - point.x,
        point.y - bounds.minimum.y,
        bounds.maximum.y - point.y,
        point.z - bounds.minimum.z,
        bounds.maximum.z - point.z,
    )


def _path_length(trajectory: Sequence[Pose3D]) -> float:
    total = 0.0
    for first, second in zip(trajectory, trajectory[1:]):
        ax, ay, az = first.position
        bx, by, bz = second.position
        total += math.dist((ax, ay, az), (bx, by, bz))
    return total


def _action_position(action: AgentAction) -> tuple[float, float, float]:
    try:
        return (
            float(action.values["x"]),
            float(action.values["y"]),
            float(action.values["z"]),
        )
    except KeyError as exc:
        raise ValueError("teleport action requires x, y, and z values") from exc


def _quat_keys() -> tuple[str, str, str, str]:
    return ("qx", "qy", "qz", "qw")


def _as_quaternion(value: tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("orientation must have four values")
    qx, qy, qz, qw = (float(component) for component in value)
    return (qx, qy, qz, qw)
