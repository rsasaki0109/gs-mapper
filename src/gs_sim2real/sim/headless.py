"""Headless Physical AI environment."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from .contract import AxisAlignedBounds, SceneEnvironment, SimulationCatalog, Vec3
from .costmap import summarize_collision_queries
from .interfaces import (
    AgentAction,
    CollisionQuery,
    KinematicState,
    Observation,
    ObservationRequest,
    PhysicalAIEnvironment,
    Pose3D,
    TrajectoryScore,
)
from .footprint import RobotFootprint
from .occupancy import OccupancyQuery, VoxelOccupancyGrid, point_to_voxel_cell
from .policy_dynamic_obstacles import DynamicObstacleTimeline
from .raw_sensor_noise import (
    RawSensorNoiseProfile,
    apply_raw_sensor_noise_to_observation,
    raw_sensor_noise_rng,
)
from .rendering import ObservationRenderer


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
    """Minimal environment that executes the simulation contract with optional rendering."""

    def __init__(
        self,
        catalog: SimulationCatalog,
        *,
        observation_renderer: ObservationRenderer | None = None,
        occupancy_grid: VoxelOccupancyGrid | None = None,
        robot_footprint: RobotFootprint | None = None,
        dynamic_obstacles: DynamicObstacleTimeline | None = None,
        raw_sensor_noise_profile: RawSensorNoiseProfile | None = None,
    ):
        self.catalog = catalog
        self.observation_renderer = observation_renderer
        self.occupancy_grid = occupancy_grid
        self.robot_footprint = robot_footprint
        self.dynamic_obstacles = dynamic_obstacles
        self.raw_sensor_noise_profile = raw_sensor_noise_profile
        self._state: HeadlessEnvironmentState | None = None
        self._reset_seed: int | None = None
        self._render_request_count: int = 0
        self._kinematic_state: KinematicState = KinematicState()

    def set_occupancy_grid(self, occupancy_grid: VoxelOccupancyGrid | None) -> None:
        """Set or clear the occupancy grid used by collision queries."""

        self.occupancy_grid = occupancy_grid

    def set_robot_footprint(self, robot_footprint: RobotFootprint | None) -> None:
        """Set or clear the robot footprint used by occupancy collision queries."""

        self.robot_footprint = robot_footprint

    def set_dynamic_obstacles(self, dynamic_obstacles: DynamicObstacleTimeline | None) -> None:
        """Set or clear the dynamic obstacle timeline consulted by collision queries."""

        self.dynamic_obstacles = dynamic_obstacles

    def set_raw_sensor_noise_profile(self, profile: RawSensorNoiseProfile | None) -> None:
        """Set or clear the raw-sensor noise profile applied to rendered observations."""

        self.raw_sensor_noise_profile = profile

    @property
    def state(self) -> HeadlessEnvironmentState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    @property
    def kinematic_state(self) -> KinematicState:
        """Return the current kinematic estimate of the agent.

        After ``reset`` and before any ``step`` the returned estimate is the
        zero state (``step_dt_seconds == 0.0``); ``step`` updates it by
        finite-differencing pose changes against the previous step.
        """

        return self._kinematic_state

    def reset(self, scene_id: str, *, seed: int | None = None) -> Mapping[str, Any]:
        scene = self.catalog.scene_by_id(scene_id)
        pose = self._initial_pose(scene, seed=seed)
        self._state = HeadlessEnvironmentState(scene_id=scene.scene_id, pose=pose, step_index=0)
        self._reset_seed = None if seed is None else int(seed)
        self._render_request_count = 0
        self._kinematic_state = KinematicState()
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
        self._kinematic_state = self._derive_kinematic_state(state.pose, next_pose, action)
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
        if request.sensor_id == "imu-proxy":
            rendered = self._render_imu_observation(scene, request)
            return self._apply_raw_sensor_noise(rendered, request)
        if self.observation_renderer is not None and self.observation_renderer.can_render(scene, request):
            rendered = self.observation_renderer.render_observation(scene, request)
            return self._apply_raw_sensor_noise(rendered, request)
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

    def _render_imu_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        """Render an IMU observation from the kinematic state finite-differenced by ``step``.

        Outputs match the format consumed by
        :func:`apply_raw_sensor_noise_to_observation`: ``angular-velocity``
        with an ``angularVelocityBase64`` block and ``linear-acceleration``
        with a ``linearAccelerationBase64`` block, both float32-le-xyz in
        the agent body frame.
        """

        kinematics = self._kinematic_state
        outputs: dict[str, Any] = {
            "mode": "kinematic-finite-diff",
            "sceneId": scene.scene_id,
            "frameId": "agent-body",
            "stepDtSeconds": float(kinematics.step_dt_seconds),
            "requestedOutputs": list(request.outputs),
        }
        if "angular-velocity" in request.outputs:
            outputs["angular-velocity"] = _imu_vector_block(
                kinematics.angular_velocity_body,
                base64_key="angularVelocityBase64",
                unit="rad/s",
            )
        if "linear-acceleration" in request.outputs:
            outputs["linear-acceleration"] = _imu_vector_block(
                kinematics.linear_acceleration_body,
                base64_key="linearAccelerationBase64",
                unit="m/s^2",
            )
        return Observation(sensor_id=request.sensor_id, pose=request.pose, outputs=outputs)

    def _derive_kinematic_state(
        self,
        previous_pose: Pose3D,
        next_pose: Pose3D,
        action: AgentAction,
    ) -> KinematicState:
        """Finite-difference the pose change to update the kinematic state.

        ``teleport`` actions reset the kinematic state to zero — a teleport
        is an instantaneous discontinuity, not a physical motion, so feeding
        its pose delta into the finite difference would produce nonsense
        velocities / accelerations.
        """

        action_type = str(action.action_type).strip().lower()
        dt = max(0.0, float(action.duration_seconds))
        if action_type == "teleport" or dt <= 0.0:
            return KinematicState(step_dt_seconds=dt)

        previous_velocity_world = self._kinematic_state.linear_velocity_world
        next_velocity_world = (
            (next_pose.position[0] - previous_pose.position[0]) / dt,
            (next_pose.position[1] - previous_pose.position[1]) / dt,
            (next_pose.position[2] - previous_pose.position[2]) / dt,
        )
        acceleration_world = (
            (next_velocity_world[0] - previous_velocity_world[0]) / dt,
            (next_velocity_world[1] - previous_velocity_world[1]) / dt,
            (next_velocity_world[2] - previous_velocity_world[2]) / dt,
        )
        angular_velocity_body = _angular_velocity_body_from_quaternion_delta(
            previous_pose.orientation_xyzw,
            next_pose.orientation_xyzw,
            dt,
        )
        linear_acceleration_body = _world_vector_to_body(
            acceleration_world,
            previous_pose.orientation_xyzw,
        )
        return KinematicState(
            linear_velocity_world=next_velocity_world,
            angular_velocity_body=angular_velocity_body,
            linear_acceleration_body=linear_acceleration_body,
            step_dt_seconds=dt,
        )

    def _apply_raw_sensor_noise(self, observation: Observation, request: ObservationRequest) -> Observation:
        """Perturb ``observation`` with the active raw-sensor noise profile if one is set."""

        profile = self.raw_sensor_noise_profile
        if profile is None or profile.is_noise_free:
            return observation
        request_index = self._render_request_count
        self._render_request_count += 1
        rng = raw_sensor_noise_rng(
            base_seed=self._reset_seed,
            profile_id=profile.profile_id,
            sensor_id=request.sensor_id,
            request_index=request_index,
            kind="obs",
        )
        return apply_raw_sensor_noise_to_observation(observation, profile, rng=rng)

    def query_collision(self, pose: Pose3D) -> CollisionQuery:
        scene = self.catalog.scene_by_id(self.state.scene_id)
        return self._query_collision_for_scene(scene, pose, step_index=self.state.step_index)

    def _query_collision_for_scene(
        self,
        scene: SceneEnvironment,
        pose: Pose3D,
        *,
        step_index: int = 0,
    ) -> CollisionQuery:
        point = Vec3.from_sequence(pose.position)
        if not scene.bounds.contains(point):
            return CollisionQuery(
                pose=pose,
                collides=True,
                reason="outside-bounds",
                clearance_meters=0.0,
            )
        if self.dynamic_obstacles is not None:
            blocking = self.dynamic_obstacles.blocking_obstacle(pose.position, step_index)
            if blocking is not None:
                centre = blocking.position_at_step(step_index, agent_position=pose.position)
                clearance = max(0.0, math.dist(tuple(pose.position), centre) - blocking.radius_meters)
                return CollisionQuery(
                    pose=pose,
                    collides=True,
                    reason=f"dynamic-obstacle:{blocking.obstacle_id}",
                    clearance_meters=clearance,
                )
        if self.occupancy_grid is not None:
            occupancy = self._query_occupancy(pose)
            if occupancy.occupied:
                return CollisionQuery(
                    pose=pose,
                    collides=True,
                    reason=f"{occupancy.reason}:{self.occupancy_grid.source}",
                    clearance_meters=occupancy.clearance_meters,
                )
            return CollisionQuery(
                pose=pose,
                collides=False,
                reason=f"{occupancy.reason}:{self.occupancy_grid.source}",
                clearance_meters=occupancy.clearance_meters,
            )
        return CollisionQuery(
            pose=pose,
            collides=False,
            reason="inside-bounds",
            clearance_meters=_bounds_clearance(scene.bounds, point),
        )

    def _query_occupancy(self, pose: Pose3D) -> OccupancyQuery:
        if self.occupancy_grid is None:
            raise RuntimeError("occupancy grid has not been set")
        if self.robot_footprint is None:
            return self.occupancy_grid.query_pose(pose)
        reference_cell = point_to_voxel_cell(pose.position, self.occupancy_grid.voxel_size_meters)
        return self.occupancy_grid.query_cells(
            self.robot_footprint.cells_for_pose(pose, self.occupancy_grid.voxel_size_meters),
            reference_cell=reference_cell,
            occupied_reason="occupied-footprint-voxel",
            free_reason="free-footprint",
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
            return TrajectoryScore(
                metrics={
                    "inside-bounds-rate": 0.0,
                    "path-length": 0.0,
                    "collision-rate": 0.0,
                    "collision-count": 0.0,
                },
                passed=False,
                notes=("empty-trajectory",),
            )

        inside = sum(scene.bounds.contains(Vec3.from_sequence(pose.position)) for pose in trajectory)
        inside_rate = inside / len(trajectory)
        path_length = _path_length(trajectory)
        collision_summary = summarize_collision_queries(
            tuple(
                self._query_collision_for_scene(scene, pose, step_index=index) for index, pose in enumerate(trajectory)
            )
        )
        metrics = {
            "inside-bounds-rate": inside_rate,
            "path-length": path_length,
            **collision_summary.metric_payload(),
        }
        notes = collision_summary.notes()
        if self.occupancy_grid is None:
            notes = ("bounds-only score", *notes)
        return TrajectoryScore(
            metrics=metrics,
            passed=inside_rate == 1.0 and collision_summary.collision_count == 0,
            notes=notes,
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


def _imu_vector_block(
    vector: tuple[float, float, float],
    *,
    base64_key: str,
    unit: str,
) -> dict[str, Any]:
    """Encode a 3-vector into the float32-le-xyz block expected by raw-sensor noise."""

    import numpy as np  # local import to keep numpy out of the cold path

    payload = np.asarray(vector, dtype="<f4").tobytes()
    return {
        "encoding": "float32-le-xyz",
        "unit": unit,
        base64_key: base64.b64encode(payload).decode("ascii"),
        "byteLength": len(payload),
    }


def _angular_velocity_body_from_quaternion_delta(
    previous_xyzw: tuple[float, float, float, float],
    next_xyzw: tuple[float, float, float, float],
    dt: float,
) -> tuple[float, float, float]:
    """Compute the body-frame angular velocity that takes ``previous`` to ``next``.

    Uses the body-frame delta quaternion ``q_prev⁻¹ ⊗ q_next``: its imaginary
    part divided by ``sin(angle/2)`` is the rotation axis in body
    coordinates, and the rotation magnitude is ``2·acos(w)``. Identity-delta
    (no rotation) returns the zero vector.
    """

    if dt <= 0.0:
        return (0.0, 0.0, 0.0)
    delta = _quaternion_multiply(_quaternion_conjugate(previous_xyzw), next_xyzw)
    qx, qy, qz, qw = delta
    qw_clamped = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw_clamped)
    if angle == 0.0:
        return (0.0, 0.0, 0.0)
    sin_half = math.sin(angle / 2.0)
    if sin_half == 0.0:
        return (0.0, 0.0, 0.0)
    rate = angle / dt
    inv_sin_half = 1.0 / sin_half
    return (qx * inv_sin_half * rate, qy * inv_sin_half * rate, qz * inv_sin_half * rate)


def _world_vector_to_body(
    vector_world: tuple[float, float, float],
    body_to_world_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate a world-frame vector into the body frame defined by ``body_to_world_xyzw``."""

    inverse = _quaternion_conjugate(body_to_world_xyzw)
    rotated = _rotate_vector_by_quaternion(vector_world, inverse)
    return rotated


def _quaternion_conjugate(
    xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = xyzw
    return (-qx, -qy, -qz, qw)


def _quaternion_multiply(
    a_xyzw: tuple[float, float, float, float],
    b_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a_xyzw
    bx, by, bz, bw = b_xyzw
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate_vector_by_quaternion(
    vector: tuple[float, float, float],
    xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate ``vector`` by the unit quaternion ``xyzw`` (Hamilton convention)."""

    vx, vy, vz = vector
    pure = (vx, vy, vz, 0.0)
    rotated = _quaternion_multiply(_quaternion_multiply(xyzw, pure), _quaternion_conjugate(xyzw))
    return (rotated[0], rotated[1], rotated[2])
