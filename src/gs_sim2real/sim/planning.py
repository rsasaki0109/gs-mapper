"""Cached planning context helpers for Physical AI simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from .interfaces import ObservationRequest, PhysicalAIEnvironment, Pose3D
from .occupancy import VoxelOccupancyGrid, build_occupancy_grid_from_lidar_observation


DEFAULT_LIDAR_OUTPUTS = ("ranges", "points")


@dataclass(frozen=True, slots=True)
class PlanningViewpointKey:
    """Stable cache key for a scene/viewpoint occupancy render."""

    scene_id: str
    sensor_id: str
    outputs: tuple[str, ...]
    position_key: tuple[int, int, int]
    orientation_key: tuple[int, int, int, int]
    voxel_size_meters: float
    inflation_radius_meters: float
    pose_resolution_meters: float
    orientation_resolution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "sensorId": self.sensor_id,
            "outputs": list(self.outputs),
            "positionKey": list(self.position_key),
            "orientationKey": list(self.orientation_key),
            "voxelSizeMeters": self.voxel_size_meters,
            "inflationRadiusMeters": self.inflation_radius_meters,
            "poseResolutionMeters": self.pose_resolution_meters,
            "orientationResolution": self.orientation_resolution,
        }


@dataclass(slots=True)
class OccupancyPlanningContext:
    """Cache occupancy grids rendered from scene/viewpoint observations."""

    voxel_size_meters: float
    inflation_radius_meters: float = 0.0
    sensor_id: str = "lidar-ray-proxy"
    outputs: tuple[str, ...] = DEFAULT_LIDAR_OUTPUTS
    pose_resolution_meters: float = 0.25
    orientation_resolution: float = 1e-4
    _cache: dict[PlanningViewpointKey, VoxelOccupancyGrid] = field(default_factory=dict, init=False, repr=False)
    _hit_count: int = field(default=0, init=False, repr=False)
    _miss_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.voxel_size_meters = _positive_float(self.voxel_size_meters, "voxel_size_meters")
        self.inflation_radius_meters = max(float(self.inflation_radius_meters), 0.0)
        self.pose_resolution_meters = _positive_float(self.pose_resolution_meters, "pose_resolution_meters")
        self.orientation_resolution = _positive_float(self.orientation_resolution, "orientation_resolution")
        self.outputs = tuple(self.outputs)
        if "points" not in self.outputs:
            raise ValueError("occupancy planning context outputs must include points")

    def key_for(self, scene_id: str, pose: Pose3D) -> PlanningViewpointKey:
        """Build the cache key for ``scene_id`` and ``pose``."""

        return PlanningViewpointKey(
            scene_id=str(scene_id),
            sensor_id=str(self.sensor_id),
            outputs=tuple(self.outputs),
            position_key=_quantize_tuple(pose.position, self.pose_resolution_meters),
            orientation_key=_quantize_tuple(pose.orientation_xyzw, self.orientation_resolution),
            voxel_size_meters=self.voxel_size_meters,
            inflation_radius_meters=self.inflation_radius_meters,
            pose_resolution_meters=self.pose_resolution_meters,
            orientation_resolution=self.orientation_resolution,
        )

    def get_or_render(
        self,
        environment: PhysicalAIEnvironment,
        *,
        scene_id: str,
        pose: Pose3D,
    ) -> VoxelOccupancyGrid:
        """Return a cached occupancy grid, rendering it from the environment on miss."""

        key = self.key_for(scene_id, pose)
        cached = self._cache.get(key)
        if cached is not None:
            self._hit_count += 1
            return cached

        self._miss_count += 1
        observation = environment.render_observation(
            ObservationRequest(pose=pose, sensor_id=self.sensor_id, outputs=self.outputs)
        )
        _validate_scene_id(scene_id, observation.outputs)
        grid = build_occupancy_grid_from_lidar_observation(
            observation,
            voxel_size_meters=self.voxel_size_meters,
            inflation_radius_meters=self.inflation_radius_meters,
        )
        self._cache[key] = grid
        return grid

    def set_environment_occupancy(
        self,
        environment: PhysicalAIEnvironment,
        *,
        scene_id: str,
        pose: Pose3D,
    ) -> VoxelOccupancyGrid:
        """Build or reuse occupancy and inject it into environments that support it."""

        grid = self.get_or_render(environment, scene_id=scene_id, pose=pose)
        setter = getattr(environment, "set_occupancy_grid", None)
        if not callable(setter):
            raise TypeError("environment must expose set_occupancy_grid to receive cached occupancy")
        setter(grid)
        return grid

    def clear(self) -> None:
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def cache_info(self) -> dict[str, int]:
        return {
            "entryCount": len(self._cache),
            "hitCount": self._hit_count,
            "missCount": self._miss_count,
        }

    def __len__(self) -> int:
        return len(self._cache)


def _validate_scene_id(expected_scene_id: str, outputs: Mapping[str, Any]) -> None:
    rendered_scene_id = outputs.get("sceneId")
    if rendered_scene_id is not None and str(rendered_scene_id) != str(expected_scene_id):
        raise ValueError(f"rendered observation sceneId {rendered_scene_id!r} does not match {expected_scene_id!r}")


def _quantize_tuple(values: tuple[float, ...], resolution: float) -> tuple[int, ...]:
    return tuple(int(round(float(value) / resolution)) for value in values)


def _positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized
