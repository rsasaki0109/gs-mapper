"""Dynamic obstacles for route policy scenario benchmarks.

The headless Physical AI environment ships with a static occupancy grid
baked in at scene load time. That is fine for coarse benchmarks but
collapses the gap between trivial direct-goal policies and policies that
have to react to the world. This module introduces a small, JSON-backed
timeline of waypointed obstacles that the environment consults during
collision queries — each obstacle's position is linearly interpolated
between the bracketing waypoints indexed by the environment's internal
``step_index``.

Design notes:

* Only library API + JSON IO. The environment integration lives in
  ``headless.py`` and threading through scenario spec / matrix lives in
  the scenario-CI modules.
* Each obstacle carries a list of ``(step_index, position)`` waypoints
  sorted strictly ascending. Queries before the first waypoint clamp to
  the first, queries after the last clamp to the last, and queries in
  between linearly interpolate component-wise.
* A ``radius_meters`` footprint turns the obstacle into a sphere in world
  coordinates. Collision tests compare the Euclidean distance between the
  query pose and the interpolated centre against the radius.
* Zero waypoints are rejected at construction time. One-waypoint
  obstacles are stationary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


ROUTE_POLICY_DYNAMIC_OBSTACLE_VERSION = "gs-mapper-route-policy-dynamic-obstacle/v1"
ROUTE_POLICY_DYNAMIC_OBSTACLE_TIMELINE_VERSION = "gs-mapper-route-policy-dynamic-obstacle-timeline/v1"


@dataclass(frozen=True, slots=True)
class DynamicObstacleWaypoint:
    """One waypoint on a dynamic obstacle's trajectory."""

    step_index: int
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if int(self.step_index) < 0:
            raise ValueError("step_index must be non-negative")
        if not isinstance(self.position, tuple) or len(self.position) != 3:
            raise ValueError("position must be a 3-tuple of floats")
        if not all(math.isfinite(float(component)) for component in self.position):
            raise ValueError("position components must be finite")
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "position", tuple(float(component) for component in self.position))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-dynamic-obstacle-waypoint",
            "stepIndex": self.step_index,
            "position": list(self.position),
        }


@dataclass(frozen=True, slots=True)
class DynamicObstacle:
    """A moving-sphere obstacle anchored to the env's internal step index.

    By default the obstacle follows its ``waypoints`` via linear interpolation.
    When ``chase_target_agent`` is ``True``, the obstacle ignores later
    waypoints and instead walks from ``waypoints[0].position`` toward the
    queried agent position at up to ``chase_speed_m_per_step`` metres per
    step, clamped at the agent once the two meet. The chase path is a pure
    function of the current agent position and the step index, so replays
    stay deterministic: no agent-pose history is retained.
    """

    obstacle_id: str
    waypoints: tuple[DynamicObstacleWaypoint, ...]
    radius_meters: float
    chase_target_agent: bool = False
    chase_speed_m_per_step: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_DYNAMIC_OBSTACLE_VERSION

    def __post_init__(self) -> None:
        if not str(self.obstacle_id):
            raise ValueError("obstacle_id must not be empty")
        if not self.waypoints:
            raise ValueError("dynamic obstacle must contain at least one waypoint")
        radius = float(self.radius_meters)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_meters must be positive and finite")
        object.__setattr__(self, "radius_meters", radius)
        chase_speed = float(self.chase_speed_m_per_step)
        if not math.isfinite(chase_speed) or chase_speed < 0.0:
            raise ValueError("chase_speed_m_per_step must be non-negative and finite")
        object.__setattr__(self, "chase_speed_m_per_step", chase_speed)
        sorted_waypoints = tuple(sorted(self.waypoints, key=lambda point: point.step_index))
        step_indices = tuple(waypoint.step_index for waypoint in sorted_waypoints)
        if len(set(step_indices)) != len(step_indices):
            raise ValueError("dynamic obstacle waypoints must have unique step indices")
        object.__setattr__(self, "waypoints", sorted_waypoints)

    @property
    def waypoint_count(self) -> int:
        return len(self.waypoints)

    def position_at_step(
        self,
        step_index: int,
        *,
        agent_position: Sequence[float] | None = None,
    ) -> tuple[float, float, float]:
        """Return the interpolated world-frame position at ``step_index``.

        When ``chase_target_agent`` is set and ``agent_position`` is provided
        the obstacle ignores later waypoints and walks from ``waypoints[0]``
        toward the agent, capped at ``chase_speed_m_per_step * max(0,
        step_index)`` metres of travel. Without ``agent_position`` the chase
        obstacle stays pinned at its first waypoint so renderers that do not
        know the agent pose still see a stable position.
        """

        if self.chase_target_agent:
            return self._chase_position(step_index, agent_position)
        return self._waypoint_position(step_index)

    def contains(
        self,
        pose_position: Sequence[float],
        step_index: int,
    ) -> bool:
        """Return True when ``pose_position`` is inside the obstacle sphere at ``step_index``."""

        centre = self.position_at_step(step_index, agent_position=pose_position)
        return math.dist(tuple(float(c) for c in pose_position), centre) <= self.radius_meters

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-dynamic-obstacle",
            "version": self.version,
            "obstacleId": self.obstacle_id,
            "radiusMeters": self.radius_meters,
            "chaseTargetAgent": bool(self.chase_target_agent),
            "chaseSpeedMPerStep": float(self.chase_speed_m_per_step),
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
            "metadata": _json_mapping(self.metadata),
        }

    def _waypoint_position(self, step_index: int) -> tuple[float, float, float]:
        waypoints = self.waypoints
        if step_index <= waypoints[0].step_index:
            return waypoints[0].position
        if step_index >= waypoints[-1].step_index:
            return waypoints[-1].position
        # Find the bracketing waypoints via linear scan (waypoint lists are
        # typically small — interpolation in tight loops can migrate to
        # bisect if that changes).
        for previous, current in zip(waypoints, waypoints[1:]):
            if previous.step_index <= step_index <= current.step_index:
                span = current.step_index - previous.step_index
                if span <= 0:
                    return current.position
                alpha = (step_index - previous.step_index) / span
                return (
                    previous.position[0] + alpha * (current.position[0] - previous.position[0]),
                    previous.position[1] + alpha * (current.position[1] - previous.position[1]),
                    previous.position[2] + alpha * (current.position[2] - previous.position[2]),
                )
        return waypoints[-1].position

    def _chase_position(
        self,
        step_index: int,
        agent_position: Sequence[float] | None,
    ) -> tuple[float, float, float]:
        start = self.waypoints[0].position
        if agent_position is None:
            return start
        target = tuple(float(c) for c in agent_position)
        delta_x = target[0] - start[0]
        delta_y = target[1] - start[1]
        delta_z = target[2] - start[2]
        gap = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
        if gap <= 0.0:
            return start
        max_travel = max(0.0, float(step_index)) * self.chase_speed_m_per_step
        distance = min(gap, max_travel)
        fraction = 0.0 if gap <= 0.0 else distance / gap
        return (
            start[0] + fraction * delta_x,
            start[1] + fraction * delta_y,
            start[2] + fraction * delta_z,
        )


@dataclass(frozen=True, slots=True)
class DynamicObstacleTimeline:
    """Ordered collection of dynamic obstacles sharing a single timeline."""

    timeline_id: str
    obstacles: tuple[DynamicObstacle, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_DYNAMIC_OBSTACLE_TIMELINE_VERSION

    def __post_init__(self) -> None:
        if not str(self.timeline_id):
            raise ValueError("timeline_id must not be empty")
        obstacle_ids = tuple(obstacle.obstacle_id for obstacle in self.obstacles)
        if len(set(obstacle_ids)) != len(obstacle_ids):
            raise ValueError("timeline must not contain duplicate obstacle ids")

    @property
    def obstacle_count(self) -> int:
        return len(self.obstacles)

    def blocking_obstacle(
        self,
        pose_position: Sequence[float],
        step_index: int,
    ) -> DynamicObstacle | None:
        """Return the first obstacle that contains ``pose_position`` at ``step_index``."""

        for obstacle in self.obstacles:
            if obstacle.contains(pose_position, step_index):
                return obstacle
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-dynamic-obstacle-timeline",
            "version": self.version,
            "timelineId": self.timeline_id,
            "obstacleCount": self.obstacle_count,
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "metadata": _json_mapping(self.metadata),
        }


def write_route_policy_dynamic_obstacle_timeline_json(
    path: str | Path,
    timeline: DynamicObstacleTimeline,
) -> Path:
    """Persist a dynamic obstacle timeline as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_dynamic_obstacle_timeline_json(path: str | Path) -> DynamicObstacleTimeline:
    """Load a dynamic obstacle timeline JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_dynamic_obstacle_timeline_from_dict(_mapping(payload, "dynamicObstacleTimeline"))


def route_policy_dynamic_obstacle_waypoint_from_dict(
    payload: Mapping[str, Any],
) -> DynamicObstacleWaypoint:
    """Rebuild a waypoint from JSON."""

    _record_type(payload, "route-policy-dynamic-obstacle-waypoint")
    position = payload.get("position")
    if not isinstance(position, Sequence) or len(position) != 3:
        raise ValueError("waypoint position must be a 3-element sequence")
    return DynamicObstacleWaypoint(
        step_index=int(payload["stepIndex"]),
        position=(float(position[0]), float(position[1]), float(position[2])),
    )


def route_policy_dynamic_obstacle_from_dict(payload: Mapping[str, Any]) -> DynamicObstacle:
    """Rebuild a single dynamic obstacle from JSON."""

    _record_type(payload, "route-policy-dynamic-obstacle")
    version = str(payload.get("version", ROUTE_POLICY_DYNAMIC_OBSTACLE_VERSION))
    if version != ROUTE_POLICY_DYNAMIC_OBSTACLE_VERSION:
        raise ValueError(f"unsupported dynamic obstacle version: {version}")
    waypoints = tuple(
        route_policy_dynamic_obstacle_waypoint_from_dict(_mapping(item, "waypoint"))
        for item in _sequence(payload.get("waypoints", ()), "waypoints")
    )
    return DynamicObstacle(
        obstacle_id=str(payload["obstacleId"]),
        waypoints=waypoints,
        radius_meters=float(payload["radiusMeters"]),
        chase_target_agent=bool(payload.get("chaseTargetAgent", False)),
        chase_speed_m_per_step=float(payload.get("chaseSpeedMPerStep", 0.0)),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def route_policy_dynamic_obstacle_timeline_from_dict(
    payload: Mapping[str, Any],
) -> DynamicObstacleTimeline:
    """Rebuild a dynamic obstacle timeline from JSON."""

    _record_type(payload, "route-policy-dynamic-obstacle-timeline")
    version = str(payload.get("version", ROUTE_POLICY_DYNAMIC_OBSTACLE_TIMELINE_VERSION))
    if version != ROUTE_POLICY_DYNAMIC_OBSTACLE_TIMELINE_VERSION:
        raise ValueError(f"unsupported dynamic obstacle timeline version: {version}")
    obstacles = tuple(
        route_policy_dynamic_obstacle_from_dict(_mapping(item, "obstacle"))
        for item in _sequence(payload.get("obstacles", ()), "obstacles")
    )
    expected_count = payload.get("obstacleCount")
    if expected_count is not None and int(expected_count) != len(obstacles):
        raise ValueError("obstacleCount does not match loaded obstacles")
    return DynamicObstacleTimeline(
        timeline_id=str(payload["timelineId"]),
        obstacles=obstacles,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_dynamic_obstacle_timeline_markdown(timeline: DynamicObstacleTimeline) -> str:
    """Render a compact Markdown summary of a dynamic obstacle timeline."""

    lines = [
        f"# Route Policy Dynamic Obstacle Timeline: {timeline.timeline_id}",
        f"- Obstacles: {timeline.obstacle_count}",
        "",
        "| Obstacle | Radius (m) | Waypoints | First step | Last step |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for obstacle in timeline.obstacles:
        first = obstacle.waypoints[0].step_index
        last = obstacle.waypoints[-1].step_index
        lines.append(
            f"| {obstacle.obstacle_id} | {obstacle.radius_meters} | {obstacle.waypoint_count} | {first} | {last} |"
        )
    return "\n".join(lines) + "\n"


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")
