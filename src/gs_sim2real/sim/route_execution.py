"""Route execution helpers for Physical AI simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .interfaces import AgentAction, PhysicalAIEnvironment, Pose3D
from .route_planning import RouteCandidate, RouteEvaluation, RoutePlan


RouteLike = RouteCandidate | RouteEvaluation | RoutePlan


@dataclass(frozen=True, slots=True)
class RouteActionStep:
    """One executable action for a segment of a route."""

    route_id: str
    action_index: int
    source_pose: Pose3D
    target_pose: Pose3D
    action: AgentAction
    distance_meters: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "routeId": self.route_id,
            "actionIndex": self.action_index,
            "sourcePose": self.source_pose.to_dict(),
            "targetPose": self.target_pose.to_dict(),
            "action": self.action.to_dict(),
            "distanceMeters": self.distance_meters,
        }


@dataclass(frozen=True, slots=True)
class RouteStepOutcome:
    """Environment transition captured after executing one route action."""

    step: RouteActionStep
    transition: Mapping[str, Any]
    applied: bool
    collides: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.to_dict(),
            "transition": dict(self.transition),
            "applied": self.applied,
            "collides": self.collides,
        }


@dataclass(frozen=True, slots=True)
class RouteRollout:
    """Executed route rollout with per-step collision outcomes."""

    route_id: str
    action_type: str
    steps: tuple[RouteActionStep, ...]
    outcomes: tuple[RouteStepOutcome, ...]

    @property
    def passed(self) -> bool:
        return len(self.outcomes) == len(self.steps) and all(
            outcome.applied and not outcome.collides for outcome in self.outcomes
        )

    @property
    def blocked_step_index(self) -> int | None:
        for outcome in self.outcomes:
            if outcome.collides or not outcome.applied:
                return outcome.step.action_index
        return None

    def metrics(self) -> dict[str, float]:
        requested = len(self.steps)
        executed = len(self.outcomes)
        applied = sum(1 for outcome in self.outcomes if outcome.applied)
        collisions = sum(1 for outcome in self.outcomes if outcome.collides)
        distance = sum(step.distance_meters for step in self.steps)
        return {
            "requested-step-count": float(requested),
            "executed-step-count": float(executed),
            "applied-step-count": float(applied),
            "collision-count": float(collisions),
            "completion-rate": _safe_rate(executed, requested),
            "applied-rate": _safe_rate(applied, requested),
            "planned-distance-meters": distance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "routeId": self.route_id,
            "actionType": self.action_type,
            "passed": self.passed,
            "blockedStepIndex": self.blocked_step_index,
            "metrics": self.metrics(),
            "steps": [step.to_dict() for step in self.steps],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def build_route_actions(
    route: RouteLike,
    *,
    action_type: str = "teleport",
    segment_duration_seconds: float = 1.0,
) -> tuple[RouteActionStep, ...]:
    """Convert a route into executable environment actions."""

    candidate = _route_candidate(route)
    trajectory = candidate.trajectory
    if not trajectory:
        raise ValueError("route trajectory must contain at least one pose")

    normalized_action_type = _normalize_action_type(action_type)
    duration = _positive_float(segment_duration_seconds, "segment_duration_seconds")
    return tuple(
        RouteActionStep(
            route_id=candidate.route_id,
            action_index=index - 1,
            source_pose=source,
            target_pose=target,
            action=_action_for_segment(
                source,
                target,
                action_type=normalized_action_type,
                duration_seconds=duration,
            ),
            distance_meters=_pose_distance(source, target),
        )
        for index, (source, target) in enumerate(zip(trajectory, trajectory[1:]), start=1)
    )


def rollout_route(
    environment: PhysicalAIEnvironment,
    route: RouteLike,
    *,
    action_type: str = "teleport",
    segment_duration_seconds: float = 1.0,
    stop_on_collision: bool = True,
) -> RouteRollout:
    """Execute route actions in an environment and capture per-step outcomes."""

    steps = build_route_actions(
        route,
        action_type=action_type,
        segment_duration_seconds=segment_duration_seconds,
    )
    normalized_action_type = _normalize_action_type(action_type)
    outcomes: list[RouteStepOutcome] = []
    for step in steps:
        transition = environment.step(step.action)
        outcome = RouteStepOutcome(
            step=step,
            transition=transition,
            applied=_transition_applied(transition),
            collides=_transition_collides(transition),
        )
        outcomes.append(outcome)
        if stop_on_collision and (outcome.collides or not outcome.applied):
            break
    return RouteRollout(
        route_id=_route_candidate(route).route_id,
        action_type=normalized_action_type,
        steps=steps,
        outcomes=tuple(outcomes),
    )


def _route_candidate(route: RouteLike) -> RouteCandidate:
    if isinstance(route, RouteCandidate):
        return route
    if isinstance(route, RouteEvaluation):
        return route.candidate
    if isinstance(route, RoutePlan):
        return route.selected.candidate
    raise TypeError("route must be a RouteCandidate, RouteEvaluation, or RoutePlan")


def _action_for_segment(
    source: Pose3D,
    target: Pose3D,
    *,
    action_type: str,
    duration_seconds: float,
) -> AgentAction:
    if action_type == "teleport":
        qx, qy, qz, qw = target.orientation_xyzw
        x, y, z = target.position
        return AgentAction(
            "teleport",
            {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw},
            duration_seconds=duration_seconds,
        )

    if action_type == "twist":
        sx, sy, sz = source.position
        tx, ty, tz = target.position
        return AgentAction(
            "twist",
            {
                "linearX": (tx - sx) / duration_seconds,
                "linearY": (ty - sy) / duration_seconds,
                "linearZ": (tz - sz) / duration_seconds,
            },
            duration_seconds=duration_seconds,
        )

    raise ValueError(f"unsupported route action type: {action_type}")


def _normalize_action_type(action_type: str) -> str:
    normalized = str(action_type).strip().lower()
    if normalized not in {"teleport", "twist"}:
        raise ValueError(f"unsupported route action type: {action_type}")
    return normalized


def _transition_collides(transition: Mapping[str, Any]) -> bool:
    collision = transition.get("collision")
    if isinstance(collision, Mapping):
        return bool(collision.get("collides", False))
    return False


def _transition_applied(transition: Mapping[str, Any]) -> bool:
    if "applied" in transition:
        return bool(transition["applied"])
    return not _transition_collides(transition)


def _pose_distance(source: Pose3D, target: Pose3D) -> float:
    return math.dist(source.position, target.position)


def _positive_float(value: float, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator
