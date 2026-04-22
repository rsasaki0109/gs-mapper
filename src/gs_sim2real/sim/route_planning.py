"""Route-level planning helpers for Physical AI simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

from .interfaces import PhysicalAIEnvironment, Pose3D, TrajectoryScore
from .planning import OccupancyPlanningContext


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    """One candidate route represented as a trajectory."""

    route_id: str
    trajectory: tuple[Pose3D, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "routeId": self.route_id,
            "trajectory": [pose.to_dict() for pose in self.trajectory],
        }


@dataclass(frozen=True, slots=True)
class RouteEvaluation:
    """Score and ranking metadata for one route candidate."""

    candidate: RouteCandidate
    score: TrajectoryScore
    rank: int
    original_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "originalIndex": self.original_index,
            "candidate": self.candidate.to_dict(),
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Ranked route-planning result."""

    scene_id: str
    evaluations: tuple[RouteEvaluation, ...]
    selected: RouteEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "selectedRouteId": self.selected.candidate.route_id,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


def select_best_route(
    environment: PhysicalAIEnvironment,
    *,
    scene_id: str,
    candidates: Sequence[RouteCandidate],
    planning_context: OccupancyPlanningContext | None = None,
    occupancy_pose: Pose3D | None = None,
) -> RoutePlan:
    """Score candidate routes and return them ranked from best to worst."""

    if not candidates:
        raise ValueError("candidates must contain at least one route")

    scored: list[tuple[int, RouteCandidate, TrajectoryScore]] = []
    for index, candidate in enumerate(candidates):
        if planning_context is not None:
            render_pose = occupancy_pose or _first_pose(candidate.trajectory)
            if render_pose is not None:
                planning_context.set_environment_occupancy(environment, scene_id=scene_id, pose=render_pose)
        score = environment.score_trajectory(scene_id, candidate.trajectory)
        scored.append((index, candidate, score))

    ranked = sorted(scored, key=lambda item: _route_sort_key(item[2], item[0]))
    evaluations = tuple(
        RouteEvaluation(candidate=candidate, score=score, rank=rank, original_index=index)
        for rank, (index, candidate, score) in enumerate(ranked, start=1)
    )
    return RoutePlan(scene_id=scene_id, evaluations=evaluations, selected=evaluations[0])


def _route_sort_key(score: TrajectoryScore, original_index: int) -> tuple[float, float, float, float, float, int]:
    metrics = score.metrics
    collision_rate = _metric(metrics, "collision-rate", 1.0)
    collision_count = _metric(metrics, "collision-count", math.inf)
    minimum_clearance = _metric(metrics, "minimum-clearance-meters", -math.inf)
    path_length = _metric(metrics, "path-length", math.inf)
    return (
        0.0 if score.passed else 1.0,
        collision_rate,
        collision_count,
        -minimum_clearance,
        path_length,
        original_index,
    )


def _metric(metrics: Any, key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default
    if math.isfinite(value):
        return value
    return default


def _first_pose(trajectory: tuple[Pose3D, ...]) -> Pose3D | None:
    return trajectory[0] if trajectory else None
