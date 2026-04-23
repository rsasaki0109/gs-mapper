"""Closed-loop route replanning helpers for Physical AI simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .interfaces import PhysicalAIEnvironment, Pose3D
from .planning import OccupancyPlanningContext
from .route_execution import RouteLike, RouteRollout, RouteStepOutcome, rollout_route
from .route_planning import RouteCandidate, RoutePlan, select_best_route


@dataclass(frozen=True, slots=True)
class RouteReplanResult:
    """One replan attempt produced after a blocked rollout step."""

    scene_id: str
    start_pose: Pose3D
    trigger: RouteStepOutcome
    candidates: tuple[RouteCandidate, ...]
    plan: RoutePlan
    rollout: RouteRollout | None = None

    @property
    def selected_route(self) -> RouteCandidate:
        return self.plan.selected.candidate

    @property
    def passed(self) -> bool:
        if self.rollout is not None:
            return self.rollout.passed
        return self.plan.selected.score.passed

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sceneId": self.scene_id,
            "startPose": self.start_pose.to_dict(),
            "trigger": self.trigger.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "plan": self.plan.to_dict(),
            "passed": self.passed,
        }
        if self.rollout is not None:
            payload["rollout"] = self.rollout.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ClosedLoopRouteRollout:
    """Route rollout with zero or more replan recovery attempts."""

    scene_id: str
    rollouts: tuple[RouteRollout, ...]
    replans: tuple[RouteReplanResult, ...]

    @property
    def final_rollout(self) -> RouteRollout:
        if not self.rollouts:
            raise ValueError("closed-loop rollout must contain at least one rollout")
        return self.rollouts[-1]

    @property
    def passed(self) -> bool:
        return self.final_rollout.passed

    def metrics(self) -> dict[str, float]:
        return {
            "rollout-count": float(len(self.rollouts)),
            "replan-count": float(len(self.replans)),
            "collision-count": float(sum(rollout.metrics()["collision-count"] for rollout in self.rollouts)),
            "planned-distance-meters": sum(rollout.metrics()["planned-distance-meters"] for rollout in self.rollouts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "passed": self.passed,
            "metrics": self.metrics(),
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
            "replans": [replan.to_dict() for replan in self.replans],
        }


def last_applied_route_pose(rollout: RouteRollout) -> Pose3D:
    """Return the pose to replan from after the latest applied route step."""

    last_applied_pose: Pose3D | None = None
    for outcome in rollout.outcomes:
        if outcome.applied and not outcome.collides:
            last_applied_pose = _state_pose_from_transition(outcome.transition) or outcome.step.target_pose
            continue
        return last_applied_pose or outcome.step.source_pose

    if last_applied_pose is not None:
        return last_applied_pose
    if rollout.steps:
        return rollout.steps[0].source_pose
    raise ValueError("rollout must contain at least one step to infer a route pose")


def reanchor_route_candidates(
    start_pose: Pose3D,
    candidates: Sequence[RouteCandidate],
) -> tuple[RouteCandidate, ...]:
    """Prepend the replan start pose to candidate continuations."""

    return tuple(
        RouteCandidate(
            route_id=candidate.route_id,
            trajectory=_reanchored_trajectory(start_pose, candidate.trajectory),
        )
        for candidate in candidates
    )


def replan_after_blocked_rollout(
    environment: PhysicalAIEnvironment,
    *,
    scene_id: str,
    rollout: RouteRollout,
    candidates: Sequence[RouteCandidate],
    planning_context: OccupancyPlanningContext | None = None,
    occupancy_pose: Pose3D | None = None,
    execute: bool = False,
    action_type: str = "teleport",
    segment_duration_seconds: float = 1.0,
    stop_on_collision: bool = True,
) -> RouteReplanResult:
    """Re-score alternative route continuations after a blocked rollout."""

    trigger = _blocked_outcome(rollout)
    start_pose = last_applied_route_pose(rollout)
    reanchored = reanchor_route_candidates(start_pose, candidates)
    plan = select_best_route(
        environment,
        scene_id=scene_id,
        candidates=reanchored,
        planning_context=planning_context,
        occupancy_pose=occupancy_pose or start_pose,
    )
    replan_rollout = None
    if execute:
        replan_rollout = rollout_route(
            environment,
            plan.selected,
            action_type=action_type,
            segment_duration_seconds=segment_duration_seconds,
            stop_on_collision=stop_on_collision,
        )
    return RouteReplanResult(
        scene_id=scene_id,
        start_pose=start_pose,
        trigger=trigger,
        candidates=reanchored,
        plan=plan,
        rollout=replan_rollout,
    )


def rollout_route_with_replanning(
    environment: PhysicalAIEnvironment,
    *,
    scene_id: str,
    initial_route: RouteLike,
    replan_candidate_batches: Sequence[Sequence[RouteCandidate]],
    planning_context: OccupancyPlanningContext | None = None,
    action_type: str = "teleport",
    segment_duration_seconds: float = 1.0,
    stop_on_collision: bool = True,
) -> ClosedLoopRouteRollout:
    """Roll out a route and retry with candidate batches when blocked."""

    current_rollout = rollout_route(
        environment,
        initial_route,
        action_type=action_type,
        segment_duration_seconds=segment_duration_seconds,
        stop_on_collision=stop_on_collision,
    )
    rollouts = [current_rollout]
    replans: list[RouteReplanResult] = []
    for candidates in replan_candidate_batches:
        if current_rollout.passed:
            break
        replan = replan_after_blocked_rollout(
            environment,
            scene_id=scene_id,
            rollout=current_rollout,
            candidates=candidates,
            planning_context=planning_context,
            execute=True,
            action_type=action_type,
            segment_duration_seconds=segment_duration_seconds,
            stop_on_collision=stop_on_collision,
        )
        replans.append(replan)
        if replan.rollout is None:
            break
        current_rollout = replan.rollout
        rollouts.append(current_rollout)

    return ClosedLoopRouteRollout(
        scene_id=scene_id,
        rollouts=tuple(rollouts),
        replans=tuple(replans),
    )


def _blocked_outcome(rollout: RouteRollout) -> RouteStepOutcome:
    for outcome in rollout.outcomes:
        if outcome.collides or not outcome.applied:
            return outcome
    raise ValueError("rollout does not contain a blocked step")


def _reanchored_trajectory(start_pose: Pose3D, trajectory: tuple[Pose3D, ...]) -> tuple[Pose3D, ...]:
    if trajectory and trajectory[0] == start_pose:
        return trajectory
    return (start_pose, *trajectory)


def _state_pose_from_transition(transition: Mapping[str, Any]) -> Pose3D | None:
    state = transition.get("state")
    if not isinstance(state, Mapping):
        return None
    pose = state.get("pose")
    if not isinstance(pose, Mapping):
        return None
    return _pose_from_payload(pose)


def _pose_from_payload(payload: Mapping[str, Any]) -> Pose3D:
    return Pose3D(
        position=_float_tuple(payload.get("position"), expected_size=3, field_name="position"),
        orientation_xyzw=_float_tuple(payload.get("orientationXyzw"), expected_size=4, field_name="orientationXyzw"),
        frame_id=str(payload.get("frameId", "world")),
        timestamp_seconds=_optional_float(payload.get("timestampSeconds")),
    )


def _float_tuple(value: Any, *, expected_size: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_size:
        raise ValueError(f"pose payload {field_name} must contain {expected_size} values")
    return tuple(float(component) for component in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
