"""Policy-facing feedback records for Physical AI route workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .route_execution import RouteRollout
from .route_planning import RouteEvaluation, RoutePlan
from .route_replanning import ClosedLoopRouteRollout, RouteReplanResult


RoutePolicySource = RoutePlan | RouteRollout | RouteReplanResult | ClosedLoopRouteRollout


@dataclass(frozen=True, slots=True)
class RoutePolicyObservation:
    """Compact numeric observation derived from route workflow records."""

    source_type: str
    source_id: str
    features: Mapping[str, float]
    events: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceType": self.source_type,
            "sourceId": self.source_id,
            "features": dict(self.features),
            "events": list(self.events),
        }


@dataclass(frozen=True, slots=True)
class RouteRewardWeights:
    """Default reward weights for route policy feedback."""

    success_reward: float = 1.0
    completion_reward: float = 0.5
    collision_penalty: float = -1.0
    distance_penalty_per_meter: float = -0.01
    step_penalty: float = -0.01
    replan_penalty: float = -0.1


@dataclass(frozen=True, slots=True)
class RoutePolicyReward:
    """Scalar reward plus weighted components."""

    reward: float
    components: Mapping[str, float]
    terminal: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "components": dict(self.components),
            "terminal": self.terminal,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class RoutePolicySample:
    """Observation/reward pair suitable for agent training or evaluation logs."""

    observation: RoutePolicyObservation
    reward: RoutePolicyReward

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
        }


def build_route_policy_observation(source: RoutePolicySource) -> RoutePolicyObservation:
    """Convert a route workflow record into compact numeric policy features."""

    if isinstance(source, RoutePlan):
        return _plan_observation(source)
    if isinstance(source, RouteRollout):
        return _rollout_observation(source)
    if isinstance(source, RouteReplanResult):
        return _replan_observation(source)
    if isinstance(source, ClosedLoopRouteRollout):
        return _closed_loop_observation(source)
    raise TypeError("unsupported route policy source")


def score_route_policy_reward(
    source: RoutePolicySource,
    *,
    weights: RouteRewardWeights | None = None,
) -> RoutePolicyReward:
    """Score a route workflow record with stable reward components."""

    reward_weights = weights or RouteRewardWeights()
    metrics = _reward_metrics(source)
    passed = bool(metrics["passed"])
    terminal = bool(metrics["terminal"])
    components = {
        "success": reward_weights.success_reward if passed else 0.0,
        "completion": reward_weights.completion_reward * metrics["completion-rate"],
        "collision": reward_weights.collision_penalty * metrics["collision-count"],
        "distance": reward_weights.distance_penalty_per_meter * metrics["planned-distance-meters"],
        "step": reward_weights.step_penalty * metrics["requested-step-count"],
        "replan": reward_weights.replan_penalty * metrics["replan-count"],
    }
    return RoutePolicyReward(
        reward=sum(components.values()),
        components=components,
        terminal=terminal,
        passed=passed,
    )


def build_route_policy_sample(
    source: RoutePolicySource,
    *,
    weights: RouteRewardWeights | None = None,
) -> RoutePolicySample:
    """Build one policy observation/reward pair from a route workflow record."""

    return RoutePolicySample(
        observation=build_route_policy_observation(source),
        reward=score_route_policy_reward(source, weights=weights),
    )


def build_route_policy_batch(
    sources: Sequence[RoutePolicySource],
    *,
    weights: RouteRewardWeights | None = None,
) -> tuple[RoutePolicySample, ...]:
    """Build policy samples for multiple route workflow records."""

    return tuple(build_route_policy_sample(source, weights=weights) for source in sources)


def _plan_observation(plan: RoutePlan) -> RoutePolicyObservation:
    selected = plan.selected
    second = plan.evaluations[1] if len(plan.evaluations) > 1 else None
    features = {
        "candidate-count": float(len(plan.evaluations)),
        "passed-candidate-count": float(sum(evaluation.score.passed for evaluation in plan.evaluations)),
        "selected-rank": float(selected.rank),
        "selected-original-index": float(selected.original_index),
        **_score_features("selected", selected),
        "selection-collision-rate-margin": _selection_margin(selected, second, "collision-rate"),
        "selection-path-length-margin": _selection_margin(selected, second, "path-length"),
    }
    return RoutePolicyObservation(
        source_type="route-plan",
        source_id=f"{plan.scene_id}:{selected.candidate.route_id}",
        features=_finite_features(features),
        events=_events("selected-route", selected.candidate.route_id, *selected.score.notes),
    )


def _rollout_observation(rollout: RouteRollout) -> RoutePolicyObservation:
    features = {
        **rollout.metrics(),
        "passed": _bool_feature(rollout.passed),
        "blocked-step-index": float(rollout.blocked_step_index if rollout.blocked_step_index is not None else -1),
        "action-type-teleport": _bool_feature(rollout.action_type == "teleport"),
        "action-type-twist": _bool_feature(rollout.action_type == "twist"),
    }
    return RoutePolicyObservation(
        source_type="route-rollout",
        source_id=rollout.route_id,
        features=_finite_features(features),
        events=("passed",) if rollout.passed else ("blocked",),
    )


def _replan_observation(replan: RouteReplanResult) -> RoutePolicyObservation:
    selected = replan.plan.selected
    features = {
        "candidate-count": float(len(replan.candidates)),
        "trigger-action-index": float(replan.trigger.step.action_index),
        "trigger-applied": _bool_feature(replan.trigger.applied),
        "trigger-collides": _bool_feature(replan.trigger.collides),
        "start-position-x": float(replan.start_pose.position[0]),
        "start-position-y": float(replan.start_pose.position[1]),
        "start-position-z": float(replan.start_pose.position[2]),
        **_score_features("selected", selected),
    }
    if replan.rollout is not None:
        features.update(_prefixed("rollout", replan.rollout.metrics()))
        features["rollout-passed"] = _bool_feature(replan.rollout.passed)
    return RoutePolicyObservation(
        source_type="route-replan",
        source_id=f"{replan.scene_id}:{selected.candidate.route_id}",
        features=_finite_features(features),
        events=_events("selected-route", selected.candidate.route_id, "executed" if replan.rollout else "planned"),
    )


def _closed_loop_observation(closed_loop: ClosedLoopRouteRollout) -> RoutePolicyObservation:
    final = closed_loop.final_rollout
    features = {
        **closed_loop.metrics(),
        "passed": _bool_feature(closed_loop.passed),
        **_prefixed("final", final.metrics()),
        "final-passed": _bool_feature(final.passed),
        "final-blocked-step-index": float(final.blocked_step_index if final.blocked_step_index is not None else -1),
    }
    return RoutePolicyObservation(
        source_type="closed-loop-route-rollout",
        source_id=f"{closed_loop.scene_id}:{final.route_id}",
        features=_finite_features(features),
        events=("passed",) if closed_loop.passed else ("blocked",),
    )


def _reward_metrics(source: RoutePolicySource) -> dict[str, float]:
    if isinstance(source, RoutePlan):
        selected = source.selected
        return {
            "passed": _bool_feature(selected.score.passed),
            "terminal": 0.0,
            "completion-rate": _bool_feature(selected.score.passed),
            "collision-count": _metric(selected.score.metrics, "collision-count", 0.0),
            "planned-distance-meters": _metric(selected.score.metrics, "path-length", 0.0),
            "requested-step-count": float(max(len(selected.candidate.trajectory) - 1, 0)),
            "replan-count": 0.0,
        }
    if isinstance(source, RouteRollout):
        metrics = source.metrics()
        return {
            "passed": _bool_feature(source.passed),
            "terminal": 1.0,
            "completion-rate": _metric(metrics, "applied-rate", 0.0),
            "collision-count": _metric(metrics, "collision-count", 0.0),
            "planned-distance-meters": _metric(metrics, "planned-distance-meters", 0.0),
            "requested-step-count": _metric(metrics, "requested-step-count", 0.0),
            "replan-count": 0.0,
        }
    if isinstance(source, RouteReplanResult):
        if source.rollout is not None:
            metrics = source.rollout.metrics()
            return {
                "passed": _bool_feature(source.rollout.passed),
                "terminal": 1.0,
                "completion-rate": _metric(metrics, "applied-rate", 0.0),
                "collision-count": _metric(metrics, "collision-count", 0.0),
                "planned-distance-meters": _metric(metrics, "planned-distance-meters", 0.0),
                "requested-step-count": _metric(metrics, "requested-step-count", 0.0),
                "replan-count": 1.0,
            }
        selected = source.plan.selected
        return {
            "passed": _bool_feature(selected.score.passed),
            "terminal": 0.0,
            "completion-rate": _bool_feature(selected.score.passed),
            "collision-count": _metric(selected.score.metrics, "collision-count", 0.0),
            "planned-distance-meters": _metric(selected.score.metrics, "path-length", 0.0),
            "requested-step-count": float(max(len(selected.candidate.trajectory) - 1, 0)),
            "replan-count": 1.0,
        }
    if isinstance(source, ClosedLoopRouteRollout):
        metrics = source.metrics()
        final_metrics = source.final_rollout.metrics()
        return {
            "passed": _bool_feature(source.passed),
            "terminal": 1.0,
            "completion-rate": _metric(final_metrics, "applied-rate", 0.0),
            "collision-count": _metric(metrics, "collision-count", 0.0),
            "planned-distance-meters": _metric(metrics, "planned-distance-meters", 0.0),
            "requested-step-count": sum(
                _metric(rollout.metrics(), "requested-step-count", 0.0) for rollout in source.rollouts
            ),
            "replan-count": _metric(metrics, "replan-count", 0.0),
        }
    raise TypeError("unsupported route policy source")


def _score_features(prefix: str, evaluation: RouteEvaluation) -> dict[str, float]:
    metrics = evaluation.score.metrics
    return {
        f"{prefix}-passed": _bool_feature(evaluation.score.passed),
        f"{prefix}-collision-rate": _metric(metrics, "collision-rate", 0.0),
        f"{prefix}-collision-count": _metric(metrics, "collision-count", 0.0),
        f"{prefix}-minimum-clearance-meters": _metric(metrics, "minimum-clearance-meters", 0.0),
        f"{prefix}-path-length": _metric(metrics, "path-length", 0.0),
    }


def _selection_margin(selected: RouteEvaluation, second: RouteEvaluation | None, key: str) -> float:
    if second is None:
        return 0.0
    return _metric(second.score.metrics, key, 0.0) - _metric(selected.score.metrics, key, 0.0)


def _metric(metrics: Mapping[str, float], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default
    if math.isfinite(value):
        return value
    return default


def _prefixed(prefix: str, features: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}-{key}": float(value) for key, value in features.items()}


def _finite_features(features: Mapping[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in sorted((str(key), float(value)) for key, value in features.items())
        if math.isfinite(value)
    }


def _events(label: str, value: str, *extra: str) -> tuple[str, ...]:
    return (f"{label}:{value}", *tuple(str(item) for item in extra if item))


def _bool_feature(value: bool) -> float:
    return 1.0 if value else 0.0
