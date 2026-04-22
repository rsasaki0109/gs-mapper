"""Costmap-style summaries for Physical AI trajectory scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math

from .interfaces import CollisionQuery


@dataclass(frozen=True, slots=True)
class TrajectoryCollisionSummary:
    """Aggregated collision and clearance information for a trajectory."""

    pose_count: int
    collision_count: int
    reason_counts: tuple[tuple[str, int], ...]
    clearance_values_meters: tuple[float, ...]

    @classmethod
    def from_queries(cls, queries: Sequence[CollisionQuery]) -> TrajectoryCollisionSummary:
        reason_counts = Counter(query.reason for query in queries)
        clearance_values = tuple(
            float(query.clearance_meters)
            for query in queries
            if query.clearance_meters is not None and math.isfinite(float(query.clearance_meters))
        )
        return cls(
            pose_count=len(queries),
            collision_count=sum(1 for query in queries if query.collides),
            reason_counts=tuple(sorted(reason_counts.items())),
            clearance_values_meters=clearance_values,
        )

    @property
    def collision_rate(self) -> float:
        if self.pose_count == 0:
            return 0.0
        return self.collision_count / self.pose_count

    @property
    def minimum_clearance_meters(self) -> float | None:
        if not self.clearance_values_meters:
            return None
        return min(self.clearance_values_meters)

    @property
    def mean_clearance_meters(self) -> float | None:
        if not self.clearance_values_meters:
            return None
        return sum(self.clearance_values_meters) / len(self.clearance_values_meters)

    def metric_payload(self) -> dict[str, float]:
        metrics = {
            "collision-rate": self.collision_rate,
            "collision-count": float(self.collision_count),
        }
        minimum_clearance = self.minimum_clearance_meters
        if minimum_clearance is not None:
            metrics["minimum-clearance-meters"] = minimum_clearance
        mean_clearance = self.mean_clearance_meters
        if mean_clearance is not None:
            metrics["mean-clearance-meters"] = mean_clearance
        return metrics

    def notes(self) -> tuple[str, ...]:
        return tuple(f"collision-reason:{reason}={count}" for reason, count in self.reason_counts)


def summarize_collision_queries(queries: Sequence[CollisionQuery]) -> TrajectoryCollisionSummary:
    """Summarize collision queries for score metrics and diagnostic notes."""

    return TrajectoryCollisionSummary.from_queries(queries)
