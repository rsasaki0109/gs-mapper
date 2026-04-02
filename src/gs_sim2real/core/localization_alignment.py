"""Stable pose-alignment interfaces for localization evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class PoseSample:
    """Normalized pose sample used for localization matching."""

    index: int
    label: str
    position: tuple[float, float, float]
    yaw_degrees: float
    timestamp_seconds: float | None
    response: dict[str, Any] | None = None
    relative_timestamp_seconds: float | None = None


@dataclass(frozen=True)
class AlignmentPair:
    """One matched ground-truth / estimate pair."""

    pair_index: int
    ground_truth: PoseSample
    estimate: PoseSample
    time_delta_seconds: float | None
    interpolation_kind: str


class PoseAlignmentStrategy(Protocol):
    """Minimal interface for interchangeable pose alignment experiments."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def align(
        self,
        ground_truth_poses: Sequence[PoseSample],
        estimate_poses: Sequence[PoseSample],
    ) -> list[AlignmentPair]:
        """Return comparable alignment pairs for one fixture."""


def normalize_signed_degrees(value: float) -> float:
    """Normalize an angle to the ``[-180, 180]`` range."""
    normalized = math.fmod(float(value), 360.0)
    if normalized <= -180.0:
        return normalized + 360.0
    if normalized > 180.0:
        return normalized - 360.0
    return normalized


def interpolate_linear(a: float, b: float, t: float) -> float:
    """Linear interpolation helper."""
    return float(a) + (float(b) - float(a)) * float(t)


def interpolate_angle_degrees(a: float, b: float, t: float) -> float:
    """Interpolate angles while respecting wraparound."""
    return normalize_signed_degrees(float(a) + normalize_signed_degrees(float(b) - float(a)) * float(t))


def build_relative_timestamp_timeline(poses: Sequence[PoseSample]) -> list[PoseSample]:
    """Build a relative timestamp timeline anchored at the first pose."""
    timed = sorted(
        (pose for pose in poses if pose.timestamp_seconds is not None),
        key=lambda pose: (float(pose.timestamp_seconds), pose.index),
    )
    if not timed:
        return []

    base_timestamp = float(timed[0].timestamp_seconds)
    return [
        PoseSample(
            index=pose.index,
            label=pose.label,
            position=pose.position,
            yaw_degrees=pose.yaw_degrees,
            timestamp_seconds=pose.timestamp_seconds,
            response=pose.response,
            relative_timestamp_seconds=float(pose.timestamp_seconds) - base_timestamp,
        )
        for pose in timed
    ]


def build_interpolated_pose(
    lower_pose: PoseSample,
    upper_pose: PoseSample,
    target_relative_timestamp_seconds: float,
    interpolation_factor: float,
) -> PoseSample:
    """Linearly interpolate position and yaw between two poses."""
    return PoseSample(
        index=lower_pose.index,
        label=f"interp:{lower_pose.index + 1}-{upper_pose.index + 1}",
        position=(
            interpolate_linear(lower_pose.position[0], upper_pose.position[0], interpolation_factor),
            interpolate_linear(lower_pose.position[1], upper_pose.position[1], interpolation_factor),
            interpolate_linear(lower_pose.position[2], upper_pose.position[2], interpolation_factor),
        ),
        yaw_degrees=interpolate_angle_degrees(lower_pose.yaw_degrees, upper_pose.yaw_degrees, interpolation_factor),
        timestamp_seconds=interpolate_linear(
            lower_pose.timestamp_seconds or 0.0,
            upper_pose.timestamp_seconds or 0.0,
            interpolation_factor,
        ),
        response=None,
        relative_timestamp_seconds=target_relative_timestamp_seconds,
    )


def build_index_aligned_pairs(
    ground_truth_poses: Sequence[PoseSample],
    estimate_poses: Sequence[PoseSample],
) -> list[AlignmentPair]:
    """Match poses by index order."""
    matched_count = min(len(ground_truth_poses), len(estimate_poses))
    pairs: list[AlignmentPair] = []
    for pair_index in range(matched_count):
        ground_truth = ground_truth_poses[pair_index]
        estimate = estimate_poses[pair_index]
        time_delta_seconds = None
        if ground_truth.timestamp_seconds is not None and estimate.timestamp_seconds is not None:
            time_delta_seconds = abs(float(estimate.timestamp_seconds) - float(ground_truth.timestamp_seconds))
        pairs.append(
            AlignmentPair(
                pair_index=pair_index,
                ground_truth=ground_truth,
                estimate=estimate,
                time_delta_seconds=time_delta_seconds,
                interpolation_kind="index",
            )
        )
    return pairs


def build_timestamp_aligned_pairs(
    ground_truth_poses: Sequence[PoseSample],
    estimate_poses: Sequence[PoseSample],
) -> list[AlignmentPair]:
    """Match poses by timestamp with linear interpolation."""
    ground_truth_timeline = build_relative_timestamp_timeline(ground_truth_poses)
    estimate_timeline = build_relative_timestamp_timeline(estimate_poses)
    if not ground_truth_timeline or not estimate_timeline:
        raise ValueError("timestamp alignment requires timestamps in both ground truth and estimate")

    pairs: list[AlignmentPair] = []
    estimate_cursor = 0
    for ground_truth_pose in ground_truth_timeline:
        target = float(ground_truth_pose.relative_timestamp_seconds or 0.0)
        if len(estimate_timeline) == 1:
            estimate_pose = estimate_timeline[0]
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=estimate_pose,
                    time_delta_seconds=abs(float(estimate_pose.relative_timestamp_seconds or 0.0) - target),
                    interpolation_kind="single-sample",
                )
            )
            continue

        while (
            estimate_cursor + 1 < len(estimate_timeline)
            and float(estimate_timeline[estimate_cursor + 1].relative_timestamp_seconds or 0.0) < target
        ):
            estimate_cursor += 1

        first_estimate = estimate_timeline[0]
        last_estimate = estimate_timeline[-1]
        if abs(float(first_estimate.relative_timestamp_seconds or 0.0) - target) < 1e-4:
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=first_estimate,
                    time_delta_seconds=0.0,
                    interpolation_kind="exact",
                )
            )
            continue
        if abs(float(last_estimate.relative_timestamp_seconds or 0.0) - target) < 1e-4:
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=last_estimate,
                    time_delta_seconds=0.0,
                    interpolation_kind="exact",
                )
            )
            continue
        if target < float(first_estimate.relative_timestamp_seconds or 0.0):
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=first_estimate,
                    time_delta_seconds=abs(float(first_estimate.relative_timestamp_seconds or 0.0) - target),
                    interpolation_kind="clamped-start",
                )
            )
            continue
        if target > float(last_estimate.relative_timestamp_seconds or 0.0):
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=last_estimate,
                    time_delta_seconds=abs(float(last_estimate.relative_timestamp_seconds or 0.0) - target),
                    interpolation_kind="clamped-end",
                )
            )
            continue

        lower_estimate = estimate_timeline[estimate_cursor]
        upper_estimate = estimate_timeline[estimate_cursor + 1]
        lower_time = float(lower_estimate.relative_timestamp_seconds or 0.0)
        upper_time = float(upper_estimate.relative_timestamp_seconds or 0.0)

        if abs(lower_time - target) < 1e-4:
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=lower_estimate,
                    time_delta_seconds=0.0,
                    interpolation_kind="exact",
                )
            )
            continue
        if abs(upper_time - target) < 1e-4:
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=upper_estimate,
                    time_delta_seconds=0.0,
                    interpolation_kind="exact",
                )
            )
            continue

        interpolation_factor = (target - lower_time) / max(1e-6, upper_time - lower_time)
        pairs.append(
            AlignmentPair(
                pair_index=len(pairs),
                ground_truth=ground_truth_pose,
                estimate=build_interpolated_pose(lower_estimate, upper_estimate, target, interpolation_factor),
                time_delta_seconds=0.0,
                interpolation_kind="linear",
            )
        )

    return pairs


class IndexAlignmentStrategy:
    """Stable index-ordered alignment."""

    name = "index"
    label = "Sequential Index"
    style = "zip-sequential"
    tier = "core"
    capabilities = {
        "worksWithoutTimestamps": True,
        "usesTimestamps": False,
        "supportsInterpolation": False,
        "clampsOutOfRange": False,
    }

    def align(
        self,
        ground_truth_poses: Sequence[PoseSample],
        estimate_poses: Sequence[PoseSample],
    ) -> list[AlignmentPair]:
        return build_index_aligned_pairs(ground_truth_poses, estimate_poses)


class TimestampLinearInterpolationStrategy:
    """Stable timestamp-aware alignment with interpolation."""

    name = "timestamp"
    label = "Timeline Interpolation"
    style = "timeline-interpolated"
    tier = "core"
    capabilities = {
        "worksWithoutTimestamps": False,
        "usesTimestamps": True,
        "supportsInterpolation": True,
        "clampsOutOfRange": True,
    }

    def align(
        self,
        ground_truth_poses: Sequence[PoseSample],
        estimate_poses: Sequence[PoseSample],
    ) -> list[AlignmentPair]:
        return build_timestamp_aligned_pairs(ground_truth_poses, estimate_poses)


CORE_ALIGNMENT_STRATEGIES: dict[str, PoseAlignmentStrategy] = {
    "index": IndexAlignmentStrategy(),
    "timestamp": TimestampLinearInterpolationStrategy(),
}


def resolve_alignment_name(
    requested_alignment: str,
    ground_truth_poses: Sequence[PoseSample],
    estimate_poses: Sequence[PoseSample],
) -> str:
    """Resolve ``auto`` into the current stable alignment mode."""
    normalized = str(requested_alignment or "auto").strip() or "auto"
    if normalized != "auto":
        if normalized not in CORE_ALIGNMENT_STRATEGIES:
            raise ValueError(
                f"unsupported alignment strategy: {normalized}. "
                f"Expected one of {', '.join(sorted(['auto', *CORE_ALIGNMENT_STRATEGIES]))}"
            )
        return normalized

    ground_truth_timestamped_count = sum(pose.timestamp_seconds is not None for pose in ground_truth_poses)
    estimate_timestamped_count = sum(pose.timestamp_seconds is not None for pose in estimate_poses)
    return "timestamp" if ground_truth_timestamped_count > 0 and estimate_timestamped_count > 0 else "index"


def align_pose_samples(
    ground_truth_poses: Sequence[PoseSample],
    estimate_poses: Sequence[PoseSample],
    *,
    alignment: str = "auto",
) -> tuple[str, list[AlignmentPair]]:
    """Resolve a stable alignment name and execute the matching strategy."""
    resolved_alignment = resolve_alignment_name(alignment, ground_truth_poses, estimate_poses)
    return resolved_alignment, CORE_ALIGNMENT_STRATEGIES[resolved_alignment].align(ground_truth_poses, estimate_poses)
