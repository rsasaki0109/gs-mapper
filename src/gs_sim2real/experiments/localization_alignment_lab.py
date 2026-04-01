"""Experiment-first lab for localization trajectory alignment."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

from gs_sim2real.core.localization_alignment import (
    AlignmentPair,
    CORE_ALIGNMENT_STRATEGIES,
    PoseAlignmentStrategy,
    PoseSample,
    build_relative_timestamp_timeline,
    normalize_signed_degrees,
)


@dataclass(frozen=True)
class AlignmentFixture:
    """Canonical input shared by every alignment experiment."""

    fixture_id: str
    label: str
    intent: str
    ground_truth: tuple[PoseSample, ...]
    estimate: tuple[PoseSample, ...]


class CoreIndexAlignmentExperiment:
    """Expose the stable index aligner to the experiment harness."""

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
        return CORE_ALIGNMENT_STRATEGIES["index"].align(ground_truth_poses, estimate_poses)


class GreedyNearestTimestampAlignmentExperiment:
    """Greedy timestamp aligner that snaps to the nearest estimate sample."""

    name = "timestamp_nearest"
    label = "Greedy Nearest Timestamp"
    style = "cursor-nearest"
    tier = "experiment"
    capabilities = {
        "worksWithoutTimestamps": False,
        "usesTimestamps": True,
        "supportsInterpolation": False,
        "clampsOutOfRange": True,
    }

    def align(
        self,
        ground_truth_poses: Sequence[PoseSample],
        estimate_poses: Sequence[PoseSample],
    ) -> list[AlignmentPair]:
        ground_truth_timeline = build_relative_timestamp_timeline(ground_truth_poses)
        estimate_timeline = build_relative_timestamp_timeline(estimate_poses)
        if not ground_truth_timeline or not estimate_timeline:
            raise ValueError("timestamp_nearest requires timestamps in both ground truth and estimate")

        pairs: list[AlignmentPair] = []
        for ground_truth_pose in ground_truth_timeline:
            target = float(ground_truth_pose.relative_timestamp_seconds or 0.0)
            estimate_pose = min(
                estimate_timeline,
                key=lambda pose: (
                    abs(float(pose.relative_timestamp_seconds or 0.0) - target),
                    pose.index,
                ),
            )
            pairs.append(
                AlignmentPair(
                    pair_index=len(pairs),
                    ground_truth=ground_truth_pose,
                    estimate=estimate_pose,
                    time_delta_seconds=abs(float(estimate_pose.relative_timestamp_seconds or 0.0) - target),
                    interpolation_kind="nearest",
                )
            )
        return pairs


class CoreTimelineInterpolationExperiment:
    """Expose the stable timestamp-interpolated aligner to the experiment harness."""

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
        return CORE_ALIGNMENT_STRATEGIES["timestamp"].align(ground_truth_poses, estimate_poses)


EXPERIMENT_ALIGNMENT_STRATEGIES: tuple[PoseAlignmentStrategy, ...] = (
    CoreIndexAlignmentExperiment(),
    GreedyNearestTimestampAlignmentExperiment(),
    CoreTimelineInterpolationExperiment(),
)


def _pose(
    index: int,
    label: str,
    *,
    position: tuple[float, float, float],
    yaw_degrees: float,
    timestamp_seconds: float | None,
) -> PoseSample:
    return PoseSample(
        index=index,
        label=label,
        position=position,
        yaw_degrees=yaw_degrees,
        timestamp_seconds=timestamp_seconds,
    )


def build_alignment_fixtures() -> list[AlignmentFixture]:
    """Build the canonical fixtures shared across all alignment strategies."""
    return [
        AlignmentFixture(
            fixture_id="ordered-index",
            label="Ordered Poses Without Timestamps",
            intent="Keep a zero-assumption path for logs that only preserve capture order.",
            ground_truth=(
                _pose(0, "gt:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=None),
                _pose(1, "gt:2", position=(1.0, 0.0, 0.0), yaw_degrees=10.0, timestamp_seconds=None),
                _pose(2, "gt:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=None),
            ),
            estimate=(
                _pose(0, "est:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=None),
                _pose(1, "est:2", position=(1.0, 0.0, 0.0), yaw_degrees=10.0, timestamp_seconds=None),
                _pose(2, "est:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=None),
            ),
        ),
        AlignmentFixture(
            fixture_id="reordered-timestamp",
            label="Reordered Timestamped Trajectory",
            intent="Penalize implementations that over-trust array order when timestamps are available.",
            ground_truth=(
                _pose(0, "gt:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=0.0),
                _pose(1, "gt:2", position=(1.0, 0.0, 0.0), yaw_degrees=10.0, timestamp_seconds=1.0),
                _pose(2, "gt:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=2.0),
            ),
            estimate=(
                _pose(0, "est:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=2.0),
                _pose(1, "est:2", position=(1.0, 0.0, 0.0), yaw_degrees=10.0, timestamp_seconds=1.0),
                _pose(2, "est:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=0.0),
            ),
        ),
        AlignmentFixture(
            fixture_id="sparse-timestamp",
            label="Sparse Timestamped Trajectory",
            intent="Expose whether an aligner can bridge missing estimate samples without discarding the middle frame.",
            ground_truth=(
                _pose(0, "gt:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=0.0),
                _pose(1, "gt:2", position=(1.0, 0.0, 0.0), yaw_degrees=10.0, timestamp_seconds=1.0),
                _pose(2, "gt:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=2.0),
            ),
            estimate=(
                _pose(0, "est:1", position=(0.0, 0.0, 0.0), yaw_degrees=0.0, timestamp_seconds=0.0),
                _pose(1, "est:3", position=(2.0, 0.0, 0.0), yaw_degrees=20.0, timestamp_seconds=2.0),
            ),
        ),
    ]


def compute_position_error_meters(a: PoseSample, b: PoseSample) -> float:
    """Euclidean distance between two positions."""
    return math.sqrt(sum((float(a.position[i]) - float(b.position[i])) ** 2 for i in range(3)))


def compute_yaw_error_degrees(a: PoseSample, b: PoseSample) -> float:
    """Absolute wrapped yaw error between two poses."""
    return abs(normalize_signed_degrees(float(b.yaw_degrees) - float(a.yaw_degrees)))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_alignment_fixture(
    strategy: PoseAlignmentStrategy,
    fixture: AlignmentFixture,
) -> dict[str, Any]:
    """Run one strategy on one canonical fixture."""
    started_at = perf_counter()
    try:
        pairs = strategy.align(fixture.ground_truth, fixture.estimate)
    except Exception as exc:
        return {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "status": "error",
            "error": str(exc),
            "runtimeMs": float((perf_counter() - started_at) * 1000.0),
        }

    runtime_ms = float((perf_counter() - started_at) * 1000.0)
    position_errors = [compute_position_error_meters(pair.ground_truth, pair.estimate) for pair in pairs]
    yaw_errors = [compute_yaw_error_degrees(pair.ground_truth, pair.estimate) for pair in pairs]
    time_deltas = [pair.time_delta_seconds for pair in pairs]
    interpolation_kinds = [pair.interpolation_kind for pair in pairs]
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "matchedCount": len(pairs),
        "groundTruthCount": len(fixture.ground_truth),
        "estimateCount": len(fixture.estimate),
        "coverage": float(len(pairs) / max(1, len(fixture.ground_truth))),
        "meanPositionErrorMeters": _mean_or_none(position_errors),
        "meanYawErrorDegrees": _mean_or_none(yaw_errors),
        "meanTimeDeltaSeconds": _mean_or_none(time_deltas),
        "interpolatedCount": sum(kind == "linear" for kind in interpolation_kinds),
        "clampedCount": sum(kind.startswith("clamped") for kind in interpolation_kinds),
        "runtimeMs": runtime_ms,
    }


def benchmark_strategy_runtime(
    strategy: PoseAlignmentStrategy,
    fixtures: Sequence[AlignmentFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure strategy runtime on the shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                strategy.align(fixture.ground_truth, fixture.estimate)
            except Exception:
                continue
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))

    if not samples_ms:
        return {"repetitions": int(repetitions), "sampleCount": 0, "meanMs": None, "medianMs": None}

    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(strategy: PoseAlignmentStrategy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(strategy.align))
    tree = ast.parse(source)
    branch_count = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in ast.walk(tree))
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", '"""', "'''", "#"))
    ]
    lines_of_code = len(lines)
    score = max(1.0, 10.0 - max(0, lines_of_code - 8) * 0.2 - max(0, branch_count - 2) * 0.8)
    notes: list[str] = []
    if lines_of_code <= 12:
        notes.append("short enough to read in one screen")
    else:
        notes.append("multiple screens or helper lookups are needed")
    if branch_count >= 4:
        notes.append("branch-heavy behavior needs fixture-level tests")
    else:
        notes.append("control flow stays shallow")
    return {
        "score": round(score, 1),
        "linesOfCode": lines_of_code,
        "branchCount": branch_count,
        "notes": notes,
    }


def evaluate_extensibility(strategy: PoseAlignmentStrategy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "worksWithoutTimestamps": 2.0,
        "usesTimestamps": 2.0,
        "supportsInterpolation": 3.0,
        "clampsOutOfRange": 3.0,
    }
    supported = [key for key, enabled in strategy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if strategy.capabilities.get(key))
    notes: list[str] = []
    if strategy.capabilities.get("worksWithoutTimestamps"):
        notes.append("can survive order-only datasets")
    if strategy.capabilities.get("usesTimestamps"):
        notes.append("understands explicit time instead of array order")
    if strategy.capabilities.get("supportsInterpolation"):
        notes.append("bridges sparse estimates without dropping frames")
    if strategy.capabilities.get("clampsOutOfRange"):
        notes.append("defines behavior outside the estimate time range")
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
        "notes": notes,
    }


def summarize_strategy(
    strategy: PoseAlignmentStrategy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one strategy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    success_rate = float(len(successful) / max(1, len(fixture_reports)))
    aggregate = {
        "successRate": success_rate,
        "meanCoverage": _mean_or_none([report.get("coverage") for report in successful]),
        "meanPositionErrorMeters": _mean_or_none([report.get("meanPositionErrorMeters") for report in successful]),
        "meanYawErrorDegrees": _mean_or_none([report.get("meanYawErrorDegrees") for report in successful]),
        "meanTimeDeltaSeconds": _mean_or_none([report.get("meanTimeDeltaSeconds") for report in successful]),
        "interpolatedCount": int(sum(int(report.get("interpolatedCount", 0)) for report in successful)),
        "failedFixtures": [report["fixtureId"] for report in fixture_reports if report["status"] != "ok"],
    }
    return {
        "name": strategy.name,
        "label": strategy.label,
        "style": strategy.style,
        "tier": strategy.tier,
        "capabilities": dict(strategy.capabilities),
        "fixtures": list(fixture_reports),
        "aggregate": aggregate,
        "runtime": runtime_report,
        "readability": evaluate_readability(strategy),
        "extensibility": evaluate_extensibility(strategy),
    }


def build_localization_alignment_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare multiple alignment strategies on shared fixtures."""
    fixtures = build_alignment_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "groundTruthCount": len(fixture.ground_truth),
            "estimateCount": len(fixture.estimate),
            "timestampedGroundTruthCount": sum(pose.timestamp_seconds is not None for pose in fixture.ground_truth),
            "timestampedEstimateCount": sum(pose.timestamp_seconds is not None for pose in fixture.estimate),
        }
        for fixture in fixtures
    ]
    strategy_reports = []
    for strategy in EXPERIMENT_ALIGNMENT_STRATEGIES:
        fixture_reports = [evaluate_alignment_fixture(strategy, fixture) for fixture in fixtures]
        runtime_report = benchmark_strategy_runtime(strategy, fixtures, repetitions=repetitions)
        strategy_reports.append(summarize_strategy(strategy, fixture_reports, runtime_report))

    best_position = min(
        (report for report in strategy_reports if report["aggregate"]["meanPositionErrorMeters"] is not None),
        key=lambda report: (
            float(report["aggregate"]["meanPositionErrorMeters"]),
            -float(report["aggregate"]["successRate"]),
        ),
    )
    fastest = min(
        (report for report in strategy_reports if report["runtime"].get("medianMs") is not None),
        key=lambda report: float(report["runtime"]["medianMs"]),
    )
    most_readable = max(strategy_reports, key=lambda report: float(report["readability"]["score"]))
    most_extensible = max(strategy_reports, key=lambda report: float(report["extensibility"]["score"]))

    return {
        "protocol": "gs-sim2real-experiment-report/v1",
        "type": "localization-alignment-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "localization-trajectory-alignment",
            "statement": (
                "Align ground-truth route captures and estimated localization poses without committing to a "
                "single universal algorithm too early."
            ),
            "stableInterface": "align_pose_samples(ground_truth_poses, estimate_poses, *, alignment='auto')",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": [
                "successRate",
                "meanCoverage",
                "meanPositionErrorMeters",
                "meanYawErrorDegrees",
                "meanTimeDeltaSeconds",
                "interpolatedCount",
            ],
            "runtime": ["meanMs", "medianMs"],
            "readability": ["score", "linesOfCode", "branchCount"],
            "extensibility": ["score", "supportedCapabilities"],
            "heuristicNotice": "Readability/extensibility scores are generated heuristics, not objective truth.",
        },
        "strategies": strategy_reports,
        "highlights": {
            "bestPositionError": {
                "strategy": best_position["name"],
                "label": best_position["label"],
                "meanPositionErrorMeters": best_position["aggregate"]["meanPositionErrorMeters"],
            },
            "fastestMedianRuntime": {
                "strategy": fastest["name"],
                "label": fastest["label"],
                "medianMs": fastest["runtime"]["medianMs"],
            },
            "mostReadable": {
                "strategy": most_readable["name"],
                "label": most_readable["label"],
                "score": most_readable["readability"]["score"],
            },
            "mostExtensible": {
                "strategy": most_extensible["name"],
                "label": most_extensible["label"],
                "score": most_extensible["extensibility"]["score"],
            },
        },
    }


def _format_metric(value: Any, *, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def build_localization_alignment_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the alignment report into a shared docs section."""
    comparison_rows = []
    for strategy in report["strategies"]:
        aggregate = strategy["aggregate"]
        runtime = strategy["runtime"]
        readability = strategy["readability"]
        extensibility = strategy["extensibility"]
        comparison_rows.append(
            [
                strategy["label"],
                strategy["tier"],
                strategy["style"],
                _format_metric(aggregate["successRate"], digits=2),
                _format_metric(aggregate["meanCoverage"], digits=2),
                _format_metric(aggregate["meanPositionErrorMeters"]),
                _format_metric(aggregate["meanYawErrorDegrees"]),
                _format_metric(aggregate["meanTimeDeltaSeconds"]),
                _format_metric(runtime["medianMs"]),
                _format_metric(readability["score"], digits=1),
                _format_metric(extensibility["score"], digits=1),
            ]
        )

    fixture_sections = []
    for fixture in report["fixtures"]:
        rows = []
        for strategy in report["strategies"]:
            fixture_report = next(item for item in strategy["fixtures"] if item["fixtureId"] == fixture["fixtureId"])
            rows.append(
                [
                    strategy["label"],
                    fixture_report["status"],
                    str(fixture_report.get("matchedCount", "n/a")),
                    _format_metric(fixture_report.get("coverage"), digits=2),
                    _format_metric(fixture_report.get("meanPositionErrorMeters")),
                    str(fixture_report.get("interpolatedCount", "n/a")),
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Strategy", "Status", "Matched", "Coverage", "Pos Err (m)", "Interpolation"],
                "rows": rows,
            }
        )

    return {
        "title": "Localization Alignment",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Strategy",
            "Tier",
            "Style",
            "Success",
            "Coverage",
            "Pos Err (m)",
            "Yaw Err (deg)",
            "Time Δ (s)",
            "Runtime (ms)",
            "Readability",
            "Extensibility",
        ],
        "comparisonRows": comparison_rows,
        "fixtureSections": fixture_sections,
        "highlights": [
            f"Best spatial fidelity: `{report['highlights']['bestPositionError']['label']}`",
            f"Fastest median runtime: `{report['highlights']['fastestMedianRuntime']['label']}`",
            f"Most readable implementation: `{report['highlights']['mostReadable']['label']}`",
            f"Broadest extension surface: `{report['highlights']['mostExtensible']['label']}`",
        ],
        "accepted": [
            "Stable code uses `gs_sim2real.core.localization_alignment.align_pose_samples()` as the only contract that production callers should depend on.",
            "`auto` keeps two stable behaviors instead of one universal algorithm: `index` for order-only logs and `timestamp` for timestamped logs.",
            "New alignment ideas must land in `src/gs_sim2real/experiments/` first and only graduate after they outperform current core behavior on shared fixtures.",
        ],
        "deferred": [
            "`timestamp_nearest` stays experimental. It is fast and simple, but it loses the middle frame on sparse trajectories where interpolation clearly wins.",
            "We are not freezing a larger abstract interface yet. The current core keeps only `PoseSample`, `AlignmentPair`, and `align_pose_samples(...)`.",
        ],
        "rules": [
            "Start with at least three concrete strategies for any new alignment problem.",
            "Keep inputs and metrics identical across strategies before discussing architecture.",
            "Promote only the minimum surface that survived comparison; delete or quarantine the rest.",
        ],
        "stableInterfaceIntro": "The stable localization-alignment surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class PoseSample:
                index: int
                label: str
                position: tuple[float, float, float]
                yaw_degrees: float
                timestamp_seconds: float | None
                response: dict[str, Any] | None = None
                relative_timestamp_seconds: float | None = None

            @dataclass(frozen=True)
            class AlignmentPair:
                pair_index: int
                ground_truth: PoseSample
                estimate: PoseSample
                time_delta_seconds: float | None
                interpolation_kind: str

            def align_pose_samples(
                ground_truth_poses: Sequence[PoseSample],
                estimate_poses: Sequence[PoseSample],
                *,
                alignment: str = 'auto',
            ) -> tuple[str, list[AlignmentPair]]: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`align(ground_truth_poses, estimate_poses) -> list[AlignmentPair]`",
        ],
        "comparableInputs": [
            "Same `PoseSample` arrays for every strategy",
            "Same canonical fixtures (`ordered-index`, `reordered-timestamp`, `sparse-timestamp`)",
            "Same evaluation axes: quality, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable, minimal, dependency surface for production code",
            "`src/gs_sim2real/experiments/`: discardable strategies and comparison harnesses",
        ],
    }


def build_experiments_markdown(report: dict[str, Any]) -> str:
    """Render the latest experiment comparison as Markdown."""
    lines = [
        "# Experiments",
        "",
        f"Updated: {report['createdAt']}",
        "",
        "This repository treats localization alignment as an exploration space, not a single final design.",
        "Every strategy below consumed the same fixtures, the same `PoseSample` shape, and the same quality/runtime metrics.",
        "",
        "## Problem",
        "",
        report["problem"]["statement"],
        "",
        "## Current Comparison",
        "",
        "| Strategy | Tier | Style | Success | Coverage | Pos Err (m) | Yaw Err (deg) | Time Δ (s) | Runtime (ms) | Readability | Extensibility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strategy in report["strategies"]:
        aggregate = strategy["aggregate"]
        runtime = strategy["runtime"]
        readability = strategy["readability"]
        extensibility = strategy["extensibility"]
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy["label"],
                    strategy["tier"],
                    strategy["style"],
                    _format_metric(aggregate["successRate"], digits=2),
                    _format_metric(aggregate["meanCoverage"], digits=2),
                    _format_metric(aggregate["meanPositionErrorMeters"]),
                    _format_metric(aggregate["meanYawErrorDegrees"]),
                    _format_metric(aggregate["meanTimeDeltaSeconds"]),
                    _format_metric(runtime["medianMs"]),
                    _format_metric(readability["score"], digits=1),
                    _format_metric(extensibility["score"], digits=1),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Fixture Notes",
            "",
        ]
    )
    for fixture in report["fixtures"]:
        lines.extend(
            [
                f"### {fixture['label']}",
                "",
                fixture["intent"],
                "",
                "| Strategy | Status | Matched | Coverage | Pos Err (m) | Interpolation |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for strategy in report["strategies"]:
            fixture_report = next(item for item in strategy["fixtures"] if item["fixtureId"] == fixture["fixtureId"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        strategy["label"],
                        fixture_report["status"],
                        str(fixture_report.get("matchedCount", "n/a")),
                        _format_metric(fixture_report.get("coverage"), digits=2),
                        _format_metric(fixture_report.get("meanPositionErrorMeters")),
                        str(fixture_report.get("interpolatedCount", "n/a")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Highlights",
            "",
            f"- Best spatial fidelity: `{report['highlights']['bestPositionError']['label']}`",
            f"- Fastest median runtime: `{report['highlights']['fastestMedianRuntime']['label']}`",
            f"- Most readable implementation: `{report['highlights']['mostReadable']['label']}`",
            f"- Broadest extension surface: `{report['highlights']['mostExtensible']['label']}`",
            "",
            "Readability/extensibility are heuristic scores generated from source shape and declared capability surface.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_decisions_markdown(report: dict[str, Any]) -> str:
    """Render the current convergence decisions as Markdown."""
    return (
        "\n".join(
            [
                "# Decisions",
                "",
                f"Updated: {report['createdAt']}",
                "",
                "## Accepted",
                "",
                "- Stable code uses `gs_sim2real.core.localization_alignment.align_pose_samples()` as the only contract that production callers should depend on.",
                "- `auto` keeps two stable behaviors instead of one universal algorithm: `index` for order-only logs and `timestamp` for timestamped logs.",
                "- New alignment ideas must land in `src/gs_sim2real/experiments/` first and only graduate after they outperform current core behavior on shared fixtures.",
                "",
                "## Deferred",
                "",
                "- `timestamp_nearest` stays experimental. It is fast and simple, but it loses the middle frame on sparse trajectories where interpolation clearly wins.",
                "- We are not freezing a larger abstract interface yet. The current core keeps only `PoseSample`, `AlignmentPair`, and `align_pose_samples(...)`.",
                "",
                "## Operating Rules",
                "",
                "1. Start with at least three concrete strategies for any new alignment problem.",
                "2. Keep inputs and metrics identical across strategies before discussing architecture.",
                "3. Promote only the minimum surface that survived comparison; delete or quarantine the rest.",
            ]
        )
        + "\n"
    )


def build_interfaces_markdown() -> str:
    """Render the current minimal stable interface as Markdown."""
    return (
        "\n".join(
            [
                "# Interfaces",
                "",
                "## Stable Core",
                "",
                "The stable localization-alignment surface is intentionally small:",
                "",
                "```python",
                "@dataclass(frozen=True)",
                "class PoseSample:",
                "    index: int",
                "    label: str",
                "    position: tuple[float, float, float]",
                "    yaw_degrees: float",
                "    timestamp_seconds: float | None",
                "    response: dict[str, Any] | None = None",
                "    relative_timestamp_seconds: float | None = None",
                "",
                "@dataclass(frozen=True)",
                "class AlignmentPair:",
                "    pair_index: int",
                "    ground_truth: PoseSample",
                "    estimate: PoseSample",
                "    time_delta_seconds: float | None",
                "    interpolation_kind: str",
                "",
                "def align_pose_samples(",
                "    ground_truth_poses: Sequence[PoseSample],",
                "    estimate_poses: Sequence[PoseSample],",
                "    *,",
                "    alignment: str = 'auto',",
                ") -> tuple[str, list[AlignmentPair]]: ...",
                "```",
                "",
                "## Experiment Contract",
                "",
                "Every experiment strategy must expose the same shape:",
                "",
                "- `name`, `label`, `style`, `tier`, `capabilities`",
                "- `align(ground_truth_poses, estimate_poses) -> list[AlignmentPair]`",
                "",
                "## Comparable Inputs",
                "",
                "- Same `PoseSample` arrays for every strategy",
                "- Same canonical fixtures (`ordered-index`, `reordered-timestamp`, `sparse-timestamp`)",
                "- Same evaluation axes: quality, runtime, readability heuristic, extensibility heuristic",
                "",
                "## Boundary",
                "",
                "- `src/gs_sim2real/core/`: stable, minimal, dependency surface for production code",
                "- `src/gs_sim2real/experiments/`: discardable strategies and comparison harnesses",
            ]
        )
        + "\n"
    )


def write_localization_alignment_docs(
    report: dict[str, Any],
    *,
    docs_dir: str | Path,
) -> dict[str, str]:
    """Write the experiment-process documents required by the repository."""
    from .report_docs import write_repo_experiment_process_docs

    return write_repo_experiment_process_docs(
        docs_dir=docs_dir,
        localization_alignment_report=report,
    )


def run_cli(args: argparse.Namespace) -> None:
    """Run the localization alignment lab and optionally refresh docs."""
    report = build_localization_alignment_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        docs = write_localization_alignment_docs(report, docs_dir=args.docs_dir)
    summary = {
        "type": report["type"],
        "strategyCount": len(report["strategies"]),
        "fixtureCount": len(report["fixtures"]),
        "bestPositionError": report["highlights"]["bestPositionError"],
        "fastestMedianRuntime": report["highlights"]["fastestMedianRuntime"],
        "docs": docs,
    }
    print(json.dumps(summary, indent=2))
