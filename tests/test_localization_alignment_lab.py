"""Tests for the experiment-first localization alignment lab."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.localization_alignment import PoseSample, align_pose_samples
from gs_sim2real.experiments.localization_alignment_lab import (
    build_localization_alignment_experiment_report,
    write_localization_alignment_docs,
)


def _pose(
    index: int,
    *,
    x: float,
    yaw_degrees: float,
    timestamp_seconds: float | None,
) -> PoseSample:
    return PoseSample(
        index=index,
        label=f"pose:{index + 1}",
        position=(x, 0.0, 0.0),
        yaw_degrees=yaw_degrees,
        timestamp_seconds=timestamp_seconds,
    )


def test_align_pose_samples_auto_prefers_timestamp_interpolation() -> None:
    ground_truth = [
        _pose(0, x=0.0, yaw_degrees=0.0, timestamp_seconds=0.0),
        _pose(1, x=1.0, yaw_degrees=10.0, timestamp_seconds=1.0),
        _pose(2, x=2.0, yaw_degrees=20.0, timestamp_seconds=2.0),
    ]
    estimate = [
        _pose(0, x=0.0, yaw_degrees=0.0, timestamp_seconds=0.0),
        _pose(1, x=2.0, yaw_degrees=20.0, timestamp_seconds=2.0),
    ]

    resolved_alignment, pairs = align_pose_samples(ground_truth, estimate, alignment="auto")

    assert resolved_alignment == "timestamp"
    assert len(pairs) == 3
    assert pairs[1].interpolation_kind == "linear"
    assert pairs[1].estimate.position == (1.0, 0.0, 0.0)


def test_localization_alignment_lab_report_compares_three_strategies() -> None:
    report = build_localization_alignment_experiment_report(repetitions=4)

    assert report["type"] == "localization-alignment-experiment-report"
    assert len(report["fixtures"]) == 3
    assert len(report["strategies"]) >= 3
    strategy_names = {strategy["name"] for strategy in report["strategies"]}
    assert {"index", "timestamp_nearest", "timestamp"}.issubset(strategy_names)
    assert report["highlights"]["bestPositionError"]["strategy"] == "timestamp"
    assert report["highlights"]["mostReadable"]["strategy"] in strategy_names
    for strategy in report["strategies"]:
        assert len(strategy["fixtures"]) == len(report["fixtures"])
        assert strategy["readability"]["score"] >= 1.0
        assert strategy["extensibility"]["score"] >= 0.0


def test_localization_alignment_lab_writes_required_docs(tmp_path: Path) -> None:
    report = build_localization_alignment_experiment_report(repetitions=2)

    outputs = write_localization_alignment_docs(report, docs_dir=tmp_path)

    experiments_path = Path(outputs["experiments"])
    decisions_path = Path(outputs["decisions"])
    interfaces_path = Path(outputs["interfaces"])
    assert experiments_path.read_text(encoding="utf-8").startswith("# Experiments")
    assert decisions_path.read_text(encoding="utf-8").startswith("# Decisions")
    interfaces_text = interfaces_path.read_text(encoding="utf-8")
    assert interfaces_text.startswith("# Interfaces")
    assert "align_pose_samples" in interfaces_text
