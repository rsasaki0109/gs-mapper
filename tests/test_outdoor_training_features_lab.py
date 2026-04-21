"""Tests for outdoor training feature-bundle comparison."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.experiments.outdoor_training_features_lab import (
    build_outdoor_training_features_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_outdoor_training_features_lab_compares_depth_appearance_pose_and_sky() -> None:
    report = build_outdoor_training_features_experiment_report()

    assert report["type"] == "outdoor-training-features-experiment-report"
    assert len(report["fixtures"]) >= 4
    profile_names = {profile["name"] for profile in report["profiles"]}
    assert {
        "rgb_only",
        "depth_supervised",
        "depth_appearance",
        "depth_appearance_pose",
        "sky_masked_full",
    }.issubset(profile_names)
    assert report["highlights"]["bestOverall"]["profile"] == "depth_appearance_pose"
    assert report["highlights"]["bestSkyFixture"]["profile"] == "sky_masked_full"


def test_sky_masked_profile_is_penalized_when_masks_are_missing() -> None:
    report = build_outdoor_training_features_experiment_report()
    sky_profile = next(profile for profile in report["profiles"] if profile["name"] == "sky_masked_full")
    autoware_fixture = next(
        fixture for fixture in sky_profile["fixtures"] if fixture["fixtureId"] == "autoware-fused-multibag"
    )

    assert autoware_fixture["status"] == "missing-inputs"
    assert autoware_fixture["missingInputs"] == ["sky_masks"]


def test_repo_experiment_docs_include_outdoor_training_features_section(tmp_path: Path) -> None:
    report = build_outdoor_training_features_experiment_report()

    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path, outdoor_training_features_report=report)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    detail_text = Path(outputs["experiments_detail"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Outdoor Training Features" in experiments_text
    assert "Depth + Appearance + Pose" in detail_text
    assert "Sky masking remains experimental" in decisions_text
    assert "depth_loss_weight" in interfaces_text
