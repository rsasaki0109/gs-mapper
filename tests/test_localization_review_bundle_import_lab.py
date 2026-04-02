"""Tests for experiment-first localization review bundle import."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.experiments.localization_review_bundle_import_lab import (
    build_localization_review_bundle_import_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_localization_review_bundle_import_lab_report_compares_three_policies() -> None:
    report = build_localization_review_bundle_import_experiment_report(repetitions=4)

    assert report["type"] == "localization-review-bundle-import-experiment-report"
    assert len(report["fixtures"]) >= 3
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"strict_canonical", "linked_capture_fallback", "alias_friendly"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "alias_friendly"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_localization_review_bundle_import_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Localization Review Bundle Import" in experiments_text
    assert "## Localization Review Bundle Import" in decisions_text
    assert "importLocalizationReviewBundleDocument" in interfaces_text
