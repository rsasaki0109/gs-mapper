"""Tests for experiment-first query cancellation policy."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.experiments.query_cancellation_policy_lab import (
    build_query_cancellation_policy_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_query_cancellation_policy_lab_report_compares_three_policies() -> None:
    report = build_query_cancellation_policy_experiment_report(repetitions=4)

    assert report["type"] == "query-cancellation-policy-experiment-report"
    assert len(report["fixtures"]) >= 3
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"ignore_orphaned", "cancel_requested_only", "cancel_source_backlog"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "cancel_source_backlog"


def test_repo_experiment_docs_include_query_cancellation_policy_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Query Cancellation Policy" in experiments_text
    assert "## Query Cancellation Policy" in decisions_text
    assert "resolve_query_cancellation_decision" in interfaces_text
