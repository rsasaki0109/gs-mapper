"""Tests for experiment-first sim2real websocket protocol import."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs
from gs_sim2real.experiments.sim2real_websocket_protocol_lab import (
    build_sim2real_websocket_protocol_experiment_report,
)


def test_sim2real_websocket_protocol_lab_report_compares_three_policies() -> None:
    report = build_sim2real_websocket_protocol_experiment_report(repetitions=4)

    assert report["type"] == "sim2real-websocket-protocol-experiment-report"
    assert len(report["fixtures"]) >= 5
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"strict_canonical", "envelope_first", "alias_friendly"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "alias_friendly"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_sim2real_websocket_protocol_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Sim2Real Websocket Protocol" in experiments_text
    assert "## Sim2Real Websocket Protocol" in decisions_text
    assert "importSim2realWebsocketMessage" in interfaces_text
