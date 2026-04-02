"""Tests for experiment-first query transport selection."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.query_transport_selection import (
    QueryTransportCapabilities,
    QueryTransportPreferences,
    QueryTransportRequest,
    select_query_transport,
)
from gs_sim2real.experiments.query_transport_selection_lab import (
    build_query_transport_selection_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_select_query_transport_balanced_disables_transport_for_publish_only_workloads() -> None:
    selection = select_query_transport(
        QueryTransportRequest(
            requested_transport="auto",
            pose_source="static",
            capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
            preferences=QueryTransportPreferences(enable_query_transport=False),
        )
    )

    assert selection.transport == "none"
    assert selection.endpoint == ""
    assert "does not request interactive queries" in selection.reason


def test_select_query_transport_balanced_prefers_browser_websocket_queries() -> None:
    selection = select_query_transport(
        QueryTransportRequest(
            requested_transport="auto",
            pose_source="query",
            capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
            preferences=QueryTransportPreferences(enable_query_transport=True, prefer_browser_clients=True),
        )
    )

    assert selection.transport == "ws"
    assert selection.endpoint == "ws://127.0.0.1:8781/sim2real"
    assert "browser-facing clients" in selection.reason


def test_select_query_transport_balanced_prefers_local_cli_zmq_queries() -> None:
    selection = select_query_transport(
        QueryTransportRequest(
            requested_transport="auto",
            pose_source="query",
            capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
            preferences=QueryTransportPreferences(enable_query_transport=True, prefer_local_cli=True),
        )
    )

    assert selection.transport == "zmq"
    assert selection.endpoint == "tcp://127.0.0.1:5588"
    assert "local query transport" in selection.reason


def test_query_transport_selection_lab_report_compares_three_policies() -> None:
    report = build_query_transport_selection_experiment_report(repetitions=4)

    assert report["type"] == "query-transport-selection-experiment-report"
    assert len(report["fixtures"]) >= 5
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"explicit_only", "balanced", "browser_first"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "balanced"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_query_transport_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Query Transport Selection" in experiments_text
    assert "## Query Transport Selection" in decisions_text
    assert "select_query_transport" in interfaces_text
