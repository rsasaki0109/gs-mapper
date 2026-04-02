"""Tests for the query source-identity experiment lab."""

from __future__ import annotations

from gs_sim2real.core.query_source_identity import (
    QuerySourceIdentityRequest,
    resolve_query_source_identity,
)
from gs_sim2real.experiments.query_source_identity_lab import (
    build_query_source_identity_experiment_report,
)


def test_query_source_identity_report_prefers_remote_observable() -> None:
    """The remote-observable policy should win the canonical shared fixtures."""
    report = build_query_source_identity_experiment_report(repetitions=4)

    assert report["highlights"]["bestFit"]["policy"] == "remote_observable"
    assert report["highlights"]["fastestMedianRuntime"]["policy"] == "serial_only"


def test_resolve_query_source_identity_uses_client_hint_fallback() -> None:
    """Client hints should remain readable while preserving per-connection uniqueness."""
    identity = resolve_query_source_identity(
        QuerySourceIdentityRequest(
            transport="ws",
            connection_serial=5,
            client_hint="Replay Panel",
        )
    )

    assert identity.source_id == "ws-replay-panel-client-5"
