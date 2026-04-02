"""Tests for the query error-mapping experiment lab."""

from __future__ import annotations

from gs_sim2real.core.query_error_mapping import (
    QueryErrorMappingRequest,
    resolve_query_error_mapping,
)
from gs_sim2real.experiments.query_error_mapping_lab import (
    build_query_error_mapping_experiment_report,
)


def test_query_error_mapping_report_prefers_structured_codes() -> None:
    """Structured codes should best match the canonical transport fixtures."""
    report = build_query_error_mapping_experiment_report(repetitions=4)

    assert report["highlights"]["bestFit"]["policy"] == "structured_codes"
    assert report["highlights"]["fastestMedianRuntime"]["policy"] in {
        "literal_passthrough",
        "action_hint",
        "structured_codes",
    }


def test_resolve_query_error_mapping_preserves_queue_reason() -> None:
    """Structured mapping should keep the canonical message plus the queue reason."""
    decision = resolve_query_error_mapping(
        QueryErrorMappingRequest(
            event="queue_dropped",
            reason="evicted lower-priority queued work in favor of an interactive request",
            request_type="localization-image-benchmark",
            transport="ws",
        )
    )

    assert decision.error == (
        "query dropped from queue: evicted lower-priority queued work in favor of an interactive request"
    )
    assert decision.error_code == "query_queue_dropped"
