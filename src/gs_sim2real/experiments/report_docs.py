"""Repository-level experiment docs aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .process_docs import write_repo_experiment_docs


def write_repo_experiment_process_docs(
    *,
    docs_dir: str | Path,
    localization_alignment_report: dict[str, Any] | None = None,
    render_backend_report: dict[str, Any] | None = None,
    localization_import_report: dict[str, Any] | None = None,
    localization_review_bundle_import_report: dict[str, Any] | None = None,
    query_cancellation_policy_report: dict[str, Any] | None = None,
    query_coalescing_policy_report: dict[str, Any] | None = None,
    query_error_mapping_report: dict[str, Any] | None = None,
    query_source_identity_report: dict[str, Any] | None = None,
    query_transport_report: dict[str, Any] | None = None,
    query_request_import_report: dict[str, Any] | None = None,
    query_queue_policy_report: dict[str, Any] | None = None,
    query_timeout_policy_report: dict[str, Any] | None = None,
    query_response_build_report: dict[str, Any] | None = None,
    live_localization_stream_import_report: dict[str, Any] | None = None,
    route_capture_bundle_import_report: dict[str, Any] | None = None,
    sim2real_websocket_protocol_report: dict[str, Any] | None = None,
    outdoor_training_features_report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write shared repo experiment docs using all currently tracked experiment seams."""
    from .localization_alignment_lab import (
        build_localization_alignment_experiment_report,
        build_localization_alignment_process_section,
    )
    from .localization_estimate_import_lab import (
        build_localization_estimate_import_experiment_report,
        build_localization_estimate_import_process_section,
    )
    from .localization_review_bundle_import_lab import (
        build_localization_review_bundle_import_experiment_report,
        build_localization_review_bundle_import_process_section,
    )
    from .query_cancellation_policy_lab import (
        build_query_cancellation_policy_experiment_report,
        build_query_cancellation_policy_process_section,
    )
    from .query_coalescing_policy_lab import (
        build_query_coalescing_policy_experiment_report,
        build_query_coalescing_policy_process_section,
    )
    from .query_error_mapping_lab import (
        build_query_error_mapping_experiment_report,
        build_query_error_mapping_process_section,
    )
    from .outdoor_training_features_lab import (
        build_outdoor_training_features_experiment_report,
        build_outdoor_training_features_process_section,
    )
    from .render_backend_selection_lab import (
        build_render_backend_selection_experiment_report,
        build_render_backend_selection_process_section,
    )
    from .query_transport_selection_lab import (
        build_query_transport_selection_experiment_report,
        build_query_transport_selection_process_section,
    )
    from .query_request_import_lab import (
        build_query_request_import_experiment_report,
        build_query_request_import_process_section,
    )
    from .query_queue_policy_lab import (
        build_query_queue_policy_experiment_report,
        build_query_queue_policy_process_section,
    )
    from .query_source_identity_lab import (
        build_query_source_identity_experiment_report,
        build_query_source_identity_process_section,
    )
    from .query_timeout_policy_lab import (
        build_query_timeout_policy_experiment_report,
        build_query_timeout_policy_process_section,
    )
    from .query_response_build_lab import (
        build_query_response_build_experiment_report,
        build_query_response_build_process_section,
    )
    from .live_localization_stream_import_lab import (
        build_live_localization_stream_import_experiment_report,
        build_live_localization_stream_import_process_section,
    )
    from .route_capture_bundle_import_lab import (
        build_route_capture_bundle_import_experiment_report,
        build_route_capture_bundle_import_process_section,
    )
    from .sim2real_websocket_protocol_lab import (
        build_sim2real_websocket_protocol_experiment_report,
        build_sim2real_websocket_protocol_process_section,
    )

    alignment_report = localization_alignment_report or build_localization_alignment_experiment_report()
    backend_report = render_backend_report or build_render_backend_selection_experiment_report()
    import_report = localization_import_report or build_localization_estimate_import_experiment_report()
    review_bundle_import_report = (
        localization_review_bundle_import_report or build_localization_review_bundle_import_experiment_report()
    )
    cancellation_policy_report = query_cancellation_policy_report or build_query_cancellation_policy_experiment_report()
    coalescing_policy_report = query_coalescing_policy_report or build_query_coalescing_policy_experiment_report()
    error_mapping_report = query_error_mapping_report or build_query_error_mapping_experiment_report()
    source_identity_report = query_source_identity_report or build_query_source_identity_experiment_report()
    transport_report = query_transport_report or build_query_transport_selection_experiment_report()
    request_import_report = query_request_import_report or build_query_request_import_experiment_report()
    queue_policy_report = query_queue_policy_report or build_query_queue_policy_experiment_report()
    timeout_policy_report = query_timeout_policy_report or build_query_timeout_policy_experiment_report()
    response_build_report = query_response_build_report or build_query_response_build_experiment_report()
    live_stream_report = (
        live_localization_stream_import_report or build_live_localization_stream_import_experiment_report()
    )
    bundle_import_report = route_capture_bundle_import_report or build_route_capture_bundle_import_experiment_report()
    websocket_protocol_report = (
        sim2real_websocket_protocol_report or build_sim2real_websocket_protocol_experiment_report()
    )
    outdoor_features_report = outdoor_training_features_report or build_outdoor_training_features_experiment_report()
    sections = [
        build_localization_alignment_process_section(alignment_report),
        build_render_backend_selection_process_section(backend_report),
        build_outdoor_training_features_process_section(outdoor_features_report),
        build_localization_estimate_import_process_section(import_report),
        build_localization_review_bundle_import_process_section(review_bundle_import_report),
        build_query_cancellation_policy_process_section(cancellation_policy_report),
        build_query_coalescing_policy_process_section(coalescing_policy_report),
        build_query_error_mapping_process_section(error_mapping_report),
        build_query_transport_selection_process_section(transport_report),
        build_query_request_import_process_section(request_import_report),
        build_query_queue_policy_process_section(queue_policy_report),
        build_query_source_identity_process_section(source_identity_report),
        build_query_timeout_policy_process_section(timeout_policy_report),
        build_query_response_build_process_section(response_build_report),
        build_live_localization_stream_import_process_section(live_stream_report),
        build_route_capture_bundle_import_process_section(bundle_import_report),
        build_sim2real_websocket_protocol_process_section(websocket_protocol_report),
    ]
    return write_repo_experiment_docs(sections, docs_dir=docs_dir)
