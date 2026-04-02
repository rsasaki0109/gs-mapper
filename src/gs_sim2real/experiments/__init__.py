"""Experimental workflows that are expected to be compared and discarded."""

from .localization_alignment_lab import (
    EXPERIMENT_ALIGNMENT_STRATEGIES,
    build_localization_alignment_experiment_report,
    build_localization_alignment_process_section,
    write_localization_alignment_docs,
)
from .localization_estimate_import_lab import (
    EXPERIMENT_LOCALIZATION_ESTIMATE_IMPORT_POLICIES,
    build_localization_estimate_import_experiment_report,
    build_localization_estimate_import_process_section,
)
from .localization_review_bundle_import_lab import (
    build_localization_review_bundle_import_experiment_report,
    build_localization_review_bundle_import_process_section,
)
from .query_cancellation_policy_lab import (
    EXPERIMENT_QUERY_CANCELLATION_POLICIES,
    build_query_cancellation_policy_experiment_report,
    build_query_cancellation_policy_process_section,
)
from .query_coalescing_policy_lab import (
    EXPERIMENT_QUERY_COALESCING_POLICIES,
    build_query_coalescing_policy_experiment_report,
    build_query_coalescing_policy_process_section,
)
from .query_error_mapping_lab import (
    EXPERIMENT_QUERY_ERROR_MAPPING_POLICIES,
    build_query_error_mapping_experiment_report,
    build_query_error_mapping_process_section,
)
from .query_transport_selection_lab import (
    EXPERIMENT_QUERY_TRANSPORT_POLICIES,
    build_query_transport_selection_experiment_report,
    build_query_transport_selection_process_section,
)
from .query_request_import_lab import (
    EXPERIMENT_QUERY_REQUEST_IMPORT_POLICIES,
    build_query_request_import_experiment_report,
    build_query_request_import_process_section,
)
from .query_queue_policy_lab import (
    EXPERIMENT_QUERY_QUEUE_POLICIES,
    build_query_queue_policy_experiment_report,
    build_query_queue_policy_process_section,
)
from .query_source_identity_lab import (
    EXPERIMENT_QUERY_SOURCE_IDENTITY_POLICIES,
    build_query_source_identity_experiment_report,
    build_query_source_identity_process_section,
)
from .query_timeout_policy_lab import (
    EXPERIMENT_QUERY_TIMEOUT_POLICIES,
    build_query_timeout_policy_experiment_report,
    build_query_timeout_policy_process_section,
)
from .query_response_build_lab import (
    EXPERIMENT_QUERY_RESPONSE_BUILD_POLICIES,
    build_query_response_build_experiment_report,
    build_query_response_build_process_section,
)
from .route_capture_bundle_import_lab import (
    EXPERIMENT_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES,
    build_route_capture_bundle_import_experiment_report,
    build_route_capture_bundle_import_process_section,
)
from .sim2real_websocket_protocol_lab import (
    build_sim2real_websocket_protocol_experiment_report,
    build_sim2real_websocket_protocol_process_section,
)
from .live_localization_stream_import_lab import (
    build_live_localization_stream_import_experiment_report,
    build_live_localization_stream_import_process_section,
)
from .render_backend_selection_lab import (
    EXPERIMENT_RENDER_BACKEND_POLICIES,
    build_render_backend_selection_experiment_report,
    build_render_backend_selection_process_section,
)

__all__ = [
    "EXPERIMENT_ALIGNMENT_STRATEGIES",
    "EXPERIMENT_LOCALIZATION_ESTIMATE_IMPORT_POLICIES",
    "EXPERIMENT_QUERY_CANCELLATION_POLICIES",
    "EXPERIMENT_QUERY_COALESCING_POLICIES",
    "EXPERIMENT_QUERY_ERROR_MAPPING_POLICIES",
    "EXPERIMENT_QUERY_QUEUE_POLICIES",
    "EXPERIMENT_QUERY_RESPONSE_BUILD_POLICIES",
    "EXPERIMENT_QUERY_SOURCE_IDENTITY_POLICIES",
    "EXPERIMENT_QUERY_TIMEOUT_POLICIES",
    "EXPERIMENT_QUERY_REQUEST_IMPORT_POLICIES",
    "EXPERIMENT_QUERY_TRANSPORT_POLICIES",
    "EXPERIMENT_RENDER_BACKEND_POLICIES",
    "EXPERIMENT_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES",
    "build_localization_alignment_experiment_report",
    "build_localization_alignment_process_section",
    "build_localization_estimate_import_experiment_report",
    "build_localization_estimate_import_process_section",
    "build_localization_review_bundle_import_experiment_report",
    "build_localization_review_bundle_import_process_section",
    "build_query_cancellation_policy_experiment_report",
    "build_query_cancellation_policy_process_section",
    "build_query_coalescing_policy_experiment_report",
    "build_query_coalescing_policy_process_section",
    "build_query_error_mapping_experiment_report",
    "build_query_error_mapping_process_section",
    "build_live_localization_stream_import_experiment_report",
    "build_live_localization_stream_import_process_section",
    "build_query_queue_policy_experiment_report",
    "build_query_queue_policy_process_section",
    "build_query_source_identity_experiment_report",
    "build_query_source_identity_process_section",
    "build_query_timeout_policy_experiment_report",
    "build_query_timeout_policy_process_section",
    "build_query_response_build_experiment_report",
    "build_query_response_build_process_section",
    "build_query_request_import_experiment_report",
    "build_query_request_import_process_section",
    "build_query_transport_selection_experiment_report",
    "build_query_transport_selection_process_section",
    "build_render_backend_selection_experiment_report",
    "build_render_backend_selection_process_section",
    "build_route_capture_bundle_import_experiment_report",
    "build_route_capture_bundle_import_process_section",
    "build_sim2real_websocket_protocol_experiment_report",
    "build_sim2real_websocket_protocol_process_section",
    "write_localization_alignment_docs",
]
