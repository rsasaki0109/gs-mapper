"""Stable query error-mapping policies for interactive sim2real transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


_DEFAULT_ERROR_SHAPES: dict[str, tuple[str, str, str]] = {
    "invalid_json": ("invalid JSON request", "ValueError", "invalid_json_request"),
    "queue_rejected": ("query queue rejected request", "RuntimeError", "query_queue_rejected"),
    "queue_dropped": ("query dropped from queue", "RuntimeError", "query_queue_dropped"),
    "query_canceled": ("query canceled", "RuntimeError", "query_canceled"),
    "query_timeout": (
        "query timed out while waiting for the render thread",
        "TimeoutError",
        "query_timeout",
    ),
    "empty_response": ("query returned no response", "RuntimeError", "query_empty_response"),
    "server_shutdown": ("render query server is shutting down", "RuntimeError", "query_server_shutdown"),
    "handler_exception": ("query handler failed", "RuntimeError", "query_handler_exception"),
    "runtime_error": ("query failed", "RuntimeError", "query_error"),
}


def _shape_for_event(event: str) -> tuple[str, str, str]:
    return _DEFAULT_ERROR_SHAPES.get(str(event or "").strip().lower(), _DEFAULT_ERROR_SHAPES["runtime_error"])


@dataclass(frozen=True)
class QueryErrorMappingRequest:
    """Stable input contract for query error mapping."""

    event: str
    reason: str = ""
    detail: str = ""
    transport: str = ""
    request_type: str = ""
    exception_type: str = ""


@dataclass(frozen=True)
class QueryErrorMappingDecision:
    """Canonical mapped error payload before response serialization."""

    error: str
    error_type: str
    error_code: str
    reason: str


class QueryErrorMappingPolicy(Protocol):
    """Minimal interface for interchangeable query error-mapping policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def map_error(self, request: QueryErrorMappingRequest) -> QueryErrorMappingDecision:
        """Map one transport/runtime event into a canonical error payload."""


def _append_suffix(message: str, suffix: str) -> str:
    detail = str(suffix or "").strip()
    if not detail:
        return message
    return f"{message}: {detail}"


class LiteralPassthroughQueryErrorMappingPolicy:
    """Preserve the literal detail or reason with minimal interpretation."""

    name = "literal_passthrough"
    label = "Literal Passthrough"
    style = "literal"
    tier = "experimental"
    capabilities = {
        "emitsStableCodes": True,
        "preservesLiteralDetail": True,
        "addsActionHints": False,
    }

    def map_error(self, request: QueryErrorMappingRequest) -> QueryErrorMappingDecision:
        _, default_type, default_code = _shape_for_event(request.event)
        message = str(request.detail or request.reason or _shape_for_event(request.event)[0])
        return QueryErrorMappingDecision(
            error=message,
            error_type=str(request.exception_type or default_type),
            error_code=default_code,
            reason="passed through literal error detail without canonical messaging",
        )


class StructuredCodesQueryErrorMappingPolicy:
    """Emit stable messages and codes without transport-specific branching."""

    name = "structured_codes"
    label = "Structured Codes"
    style = "canonical"
    tier = "core"
    capabilities = {
        "emitsStableCodes": True,
        "preservesLiteralDetail": True,
        "addsActionHints": False,
    }

    def map_error(self, request: QueryErrorMappingRequest) -> QueryErrorMappingDecision:
        base_message, default_type, error_code = _shape_for_event(request.event)
        event = str(request.event or "").strip().lower()
        if event == "invalid_json":
            error = _append_suffix(base_message, request.detail)
        elif event in {"queue_rejected", "queue_dropped", "query_canceled"}:
            error = _append_suffix(base_message, request.reason or request.detail)
        elif event == "handler_exception":
            error = _append_suffix(base_message, request.detail)
        elif request.detail and event not in {"query_timeout", "empty_response", "server_shutdown"}:
            error = _append_suffix(base_message, request.detail)
        else:
            error = base_message
        return QueryErrorMappingDecision(
            error=error,
            error_type=str(request.exception_type or default_type),
            error_code=error_code,
            reason="mapped the event into a stable canonical message and error code",
        )


class ActionHintQueryErrorMappingPolicy:
    """Keep stable codes while making messages explicitly actionable for browser users."""

    name = "action_hint"
    label = "Action Hint"
    style = "actionable"
    tier = "experimental"
    capabilities = {
        "emitsStableCodes": True,
        "preservesLiteralDetail": False,
        "addsActionHints": True,
    }

    def map_error(self, request: QueryErrorMappingRequest) -> QueryErrorMappingDecision:
        _, default_type, error_code = _shape_for_event(request.event)
        event = str(request.event or "").strip().lower()
        if event == "invalid_json":
            error = "invalid JSON request; send a valid sim2real query document"
        elif event == "queue_rejected":
            error = "query queue rejected the request; wait for current work to finish and retry"
        elif event == "queue_dropped":
            error = "queued render was superseded by newer work from the same interaction"
        elif event == "query_canceled":
            error = "queued request was canceled before dispatch"
        elif event == "query_timeout":
            error = "query timed out; reduce workload size or retry after the queue drains"
        elif event == "empty_response":
            error = "render thread returned no response; retry the request"
        elif event == "server_shutdown":
            error = "render query server is shutting down; reconnect after restart"
        elif event == "handler_exception":
            error = _append_suffix("render query failed", request.detail or request.reason)
        else:
            error = _append_suffix("query failed", request.detail or request.reason)
        return QueryErrorMappingDecision(
            error=error,
            error_type=str(request.exception_type or default_type),
            error_code=error_code,
            reason="mapped the event into an action-oriented browser-facing error message",
        )


CORE_QUERY_ERROR_MAPPING_POLICIES: tuple[QueryErrorMappingPolicy, ...] = (
    LiteralPassthroughQueryErrorMappingPolicy(),
    StructuredCodesQueryErrorMappingPolicy(),
    ActionHintQueryErrorMappingPolicy(),
)


def resolve_query_error_mapping_policy(
    policy: str | QueryErrorMappingPolicy = "structured_codes",
) -> QueryErrorMappingPolicy:
    """Resolve an error-mapping policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "structured_codes"
    for candidate in CORE_QUERY_ERROR_MAPPING_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query error mapping policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_ERROR_MAPPING_POLICIES)}"
    )


def resolve_query_error_mapping(
    request: QueryErrorMappingRequest,
    *,
    policy: str | QueryErrorMappingPolicy = "structured_codes",
) -> QueryErrorMappingDecision:
    """Resolve one query error payload using the selected policy."""
    return resolve_query_error_mapping_policy(policy).map_error(request)


__all__ = [
    "ActionHintQueryErrorMappingPolicy",
    "CORE_QUERY_ERROR_MAPPING_POLICIES",
    "LiteralPassthroughQueryErrorMappingPolicy",
    "QueryErrorMappingDecision",
    "QueryErrorMappingPolicy",
    "QueryErrorMappingRequest",
    "StructuredCodesQueryErrorMappingPolicy",
    "resolve_query_error_mapping",
    "resolve_query_error_mapping_policy",
]
