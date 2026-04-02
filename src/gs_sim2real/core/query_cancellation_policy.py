"""Stable query cancellation policy interfaces for sim2real interactive queues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .query_queue_policy import QueuedQueryItem


@dataclass(frozen=True)
class QueryCancellationRequest:
    """Stable input contract for query cancellation decisions."""

    pending_items: tuple[QueuedQueryItem, ...]
    event: str
    target_request_id: str | None = None
    source_id: str = ""


@dataclass(frozen=True)
class QueryCancellationDecision:
    """Resolved cancellation decision for queued queries."""

    canceled_request_ids: tuple[str, ...]
    reason: str


class QueryCancellationPolicy(Protocol):
    """Minimal interface for interchangeable cancellation policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def cancel(self, request: QueryCancellationRequest) -> QueryCancellationDecision:
        """Resolve which queued requests should be canceled."""


def _resolve_shutdown_cancel_ids(request: QueryCancellationRequest) -> tuple[str, ...]:
    if request.event == "shutdown":
        return tuple(item.request_id for item in request.pending_items)
    return ()


class IgnoreOrphanedQueryCancellationPolicy:
    """Only cancel on shutdown; otherwise keep orphaned work queued."""

    name = "ignore_orphaned"
    label = "Ignore Orphaned"
    style = "minimal"
    tier = "experimental"
    capabilities = {
        "cancelsRequestedOnly": False,
        "cancelsSourceBacklog": False,
        "cancelsOnShutdown": True,
    }

    def cancel(self, request: QueryCancellationRequest) -> QueryCancellationDecision:
        shutdown_cancel_ids = _resolve_shutdown_cancel_ids(request)
        if shutdown_cancel_ids:
            return QueryCancellationDecision(shutdown_cancel_ids, "shutdown drains all queued requests")
        return QueryCancellationDecision((), "policy keeps orphaned queued work")


class CancelRequestedOnlyQueryCancellationPolicy:
    """Cancel only the targeted queued request unless the server is shutting down."""

    name = "cancel_requested_only"
    label = "Cancel Requested Only"
    style = "targeted"
    tier = "experimental"
    capabilities = {
        "cancelsRequestedOnly": True,
        "cancelsSourceBacklog": False,
        "cancelsOnShutdown": True,
    }

    def cancel(self, request: QueryCancellationRequest) -> QueryCancellationDecision:
        shutdown_cancel_ids = _resolve_shutdown_cancel_ids(request)
        if shutdown_cancel_ids:
            return QueryCancellationDecision(shutdown_cancel_ids, "shutdown drains all queued requests")
        if not request.target_request_id:
            return QueryCancellationDecision((), "no targeted queued request to cancel")
        cancel_ids = tuple(
            item.request_id for item in request.pending_items if item.request_id == request.target_request_id
        )
        return QueryCancellationDecision(cancel_ids, "canceled only the targeted queued request")


class CancelSourceBacklogQueryCancellationPolicy:
    """Cancel the requested queue item and the rest of the same source backlog."""

    name = "cancel_source_backlog"
    label = "Cancel Source Backlog"
    style = "source_scoped"
    tier = "core"
    capabilities = {
        "cancelsRequestedOnly": True,
        "cancelsSourceBacklog": True,
        "cancelsOnShutdown": True,
    }

    def cancel(self, request: QueryCancellationRequest) -> QueryCancellationDecision:
        shutdown_cancel_ids = _resolve_shutdown_cancel_ids(request)
        if shutdown_cancel_ids:
            return QueryCancellationDecision(shutdown_cancel_ids, "shutdown drains all queued requests")
        source_id = request.source_id
        target_item = next(
            (item for item in request.pending_items if item.request_id == request.target_request_id),
            None,
        )
        if source_id:
            cancel_ids = tuple(item.request_id for item in request.pending_items if item.source_id == source_id)
            return QueryCancellationDecision(cancel_ids, "canceled all queued requests for the disconnected source")
        if target_item is not None and target_item.source_id:
            cancel_ids = tuple(
                item.request_id for item in request.pending_items if item.source_id == target_item.source_id
            )
            return QueryCancellationDecision(cancel_ids, "canceled all queued requests for the targeted source")
        if request.target_request_id:
            return QueryCancellationDecision(
                tuple(
                    item.request_id for item in request.pending_items if item.request_id == request.target_request_id
                ),
                "canceled only the targeted queued request because no source scope was available",
            )
        return QueryCancellationDecision((), "no matching queued source to cancel")


CORE_QUERY_CANCELLATION_POLICIES: tuple[QueryCancellationPolicy, ...] = (
    IgnoreOrphanedQueryCancellationPolicy(),
    CancelRequestedOnlyQueryCancellationPolicy(),
    CancelSourceBacklogQueryCancellationPolicy(),
)


def resolve_query_cancellation_policy(
    policy: str | QueryCancellationPolicy = "cancel_source_backlog",
) -> QueryCancellationPolicy:
    """Resolve a cancellation policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "cancel_source_backlog"
    for candidate in CORE_QUERY_CANCELLATION_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query cancellation policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_CANCELLATION_POLICIES)}"
    )


def resolve_query_cancellation_decision(
    request: QueryCancellationRequest,
    *,
    policy: str | QueryCancellationPolicy = "cancel_source_backlog",
) -> QueryCancellationDecision:
    """Resolve one cancellation decision using the selected policy."""
    return resolve_query_cancellation_policy(policy).cancel(request)


__all__ = [
    "CORE_QUERY_CANCELLATION_POLICIES",
    "CancelRequestedOnlyQueryCancellationPolicy",
    "CancelSourceBacklogQueryCancellationPolicy",
    "IgnoreOrphanedQueryCancellationPolicy",
    "QueryCancellationDecision",
    "QueryCancellationPolicy",
    "QueryCancellationRequest",
    "resolve_query_cancellation_decision",
    "resolve_query_cancellation_policy",
]
