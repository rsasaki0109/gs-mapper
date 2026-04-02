"""Stable query dedupe/coalescing policy interfaces for sim2real queues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .query_queue_policy import QueuedQueryItem


@dataclass(frozen=True)
class QueryCoalescingRequest:
    """Stable input contract for query coalescing decisions."""

    pending_items: tuple[QueuedQueryItem, ...]
    incoming_item: QueuedQueryItem


@dataclass(frozen=True)
class QueryCoalescingDecision:
    """Resolved coalescing decision before queue admission."""

    accepted: bool
    pending_request_ids: tuple[str, ...]
    evicted_request_ids: tuple[str, ...] = ()
    rejected_request_id: str | None = None
    reason: str = ""


class QueryCoalescingPolicy(Protocol):
    """Minimal interface for interchangeable coalescing policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def coalesce(self, request: QueryCoalescingRequest) -> QueryCoalescingDecision:
        """Resolve one dedupe/coalescing decision."""


class KeepAllQueryCoalescingPolicy:
    """Accept every incoming request without dedupe."""

    name = "keep_all"
    label = "Keep All"
    style = "append_only"
    tier = "experimental"
    capabilities = {
        "dedupesExactRender": False,
        "replacesOlderRenderFromSource": False,
        "preservesBackgroundBenchmark": True,
    }

    def coalesce(self, request: QueryCoalescingRequest) -> QueryCoalescingDecision:
        pending_ids = tuple(item.request_id for item in request.pending_items) + (request.incoming_item.request_id,)
        return QueryCoalescingDecision(
            accepted=True,
            pending_request_ids=pending_ids,
            reason="accepted without coalescing",
        )


class ExactRenderDropNewQueryCoalescingPolicy:
    """Reject a new render when an identical pending render already exists for the same source."""

    name = "exact_render_drop_new"
    label = "Exact Render Drop New"
    style = "exact_dedupe"
    tier = "experimental"
    capabilities = {
        "dedupesExactRender": True,
        "replacesOlderRenderFromSource": False,
        "preservesBackgroundBenchmark": True,
    }

    def coalesce(self, request: QueryCoalescingRequest) -> QueryCoalescingDecision:
        incoming = request.incoming_item
        if incoming.request_type == "render" and incoming.dedupe_key:
            for item in request.pending_items:
                if (
                    item.request_type == "render"
                    and item.source_id == incoming.source_id
                    and item.dedupe_key == incoming.dedupe_key
                ):
                    return QueryCoalescingDecision(
                        accepted=False,
                        pending_request_ids=tuple(current.request_id for current in request.pending_items),
                        rejected_request_id=incoming.request_id,
                        reason="identical render request is already pending for this source",
                    )
        return KeepAllQueryCoalescingPolicy().coalesce(request)


class LatestRenderPerSourceQueryCoalescingPolicy:
    """Keep the latest pending render per source while preserving background work."""

    name = "latest_render_per_source"
    label = "Latest Render Per Source"
    style = "latest_render"
    tier = "core"
    capabilities = {
        "dedupesExactRender": True,
        "replacesOlderRenderFromSource": True,
        "preservesBackgroundBenchmark": True,
    }

    def coalesce(self, request: QueryCoalescingRequest) -> QueryCoalescingDecision:
        incoming = request.incoming_item
        if incoming.request_type != "render" or not incoming.source_id:
            return KeepAllQueryCoalescingPolicy().coalesce(request)
        kept_items: list[QueuedQueryItem] = []
        evicted_ids: list[str] = []
        for item in request.pending_items:
            if item.request_type == "render" and item.source_id == incoming.source_id:
                evicted_ids.append(item.request_id)
                continue
            kept_items.append(item)
        pending_ids = tuple(item.request_id for item in kept_items) + (incoming.request_id,)
        return QueryCoalescingDecision(
            accepted=True,
            pending_request_ids=pending_ids,
            evicted_request_ids=tuple(evicted_ids),
            reason=(
                "replaced older pending render requests from the same source"
                if evicted_ids
                else "accepted latest render for this source"
            ),
        )


CORE_QUERY_COALESCING_POLICIES: tuple[QueryCoalescingPolicy, ...] = (
    KeepAllQueryCoalescingPolicy(),
    ExactRenderDropNewQueryCoalescingPolicy(),
    LatestRenderPerSourceQueryCoalescingPolicy(),
)


def resolve_query_coalescing_policy(
    policy: str | QueryCoalescingPolicy = "latest_render_per_source",
) -> QueryCoalescingPolicy:
    """Resolve a coalescing policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "latest_render_per_source"
    for candidate in CORE_QUERY_COALESCING_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query coalescing policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_COALESCING_POLICIES)}"
    )


def resolve_query_coalescing_decision(
    request: QueryCoalescingRequest,
    *,
    policy: str | QueryCoalescingPolicy = "latest_render_per_source",
) -> QueryCoalescingDecision:
    """Resolve one coalescing decision using the selected policy."""
    return resolve_query_coalescing_policy(policy).coalesce(request)


__all__ = [
    "CORE_QUERY_COALESCING_POLICIES",
    "ExactRenderDropNewQueryCoalescingPolicy",
    "KeepAllQueryCoalescingPolicy",
    "LatestRenderPerSourceQueryCoalescingPolicy",
    "QueryCoalescingDecision",
    "QueryCoalescingPolicy",
    "QueryCoalescingRequest",
    "resolve_query_coalescing_decision",
    "resolve_query_coalescing_policy",
]
