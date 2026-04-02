"""Stable query queue policy interfaces for sim2real interactive transports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .query_request_import import (
    QueryRequestImportRequest,
    RenderQueryDefaults,
    canonicalize_query_request_type,
    import_query_request,
)


@dataclass(frozen=True)
class QueuedQueryItem:
    """Normalized queue item independent of the transport implementation."""

    request_id: str
    request_type: str
    transport: str
    submitted_order: int
    source_id: str = ""
    dedupe_key: str = ""
    expected_work_units: int = 1


@dataclass(frozen=True)
class QueryQueueState:
    """Serializable queue state compared by queue policies."""

    pending_items: tuple[QueuedQueryItem, ...]
    max_pending: int


@dataclass(frozen=True)
class QueryQueueAdmitDecision:
    """Result of admitting a new query item into the queue."""

    accepted: bool
    pending_request_ids: tuple[str, ...]
    evicted_request_ids: tuple[str, ...] = ()
    rejected_request_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class QueryQueueDispatchDecision:
    """Result of selecting the next query item to dispatch."""

    dispatch_request_id: str | None
    pending_request_ids: tuple[str, ...]
    reason: str = ""


class QueryQueuePolicy(Protocol):
    """Minimal interface for interchangeable queue policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def admit(self, state: QueryQueueState, item: QueuedQueryItem) -> QueryQueueAdmitDecision:
        """Admit one queue item into the current state."""

    def dispatch(self, state: QueryQueueState) -> QueryQueueDispatchDecision:
        """Choose the next queue item to dispatch."""


def _sorted_fifo_ids(items: tuple[QueuedQueryItem, ...]) -> tuple[str, ...]:
    return tuple(item.request_id for item in sorted(items, key=lambda item: item.submitted_order))


def _priority_key(item: QueuedQueryItem) -> tuple[int, int, int]:
    request_priority = 0 if item.request_type == "render" else 1
    workload_priority = min(max(1, int(item.expected_work_units)), 1_000_000)
    return (request_priority, workload_priority, item.submitted_order)


class FifoUnboundedQueryQueuePolicy:
    """Accept every request and dispatch strictly by arrival order."""

    name = "fifo_unbounded"
    label = "FIFO Unbounded"
    style = "fifo"
    tier = "experimental"
    capabilities = {
        "boundsQueue": False,
        "prioritizesInteractiveRender": False,
        "evictsBackgroundWork": False,
    }

    def admit(self, state: QueryQueueState, item: QueuedQueryItem) -> QueryQueueAdmitDecision:
        pending_ids = _sorted_fifo_ids(state.pending_items + (item,))
        return QueryQueueAdmitDecision(
            accepted=True,
            pending_request_ids=pending_ids,
            reason="accepted with unbounded FIFO queue",
        )

    def dispatch(self, state: QueryQueueState) -> QueryQueueDispatchDecision:
        pending = sorted(state.pending_items, key=lambda current: current.submitted_order)
        if not pending:
            return QueryQueueDispatchDecision(dispatch_request_id=None, pending_request_ids=(), reason="queue empty")
        dispatch_item = pending[0]
        return QueryQueueDispatchDecision(
            dispatch_request_id=dispatch_item.request_id,
            pending_request_ids=tuple(item.request_id for item in pending[1:]),
            reason="dispatch oldest queued request",
        )


class BoundedFifoQueryQueuePolicy:
    """Keep strict FIFO ordering while bounding queue growth."""

    name = "bounded_fifo"
    label = "Bounded FIFO"
    style = "fifo_bounded"
    tier = "experimental"
    capabilities = {
        "boundsQueue": True,
        "prioritizesInteractiveRender": False,
        "evictsBackgroundWork": False,
    }

    def admit(self, state: QueryQueueState, item: QueuedQueryItem) -> QueryQueueAdmitDecision:
        pending = sorted(state.pending_items, key=lambda current: current.submitted_order)
        if len(pending) >= max(1, int(state.max_pending)):
            return QueryQueueAdmitDecision(
                accepted=False,
                pending_request_ids=tuple(existing.request_id for existing in pending),
                rejected_request_id=item.request_id,
                reason="queue is full under bounded FIFO policy",
            )
        pending.append(item)
        return QueryQueueAdmitDecision(
            accepted=True,
            pending_request_ids=tuple(existing.request_id for existing in pending),
            reason="accepted without reordering under bounded FIFO policy",
        )

    def dispatch(self, state: QueryQueueState) -> QueryQueueDispatchDecision:
        return FifoUnboundedQueryQueuePolicy().dispatch(state)


class InteractiveFirstQueryQueuePolicy:
    """Prefer short render interactions over heavy background benchmark work."""

    name = "interactive_first"
    label = "Interactive First"
    style = "priority_bounded"
    tier = "core"
    capabilities = {
        "boundsQueue": True,
        "prioritizesInteractiveRender": True,
        "evictsBackgroundWork": True,
    }

    def admit(self, state: QueryQueueState, item: QueuedQueryItem) -> QueryQueueAdmitDecision:
        pending = list(state.pending_items)
        capacity = max(1, int(state.max_pending))
        if len(pending) < capacity:
            pending.append(item)
            ordered = tuple(entry.request_id for entry in sorted(pending, key=_priority_key))
            return QueryQueueAdmitDecision(
                accepted=True,
                pending_request_ids=ordered,
                reason="accepted and ranked by interactive priority",
            )

        eviction_candidates = sorted(pending, key=_priority_key, reverse=True)
        worst = eviction_candidates[0]
        if _priority_key(item) >= _priority_key(worst):
            ordered = tuple(entry.request_id for entry in sorted(pending, key=_priority_key))
            return QueryQueueAdmitDecision(
                accepted=False,
                pending_request_ids=ordered,
                rejected_request_id=item.request_id,
                reason="incoming request is lower priority than current queue contents",
            )

        kept = [entry for entry in pending if entry.request_id != worst.request_id]
        kept.append(item)
        ordered = tuple(entry.request_id for entry in sorted(kept, key=_priority_key))
        return QueryQueueAdmitDecision(
            accepted=True,
            pending_request_ids=ordered,
            evicted_request_ids=(worst.request_id,),
            reason="evicted lower-priority queued work in favor of an interactive request",
        )

    def dispatch(self, state: QueryQueueState) -> QueryQueueDispatchDecision:
        pending = sorted(state.pending_items, key=_priority_key)
        if not pending:
            return QueryQueueDispatchDecision(dispatch_request_id=None, pending_request_ids=(), reason="queue empty")
        dispatch_item = pending[0]
        return QueryQueueDispatchDecision(
            dispatch_request_id=dispatch_item.request_id,
            pending_request_ids=tuple(item.request_id for item in pending[1:]),
            reason="dispatch highest-priority interactive request first",
        )


CORE_QUERY_QUEUE_POLICIES: tuple[QueryQueuePolicy, ...] = (
    FifoUnboundedQueryQueuePolicy(),
    BoundedFifoQueryQueuePolicy(),
    InteractiveFirstQueryQueuePolicy(),
)


def resolve_query_queue_policy(
    policy: str | QueryQueuePolicy = "interactive_first",
) -> QueryQueuePolicy:
    """Resolve a queue policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "interactive_first"
    for candidate in CORE_QUERY_QUEUE_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query queue policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_QUEUE_POLICIES)}"
    )


def admit_query_queue_item(
    state: QueryQueueState,
    item: QueuedQueryItem,
    *,
    policy: str | QueryQueuePolicy = "interactive_first",
) -> QueryQueueAdmitDecision:
    """Admit one queue item using the resolved policy."""
    return resolve_query_queue_policy(policy).admit(state, item)


def dispatch_query_queue_item(
    state: QueryQueueState,
    *,
    policy: str | QueryQueuePolicy = "interactive_first",
) -> QueryQueueDispatchDecision:
    """Select one queued item for dispatch using the resolved policy."""
    return resolve_query_queue_policy(policy).dispatch(state)


def build_queued_query_item_from_payload(
    payload: Any,
    *,
    request_id: str,
    submitted_order: int,
    transport: str,
    source_id: str = "",
) -> QueuedQueryItem:
    """Build a queue item summary from a raw query payload."""
    request_type = "render"
    expected_work_units = 1
    dedupe_key = ""

    if isinstance(payload, dict):
        raw_request_type = payload.get("type", payload.get("requestType", "render"))
        if isinstance(raw_request_type, str):
            try:
                request_type = canonicalize_query_request_type(raw_request_type, default="render")
            except ValueError:
                request_type = "render"

    try:
        imported = import_query_request(
            QueryRequestImportRequest(
                payload=payload,
                defaults=RenderQueryDefaults(
                    width=1,
                    height=1,
                    fov_degrees=60.0,
                    near_clip=0.05,
                    far_clip=50.0,
                    point_radius=1,
                ),
            )
        )
        request_type = imported.request_type
        if imported.render is not None:
            dedupe_key = json.dumps(
                {
                    "requestType": imported.request_type,
                    "position": list(imported.render.position),
                    "orientation": list(imported.render.orientation),
                    "width": imported.render.width,
                    "height": imported.render.height,
                    "fovDegrees": imported.render.fov_degrees,
                    "nearClip": imported.render.near_clip,
                    "farClip": imported.render.far_clip,
                    "pointRadius": imported.render.point_radius,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        if imported.image_benchmark is not None:
            captures = imported.image_benchmark.ground_truth_bundle.get("captures")
            capture_count = len(captures) if isinstance(captures, list) else 0
            expected_work_units = (
                int(imported.image_benchmark.max_frames)
                if imported.image_benchmark.max_frames is not None
                else max(1, capture_count)
            )
    except Exception:
        pass

    return QueuedQueryItem(
        request_id=str(request_id),
        request_type=request_type,
        transport=str(transport or "unknown"),
        submitted_order=int(submitted_order),
        source_id=str(source_id or ""),
        dedupe_key=str(dedupe_key or ""),
        expected_work_units=max(1, int(expected_work_units)),
    )


__all__ = [
    "CORE_QUERY_QUEUE_POLICIES",
    "BoundedFifoQueryQueuePolicy",
    "FifoUnboundedQueryQueuePolicy",
    "InteractiveFirstQueryQueuePolicy",
    "QueryQueueAdmitDecision",
    "QueryQueueDispatchDecision",
    "QueryQueuePolicy",
    "QueryQueueState",
    "QueuedQueryItem",
    "admit_query_queue_item",
    "build_queued_query_item_from_payload",
    "dispatch_query_queue_item",
    "resolve_query_queue_policy",
]
