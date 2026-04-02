"""Stable query-transport selection interfaces for sim2real render servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


DEFAULT_QUERY_TRANSPORT_ENDPOINTS = {
    "zmq": "tcp://127.0.0.1:5588",
    "ws": "ws://127.0.0.1:8781/sim2real",
}


@dataclass(frozen=True)
class QueryTransportCapabilities:
    """Runtime capabilities that constrain query transport selection."""

    zmq_available: bool
    ws_available: bool


@dataclass(frozen=True)
class QueryTransportPreferences:
    """Optional workload preferences for query transport selection."""

    enable_query_transport: bool = False
    prefer_browser_clients: bool = False
    prefer_local_cli: bool = False


@dataclass(frozen=True)
class QueryTransportRequest:
    """Stable input contract for query transport decisions."""

    requested_transport: str
    pose_source: str
    endpoint: str = ""
    capabilities: QueryTransportCapabilities = QueryTransportCapabilities(
        zmq_available=False,
        ws_available=False,
    )
    preferences: QueryTransportPreferences = QueryTransportPreferences()


@dataclass(frozen=True)
class QueryTransportSelection:
    """Resolved query transport and endpoint."""

    transport: str
    endpoint: str
    reason: str


class QueryTransportPolicy(Protocol):
    """Minimal interface for interchangeable query transport policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def select(self, request: QueryTransportRequest) -> QueryTransportSelection:
        """Choose a query transport or raise when the request cannot be satisfied."""


def resolve_query_transport_endpoint(transport: str, endpoint: str) -> str:
    """Resolve the bind endpoint for the selected transport."""
    normalized_transport = str(transport or "").strip()
    candidate = str(endpoint or "").strip()
    if normalized_transport == "none":
        return ""
    if candidate:
        if normalized_transport == "ws" and candidate == DEFAULT_QUERY_TRANSPORT_ENDPOINTS["zmq"]:
            return DEFAULT_QUERY_TRANSPORT_ENDPOINTS["ws"]
        if normalized_transport == "zmq" and candidate == DEFAULT_QUERY_TRANSPORT_ENDPOINTS["ws"]:
            return DEFAULT_QUERY_TRANSPORT_ENDPOINTS["zmq"]
        return candidate
    return DEFAULT_QUERY_TRANSPORT_ENDPOINTS.get(normalized_transport, candidate)


class BalancedQueryTransportPolicy:
    """Stable query transport policy used by production code."""

    name = "balanced"
    label = "Balanced Interactive Transport"
    style = "workload-aware"
    tier = "core"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": True,
        "supportsBrowserFirst": True,
    }

    def select(self, request: QueryTransportRequest) -> QueryTransportSelection:
        requested_transport = str(request.requested_transport or "auto").strip() or "auto"
        pose_source = str(request.pose_source or "").strip()
        runtime = request.capabilities
        preferences = request.preferences
        query_required = pose_source == "query"

        if requested_transport == "none":
            if query_required:
                raise RuntimeError("pose-source=query requires --query-transport zmq, ws, or auto")
            return QueryTransportSelection(
                transport="none",
                endpoint="",
                reason="forced by --query-transport none",
            )

        if requested_transport in {"zmq", "ws"}:
            if requested_transport == "zmq" and not runtime.zmq_available:
                raise RuntimeError("query-transport=zmq requires the optional `pyzmq` package")
            if requested_transport == "ws" and not runtime.ws_available:
                raise RuntimeError("query-transport=ws requires the optional `websockets` package")
            return QueryTransportSelection(
                transport=requested_transport,
                endpoint=resolve_query_transport_endpoint(requested_transport, request.endpoint),
                reason=f"forced by --query-transport {requested_transport}",
            )

        if requested_transport != "auto":
            raise RuntimeError(
                f"unsupported query transport: {requested_transport}. Expected one of auto, none, zmq, ws"
            )

        should_enable_transport = preferences.enable_query_transport or query_required
        if not should_enable_transport:
            return QueryTransportSelection(
                transport="none",
                endpoint="",
                reason="auto-selected no query transport because this workload does not request interactive queries",
            )

        candidate_order: list[str]
        if preferences.prefer_local_cli and not preferences.prefer_browser_clients:
            candidate_order = ["zmq", "ws"]
        else:
            candidate_order = ["ws", "zmq"]

        for candidate_transport in candidate_order:
            if candidate_transport == "ws" and runtime.ws_available:
                return QueryTransportSelection(
                    transport="ws",
                    endpoint=resolve_query_transport_endpoint("ws", request.endpoint),
                    reason="auto-selected ws because the workload prefers browser-facing clients",
                )
            if candidate_transport == "zmq" and runtime.zmq_available:
                return QueryTransportSelection(
                    transport="zmq",
                    endpoint=resolve_query_transport_endpoint("zmq", request.endpoint),
                    reason="auto-selected zmq because it is the first available local query transport",
                )

        if query_required:
            raise RuntimeError("pose-source=query requires the optional `websockets` or `pyzmq` packages")
        return QueryTransportSelection(
            transport="none",
            endpoint="",
            reason="auto-selected no query transport because optional transport dependencies are unavailable",
        )


CORE_QUERY_TRANSPORT_POLICIES: dict[str, QueryTransportPolicy] = {
    "balanced": BalancedQueryTransportPolicy(),
}


def select_query_transport(
    request: QueryTransportRequest,
    *,
    policy: str = "balanced",
) -> QueryTransportSelection:
    """Select the query transport under the current runtime and workload constraints."""
    if policy not in CORE_QUERY_TRANSPORT_POLICIES:
        raise RuntimeError(
            f"unsupported query transport policy: {policy}. "
            f"Expected one of {', '.join(sorted(CORE_QUERY_TRANSPORT_POLICIES))}"
        )
    return CORE_QUERY_TRANSPORT_POLICIES[policy].select(request)
