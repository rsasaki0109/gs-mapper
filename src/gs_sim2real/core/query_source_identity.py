"""Stable query source-identity policies for interactive sim2real transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse


def _normalize_token(value: str, *, allow_dot: bool = False) -> str:
    text = str(value or "").strip().lower()
    normalized: list[str] = []
    last_was_dash = False
    for char in text:
        if char.isalnum() or char in ("_",) or (allow_dot and char == "."):
            normalized.append(char)
            last_was_dash = False
            continue
        if not last_was_dash:
            normalized.append("-")
            last_was_dash = True
    token = "".join(normalized).strip("-")
    return token or "unknown"


def _build_endpoint_scope(endpoint: str) -> str:
    parsed = urlparse(str(endpoint or ""))
    parts: list[str] = []
    if parsed.hostname:
        parts.append(_normalize_token(parsed.hostname, allow_dot=True))
    if parsed.port is not None:
        parts.append(str(int(parsed.port)))
    if parsed.path and parsed.path != "/":
        parts.append(_normalize_token(parsed.path.strip("/")))
    return "-".join(part for part in parts if part)


@dataclass(frozen=True)
class QuerySourceIdentityRequest:
    """Stable input contract for source-identity decisions."""

    transport: str
    connection_serial: int
    endpoint: str = ""
    remote_host: str = ""
    remote_port: int | None = None
    client_hint: str = ""


@dataclass(frozen=True)
class QuerySourceIdentity:
    """Resolved source identity used by queue, coalescing, and cancellation layers."""

    source_id: str
    reason: str


class QuerySourceIdentityPolicy(Protocol):
    """Minimal interface for interchangeable source-identity policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def identify(self, request: QuerySourceIdentityRequest) -> QuerySourceIdentity:
        """Resolve one stable source identity for the incoming transport session."""


def _transport_prefix(request: QuerySourceIdentityRequest) -> str:
    return _normalize_token(request.transport) or "transport"


class SerialOnlyQuerySourceIdentityPolicy:
    """Keep source identity opaque and scoped only by connection serial."""

    name = "serial_only"
    label = "Serial Only"
    style = "opaque"
    tier = "experimental"
    capabilities = {
        "usesRemoteAddress": False,
        "usesEndpoint": False,
        "usesClientHint": False,
    }

    def identify(self, request: QuerySourceIdentityRequest) -> QuerySourceIdentity:
        source_id = f"{_transport_prefix(request)}-client-{max(1, int(request.connection_serial))}"
        return QuerySourceIdentity(source_id, "built an opaque source id from the transport and connection serial")


class EndpointScopedQuerySourceIdentityPolicy:
    """Attach endpoint scope so multi-endpoint servers remain distinguishable."""

    name = "endpoint_scoped"
    label = "Endpoint Scoped"
    style = "endpoint_scoped"
    tier = "experimental"
    capabilities = {
        "usesRemoteAddress": False,
        "usesEndpoint": True,
        "usesClientHint": False,
    }

    def identify(self, request: QuerySourceIdentityRequest) -> QuerySourceIdentity:
        prefix = _transport_prefix(request)
        endpoint_scope = _build_endpoint_scope(request.endpoint)
        if endpoint_scope:
            source_id = f"{prefix}-{endpoint_scope}-client-{max(1, int(request.connection_serial))}"
            return QuerySourceIdentity(
                source_id, "built a source id from the transport, endpoint, and connection serial"
            )
        return SerialOnlyQuerySourceIdentityPolicy().identify(request)


class RemoteObservableQuerySourceIdentityPolicy:
    """Prefer observable remote address data while keeping deterministic fallbacks."""

    name = "remote_observable"
    label = "Remote Observable"
    style = "connection_observable"
    tier = "core"
    capabilities = {
        "usesRemoteAddress": True,
        "usesEndpoint": True,
        "usesClientHint": True,
    }

    def identify(self, request: QuerySourceIdentityRequest) -> QuerySourceIdentity:
        prefix = _transport_prefix(request)
        if request.remote_host and request.remote_port is not None:
            host = _normalize_token(request.remote_host, allow_dot=True)
            port = str(int(request.remote_port))
            return QuerySourceIdentity(
                f"{prefix}-{host}-{port}",
                "built a source id from the observable remote socket address",
            )
        if request.client_hint:
            hint = _normalize_token(request.client_hint)
            return QuerySourceIdentity(
                f"{prefix}-{hint}-client-{max(1, int(request.connection_serial))}",
                "built a source id from the client hint and connection serial",
            )
        endpoint_identity = EndpointScopedQuerySourceIdentityPolicy().identify(request)
        return QuerySourceIdentity(
            endpoint_identity.source_id,
            "fell back to endpoint-scoped identity because no remote address or client hint was available",
        )


CORE_QUERY_SOURCE_IDENTITY_POLICIES: tuple[QuerySourceIdentityPolicy, ...] = (
    SerialOnlyQuerySourceIdentityPolicy(),
    EndpointScopedQuerySourceIdentityPolicy(),
    RemoteObservableQuerySourceIdentityPolicy(),
)


def resolve_query_source_identity_policy(
    policy: str | QuerySourceIdentityPolicy = "remote_observable",
) -> QuerySourceIdentityPolicy:
    """Resolve a source-identity policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "remote_observable"
    for candidate in CORE_QUERY_SOURCE_IDENTITY_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query source identity policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_SOURCE_IDENTITY_POLICIES)}"
    )


def resolve_query_source_identity(
    request: QuerySourceIdentityRequest,
    *,
    policy: str | QuerySourceIdentityPolicy = "remote_observable",
) -> QuerySourceIdentity:
    """Resolve one stable source identity using the selected policy."""
    return resolve_query_source_identity_policy(policy).identify(request)


__all__ = [
    "CORE_QUERY_SOURCE_IDENTITY_POLICIES",
    "EndpointScopedQuerySourceIdentityPolicy",
    "QuerySourceIdentity",
    "QuerySourceIdentityPolicy",
    "QuerySourceIdentityRequest",
    "RemoteObservableQuerySourceIdentityPolicy",
    "SerialOnlyQuerySourceIdentityPolicy",
    "resolve_query_source_identity",
    "resolve_query_source_identity_policy",
]
