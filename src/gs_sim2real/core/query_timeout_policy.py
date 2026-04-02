"""Stable query timeout policy interfaces for sim2real query round-trips."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .query_request_import import (
    QueryRequestImportRequest,
    RenderQueryDefaults,
    canonicalize_query_request_type,
    import_query_request,
)


@dataclass(frozen=True)
class QueryTimeoutPolicyRequest:
    """Stable input contract for query timeout policy decisions."""

    request_type: str
    transport: str
    explicit_server_timeout_seconds: float | None = None
    explicit_client_timeout_ms: int | None = None
    expected_work_units: int = 1
    allow_retry: bool = False
    default_server_timeout_seconds: float = 30.0
    maximum_server_timeout_seconds: float = 300.0
    default_client_timeout_ms: int = 10_000
    minimum_client_timeout_ms: int = 1_000


@dataclass(frozen=True)
class QueryTimeoutPlan:
    """Resolved timeout and retry plan for one query round-trip."""

    server_timeout_seconds: float
    client_timeout_ms: int
    attempt_timeout_ms: int
    max_attempts: int
    retry_backoff_ms: int
    reason: str


class QueryTimeoutPolicy(Protocol):
    """Minimal interface for interchangeable query timeout policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def resolve_timeout_plan(self, request: QueryTimeoutPolicyRequest) -> QueryTimeoutPlan:
        """Resolve one timeout plan."""


def _clamp_timeout_seconds(value: float, request: QueryTimeoutPolicyRequest) -> float:
    maximum = max(float(request.maximum_server_timeout_seconds), 1.0)
    minimum = min(maximum, 1.0)
    return min(maximum, max(minimum, float(value)))


def _clamp_timeout_ms(value: int, request: QueryTimeoutPolicyRequest) -> int:
    minimum = max(1, int(request.minimum_client_timeout_ms))
    return max(minimum, int(value))


def _normalize_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized <= 0.0:
        return None
    return float(normalized)


def _normalize_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    if normalized <= 0:
        return None
    return int(normalized)


def _normalize_transport(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"ws", "wss", "tcp"}:
        return normalized
    return "unknown"


def _workload_floor_seconds(request: QueryTimeoutPolicyRequest) -> float:
    if request.request_type != "localization-image-benchmark":
        return float(request.default_server_timeout_seconds)
    return min(
        float(request.maximum_server_timeout_seconds),
        max(float(request.default_server_timeout_seconds), float(max(1, int(request.expected_work_units))) * 8.0),
    )


class FixedDeadlineQueryTimeoutPolicy:
    """Ignore request hints and keep one conservative deadline."""

    name = "fixed_deadline"
    label = "Fixed Deadline"
    style = "fixed"
    tier = "experimental"
    capabilities = {
        "respectsExplicitServerHint": False,
        "scalesForWorkload": False,
        "supportsRetryBudget": False,
    }

    def resolve_timeout_plan(self, request: QueryTimeoutPolicyRequest) -> QueryTimeoutPlan:
        client_timeout_ms = _clamp_timeout_ms(
            request.explicit_client_timeout_ms
            if request.explicit_client_timeout_ms is not None
            else int(request.default_client_timeout_ms),
            request,
        )
        return QueryTimeoutPlan(
            server_timeout_seconds=_clamp_timeout_seconds(request.default_server_timeout_seconds, request),
            client_timeout_ms=client_timeout_ms,
            attempt_timeout_ms=client_timeout_ms,
            max_attempts=1,
            retry_backoff_ms=0,
            reason="fixed default deadline",
        )


class HintBoundedQueryTimeoutPolicy:
    """Respect explicit timeout hints but keep one attempt."""

    name = "hint_bounded"
    label = "Hint Bounded"
    style = "hint_driven"
    tier = "experimental"
    capabilities = {
        "respectsExplicitServerHint": True,
        "scalesForWorkload": False,
        "supportsRetryBudget": False,
    }

    def resolve_timeout_plan(self, request: QueryTimeoutPolicyRequest) -> QueryTimeoutPlan:
        server_timeout_seconds = _clamp_timeout_seconds(
            request.explicit_server_timeout_seconds
            if request.explicit_server_timeout_seconds is not None
            else request.default_server_timeout_seconds,
            request,
        )
        client_timeout_ms = _clamp_timeout_ms(
            request.explicit_client_timeout_ms
            if request.explicit_client_timeout_ms is not None
            else int(round(server_timeout_seconds * 1000.0)),
            request,
        )
        return QueryTimeoutPlan(
            server_timeout_seconds=server_timeout_seconds,
            client_timeout_ms=client_timeout_ms,
            attempt_timeout_ms=client_timeout_ms,
            max_attempts=1,
            retry_backoff_ms=0,
            reason="explicit timeout hints bounded to safe limits",
        )


class WorkloadAwareRetryQueryTimeoutPolicy:
    """Respect hints, protect benchmark workloads, and allow render retries."""

    name = "workload_aware_retry"
    label = "Workload Aware Retry"
    style = "adaptive"
    tier = "core"
    capabilities = {
        "respectsExplicitServerHint": True,
        "scalesForWorkload": True,
        "supportsRetryBudget": True,
    }

    def resolve_timeout_plan(self, request: QueryTimeoutPolicyRequest) -> QueryTimeoutPlan:
        workload_floor_seconds = _workload_floor_seconds(request)
        requested_server_timeout = (
            request.explicit_server_timeout_seconds
            if request.explicit_server_timeout_seconds is not None
            else request.default_server_timeout_seconds
        )
        server_timeout_seconds = _clamp_timeout_seconds(
            max(float(requested_server_timeout), float(workload_floor_seconds)),
            request,
        )
        if request.request_type == "localization-image-benchmark":
            transport_slack_ms = 5_000 if _normalize_transport(request.transport) in {"ws", "wss"} else 2_000
            client_timeout_candidate = max(
                int(round(server_timeout_seconds * 1000.0)) + transport_slack_ms,
                int(request.explicit_client_timeout_ms or 0),
            )
        else:
            client_timeout_candidate = int(
                request.explicit_client_timeout_ms
                if request.explicit_client_timeout_ms is not None
                else request.default_client_timeout_ms
            )
        client_timeout_ms = _clamp_timeout_ms(client_timeout_candidate, request)
        max_attempts = (
            2
            if request.allow_retry
            and request.request_type == "render"
            and _normalize_transport(request.transport) in {"ws", "wss"}
            else 1
        )
        retry_backoff_ms = 150 if max_attempts > 1 else 0
        attempt_timeout_ms = client_timeout_ms
        if max_attempts > 1:
            attempt_timeout_ms = _clamp_timeout_ms(
                max(
                    request.minimum_client_timeout_ms,
                    (client_timeout_ms - retry_backoff_ms * (max_attempts - 1)) // max_attempts,
                ),
                request,
            )
        return QueryTimeoutPlan(
            server_timeout_seconds=server_timeout_seconds,
            client_timeout_ms=client_timeout_ms,
            attempt_timeout_ms=attempt_timeout_ms,
            max_attempts=max_attempts,
            retry_backoff_ms=retry_backoff_ms,
            reason="workload floor with bounded render retries",
        )


CORE_QUERY_TIMEOUT_POLICIES: tuple[QueryTimeoutPolicy, ...] = (
    FixedDeadlineQueryTimeoutPolicy(),
    HintBoundedQueryTimeoutPolicy(),
    WorkloadAwareRetryQueryTimeoutPolicy(),
)


def resolve_query_timeout_policy(
    policy: str | QueryTimeoutPolicy = "workload_aware_retry",
) -> QueryTimeoutPolicy:
    """Resolve a query timeout policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "workload_aware_retry"
    for candidate in CORE_QUERY_TIMEOUT_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query timeout policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_TIMEOUT_POLICIES)}"
    )


def resolve_query_timeout_plan(
    request: QueryTimeoutPolicyRequest,
    *,
    policy: str | QueryTimeoutPolicy = "workload_aware_retry",
) -> QueryTimeoutPlan:
    """Resolve a timeout plan for one query request."""
    return resolve_query_timeout_policy(policy).resolve_timeout_plan(request)


def build_query_timeout_policy_request(
    payload: Any,
    *,
    transport: str,
    explicit_client_timeout_ms: int | None = None,
    allow_retry: bool = False,
    default_server_timeout_seconds: float = 30.0,
    maximum_server_timeout_seconds: float = 300.0,
    default_client_timeout_ms: int = 10_000,
    minimum_client_timeout_ms: int = 1_000,
) -> QueryTimeoutPolicyRequest:
    """Build a stable timeout-policy request from a raw query payload."""
    request_type = "render"
    expected_work_units = 1
    explicit_server_timeout_seconds = _normalize_positive_float(
        payload.get("responseTimeoutSeconds") if isinstance(payload, dict) else None
    )

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
                    timeout_ms=int(default_client_timeout_ms),
                ),
            )
        )
        request_type = imported.request_type
        if imported.image_benchmark is not None:
            captures = imported.image_benchmark.ground_truth_bundle.get("captures")
            capture_count = len(captures) if isinstance(captures, list) else 0
            expected_work_units = (
                int(imported.image_benchmark.max_frames)
                if imported.image_benchmark.max_frames is not None
                else max(1, capture_count)
            )
            if explicit_server_timeout_seconds is None and imported.image_benchmark.timeout_ms > 0:
                explicit_server_timeout_seconds = float(imported.image_benchmark.timeout_ms) / 1000.0
    except Exception:
        pass

    return QueryTimeoutPolicyRequest(
        request_type=request_type,
        transport=_normalize_transport(transport),
        explicit_server_timeout_seconds=explicit_server_timeout_seconds,
        explicit_client_timeout_ms=_normalize_positive_int(explicit_client_timeout_ms),
        expected_work_units=max(1, int(expected_work_units)),
        allow_retry=bool(allow_retry),
        default_server_timeout_seconds=float(default_server_timeout_seconds),
        maximum_server_timeout_seconds=float(maximum_server_timeout_seconds),
        default_client_timeout_ms=int(default_client_timeout_ms),
        minimum_client_timeout_ms=int(minimum_client_timeout_ms),
    )


__all__ = [
    "CORE_QUERY_TIMEOUT_POLICIES",
    "FixedDeadlineQueryTimeoutPolicy",
    "HintBoundedQueryTimeoutPolicy",
    "QueryTimeoutPlan",
    "QueryTimeoutPolicy",
    "QueryTimeoutPolicyRequest",
    "WorkloadAwareRetryQueryTimeoutPolicy",
    "build_query_timeout_policy_request",
    "resolve_query_timeout_plan",
    "resolve_query_timeout_policy",
]
