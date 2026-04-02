"""Stable query-response builders for sim2real transports."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol


QUERY_RESPONSE_PROTOCOL_ID = "dreamwalker-sim2real-query/v1"


@dataclass(frozen=True)
class QueryRenderResultResponseInput:
    """Normalized source payload for one render-result response."""

    frame_id: str
    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    camera_info: dict[str, Any]
    color_jpeg_bytes: bytes
    depth_float32_bytes: bytes
    protocol: str = QUERY_RESPONSE_PROTOCOL_ID
    color_encoding: str = "jpeg"
    depth_encoding: str = "32FC1"


@dataclass(frozen=True)
class QueryReadyDefaults:
    """Shared query defaults announced by interactive transports."""

    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int


@dataclass(frozen=True)
class QueryReadyResponseInput:
    """Normalized source payload for one query-ready response."""

    transport: str
    endpoint: str
    frame_id: str
    renderer: str
    renderer_reason: str
    request_types: tuple[str, ...]
    defaults: QueryReadyDefaults
    protocol: str = QUERY_RESPONSE_PROTOCOL_ID


@dataclass(frozen=True)
class QueryErrorResponseInput:
    """Normalized source payload for one error response."""

    error: str
    error_type: str = ""
    error_code: str = ""
    protocol: str = QUERY_RESPONSE_PROTOCOL_ID


class QueryResponseBuildPolicy(Protocol):
    """Minimal interface for interchangeable query-response build policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def build_render_result(self, response_input: QueryRenderResultResponseInput) -> dict[str, Any]:
        """Build one render-result response document."""

    def build_query_ready(self, response_input: QueryReadyResponseInput) -> dict[str, Any]:
        """Build one query-ready handshake document."""

    def build_query_error(self, response_input: QueryErrorResponseInput) -> dict[str, Any]:
        """Build one transport-safe error document."""


def _encode_base64_ascii(payload: bytes) -> str:
    return base64.b64encode(bytes(payload)).decode("ascii")


def _copy_camera_info(camera_info: dict[str, Any]) -> dict[str, Any]:
    return dict(camera_info)


class MinimalEnvelopeQueryResponseBuildPolicy:
    """Prefer the smallest payload that still carries render data."""

    name = "minimal_envelope"
    label = "Minimal Envelope"
    style = "minimal"
    tier = "experimental"
    capabilities = {
        "preservesCanonicalProtocol": True,
        "includesRenderSettings": False,
        "includesRequestCatalog": False,
        "includesDiagnosticMeta": False,
    }

    def build_render_result(self, response_input: QueryRenderResultResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "render-result",
            "width": int(response_input.width),
            "height": int(response_input.height),
            "pose": {
                "position": list(response_input.position),
                "orientation": list(response_input.orientation),
            },
            "cameraInfo": _copy_camera_info(response_input.camera_info),
            "colorJpegBase64": _encode_base64_ascii(response_input.color_jpeg_bytes),
            "depthBase64": _encode_base64_ascii(response_input.depth_float32_bytes),
        }

    def build_query_ready(self, response_input: QueryReadyResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "query-ready",
            "transport": response_input.transport,
            "endpoint": response_input.endpoint,
        }

    def build_query_error(self, response_input: QueryErrorResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "error",
            "error": str(response_input.error),
        }


class BrowserObservableQueryResponseBuildPolicy:
    """Preserve the current browser-facing payload shape."""

    name = "browser_observable"
    label = "Browser Observable"
    style = "canonical"
    tier = "core"
    capabilities = {
        "preservesCanonicalProtocol": True,
        "includesRenderSettings": True,
        "includesRequestCatalog": True,
        "includesDiagnosticMeta": False,
    }

    def build_render_result(self, response_input: QueryRenderResultResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "render-result",
            "frameId": response_input.frame_id,
            "width": int(response_input.width),
            "height": int(response_input.height),
            "fovDegrees": float(response_input.fov_degrees),
            "nearClip": float(response_input.near_clip),
            "farClip": float(response_input.far_clip),
            "pointRadius": int(response_input.point_radius),
            "pose": {
                "position": list(response_input.position),
                "orientation": list(response_input.orientation),
            },
            "cameraInfo": _copy_camera_info(response_input.camera_info),
            "colorEncoding": response_input.color_encoding,
            "colorJpegBase64": _encode_base64_ascii(response_input.color_jpeg_bytes),
            "depthEncoding": response_input.depth_encoding,
            "depthBase64": _encode_base64_ascii(response_input.depth_float32_bytes),
        }

    def build_query_ready(self, response_input: QueryReadyResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "query-ready",
            "transport": response_input.transport,
            "endpoint": response_input.endpoint,
            "frameId": response_input.frame_id,
            "renderer": response_input.renderer,
            "rendererReason": response_input.renderer_reason,
            "requestTypes": list(response_input.request_types),
            "defaults": {
                "width": int(response_input.defaults.width),
                "height": int(response_input.defaults.height),
                "fovDegrees": float(response_input.defaults.fov_degrees),
                "nearClip": float(response_input.defaults.near_clip),
                "farClip": float(response_input.defaults.far_clip),
                "pointRadius": int(response_input.defaults.point_radius),
            },
        }

    def build_query_error(self, response_input: QueryErrorResponseInput) -> dict[str, Any]:
        return {
            "protocol": response_input.protocol,
            "type": "error",
            "error": str(response_input.error),
        }


class DiagnosticMetaQueryResponseBuildPolicy:
    """Keep the canonical payload while adding deterministic diagnostics."""

    name = "diagnostic_meta"
    label = "Diagnostic Meta"
    style = "telemetry_rich"
    tier = "experimental"
    capabilities = {
        "preservesCanonicalProtocol": True,
        "includesRenderSettings": True,
        "includesRequestCatalog": True,
        "includesDiagnosticMeta": True,
    }

    def build_render_result(self, response_input: QueryRenderResultResponseInput) -> dict[str, Any]:
        response = BrowserObservableQueryResponseBuildPolicy().build_render_result(response_input)
        response["meta"] = {
            "policy": self.name,
            "colorBytes": len(response_input.color_jpeg_bytes),
            "depthBytes": len(response_input.depth_float32_bytes),
        }
        return response

    def build_query_ready(self, response_input: QueryReadyResponseInput) -> dict[str, Any]:
        response = BrowserObservableQueryResponseBuildPolicy().build_query_ready(response_input)
        response["meta"] = {
            "policy": self.name,
            "requestTypeCount": len(response_input.request_types),
            "supportsBenchmarkRequests": "localization-image-benchmark" in response_input.request_types,
        }
        return response

    def build_query_error(self, response_input: QueryErrorResponseInput) -> dict[str, Any]:
        response = BrowserObservableQueryResponseBuildPolicy().build_query_error(response_input)
        response["meta"] = {
            "policy": self.name,
            "errorType": response_input.error_type or "RuntimeError",
            "errorCode": response_input.error_code or "query_error",
        }
        return response


CORE_QUERY_RESPONSE_BUILD_POLICIES: tuple[QueryResponseBuildPolicy, ...] = (
    MinimalEnvelopeQueryResponseBuildPolicy(),
    BrowserObservableQueryResponseBuildPolicy(),
    DiagnosticMetaQueryResponseBuildPolicy(),
)


def resolve_query_response_build_policy(
    policy: str | QueryResponseBuildPolicy = "browser_observable",
) -> QueryResponseBuildPolicy:
    """Resolve a query-response build policy by name."""
    if not isinstance(policy, str):
        return policy
    normalized = policy.strip().lower() or "browser_observable"
    for candidate in CORE_QUERY_RESPONSE_BUILD_POLICIES:
        if candidate.name == normalized:
            return candidate
    raise ValueError(
        "unknown query response build policy: "
        f"{policy}. Expected one of {', '.join(item.name for item in CORE_QUERY_RESPONSE_BUILD_POLICIES)}"
    )


def build_render_result_response_document(
    response_input: QueryRenderResultResponseInput,
    *,
    policy: str | QueryResponseBuildPolicy = "browser_observable",
) -> dict[str, Any]:
    """Build a render-result response document."""
    return resolve_query_response_build_policy(policy).build_render_result(response_input)


def build_query_ready_response_document(
    response_input: QueryReadyResponseInput,
    *,
    policy: str | QueryResponseBuildPolicy = "browser_observable",
) -> dict[str, Any]:
    """Build a query-ready response document."""
    return resolve_query_response_build_policy(policy).build_query_ready(response_input)


def build_query_error_response_document(
    response_input: QueryErrorResponseInput,
    *,
    policy: str | QueryResponseBuildPolicy = "browser_observable",
) -> dict[str, Any]:
    """Build an error response document."""
    return resolve_query_response_build_policy(policy).build_query_error(response_input)


__all__ = [
    "BrowserObservableQueryResponseBuildPolicy",
    "CORE_QUERY_RESPONSE_BUILD_POLICIES",
    "DiagnosticMetaQueryResponseBuildPolicy",
    "MinimalEnvelopeQueryResponseBuildPolicy",
    "QUERY_RESPONSE_PROTOCOL_ID",
    "QueryErrorResponseInput",
    "QueryReadyDefaults",
    "QueryReadyResponseInput",
    "QueryRenderResultResponseInput",
    "QueryResponseBuildPolicy",
    "build_query_error_response_document",
    "build_query_ready_response_document",
    "build_render_result_response_document",
    "resolve_query_response_build_policy",
]
