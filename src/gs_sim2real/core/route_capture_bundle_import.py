"""Stable route-capture bundle import interfaces for sim2real benchmarking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence


def read_non_empty_string(value: Any) -> str:
    """Return a stripped string or an empty string."""
    return value.strip() if isinstance(value, str) else ""


def parse_timestamp_seconds_candidate(value: Any) -> float | None:
    """Parse numeric or ISO-like timestamps into seconds."""
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None

    text = read_non_empty_string(value)
    if not text:
        return None

    try:
        numeric_value = float(text)
    except ValueError:
        numeric_value = None
    if numeric_value is not None and math.isfinite(numeric_value):
        return numeric_value

    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def normalize_vector(value: Any, *, field_name: str, length: int) -> tuple[float, ...]:
    """Normalize a finite vector payload."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of {length} finite numbers")
    normalized = tuple(float(item) for item in value[:length])
    if len(normalized) != length or any(not math.isfinite(item) for item in normalized):
        raise ValueError(f"{field_name} must be a sequence of {length} finite numbers")
    return normalized


def normalize_optional_metric_number(value: Any) -> float | None:
    """Normalize a finite metric value or return ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def quaternion_to_yaw_degrees(quaternion_like: Any) -> float | None:
    """Extract yaw in degrees from a quaternion-like payload."""
    quaternion: tuple[float, float, float, float] | None = None
    if isinstance(quaternion_like, Sequence) and not isinstance(quaternion_like, (str, bytes)):
        values = tuple(float(item) for item in quaternion_like[:4])
        if len(values) == 4 and all(math.isfinite(item) for item in values):
            quaternion = values
    elif isinstance(quaternion_like, dict):
        values = (
            float(quaternion_like.get("x", float("nan"))),
            float(quaternion_like.get("y", float("nan"))),
            float(quaternion_like.get("z", float("nan"))),
            float(quaternion_like.get("w", float("nan"))),
        )
        if all(math.isfinite(item) for item in values):
            quaternion = values

    if quaternion is None:
        return None

    x, y, z, w = quaternion
    sin_yaw = 2.0 * (w * y + x * z)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(sin_yaw, cos_yaw))


@dataclass(frozen=True)
class RouteCaptureBundleImportRequest:
    """Stable input contract for route capture bundle import decisions."""

    input_like: Any


class RouteCaptureBundleImportPolicy(Protocol):
    """Minimal interface for interchangeable route capture bundle import policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def import_bundle(self, request: RouteCaptureBundleImportRequest) -> dict[str, Any]:
        """Import one raw route-capture bundle into the normalized schema."""


def _require_bundle_mapping(input_like: Any) -> dict[str, Any]:
    bundle = input_like if isinstance(input_like, dict) else {}
    if bundle.get("type") != "route-capture-bundle":
        raise ValueError("ground truth must be a route-capture-bundle")
    return bundle


def _require_captures(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    captures_like = bundle.get("captures")
    if not isinstance(captures_like, list) or not captures_like:
        raise ValueError("image benchmark ground truth requires capture responses with RGB frames")
    return [capture if isinstance(capture, dict) else {} for capture in captures_like]


def _extract_route_entries(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    route_like = bundle.get("route")
    if not isinstance(route_like, list):
        return []
    return [entry if isinstance(entry, dict) else {} for entry in route_like]


def _extract_render_response(capture: dict[str, Any]) -> dict[str, Any]:
    response = capture.get("response")
    if not isinstance(response, dict) or response.get("type") != "render-result":
        raise ValueError("capture bundle entries must include render-result responses")
    if not read_non_empty_string(response.get("colorJpegBase64")):
        raise ValueError("capture responses must include colorJpegBase64")
    return response


def _resolve_position_from_mapping(mapping: dict[str, Any], *, field_name: str) -> list[float] | None:
    value = mapping.get("position")
    if value is None:
        return None
    return list(normalize_vector(value, field_name=field_name, length=3))


def _resolve_pose_from_capture_strict(capture: dict[str, Any], *, index: int) -> dict[str, Any]:
    pose_like = capture.get("pose")
    if not isinstance(pose_like, dict):
        raise ValueError(f"capture bundle entries must include pose objects (capture {index + 1})")
    position = _resolve_position_from_mapping(
        pose_like,
        field_name="capture.pose.position",
    )
    if position is None:
        raise ValueError(f"capture bundle entries must include pose.position (capture {index + 1})")
    return {
        "position": position,
        "yawDegrees": float(pose_like.get("yawDegrees", 0.0)),
    }


def _resolve_pose_from_response_fallback(
    capture: dict[str, Any], response: dict[str, Any], *, index: int
) -> dict[str, Any]:
    pose_like = capture.get("pose") if isinstance(capture.get("pose"), dict) else {}
    position = _resolve_position_from_mapping(
        pose_like,
        field_name="capture.pose.position",
    )
    response_pose = response.get("pose") if isinstance(response.get("pose"), dict) else {}
    if position is None:
        position = _resolve_position_from_mapping(
            response_pose,
            field_name="response.pose.position",
        )
    if position is None:
        raise ValueError(
            f"capture bundle entries must include pose.position or response.pose.position (capture {index + 1})"
        )

    yaw_degrees = normalize_optional_metric_number(pose_like.get("yawDegrees"))
    if yaw_degrees is None:
        yaw_degrees = quaternion_to_yaw_degrees(response_pose.get("orientation"))
    return {
        "position": position,
        "yawDegrees": float(yaw_degrees or 0.0),
    }


def _resolve_pose_from_route_aware(
    capture: dict[str, Any],
    response: dict[str, Any],
    route_entry: dict[str, Any] | None,
    *,
    index: int,
) -> dict[str, Any]:
    pose_like = capture.get("pose") if isinstance(capture.get("pose"), dict) else {}
    route_pose = route_entry.get("pose") if route_entry and isinstance(route_entry.get("pose"), dict) else {}
    response_pose = response.get("pose") if isinstance(response.get("pose"), dict) else {}

    position = _resolve_position_from_mapping(
        pose_like,
        field_name="capture.pose.position",
    )
    if position is None and route_entry:
        position = _resolve_position_from_mapping(
            route_entry,
            field_name="route.position",
        )
    if position is None and route_pose:
        position = _resolve_position_from_mapping(
            route_pose,
            field_name="route.pose.position",
        )
    if position is None:
        position = _resolve_position_from_mapping(
            response_pose,
            field_name="response.pose.position",
        )
    if position is None:
        raise ValueError(
            f"capture bundle entries must include pose.position, route position, or response.pose.position (capture {index + 1})"
        )

    yaw_degrees = normalize_optional_metric_number(pose_like.get("yawDegrees"))
    if yaw_degrees is None and route_entry:
        yaw_degrees = normalize_optional_metric_number(route_entry.get("yawDegrees"))
    if yaw_degrees is None and route_pose:
        yaw_degrees = normalize_optional_metric_number(route_pose.get("yawDegrees"))
    if yaw_degrees is None:
        yaw_degrees = quaternion_to_yaw_degrees(response_pose.get("orientation"))
    return {
        "position": position,
        "yawDegrees": float(yaw_degrees or 0.0),
    }


def _build_normalized_bundle(bundle: dict[str, Any], captures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": read_non_empty_string(bundle.get("protocol")) or "dreamwalker-sim2real-capture/v1",
        "type": "route-capture-bundle",
        "capturedAt": read_non_empty_string(bundle.get("capturedAt")),
        "fragmentId": read_non_empty_string(bundle.get("fragmentId")),
        "fragmentLabel": read_non_empty_string(bundle.get("fragmentLabel")),
        "endpoint": read_non_empty_string(bundle.get("endpoint")),
        "request": bundle.get("request") if isinstance(bundle.get("request"), dict) else {},
        "captures": captures,
    }


class StrictCanonicalRouteCaptureBundleImportPolicy:
    """Accept only canonical capture bundles with explicit capture poses."""

    name = "strict_canonical"
    label = "Strict Canonical"
    style = "exact-contract"
    tier = "experiment"
    capabilities = {
        "supportsCanonicalCapturePose": True,
        "supportsResponsePoseFallback": False,
        "supportsRoutePoseFallback": False,
    }

    def import_bundle(self, request: RouteCaptureBundleImportRequest) -> dict[str, Any]:
        bundle = _require_bundle_mapping(request.input_like)
        captures: list[dict[str, Any]] = []
        for index, capture in enumerate(_require_captures(bundle)):
            response = _extract_render_response(capture)
            captures.append(
                {
                    "index": int(capture.get("index", index)),
                    "label": read_non_empty_string(capture.get("label")) or f"capture:{index + 1}",
                    "capturedAt": read_non_empty_string(capture.get("capturedAt")),
                    "relativeTimeSeconds": parse_timestamp_seconds_candidate(capture.get("relativeTimeSeconds")),
                    "pose": _resolve_pose_from_capture_strict(capture, index=index),
                    "response": response,
                }
            )
        return _build_normalized_bundle(bundle, captures)


class ResponsePoseFallbackRouteCaptureBundleImportPolicy:
    """Recover capture poses from render-result responses when capture.pose is partial."""

    name = "response_pose_fallback"
    label = "Response Pose Fallback"
    style = "response-oriented"
    tier = "experiment"
    capabilities = {
        "supportsCanonicalCapturePose": True,
        "supportsResponsePoseFallback": True,
        "supportsRoutePoseFallback": False,
    }

    def import_bundle(self, request: RouteCaptureBundleImportRequest) -> dict[str, Any]:
        bundle = _require_bundle_mapping(request.input_like)
        captures: list[dict[str, Any]] = []
        for index, capture in enumerate(_require_captures(bundle)):
            response = _extract_render_response(capture)
            captures.append(
                {
                    "index": int(capture.get("index", index)),
                    "label": read_non_empty_string(capture.get("label")) or f"capture:{index + 1}",
                    "capturedAt": read_non_empty_string(capture.get("capturedAt")),
                    "relativeTimeSeconds": parse_timestamp_seconds_candidate(capture.get("relativeTimeSeconds")),
                    "pose": _resolve_pose_from_response_fallback(capture, response, index=index),
                    "response": response,
                }
            )
        return _build_normalized_bundle(bundle, captures)


class RouteAwareRouteCaptureBundleImportPolicy:
    """Recover capture poses from capture.pose, route entries, or render-result responses."""

    name = "route_aware"
    label = "Route Aware"
    style = "bundle-aware"
    tier = "core"
    capabilities = {
        "supportsCanonicalCapturePose": True,
        "supportsResponsePoseFallback": True,
        "supportsRoutePoseFallback": True,
    }

    def import_bundle(self, request: RouteCaptureBundleImportRequest) -> dict[str, Any]:
        bundle = _require_bundle_mapping(request.input_like)
        route_entries = _extract_route_entries(bundle)
        captures: list[dict[str, Any]] = []
        for index, capture in enumerate(_require_captures(bundle)):
            response = _extract_render_response(capture)
            route_entry = route_entries[index] if index < len(route_entries) else None
            captures.append(
                {
                    "index": int(capture.get("index", index)),
                    "label": read_non_empty_string(capture.get("label")) or f"capture:{index + 1}",
                    "capturedAt": read_non_empty_string(capture.get("capturedAt")),
                    "relativeTimeSeconds": parse_timestamp_seconds_candidate(capture.get("relativeTimeSeconds")),
                    "pose": _resolve_pose_from_route_aware(capture, response, route_entry, index=index),
                    "response": response,
                }
            )
        return _build_normalized_bundle(bundle, captures)


CORE_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES: dict[str, RouteCaptureBundleImportPolicy] = {
    "route_aware": RouteAwareRouteCaptureBundleImportPolicy(),
}


def import_route_capture_bundle(
    request: RouteCaptureBundleImportRequest,
    *,
    policy: str = "route_aware",
) -> dict[str, Any]:
    """Import one route-capture bundle under the selected policy."""
    policies: dict[str, RouteCaptureBundleImportPolicy] = {
        "strict_canonical": StrictCanonicalRouteCaptureBundleImportPolicy(),
        "response_pose_fallback": ResponsePoseFallbackRouteCaptureBundleImportPolicy(),
        **CORE_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES,
    }
    if policy not in policies:
        raise RuntimeError(
            f"unsupported route capture bundle import policy: {policy}. Expected one of {', '.join(sorted(policies))}"
        )
    return policies[policy].import_bundle(request)
