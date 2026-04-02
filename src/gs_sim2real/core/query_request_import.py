"""Stable query-request import interfaces for sim2real render servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RenderQueryDefaults:
    """Default render settings applied when the request omits optional fields."""

    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int
    timeout_ms: int = 10_000


@dataclass(frozen=True)
class RenderQuerySpec:
    """Normalized render request."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int


@dataclass(frozen=True)
class LocalizationImageBenchmarkQuerySpec:
    """Normalized localization image benchmark request."""

    ground_truth_bundle: dict[str, Any]
    estimate: Any
    alignment: str
    timeout_ms: int
    max_frames: int | None
    metrics: tuple[str, ...]
    lpips_net: str
    device: str


@dataclass(frozen=True)
class ImportedQueryRequest:
    """Normalized query request envelope."""

    request_type: str
    render: RenderQuerySpec | None = None
    image_benchmark: LocalizationImageBenchmarkQuerySpec | None = None


@dataclass(frozen=True)
class QueryRequestImportRequest:
    """Stable input contract for query request import decisions."""

    payload: Any
    defaults: RenderQueryDefaults


class QueryRequestImportPolicy(Protocol):
    """Minimal interface for interchangeable query request import policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def import_request(self, request: QueryRequestImportRequest) -> ImportedQueryRequest:
        """Import one raw payload into the normalized query-request schema."""


REQUEST_TYPE_ALIASES = {
    "render": "render",
    "localization-image-benchmark": "localization-image-benchmark",
    "image-benchmark": "localization-image-benchmark",
    "benchmark": "localization-image-benchmark",
}


def read_non_empty_string(value: Any) -> str:
    """Return a stripped string value or an empty string."""
    return value.strip() if isinstance(value, str) else ""


def normalize_number(value: Any, field_name: str) -> float:
    """Normalize a finite numeric field from a query payload."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    normalized = float(value)
    if not normalized == normalized or normalized in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    return normalized


def normalize_positive_number(value: Any, field_name: str) -> float:
    """Normalize a positive finite numeric field from a query payload."""
    normalized = normalize_number(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def normalize_positive_int(value: Any, field_name: str) -> int:
    """Normalize a positive integer field from a query payload."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be an integer")
    normalized = int(value)
    if float(normalized) != float(value):
        raise ValueError(f"{field_name} must be an integer")
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def normalize_non_negative_int(value: Any, field_name: str) -> int:
    """Normalize a non-negative integer field from a query payload."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        return 0
    return normalized


def normalize_vector(value: Any, field_name: str, *, length: int) -> tuple[float, ...]:
    """Normalize a fixed-length numeric vector from a query payload."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must be a list of length {length}")
    return tuple(normalize_number(component, f"{field_name}[{index}]") for index, component in enumerate(value))


def canonicalize_query_request_type(raw_value: str, *, default: str = "render") -> str:
    """Resolve query request type aliases to the stable request names."""
    normalized = read_non_empty_string(raw_value) or default
    canonical = REQUEST_TYPE_ALIASES.get(normalized)
    if canonical:
        return canonical
    raise ValueError(
        f"unsupported query payload type: {normalized}. Expected one of {', '.join(sorted(REQUEST_TYPE_ALIASES))}"
    )


def _require_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("query payload must be a JSON object")
    return payload


def _first_mapping(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any] | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        if key in mapping:
            return key, mapping.get(key)
    return None


def _first_scalar_or_container(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    found = _first_present(mapping, keys)
    return found[1] if found else None


def _normalize_metrics(metrics_payload: Any) -> tuple[str, ...]:
    if metrics_payload is None:
        return ("psnr", "ssim", "lpips")
    if not isinstance(metrics_payload, (list, tuple)) or not metrics_payload:
        raise ValueError("localization-image-benchmark metrics must be a non-empty list")
    metrics = tuple(
        read_non_empty_string(metric).lower() for metric in metrics_payload if read_non_empty_string(metric)
    )
    if not metrics:
        raise ValueError("localization-image-benchmark metrics must include at least one metric name")
    return metrics


def _parse_render_query_spec(
    body: dict[str, Any],
    *,
    defaults: RenderQueryDefaults,
    pose_keys: tuple[str, ...],
    position_keys: tuple[str, ...],
    orientation_keys: tuple[str, ...],
    width_keys: tuple[str, ...],
    height_keys: tuple[str, ...],
    fov_keys: tuple[str, ...],
    near_clip_keys: tuple[str, ...],
    far_clip_keys: tuple[str, ...],
    point_radius_keys: tuple[str, ...],
    allow_pose_shortcut: bool,
) -> RenderQuerySpec:
    pose_payload = _first_mapping(body, pose_keys)
    if pose_payload is None:
        if allow_pose_shortcut:
            pose_payload = body
        else:
            raise ValueError("query payload must include a pose object")

    position_value = _first_scalar_or_container(pose_payload, position_keys)
    orientation_value = _first_scalar_or_container(pose_payload, orientation_keys)
    position = normalize_vector(position_value, "pose.position", length=3)
    orientation = normalize_vector(orientation_value, "pose.orientation", length=4)

    width_value = _first_scalar_or_container(body, width_keys)
    height_value = _first_scalar_or_container(body, height_keys)
    fov_value = _first_scalar_or_container(body, fov_keys)
    near_clip_value = _first_scalar_or_container(body, near_clip_keys)
    far_clip_value = _first_scalar_or_container(body, far_clip_keys)
    point_radius_value = _first_scalar_or_container(body, point_radius_keys)

    return RenderQuerySpec(
        position=position,
        orientation=orientation,
        width=normalize_positive_int(defaults.width if width_value is None else width_value, "width"),
        height=normalize_positive_int(defaults.height if height_value is None else height_value, "height"),
        fov_degrees=normalize_positive_number(
            defaults.fov_degrees if fov_value is None else fov_value,
            "fovDegrees",
        ),
        near_clip=normalize_positive_number(
            defaults.near_clip if near_clip_value is None else near_clip_value,
            "nearClip",
        ),
        far_clip=normalize_positive_number(
            defaults.far_clip if far_clip_value is None else far_clip_value,
            "farClip",
        ),
        point_radius=normalize_non_negative_int(
            defaults.point_radius if point_radius_value is None else point_radius_value,
            "pointRadius",
        ),
    )


def _parse_benchmark_query_spec(
    body: dict[str, Any],
    *,
    defaults: RenderQueryDefaults,
    ground_truth_keys: tuple[str, ...],
    estimate_keys: tuple[str, ...],
    alignment_keys: tuple[str, ...],
    timeout_keys: tuple[str, ...],
    max_frames_keys: tuple[str, ...],
    metrics_keys: tuple[str, ...],
    lpips_net_keys: tuple[str, ...],
    device_keys: tuple[str, ...],
) -> LocalizationImageBenchmarkQuerySpec:
    ground_truth_value = _first_scalar_or_container(body, ground_truth_keys)
    estimate_value = _first_scalar_or_container(body, estimate_keys)

    if not isinstance(ground_truth_value, dict):
        raise ValueError("localization-image-benchmark requires groundTruthBundle")
    if estimate_value is None:
        raise ValueError("localization-image-benchmark requires estimate")

    alignment_value = _first_scalar_or_container(body, alignment_keys)
    timeout_value = _first_scalar_or_container(body, timeout_keys)
    max_frames_value = _first_scalar_or_container(body, max_frames_keys)
    metrics_value = _first_scalar_or_container(body, metrics_keys)
    lpips_net_value = _first_scalar_or_container(body, lpips_net_keys)
    device_value = _first_scalar_or_container(body, device_keys)

    return LocalizationImageBenchmarkQuerySpec(
        ground_truth_bundle=ground_truth_value,
        estimate=estimate_value,
        alignment=read_non_empty_string(alignment_value) or "auto",
        timeout_ms=(
            defaults.timeout_ms if timeout_value is None else normalize_positive_int(timeout_value, "timeoutMs")
        ),
        max_frames=(None if max_frames_value is None else normalize_positive_int(max_frames_value, "maxFrames")),
        metrics=_normalize_metrics(metrics_value),
        lpips_net=read_non_empty_string(lpips_net_value) or "alex",
        device=read_non_empty_string(device_value) or "cpu",
    )


class StrictSchemaQueryRequestImportPolicy:
    """Accept only the current canonical payload schema."""

    name = "strict_schema"
    label = "Strict Schema"
    style = "exact-contract"
    tier = "experiment"
    capabilities = {
        "respectsCanonicalSchema": True,
        "supportsEnvelopeWrappers": False,
        "supportsAliasKeys": False,
        "supportsPoseShortcuts": False,
    }

    def import_request(self, request: QueryRequestImportRequest) -> ImportedQueryRequest:
        payload = _require_mapping(request.payload)
        request_type = canonicalize_query_request_type(read_non_empty_string(payload.get("type")) or "render")
        if request_type == "render":
            return ImportedQueryRequest(
                request_type="render",
                render=_parse_render_query_spec(
                    payload,
                    defaults=request.defaults,
                    pose_keys=("pose",),
                    position_keys=("position",),
                    orientation_keys=("orientation",),
                    width_keys=("width",),
                    height_keys=("height",),
                    fov_keys=("fovDegrees",),
                    near_clip_keys=("nearClip",),
                    far_clip_keys=("farClip",),
                    point_radius_keys=("pointRadius",),
                    allow_pose_shortcut=False,
                ),
            )

        return ImportedQueryRequest(
            request_type="localization-image-benchmark",
            image_benchmark=_parse_benchmark_query_spec(
                payload,
                defaults=request.defaults,
                ground_truth_keys=("groundTruthBundle",),
                estimate_keys=("estimate",),
                alignment_keys=("alignment",),
                timeout_keys=("timeoutMs",),
                max_frames_keys=("maxFrames",),
                metrics_keys=("metrics",),
                lpips_net_keys=("lpipsNet",),
                device_keys=("device",),
            ),
        )


class EnvelopeFirstQueryRequestImportPolicy:
    """Prefer explicit request envelopes while still allowing some aliases inside them."""

    name = "envelope_first"
    label = "Envelope First"
    style = "wrapper-oriented"
    tier = "experiment"
    capabilities = {
        "respectsCanonicalSchema": True,
        "supportsEnvelopeWrappers": True,
        "supportsAliasKeys": True,
        "supportsPoseShortcuts": False,
    }

    def import_request(self, request: QueryRequestImportRequest) -> ImportedQueryRequest:
        payload = _require_mapping(request.payload)
        explicit_type = (
            read_non_empty_string(payload.get("type"))
            or read_non_empty_string(payload.get("requestType"))
            or read_non_empty_string(payload.get("kind"))
        )
        if explicit_type:
            request_type = canonicalize_query_request_type(explicit_type)
        elif _first_mapping(payload, ("benchmark", "localizationImageBenchmark")) is not None:
            request_type = "localization-image-benchmark"
        else:
            request_type = "render"

        if request_type == "render":
            body = _first_mapping(payload, ("render", "renderRequest", "request")) or payload
            return ImportedQueryRequest(
                request_type="render",
                render=_parse_render_query_spec(
                    body,
                    defaults=request.defaults,
                    pose_keys=("pose", "cameraPose"),
                    position_keys=("position",),
                    orientation_keys=("orientation", "quaternion"),
                    width_keys=("width", "imageWidth"),
                    height_keys=("height", "imageHeight"),
                    fov_keys=("fovDegrees", "fov"),
                    near_clip_keys=("nearClip", "near"),
                    far_clip_keys=("farClip", "far"),
                    point_radius_keys=("pointRadius", "radius"),
                    allow_pose_shortcut=False,
                ),
            )

        body = _first_mapping(payload, ("benchmark", "localizationImageBenchmark", "request")) or payload
        return ImportedQueryRequest(
            request_type="localization-image-benchmark",
            image_benchmark=_parse_benchmark_query_spec(
                body,
                defaults=request.defaults,
                ground_truth_keys=("groundTruthBundle", "groundTruth", "captureBundle"),
                estimate_keys=("estimate", "trajectory"),
                alignment_keys=("alignment", "alignmentMode"),
                timeout_keys=("timeoutMs", "responseTimeoutMs"),
                max_frames_keys=("maxFrames", "frameLimit"),
                metrics_keys=("metrics", "metricNames"),
                lpips_net_keys=("lpipsNet", "lpipsNetName"),
                device_keys=("device", "computeDevice"),
            ),
        )


class AliasFriendlyQueryRequestImportPolicy:
    """Accept thin client wrappers and common field aliases without changing the stable output."""

    name = "alias_friendly"
    label = "Alias Friendly"
    style = "compatibility-first"
    tier = "core"
    capabilities = {
        "respectsCanonicalSchema": True,
        "supportsEnvelopeWrappers": True,
        "supportsAliasKeys": True,
        "supportsPoseShortcuts": True,
    }

    def import_request(self, request: QueryRequestImportRequest) -> ImportedQueryRequest:
        payload = _require_mapping(request.payload)
        explicit_type = (
            read_non_empty_string(payload.get("type"))
            or read_non_empty_string(payload.get("requestType"))
            or read_non_empty_string(payload.get("kind"))
        )
        if explicit_type:
            request_type = canonicalize_query_request_type(explicit_type)
        elif any(
            key in payload
            for key in (
                "benchmark",
                "localizationImageBenchmark",
                "groundTruthBundle",
                "groundTruth",
                "estimate",
                "trajectory",
                "estimateTrajectory",
            )
        ):
            request_type = "localization-image-benchmark"
        else:
            request_type = "render"

        if request_type == "render":
            body = _first_mapping(payload, ("render", "renderRequest", "request", "payload")) or payload
            return ImportedQueryRequest(
                request_type="render",
                render=_parse_render_query_spec(
                    body,
                    defaults=request.defaults,
                    pose_keys=("pose", "cameraPose", "camera"),
                    position_keys=("position", "translation"),
                    orientation_keys=("orientation", "quaternion", "rotation"),
                    width_keys=("width", "imageWidth"),
                    height_keys=("height", "imageHeight"),
                    fov_keys=("fovDegrees", "fov", "fovDeg"),
                    near_clip_keys=("nearClip", "near"),
                    far_clip_keys=("farClip", "far"),
                    point_radius_keys=("pointRadius", "radius", "pixelRadius"),
                    allow_pose_shortcut=True,
                ),
            )

        body = _first_mapping(payload, ("benchmark", "localizationImageBenchmark", "request", "payload")) or payload
        return ImportedQueryRequest(
            request_type="localization-image-benchmark",
            image_benchmark=_parse_benchmark_query_spec(
                body,
                defaults=request.defaults,
                ground_truth_keys=("groundTruthBundle", "groundTruth", "captureBundle"),
                estimate_keys=("estimate", "trajectory", "estimateTrajectory"),
                alignment_keys=("alignment", "alignmentMode", "alignmentStrategy"),
                timeout_keys=("timeoutMs", "responseTimeoutMs"),
                max_frames_keys=("maxFrames", "frameLimit"),
                metrics_keys=("metrics", "metricNames", "imageMetrics"),
                lpips_net_keys=("lpipsNet", "lpipsNetName"),
                device_keys=("device", "computeDevice"),
            ),
        )


CORE_QUERY_REQUEST_IMPORT_POLICIES: dict[str, QueryRequestImportPolicy] = {
    "alias_friendly": AliasFriendlyQueryRequestImportPolicy(),
}


def import_query_request(
    request: QueryRequestImportRequest,
    *,
    policy: str = "alias_friendly",
) -> ImportedQueryRequest:
    """Import one raw query payload under the selected policy."""
    policies: dict[str, QueryRequestImportPolicy] = {
        "strict_schema": StrictSchemaQueryRequestImportPolicy(),
        "envelope_first": EnvelopeFirstQueryRequestImportPolicy(),
        **CORE_QUERY_REQUEST_IMPORT_POLICIES,
    }
    if policy not in policies:
        raise RuntimeError(
            f"unsupported query request import policy: {policy}. Expected one of {', '.join(sorted(policies))}"
        )
    return policies[policy].import_request(request)
