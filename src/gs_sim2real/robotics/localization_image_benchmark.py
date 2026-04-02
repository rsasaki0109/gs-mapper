"""Render-based image benchmarking for localization trajectories."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from gs_sim2real.core.localization_estimate_import import (
    LocalizationEstimateImportRequest,
    import_localization_estimate_document,
    normalize_localization_estimate as core_normalize_localization_estimate,
    parse_text_trajectory as core_parse_text_trajectory,
)
from gs_sim2real.core.localization_alignment import PoseSample, align_pose_samples
from gs_sim2real.core.route_capture_bundle_import import (
    RouteCaptureBundleImportRequest,
    import_route_capture_bundle,
)

from .gsplat_render_server import yaw_to_quaternion
from .render_query_client import decode_render_query_response, send_render_query


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


def resolve_position_vector(sample: dict[str, Any]) -> tuple[float, float, float] | None:
    """Extract a position vector from a pose-like payload."""
    for key in ("position", "translation"):
        value = sample.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized = tuple(float(item) for item in value[:3])
            if len(normalized) == 3 and all(math.isfinite(item) for item in normalized):
                return normalized

    values = (sample.get("x"), sample.get("y"), sample.get("z"))
    if all(value is not None for value in values):
        normalized = tuple(float(value) for value in values)
        if all(math.isfinite(item) for item in normalized):
            return normalized

    return None


def resolve_yaw_degrees(sample: dict[str, Any]) -> float:
    """Resolve yaw from explicit degrees, radians, or quaternion."""
    yaw_degrees = normalize_optional_metric_number(sample.get("yawDegrees"))
    if yaw_degrees is not None:
        return yaw_degrees

    yaw_radians = normalize_optional_metric_number(sample.get("yawRadians"))
    if yaw_radians is not None:
        return math.degrees(yaw_radians)

    return (
        quaternion_to_yaw_degrees(sample.get("orientation") or sample.get("rotation") or sample.get("quaternion"))
        or 0.0
    )


def resolve_timestamp_seconds(container: dict[str, Any], sample: dict[str, Any]) -> float | None:
    """Resolve the first usable timestamp candidate from a pose container."""
    candidates = (
        container.get("relativeTimeSeconds"),
        sample.get("relativeTimeSeconds"),
        container.get("timestampSeconds"),
        sample.get("timestampSeconds"),
        container.get("timestamp"),
        sample.get("timestamp"),
        container.get("timeSeconds"),
        sample.get("timeSeconds"),
        container.get("time"),
        sample.get("time"),
        container.get("capturedAt"),
        sample.get("capturedAt"),
        container.get("response", {}).get("relativeTimeSeconds")
        if isinstance(container.get("response"), dict)
        else None,
        container.get("response", {}).get("timestampSeconds") if isinstance(container.get("response"), dict) else None,
        container.get("response", {}).get("timestamp") if isinstance(container.get("response"), dict) else None,
        container.get("response", {}).get("capturedAt") if isinstance(container.get("response"), dict) else None,
    )
    for candidate in candidates:
        parsed = parse_timestamp_seconds_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def normalize_pose_sample(sample_like: Any, index: int) -> PoseSample:
    """Normalize one pose-like payload into a ``PoseSample``."""
    container = sample_like if isinstance(sample_like, dict) else {}
    pose = container.get("pose") if isinstance(container.get("pose"), dict) else container
    if not isinstance(pose, dict):
        raise ValueError(f"pose sample {index + 1} must be a JSON object")

    position = resolve_position_vector(pose)
    if position is None:
        raise ValueError(f"pose sample {index + 1} must include a finite 3D position")

    response = container.get("response") if isinstance(container.get("response"), dict) else None
    return PoseSample(
        index=index,
        label=read_non_empty_string(container.get("label")) or f"pose:{index + 1}",
        position=position,
        yaw_degrees=resolve_yaw_degrees(pose),
        timestamp_seconds=resolve_timestamp_seconds(container, pose),
        response=response,
    )


def parse_text_trajectory(raw_text: str, *, file_name: str | None = None) -> dict[str, Any]:
    """Parse a TUM/ORB-SLAM style text trajectory."""
    return core_parse_text_trajectory(raw_text, file_name=file_name)


def extract_estimate_pose_list(input_like: Any) -> tuple[list[Any], str]:
    """Extract an estimate pose list and a source label."""
    if isinstance(input_like, list):
        return input_like, "array"

    input_object = input_like if isinstance(input_like, dict) else {}
    if input_object.get("type") == "route-capture-bundle" and isinstance(input_object.get("captures"), list):
        return list(input_object["captures"]), "route-capture-bundle"

    for key in ("poses", "trajectory", "route", "samples", "estimates", "captures"):
        value = input_object.get(key)
        if isinstance(value, list):
            return list(value), key

    pose = input_object.get("pose")
    if isinstance(pose, dict):
        return [pose], "single-pose"

    return [], read_non_empty_string(input_object.get("type")) or "object"


def normalize_localization_estimate(input_like: Any) -> dict[str, Any]:
    """Normalize a localization estimate object into a stable schema."""
    return core_normalize_localization_estimate(input_like)


def parse_localization_estimate_document(raw_text: str, *, file_name: str | None = None) -> dict[str, Any]:
    """Parse JSON or text trajectory documents into a localization estimate."""
    return import_localization_estimate_document(
        LocalizationEstimateImportRequest(raw_text=str(raw_text or ""), file_name=file_name)
    )


def normalize_route_capture_bundle(bundle_like: Any) -> dict[str, Any]:
    """Normalize a route capture bundle for image benchmarking."""
    return import_route_capture_bundle(
        RouteCaptureBundleImportRequest(bundle_like),
    )


def extract_ground_truth_captures(bundle_like: Any) -> list[PoseSample]:
    """Extract normalized capture poses from a route capture bundle."""
    bundle = normalize_route_capture_bundle(bundle_like)
    return [normalize_pose_sample(capture, index) for index, capture in enumerate(bundle["captures"])]


def build_statistics(values: Sequence[float | None]) -> dict[str, float] | None:
    """Compute min/max/mean/median statistics for finite values."""
    finite_values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite_values:
        return None

    values_array = np.asarray(finite_values, dtype=np.float64)
    return {
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
    }


def build_metric_highlights(frame_artifacts: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build worst-frame highlights for each supported image metric."""
    highlights: dict[str, dict[str, Any]] = {}
    ordering = {
        "lpips": "max",
        "psnr": "min",
        "ssim": "min",
    }

    for metric_name, selection_mode in ordering.items():
        candidates = [
            artifact
            for artifact in frame_artifacts
            if math.isfinite(float(artifact.get("metrics", {}).get(metric_name, float("nan"))))
        ]
        if not candidates:
            continue

        selected = (
            max(candidates, key=lambda artifact: float(artifact["metrics"][metric_name]))
            if selection_mode == "max"
            else min(candidates, key=lambda artifact: float(artifact["metrics"][metric_name]))
        )
        highlights[metric_name] = {
            "ordering": selection_mode,
            "frameIndex": int(selected["frameIndex"]),
            "value": float(selected["metrics"][metric_name]),
            "groundTruthLabel": selected["groundTruthLabel"],
            "estimateLabel": selected["estimateLabel"],
            "interpolationKind": selected["interpolationKind"],
            "timeDeltaSeconds": selected["timeDeltaSeconds"],
            "groundTruthColorJpegBase64": selected["groundTruthColorJpegBase64"],
            "renderedColorJpegBase64": selected["renderedColorJpegBase64"],
        }

    return highlights


def decode_rgb_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    """Decode RGB JPEG bytes into a float32 array in the ``[0, 1]`` range."""
    image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def decode_rgb_jpeg_base64(payload: str) -> np.ndarray:
    """Decode a base64-encoded JPEG payload."""
    return decode_rgb_jpeg_bytes(base64.b64decode(payload, validate=True))


def compute_psnr(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
    """Compute PSNR in dB, clamped for exact matches."""
    mse = float(
        np.mean(np.square(np.asarray(candidate_rgb, dtype=np.float32) - np.asarray(reference_rgb, dtype=np.float32)))
    )
    if mse <= 1e-12:
        return 120.0
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
    """Compute a simplified SSIM metric using torch convolutions."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(reference_rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(np.asarray(candidate_rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)

    c1 = 0.01**2
    c2 = 0.03**2
    window_size = 11
    padding = window_size // 2
    window = torch.ones(1, 1, window_size, window_size, dtype=torch.float32) / (window_size * window_size)
    window = window.expand(3, 1, -1, -1)

    mu_x = F.conv2d(x, window, padding=padding, groups=3)
    mu_y = F.conv2d(y, window, padding=padding, groups=3)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=3) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=3) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=3) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(ssim_map.mean().item())


class LPIPSMetric:
    """Optional LPIPS metric wrapper with lazy loading."""

    def __init__(self, *, net: str, device: str) -> None:
        try:
            import lpips
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LPIPS requires the optional `lpips` package. Install with `python3 -m pip install lpips`."
            ) from exc

        self._torch = torch
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        # Upstream `lpips` still calls torchvision's deprecated `pretrained=`
        # AlexNet path; constrain the suppression to those two warnings only.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            self.model = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.model.eval()

    def __call__(self, reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
        torch = self._torch
        reference = (
            torch.from_numpy(np.asarray(reference_rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        candidate = (
            torch.from_numpy(np.asarray(candidate_rgb, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        reference = reference * 2.0 - 1.0
        candidate = candidate * 2.0 - 1.0

        # LPIPS backbones such as AlexNet are unstable on tiny fixtures; upsample
        # both images together so small validation frames still produce a score.
        height = int(reference.shape[-2])
        width = int(reference.shape[-1])
        if min(height, width) < 64:
            scale = max(64.0 / max(height, 1), 64.0 / max(width, 1))
            target_height = max(64, int(math.ceil(height * scale)))
            target_width = max(64, int(math.ceil(width * scale)))
            reference = torch.nn.functional.interpolate(
                reference,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            candidate = torch.nn.functional.interpolate(
                candidate,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        with torch.inference_mode():
            return float(self.model(reference, candidate).item())


def ensure_metric_support(
    metrics: Sequence[str], *, lpips_net: str, device: str
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Resolve requested image metrics into callables."""
    supported: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
    for metric in metrics:
        if metric == "psnr":
            supported[metric] = compute_psnr
        elif metric == "ssim":
            supported[metric] = compute_ssim
        elif metric == "lpips":
            supported[metric] = LPIPSMetric(net=lpips_net, device=device)
        else:
            raise ValueError(f"unsupported image metric: {metric}")
    return supported


def build_render_payload_for_pose(
    estimate_pose: PoseSample,
    *,
    ground_truth_response: dict[str, Any],
    bundle_request: dict[str, Any],
) -> dict[str, Any]:
    """Build a render query payload aligned to the ground-truth frame settings."""
    width = int(ground_truth_response.get("width") or bundle_request.get("width") or 640)
    height = int(ground_truth_response.get("height") or bundle_request.get("height") or 480)
    fov_degrees = float(ground_truth_response.get("fovDegrees") or bundle_request.get("fovDegrees") or 60.0)
    near_clip = float(ground_truth_response.get("nearClip") or bundle_request.get("nearClip") or 0.05)
    far_clip = float(ground_truth_response.get("farClip") or bundle_request.get("farClip") or 50.0)
    point_radius = int(ground_truth_response.get("pointRadius") or bundle_request.get("pointRadius") or 1)

    return {
        "type": "render",
        "pose": {
            "position": [float(value) for value in estimate_pose.position],
            "orientation": [float(value) for value in yaw_to_quaternion(math.radians(estimate_pose.yaw_degrees))],
        },
        "width": width,
        "height": height,
        "fovDegrees": fov_degrees,
        "nearClip": near_clip,
        "farClip": far_clip,
        "pointRadius": point_radius,
    }


def normalize_benchmark_inputs(
    *,
    run_document: dict[str, Any] | None,
    ground_truth_document: dict[str, Any] | None,
    estimate_document: dict[str, Any] | None,
    estimate_file_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Resolve run snapshot or explicit ground-truth/estimate inputs."""
    if run_document is not None:
        if run_document.get("type") != "localization-run-snapshot":
            raise ValueError("run input must be a localization-run-snapshot JSON export")
        ground_truth_bundle = run_document.get("groundTruth", {}).get("bundle")
        if ground_truth_bundle is None and ground_truth_document is None:
            raise ValueError("run snapshot does not include groundTruth.bundle; provide --ground-truth explicitly")
        estimate_input = run_document.get("estimate")
        if estimate_input is None and estimate_document is None:
            raise ValueError("run snapshot does not include estimate; provide --estimate explicitly")
        return (
            normalize_route_capture_bundle(ground_truth_document or ground_truth_bundle),
            normalize_localization_estimate(estimate_document or estimate_input),
            read_non_empty_string(run_document.get("benchmark", {}).get("requestedAlignment")) or None,
        )

    if ground_truth_document is None:
        raise ValueError("--ground-truth is required unless --run already includes ground truth")
    if estimate_document is None:
        raise ValueError("--estimate is required unless --run already includes estimate")

    if isinstance(estimate_document, str):
        normalized_estimate = parse_localization_estimate_document(estimate_document, file_name=estimate_file_name)
    else:
        normalized_estimate = normalize_localization_estimate(estimate_document)

    return normalize_route_capture_bundle(ground_truth_document), normalized_estimate, None


def benchmark_localization_images(
    *,
    endpoint: str,
    ground_truth_bundle: dict[str, Any],
    estimate_input: dict[str, Any],
    alignment: str = "auto",
    timeout_ms: int = 10000,
    max_frames: int | None = None,
    metrics: Sequence[str] = ("psnr", "ssim", "lpips"),
    lpips_net: str = "alex",
    device: str = "cpu",
    query_fn: Callable[[str, dict[str, Any], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render estimate poses and compare them against ground-truth RGB captures."""
    normalized_ground_truth_bundle = normalize_route_capture_bundle(ground_truth_bundle)
    ground_truth_captures = extract_ground_truth_captures(normalized_ground_truth_bundle)
    normalized_estimate = normalize_localization_estimate(estimate_input)

    requested_alignment = read_non_empty_string(alignment) or "auto"
    ground_truth_timestamped_count = sum(pose.timestamp_seconds is not None for pose in ground_truth_captures)
    estimate_timestamped_count = sum(pose.timestamp_seconds is not None for pose in normalized_estimate["poses"])
    resolved_alignment, matched_pairs = align_pose_samples(
        ground_truth_captures,
        normalized_estimate["poses"],
        alignment=requested_alignment,
    )
    if max_frames is not None:
        matched_pairs = matched_pairs[: max(1, int(max_frames))]
    if not matched_pairs:
        raise ValueError("image benchmark requires at least one matched pose")

    metric_functions = ensure_metric_support(metrics, lpips_net=lpips_net, device=device)
    request_fn = query_fn or (
        lambda query_endpoint, payload, current_timeout_ms: send_render_query(
            query_endpoint, payload, timeout_ms=current_timeout_ms
        )
    )

    frames: list[dict[str, Any]] = []
    frame_artifacts: list[dict[str, Any]] = []
    metric_values: dict[str, list[float | None]] = {metric_name: [] for metric_name in metric_functions}

    for pair in matched_pairs:
        ground_truth_pose = pair.ground_truth
        if ground_truth_pose.response is None:
            raise ValueError("ground truth captures must include render-result responses")
        payload = build_render_payload_for_pose(
            pair.estimate,
            ground_truth_response=ground_truth_pose.response,
            bundle_request=normalized_ground_truth_bundle["request"],
        )
        render_response = request_fn(endpoint, payload, timeout_ms)
        decoded_render = decode_render_query_response(render_response)
        ground_truth_rgb = decode_rgb_jpeg_base64(str(ground_truth_pose.response["colorJpegBase64"]))
        rendered_rgb = decode_rgb_jpeg_bytes(decoded_render.color_jpeg)

        if rendered_rgb.shape != ground_truth_rgb.shape:
            image = Image.fromarray((np.clip(rendered_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
            image = image.resize((ground_truth_rgb.shape[1], ground_truth_rgb.shape[0]), resample=Image.BILINEAR)
            rendered_rgb = np.asarray(image, dtype=np.float32) / 255.0

        frame_metrics: dict[str, float] = {}
        for metric_name, metric_fn in metric_functions.items():
            metric_value = metric_fn(ground_truth_rgb, rendered_rgb)
            frame_metrics[metric_name] = float(metric_value)
            metric_values[metric_name].append(float(metric_value))

        frames.append(
            {
                "index": len(frames),
                "pairIndex": pair.pair_index,
                "groundTruthLabel": ground_truth_pose.label,
                "estimateLabel": pair.estimate.label,
                "timeDeltaSeconds": pair.time_delta_seconds,
                "interpolationKind": pair.interpolation_kind,
                "groundTruth": {
                    "position": list(ground_truth_pose.position),
                    "yawDegrees": ground_truth_pose.yaw_degrees,
                    "timestampSeconds": ground_truth_pose.timestamp_seconds,
                },
                "estimate": {
                    "position": list(pair.estimate.position),
                    "yawDegrees": pair.estimate.yaw_degrees,
                    "timestampSeconds": pair.estimate.timestamp_seconds,
                },
                "request": payload,
                "renderResult": {
                    "width": decoded_render.width,
                    "height": decoded_render.height,
                    "frameId": decoded_render.camera_info.get("frameId"),
                    "colorBytes": len(decoded_render.color_jpeg),
                    "depthBytes": int(decoded_render.depth.nbytes),
                },
                "metrics": frame_metrics,
            }
        )
        frame_artifacts.append(
            {
                "frameIndex": len(frames) - 1,
                "groundTruthLabel": ground_truth_pose.label,
                "estimateLabel": pair.estimate.label,
                "interpolationKind": pair.interpolation_kind,
                "timeDeltaSeconds": pair.time_delta_seconds,
                "metrics": frame_metrics,
                "groundTruthColorJpegBase64": str(ground_truth_pose.response["colorJpegBase64"]),
                "renderedColorJpegBase64": str(render_response["colorJpegBase64"]),
            }
        )

    summary_metrics = {metric_name: build_statistics(values) for metric_name, values in metric_values.items()}

    return {
        "protocol": "dreamwalker-localization-image-benchmark/v1",
        "type": "localization-image-benchmark-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "alignment": resolved_alignment,
        "requestedAlignment": requested_alignment,
        "groundTruth": {
            "fragmentId": normalized_ground_truth_bundle["fragmentId"],
            "fragmentLabel": normalized_ground_truth_bundle["fragmentLabel"],
            "capturedAt": normalized_ground_truth_bundle["capturedAt"],
            "poseCount": len(ground_truth_captures),
            "timestampedPoseCount": ground_truth_timestamped_count,
        },
        "estimate": {
            "label": normalized_estimate["label"],
            "sourceType": normalized_estimate["sourceType"],
            "poseCount": len(normalized_estimate["poses"]),
            "timestampedPoseCount": estimate_timestamped_count,
        },
        "matching": {
            "matchedCount": len(matched_pairs),
            "groundTruthCount": len(ground_truth_captures),
            "estimateCount": len(normalized_estimate["poses"]),
        },
        "metrics": {
            "requested": list(metrics),
            "summary": summary_metrics,
            "highlights": build_metric_highlights(frame_artifacts),
        },
        "frames": frames,
    }


def load_json_document(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a Python object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_benchmark_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact stdout summary from a benchmark report."""
    metric_summary = report.get("metrics", {}).get("summary", {})
    return {
        "type": "localization-image-benchmark-summary",
        "endpoint": report.get("endpoint"),
        "alignment": report.get("alignment"),
        "estimateLabel": report.get("estimate", {}).get("label"),
        "matchedCount": report.get("matching", {}).get("matchedCount"),
        "psnrMean": metric_summary.get("psnr", {}).get("mean")
        if isinstance(metric_summary.get("psnr"), dict)
        else None,
        "ssimMean": metric_summary.get("ssim", {}).get("mean")
        if isinstance(metric_summary.get("ssim"), dict)
        else None,
        "lpipsMean": metric_summary.get("lpips", {}).get("mean")
        if isinstance(metric_summary.get("lpips"), dict)
        else None,
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the localization image benchmark CLI."""
    run_document = load_json_document(args.run) if args.run else None
    ground_truth_document = load_json_document(args.ground_truth) if args.ground_truth else None
    estimate_document: dict[str, Any] | str | None = None
    estimate_file_name: str | None = None
    if args.estimate:
        estimate_path = Path(args.estimate)
        estimate_file_name = estimate_path.name
        text = estimate_path.read_text(encoding="utf-8")
        estimate_document = text if text.strip() and not text.lstrip().startswith(("{", "[")) else json.loads(text)

    ground_truth_bundle, estimate_input, preferred_alignment = normalize_benchmark_inputs(
        run_document=run_document,
        ground_truth_document=ground_truth_document,
        estimate_document=estimate_document,
        estimate_file_name=estimate_file_name,
    )
    report = benchmark_localization_images(
        endpoint=args.endpoint,
        ground_truth_bundle=ground_truth_bundle,
        estimate_input=estimate_input,
        alignment=preferred_alignment if args.alignment == "auto" and preferred_alignment else args.alignment,
        timeout_ms=args.timeout_ms,
        max_frames=args.max_frames,
        metrics=args.metrics,
        lpips_net=args.lpips_net,
        device=args.device,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(build_benchmark_summary(report), indent=2))
