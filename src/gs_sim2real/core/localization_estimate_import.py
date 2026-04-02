"""Stable importer interfaces for localization estimate documents."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from .localization_alignment import PoseSample


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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


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


def derive_estimate_label(file_name: str | None) -> str:
    """Build a stable fallback label from a file name."""
    if file_name:
        stem = Path(file_name).stem.strip()
        if stem:
            return stem
    return "Localization Estimate"


def strip_leading_comment_lines(raw_text: str) -> str:
    """Drop leading comment lines before attempting JSON parse."""
    lines = str(raw_text or "").splitlines()
    start_index = 0
    while start_index < len(lines):
        candidate = lines[start_index].strip()
        if not candidate or candidate.startswith(("#", "//", "%")):
            start_index += 1
            continue
        break
    return "\n".join(lines[start_index:]).strip()


def parse_text_trajectory(raw_text: str, *, file_name: str | None = None) -> dict[str, Any]:
    """Parse a TUM/ORB-SLAM style text trajectory."""
    poses: list[dict[str, Any]] = []
    for line in str(raw_text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "%")):
            continue
        tokens = [token for token in stripped.replace(",", " ").split() if token]
        if len(tokens) < 7:
            continue
        try:
            numeric_tokens = [float(token) for token in tokens]
        except ValueError:
            continue

        if len(numeric_tokens) >= 8:
            poses.append(
                {
                    "index": len(poses),
                    "label": f"pose:{len(poses) + 1}",
                    "timestamp": numeric_tokens[0],
                    "position": numeric_tokens[1:4],
                    "orientation": numeric_tokens[4:8],
                }
            )
        elif len(numeric_tokens) >= 7:
            poses.append(
                {
                    "index": len(poses),
                    "label": f"pose:{len(poses) + 1}",
                    "position": numeric_tokens[0:3],
                    "orientation": numeric_tokens[3:7],
                }
            )

    if not poses:
        raise ValueError("text trajectory must contain lines like: timestamp tx ty tz qx qy qz qw")

    return {
        "protocol": "tum-trajectory-text/v1",
        "type": "localization-estimate",
        "sourceType": "tum-trajectory-text",
        "label": derive_estimate_label(file_name),
        "poses": poses,
    }


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
    input_object = input_like if isinstance(input_like, dict) else {}
    if (
        input_object.get("type") == "localization-estimate"
        and isinstance(input_object.get("poses"), list)
        and input_object["poses"]
    ):
        poses = [normalize_pose_sample(pose, index) for index, pose in enumerate(input_object["poses"])]
        return {
            "protocol": read_non_empty_string(input_object.get("protocol")),
            "type": "localization-estimate",
            "sourceType": read_non_empty_string(input_object.get("sourceType")) or "poses",
            "label": (
                read_non_empty_string(input_object.get("label"))
                or read_non_empty_string(input_object.get("name"))
                or read_non_empty_string(input_object.get("runLabel"))
                or read_non_empty_string(input_object.get("fragmentLabel"))
                or read_non_empty_string(input_object.get("fragmentId"))
                or "Localization Estimate"
            ),
            "poses": poses,
        }

    poses_like, source_type = extract_estimate_pose_list(input_like)
    if not poses_like:
        raise ValueError("localization estimate must include at least one pose")

    poses = [normalize_pose_sample(pose, index) for index, pose in enumerate(poses_like)]
    return {
        "protocol": read_non_empty_string(input_object.get("protocol")),
        "type": "localization-estimate",
        "sourceType": source_type,
        "label": (
            read_non_empty_string(input_object.get("label"))
            or read_non_empty_string(input_object.get("name"))
            or read_non_empty_string(input_object.get("runLabel"))
            or read_non_empty_string(input_object.get("fragmentLabel"))
            or read_non_empty_string(input_object.get("fragmentId"))
            or read_non_empty_string(input_object.get("type"))
            or "Localization Estimate"
        ),
        "poses": poses,
    }


@dataclass(frozen=True)
class LocalizationEstimateImportRequest:
    """Stable input for localization estimate document import."""

    raw_text: str
    file_name: str | None = None


class LocalizationEstimateImportPolicy(Protocol):
    """Minimal interface for interchangeable document import policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def import_document(self, request: LocalizationEstimateImportRequest) -> dict[str, Any]:
        """Import one raw document into the normalized estimate schema."""


def _normalize_with_label(parsed: Any, *, file_name: str | None) -> dict[str, Any]:
    normalized = normalize_localization_estimate(parsed)
    if normalized["label"] == "Localization Estimate" and file_name:
        normalized["label"] = derive_estimate_label(file_name)
    return normalized


class StrictContentGateImportPolicy:
    """Use one parser path based on the first non-space character."""

    name = "strict_content_gate"
    label = "Strict Content Gate"
    style = "single-branch"
    tier = "experiment"
    capabilities = {
        "usesFileNameHints": False,
        "supportsCommentRepair": False,
        "fallsBackAcrossFormats": False,
    }

    def import_document(self, request: LocalizationEstimateImportRequest) -> dict[str, Any]:
        text = str(request.raw_text or "").strip()
        if not text:
            raise ValueError("localization estimate file is empty")
        if text.startswith("{") or text.startswith("["):
            return _normalize_with_label(json.loads(text), file_name=request.file_name)
        return normalize_localization_estimate(parse_text_trajectory(text, file_name=request.file_name))


class FallbackCascadeImportPolicy:
    """Try JSON when it looks like JSON, then fall back to text."""

    name = "fallback_cascade"
    label = "Fallback Cascade"
    style = "json-then-text"
    tier = "experiment"
    capabilities = {
        "usesFileNameHints": False,
        "supportsCommentRepair": False,
        "fallsBackAcrossFormats": True,
    }

    def import_document(self, request: LocalizationEstimateImportRequest) -> dict[str, Any]:
        text = str(request.raw_text or "").strip()
        if not text:
            raise ValueError("localization estimate file is empty")
        if text.startswith("{") or text.startswith("["):
            try:
                return _normalize_with_label(json.loads(text), file_name=request.file_name)
            except Exception as json_error:
                try:
                    return normalize_localization_estimate(parse_text_trajectory(text, file_name=request.file_name))
                except Exception as text_error:
                    raise ValueError(f"failed to parse localization estimate JSON: {json_error}") from text_error
        return normalize_localization_estimate(parse_text_trajectory(text, file_name=request.file_name))


class SuffixAwareRepairImportPolicy:
    """Use suffix hints, repair leading comments, then fall back conservatively."""

    name = "suffix_aware"
    label = "Suffix-Aware Repair"
    style = "hinted-repairing"
    tier = "core"
    capabilities = {
        "usesFileNameHints": True,
        "supportsCommentRepair": True,
        "fallsBackAcrossFormats": True,
    }

    def import_document(self, request: LocalizationEstimateImportRequest) -> dict[str, Any]:
        text = str(request.raw_text or "").strip()
        if not text:
            raise ValueError("localization estimate file is empty")

        suffix = Path(request.file_name).suffix.lower() if request.file_name else ""
        json_like = text.startswith("{") or text.startswith("[")
        comment_repaired_text = strip_leading_comment_lines(text)
        repaired_json_like = comment_repaired_text.startswith("{") or comment_repaired_text.startswith("[")
        likely_text_suffix = suffix in {".txt", ".traj", ".tum", ".log"}
        likely_json_suffix = suffix in {".json", ".jsonl"}

        parse_order: list[str] = []
        if likely_json_suffix or repaired_json_like:
            parse_order.extend(["json", "text"])
        elif likely_text_suffix:
            parse_order.extend(["text", "json"])
        elif json_like:
            parse_order.extend(["json", "text"])
        else:
            parse_order.extend(["text", "json"])

        errors: list[str] = []
        for parse_mode in parse_order:
            try:
                if parse_mode == "json":
                    candidate_text = comment_repaired_text if repaired_json_like else text
                    return _normalize_with_label(json.loads(candidate_text), file_name=request.file_name)
                return normalize_localization_estimate(parse_text_trajectory(text, file_name=request.file_name))
            except Exception as error:  # pragma: no cover - exercised through fallback behavior
                errors.append(str(error))
        raise ValueError("; ".join(errors) if errors else "failed to import localization estimate document")


CORE_LOCALIZATION_ESTIMATE_IMPORT_POLICIES: dict[str, LocalizationEstimateImportPolicy] = {
    "suffix_aware": SuffixAwareRepairImportPolicy(),
}


def import_localization_estimate_document(
    request: LocalizationEstimateImportRequest,
    *,
    policy: str = "suffix_aware",
) -> dict[str, Any]:
    """Import one estimate document through the selected stable policy."""
    if policy not in CORE_LOCALIZATION_ESTIMATE_IMPORT_POLICIES:
        raise ValueError(
            f"unsupported localization estimate import policy: {policy}. "
            f"Expected one of {', '.join(sorted(CORE_LOCALIZATION_ESTIMATE_IMPORT_POLICIES))}"
        )
    return CORE_LOCALIZATION_ESTIMATE_IMPORT_POLICIES[policy].import_document(request)
