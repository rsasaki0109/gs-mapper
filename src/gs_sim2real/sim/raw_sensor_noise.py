"""Raw sensor noise profiles for Physical AI observation renderers.

The pose / goal / heading facing :class:`~gs_sim2real.sim.policy_sensor_noise.RoutePolicySensorNoiseProfile`
perturbs what the route policy observes. Real robotics stacks also see
sensor streams (RGB frames, depth maps, LiDAR ranges) through channels with
non-zero noise before any policy-facing feature is computed. This module adds
a sibling :class:`RawSensorNoiseProfile` and helpers that perturb the raw
arrays inside an :class:`~gs_sim2real.sim.interfaces.Observation` produced by
an :class:`~gs_sim2real.sim.rendering.ObservationRenderer`.

Scope boundary:

- Noise is added to the decoded arrays and the outputs dict is re-encoded in
  place, so downstream consumers keep reading the existing base64 fields.
- The base renderer stays pure — noise is wired in by wrapping a renderer in
  :class:`NoisyObservationRenderer` or by calling
  :func:`apply_raw_sensor_noise_to_observation` directly.
- IMU noise is intentionally out of scope: no observation renderer currently
  produces IMU readings, so there is nowhere to attach it. Adding an IMU
  output is a separate piece of work.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from .contract import SceneEnvironment
from .interfaces import Observation, ObservationRequest
from .rendering import ObservationRenderer


RAW_SENSOR_NOISE_PROFILE_VERSION = "gs-mapper-raw-sensor-noise-profile/v1"


@dataclass(frozen=True, slots=True)
class RawSensorNoiseProfile:
    """Gaussian noise budget for raw camera / depth / LiDAR observations.

    All ``*_std`` values are interpreted as the standard deviation of a
    zero-mean Gaussian applied additively to the respective quantity. A
    value of ``0.0`` disables that axis.

    - ``rgb_intensity_std`` perturbs JPEG-encoded RGB frames on the 0–255
      uint8 scale before re-encoding. Values are clipped to [0, 255].
    - ``depth_range_std_meters`` perturbs each pixel of the float32 depth
      map. Values are clamped to the far-clip horizon when the observation
      advertises one so validity masks stay meaningful.
    - ``lidar_range_std_meters`` perturbs each LiDAR ray range. Negative
      draws are clipped to zero so ranges stay physical.
    """

    profile_id: str
    rgb_intensity_std: float = 0.0
    depth_range_std_meters: float = 0.0
    lidar_range_std_meters: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = RAW_SENSOR_NOISE_PROFILE_VERSION

    def __post_init__(self) -> None:
        if not str(self.profile_id):
            raise ValueError("profile_id must not be empty")
        _non_negative_float(self.rgb_intensity_std, "rgb_intensity_std")
        _non_negative_float(self.depth_range_std_meters, "depth_range_std_meters")
        _non_negative_float(self.lidar_range_std_meters, "lidar_range_std_meters")

    @property
    def is_noise_free(self) -> bool:
        return (
            self.rgb_intensity_std == 0.0 and self.depth_range_std_meters == 0.0 and self.lidar_range_std_meters == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "raw-sensor-noise-profile",
            "version": self.version,
            "profileId": self.profile_id,
            "rgbIntensityStd": float(self.rgb_intensity_std),
            "depthRangeStdMeters": float(self.depth_range_std_meters),
            "lidarRangeStdMeters": float(self.lidar_range_std_meters),
            "metadata": _json_mapping(self.metadata),
        }


def write_raw_sensor_noise_profile_json(
    path: str | Path,
    profile: RawSensorNoiseProfile,
) -> Path:
    """Persist a raw sensor noise profile as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_raw_sensor_noise_profile_json(path: str | Path) -> RawSensorNoiseProfile:
    """Load a raw sensor noise profile JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw_sensor_noise_profile_from_dict(_mapping(payload, "rawSensorNoiseProfile"))


def raw_sensor_noise_profile_from_dict(payload: Mapping[str, Any]) -> RawSensorNoiseProfile:
    """Rebuild a raw sensor noise profile from JSON."""

    _record_type(payload, "raw-sensor-noise-profile")
    version = str(payload.get("version", RAW_SENSOR_NOISE_PROFILE_VERSION))
    if version != RAW_SENSOR_NOISE_PROFILE_VERSION:
        raise ValueError(f"unsupported raw sensor noise profile version: {version}")
    return RawSensorNoiseProfile(
        profile_id=str(payload["profileId"]),
        rgb_intensity_std=float(payload.get("rgbIntensityStd", 0.0)),
        depth_range_std_meters=float(payload.get("depthRangeStdMeters", 0.0)),
        lidar_range_std_meters=float(payload.get("lidarRangeStdMeters", 0.0)),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_raw_sensor_noise_profile_markdown(profile: RawSensorNoiseProfile) -> str:
    """Render a compact Markdown summary for a raw sensor noise profile."""

    lines = [
        f"# Raw Sensor Noise Profile: {profile.profile_id}",
        f"- RGB intensity σ (0-255): {profile.rgb_intensity_std}",
        f"- Depth σ (m): {profile.depth_range_std_meters}",
        f"- LiDAR range σ (m): {profile.lidar_range_std_meters}",
        f"- Noise free: {'yes' if profile.is_noise_free else 'no'}",
    ]
    if profile.metadata:
        lines.append("")
        lines.append("| Metadata | Value |")
        lines.append("| --- | --- |")
        for key in sorted(profile.metadata):
            lines.append(f"| {key} | {profile.metadata[key]} |")
    return "\n".join(lines) + "\n"


def raw_sensor_noise_rng(
    *,
    base_seed: int | None,
    profile_id: str,
    sensor_id: str,
    request_index: int,
    kind: str,
) -> random.Random:
    """Return a ``random.Random`` seeded from the raw-sensor noise context.

    Seeding is SHA-256 over the context tuple, matching the policy-facing
    :func:`sensor_noise_rng` so replays stay bit-identical across Python
    interpreter restarts.
    """

    resolved_base = "none" if base_seed is None else str(int(base_seed))
    digest = hashlib.sha256(
        f"{resolved_base}|{profile_id}|{sensor_id}|{int(request_index)}|{kind}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def apply_raw_sensor_noise_to_observation(
    observation: Observation,
    profile: RawSensorNoiseProfile,
    *,
    rng: random.Random,
) -> Observation:
    """Return a copy of ``observation`` with raw sensor noise applied to its outputs.

    The decoded arrays are perturbed and re-encoded in place so downstream
    consumers keep reading the existing base64 fields. When ``profile`` has
    no noise the outputs dict is returned unchanged (identity).
    """

    if profile.is_noise_free:
        return observation

    outputs = dict(observation.outputs)

    rgb_block = outputs.get("rgb")
    if isinstance(rgb_block, Mapping) and profile.rgb_intensity_std > 0.0:
        outputs["rgb"] = _perturb_rgb_block(rgb_block, profile.rgb_intensity_std, rng)

    depth_block = outputs.get("depth")
    if isinstance(depth_block, Mapping) and profile.depth_range_std_meters > 0.0:
        outputs["depth"] = _perturb_depth_block(depth_block, profile.depth_range_std_meters, rng)

    ranges_block = outputs.get("ranges")
    if isinstance(ranges_block, Mapping) and profile.lidar_range_std_meters > 0.0:
        outputs["ranges"] = _perturb_ranges_block(ranges_block, profile.lidar_range_std_meters, rng)

    return Observation(sensor_id=observation.sensor_id, pose=observation.pose, outputs=outputs)


class NoisyObservationRenderer:
    """:class:`ObservationRenderer` wrapper that injects raw-sensor noise.

    The wrapper defers ``can_render`` to the wrapped renderer and applies
    noise on the way out. ``rng_provider`` receives the original
    :class:`ObservationRequest` and should return a seeded
    :class:`random.Random` — use :func:`raw_sensor_noise_rng` for a
    deterministic default.
    """

    def __init__(
        self,
        renderer: ObservationRenderer,
        *,
        profile: RawSensorNoiseProfile,
        rng_provider: Callable[[ObservationRequest], random.Random],
    ) -> None:
        self._renderer = renderer
        self._profile = profile
        self._rng_provider = rng_provider

    def can_render(self, scene: SceneEnvironment, request: ObservationRequest) -> bool:
        return self._renderer.can_render(scene, request)

    def render_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        observation = self._renderer.render_observation(scene, request)
        return apply_raw_sensor_noise_to_observation(
            observation,
            self._profile,
            rng=self._rng_provider(request),
        )


def _perturb_rgb_block(block: Mapping[str, Any], std: float, rng: random.Random) -> dict[str, Any]:
    encoding = block.get("encoding")
    if encoding != "jpeg":
        return dict(block)
    payload = block.get("jpegBase64")
    if not isinstance(payload, str):
        return dict(block)
    jpeg_bytes = base64.b64decode(payload.encode("ascii"))
    with PILImage.open(io.BytesIO(jpeg_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    gauss = _sample_gaussian_array(rgb.shape, std, rng).astype(np.int16)
    noisy = np.clip(rgb + gauss, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    PILImage.fromarray(noisy, mode="RGB").save(buffer, format="JPEG", quality=85, optimize=False)
    new_bytes = buffer.getvalue()
    return {
        **block,
        "jpegBase64": base64.b64encode(new_bytes).decode("ascii"),
        "byteLength": len(new_bytes),
    }


def _perturb_depth_block(block: Mapping[str, Any], std: float, rng: random.Random) -> dict[str, Any]:
    payload = block.get("depthBase64")
    if not isinstance(payload, str):
        return dict(block)
    far_clip = block.get("farClipMeters")
    depth = np.frombuffer(base64.b64decode(payload.encode("ascii")), dtype="<f4").copy()
    noise = _sample_gaussian_array(depth.shape, std, rng).astype(np.float32)
    noisy = depth + noise
    if isinstance(far_clip, (int, float)):
        noisy = np.clip(noisy, 0.0, float(far_clip))
    else:
        noisy = np.clip(noisy, 0.0, None)
    encoded = noisy.astype("<f4").tobytes()
    return {
        **block,
        "depthBase64": base64.b64encode(encoded).decode("ascii"),
        "byteLength": len(encoded),
    }


def _perturb_ranges_block(block: Mapping[str, Any], std: float, rng: random.Random) -> dict[str, Any]:
    payload = block.get("rangesBase64")
    if not isinstance(payload, str):
        return dict(block)
    ranges = np.frombuffer(base64.b64decode(payload.encode("ascii")), dtype="<f4").copy()
    noise = _sample_gaussian_array(ranges.shape, std, rng).astype(np.float32)
    noisy = np.clip(ranges + noise, 0.0, None).astype(np.float32)
    encoded = noisy.tobytes()
    return {
        **block,
        "rangesBase64": base64.b64encode(encoded).decode("ascii"),
        "byteLength": len(encoded),
    }


def _sample_gaussian_array(shape: tuple[int, ...], std: float, rng: random.Random) -> np.ndarray:
    count = 1
    for dim in shape:
        count *= int(dim)
    if count <= 0 or std <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    values = np.fromiter(
        (rng.gauss(0.0, float(std)) for _ in range(count)),
        dtype=np.float32,
        count=count,
    )
    return values.reshape(shape)


def _non_negative_float(value: float, field_name: str) -> None:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "NoisyObservationRenderer",
    "RAW_SENSOR_NOISE_PROFILE_VERSION",
    "RawSensorNoiseProfile",
    "apply_raw_sensor_noise_to_observation",
    "load_raw_sensor_noise_profile_json",
    "raw_sensor_noise_profile_from_dict",
    "raw_sensor_noise_rng",
    "render_raw_sensor_noise_profile_markdown",
    "write_raw_sensor_noise_profile_json",
]
