"""Tests for raw camera / depth / LiDAR sensor noise profiles."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from gs_sim2real.sim import (
    NoisyObservationRenderer,
    Observation,
    ObservationRequest,
    Pose3D,
    RawSensorNoiseProfile,
    apply_raw_sensor_noise_to_observation,
    load_raw_sensor_noise_profile_json,
    raw_sensor_noise_profile_from_dict,
    raw_sensor_noise_rng,
    render_raw_sensor_noise_profile_markdown,
    write_raw_sensor_noise_profile_json,
)


def _unit_pose() -> Pose3D:
    return Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0))


def _encode_rgb_jpeg(rgb: np.ndarray) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=95, optimize=False)
    raw = buffer.getvalue()
    return raw, base64.b64encode(raw).decode("ascii")


def _encode_float32(values: np.ndarray) -> tuple[bytes, str]:
    raw = values.astype("<f4").tobytes()
    return raw, base64.b64encode(raw).decode("ascii")


def _sample_observation() -> Observation:
    rgb = np.full((4, 6, 3), fill_value=128, dtype=np.uint8)
    _, rgb_b64 = _encode_rgb_jpeg(rgb)
    depth = np.full((4, 6), fill_value=10.0, dtype=np.float32)
    depth_bytes, depth_b64 = _encode_float32(depth)
    ranges = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
    ranges_bytes, ranges_b64 = _encode_float32(ranges)
    outputs = {
        "mode": "splat-raster-rgb",
        "rgb": {
            "encoding": "jpeg",
            "width": 6,
            "height": 4,
            "jpegBase64": rgb_b64,
            "byteLength": len(rgb_b64),
        },
        "depth": {
            "encoding": "float32-le",
            "unit": "meter",
            "width": 6,
            "height": 4,
            "depthBase64": depth_b64,
            "byteLength": len(depth_bytes),
            "nearClipMeters": 0.05,
            "farClipMeters": 80.0,
        },
        "ranges": {
            "encoding": "float32-le",
            "unit": "meter",
            "count": int(ranges.size),
            "rangesBase64": ranges_b64,
            "byteLength": len(ranges_bytes),
        },
    }
    return Observation(sensor_id="multi", pose=_unit_pose(), outputs=outputs)


def test_raw_sensor_noise_profile_round_trips_through_json(tmp_path: Path) -> None:
    profile = RawSensorNoiseProfile(
        profile_id="raw-unit",
        rgb_intensity_std=2.0,
        depth_range_std_meters=0.1,
        lidar_range_std_meters=0.05,
        metadata={"source": "spec-sheet"},
    )
    path = write_raw_sensor_noise_profile_json(tmp_path / "profile.json", profile)
    loaded = load_raw_sensor_noise_profile_json(path)

    assert loaded == profile
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["recordType"] == "raw-sensor-noise-profile"
    rebuilt = raw_sensor_noise_profile_from_dict(payload)
    assert rebuilt == profile


def test_raw_sensor_noise_profile_rejects_negative_std() -> None:
    with pytest.raises(ValueError):
        RawSensorNoiseProfile(profile_id="bad", rgb_intensity_std=-1.0)
    with pytest.raises(ValueError):
        RawSensorNoiseProfile(profile_id="bad", depth_range_std_meters=float("nan"))
    with pytest.raises(ValueError):
        RawSensorNoiseProfile(profile_id="", rgb_intensity_std=0.0)


def test_raw_sensor_noise_profile_is_noise_free_when_all_stds_zero() -> None:
    assert RawSensorNoiseProfile(profile_id="zero").is_noise_free is True
    assert RawSensorNoiseProfile(profile_id="lidar", lidar_range_std_meters=0.05).is_noise_free is False


def test_apply_raw_sensor_noise_is_identity_when_profile_is_noise_free() -> None:
    observation = _sample_observation()
    profile = RawSensorNoiseProfile(profile_id="zero")
    result = apply_raw_sensor_noise_to_observation(
        observation,
        profile,
        rng=raw_sensor_noise_rng(base_seed=1, profile_id="zero", sensor_id="multi", request_index=0, kind="obs"),
    )
    assert result is observation


def test_apply_raw_sensor_noise_perturbs_depth_and_ranges_deterministically() -> None:
    observation = _sample_observation()
    profile = RawSensorNoiseProfile(
        profile_id="raw-depth-and-lidar",
        depth_range_std_meters=0.2,
        lidar_range_std_meters=0.1,
    )
    rng1 = raw_sensor_noise_rng(
        base_seed=42, profile_id=profile.profile_id, sensor_id="multi", request_index=0, kind="obs"
    )
    rng2 = raw_sensor_noise_rng(
        base_seed=42, profile_id=profile.profile_id, sensor_id="multi", request_index=0, kind="obs"
    )

    first = apply_raw_sensor_noise_to_observation(observation, profile, rng=rng1)
    second = apply_raw_sensor_noise_to_observation(observation, profile, rng=rng2)

    assert first.outputs["depth"]["depthBase64"] == second.outputs["depth"]["depthBase64"]
    assert first.outputs["ranges"]["rangesBase64"] == second.outputs["ranges"]["rangesBase64"]
    # RGB block is unchanged when rgb_intensity_std == 0.
    assert first.outputs["rgb"]["jpegBase64"] == observation.outputs["rgb"]["jpegBase64"]

    perturbed_depth = np.frombuffer(
        base64.b64decode(first.outputs["depth"]["depthBase64"]),
        dtype="<f4",
    )
    original_depth = np.frombuffer(
        base64.b64decode(observation.outputs["depth"]["depthBase64"]),
        dtype="<f4",
    )
    assert perturbed_depth.shape == original_depth.shape
    assert not np.array_equal(perturbed_depth, original_depth)
    # Values stay clamped to the advertised far-clip horizon and are non-negative.
    assert float(perturbed_depth.min()) >= 0.0
    assert float(perturbed_depth.max()) <= 80.0


def test_apply_raw_sensor_noise_to_rgb_clips_and_changes_bytes() -> None:
    observation = _sample_observation()
    profile = RawSensorNoiseProfile(profile_id="raw-rgb", rgb_intensity_std=20.0)
    rng = raw_sensor_noise_rng(
        base_seed=7,
        profile_id=profile.profile_id,
        sensor_id="multi",
        request_index=1,
        kind="obs",
    )
    result = apply_raw_sensor_noise_to_observation(observation, profile, rng=rng)
    assert result.outputs["rgb"]["jpegBase64"] != observation.outputs["rgb"]["jpegBase64"]
    jpeg_bytes = base64.b64decode(result.outputs["rgb"]["jpegBase64"])
    with PILImage.open(io.BytesIO(jpeg_bytes)) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    assert decoded.shape == (4, 6, 3)
    assert int(decoded.min()) >= 0
    assert int(decoded.max()) <= 255


def test_apply_raw_sensor_noise_to_lidar_keeps_ranges_non_negative() -> None:
    observation = _sample_observation()
    # With huge LiDAR std some draws would go negative; the helper must clip.
    profile = RawSensorNoiseProfile(profile_id="raw-lidar-clip", lidar_range_std_meters=50.0)
    rng = raw_sensor_noise_rng(
        base_seed=3,
        profile_id=profile.profile_id,
        sensor_id="multi",
        request_index=0,
        kind="obs",
    )
    result = apply_raw_sensor_noise_to_observation(observation, profile, rng=rng)
    ranges = np.frombuffer(base64.b64decode(result.outputs["ranges"]["rangesBase64"]), dtype="<f4")
    assert float(ranges.min()) >= 0.0


def test_raw_sensor_noise_rng_varies_across_kind_and_is_reproducible() -> None:
    def draw(kind: str, base_seed: int = 1) -> float:
        rng = raw_sensor_noise_rng(base_seed=base_seed, profile_id="p", sensor_id="s", request_index=0, kind=kind)
        return rng.random()

    # Different kinds produce different draws.
    assert draw("rgb") != draw("depth")
    # Same context reproduces the same draw across calls.
    assert draw("rgb") == draw("rgb")
    # Different base seeds produce different draws for the same kind.
    assert draw("rgb", base_seed=1) != draw("rgb", base_seed=2)


def test_render_raw_sensor_noise_profile_markdown_includes_fields() -> None:
    profile = RawSensorNoiseProfile(
        profile_id="unit-md",
        rgb_intensity_std=3.0,
        depth_range_std_meters=0.15,
        lidar_range_std_meters=0.07,
        metadata={"note": "spec-sheet"},
    )
    text = render_raw_sensor_noise_profile_markdown(profile)
    assert "Raw Sensor Noise Profile: unit-md" in text
    assert "RGB intensity σ" in text
    assert "Depth σ" in text
    assert "LiDAR range σ" in text
    assert "Noise free: no" in text
    assert "| note | spec-sheet |" in text


class _IdentityRenderer:
    def can_render(self, scene: object, request: ObservationRequest) -> bool:  # noqa: D401
        return True

    def render_observation(self, scene: object, request: ObservationRequest) -> Observation:
        return _sample_observation()


def test_noisy_observation_renderer_defers_can_render_and_injects_noise() -> None:
    profile = RawSensorNoiseProfile(profile_id="wrapper", depth_range_std_meters=0.3)
    base = _IdentityRenderer()

    def rng_provider(request: ObservationRequest) -> object:
        return raw_sensor_noise_rng(
            base_seed=11,
            profile_id=profile.profile_id,
            sensor_id=request.sensor_id,
            request_index=0,
            kind="obs",
        )

    wrapper = NoisyObservationRenderer(base, profile=profile, rng_provider=rng_provider)
    request = ObservationRequest(pose=_unit_pose(), sensor_id="multi", outputs=("rgb", "depth"))
    assert wrapper.can_render(object(), request) is True
    observation = wrapper.render_observation(object(), request)
    noiseless_depth = np.frombuffer(
        base64.b64decode(_sample_observation().outputs["depth"]["depthBase64"]),
        dtype="<f4",
    )
    perturbed_depth = np.frombuffer(
        base64.b64decode(observation.outputs["depth"]["depthBase64"]),
        dtype="<f4",
    )
    assert not np.array_equal(noiseless_depth, perturbed_depth)
