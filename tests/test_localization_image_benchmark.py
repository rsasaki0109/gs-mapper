"""Tests for localization image benchmarking via sim2real render queries."""

from __future__ import annotations

import base64
import warnings
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from gs_sim2real.robotics.localization_image_benchmark import (
    LPIPSMetric,
    benchmark_localization_images,
    normalize_benchmark_inputs,
    normalize_route_capture_bundle,
    parse_localization_estimate_document,
)


def _encode_rgb_base64(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_response(rgb: np.ndarray, *, position: list[float]) -> dict[str, object]:
    height, width, _ = rgb.shape
    depth = np.full((height, width), 1.0, dtype="<f4")
    return {
        "protocol": "dreamwalker-sim2real-query/v1",
        "type": "render-result",
        "frameId": "dreamwalker_map",
        "width": width,
        "height": height,
        "fovDegrees": 60.0,
        "nearClip": 0.05,
        "farClip": 50.0,
        "pointRadius": 1,
        "pose": {
            "position": position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        "cameraInfo": {
            "frameId": "dreamwalker_map",
            "width": width,
            "height": height,
        },
        "colorJpegBase64": _encode_rgb_base64(rgb),
        "depthBase64": base64.b64encode(depth.tobytes()).decode("ascii"),
    }


def _ground_truth_bundle() -> tuple[dict[str, object], list[np.ndarray]]:
    frame_a = np.full((8, 8, 3), [32, 96, 160], dtype=np.uint8)
    frame_b = np.full((8, 8, 3), [120, 48, 200], dtype=np.uint8)
    bundle = {
        "protocol": "dreamwalker-sim2real-capture/v1",
        "type": "route-capture-bundle",
        "capturedAt": "2026-04-02T00:00:00Z",
        "fragmentId": "residency",
        "fragmentLabel": "Residency",
        "request": {
            "width": 8,
            "height": 8,
            "fovDegrees": 60.0,
            "nearClip": 0.05,
            "farClip": 50.0,
            "pointRadius": 1,
        },
        "captures": [
            {
                "index": 0,
                "label": "gt:1",
                "capturedAt": "2026-04-02T00:00:00Z",
                "relativeTimeSeconds": 0.0,
                "pose": {
                    "position": [0.0, 0.0, 0.0],
                    "yawDegrees": 0.0,
                },
                "response": _render_response(frame_a, position=[0.0, 0.0, 0.0]),
            },
            {
                "index": 1,
                "label": "gt:2",
                "capturedAt": "2026-04-02T00:00:01Z",
                "relativeTimeSeconds": 1.0,
                "pose": {
                    "position": [1.0, 0.0, 0.0],
                    "yawDegrees": 0.0,
                },
                "response": _render_response(frame_b, position=[1.0, 0.0, 0.0]),
            },
        ],
    }
    return bundle, [frame_a, frame_b]


def test_benchmark_localization_images_matches_identical_frames() -> None:
    bundle, frames = _ground_truth_bundle()
    estimate = {
        "type": "localization-estimate",
        "label": "ORB-SLAM3",
        "sourceType": "poses",
        "poses": [
            {
                "position": [0.0, 0.0, 0.0],
                "yawDegrees": 0.0,
                "timestampSeconds": 0.0,
            },
            {
                "position": [1.0, 0.0, 0.0],
                "yawDegrees": 0.0,
                "timestampSeconds": 1.0,
            },
        ],
    }

    def query_fn(endpoint: str, payload: dict[str, object], timeout_ms: int) -> dict[str, object]:
        assert endpoint == "ws://127.0.0.1:8781/sim2real"
        assert timeout_ms == 2500
        pose = payload["pose"]  # type: ignore[index]
        position = pose["position"]  # type: ignore[index]
        frame = frames[0] if position[0] < 0.5 else frames[1]
        return _render_response(frame, position=position)

    report = benchmark_localization_images(
        endpoint="ws://127.0.0.1:8781/sim2real",
        ground_truth_bundle=bundle,
        estimate_input=estimate,
        alignment="timestamp",
        timeout_ms=2500,
        metrics=("psnr", "ssim"),
        query_fn=query_fn,
    )

    assert report["type"] == "localization-image-benchmark-report"
    assert report["matching"]["matchedCount"] == 2
    assert report["alignment"] == "timestamp"
    assert report["estimate"]["label"] == "ORB-SLAM3"
    assert report["metrics"]["summary"]["psnr"]["mean"] == 120.0
    assert report["metrics"]["summary"]["ssim"]["mean"] > 0.99
    assert report["frames"][0]["metrics"]["psnr"] == 120.0
    assert report["metrics"]["highlights"]["psnr"]["frameIndex"] == 0
    assert report["metrics"]["highlights"]["psnr"]["ordering"] == "min"
    assert report["metrics"]["highlights"]["psnr"]["groundTruthColorJpegBase64"]
    assert report["metrics"]["highlights"]["psnr"]["renderedColorJpegBase64"]


def test_normalize_benchmark_inputs_reads_run_snapshot_bundle_and_estimate() -> None:
    bundle, _ = _ground_truth_bundle()
    run_snapshot = {
        "protocol": "dreamwalker-localization-run/v1",
        "type": "localization-run-snapshot",
        "label": "ORB-SLAM3 / Run A",
        "groundTruth": {
            "sourceId": "current-capture",
            "label": "Current Capture / 2 frames",
            "bundle": bundle,
        },
        "estimate": {
            "type": "localization-estimate",
            "label": "ORB-SLAM3 / Run A",
            "sourceType": "live-stream",
            "poses": [
                {
                    "position": [0.0, 0.0, 0.0],
                    "yawDegrees": 0.0,
                    "timestampSeconds": 0.0,
                }
            ],
        },
        "benchmark": {
            "requestedAlignment": "timestamp",
        },
    }

    ground_truth_bundle, estimate_input, preferred_alignment = normalize_benchmark_inputs(
        run_document=run_snapshot,
        ground_truth_document=None,
        estimate_document=None,
    )

    assert ground_truth_bundle["type"] == "route-capture-bundle"
    assert ground_truth_bundle["captures"][0]["response"]["type"] == "render-result"
    assert estimate_input["label"] == "ORB-SLAM3 / Run A"
    assert estimate_input["sourceType"] == "live-stream"
    assert preferred_alignment == "timestamp"


def test_parse_localization_estimate_document_reads_commented_json_export() -> None:
    parsed = parse_localization_estimate_document(
        "\n".join(
            [
                "// exported from review bundle",
                '{"type":"localization-estimate","label":"Commented Export","sourceType":"poses","poses":[{"position":[0,0,0],"orientation":[0,0,0,1],"timestampSeconds":0}]}',
            ]
        ),
        file_name="commented_export.json",
    )

    assert parsed["label"] == "Commented Export"
    assert parsed["sourceType"] == "poses"
    assert len(parsed["poses"]) == 1


def test_lpips_metric_handles_small_images_when_installed() -> None:
    pytest.importorskip("lpips")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric = LPIPSMetric(net="alex", device="cpu")
    image = np.zeros((8, 8, 3), dtype=np.float32)
    value = metric(image, image)

    assert not caught
    assert value >= 0.0


def test_normalize_route_capture_bundle_recovers_pose_from_route_when_capture_pose_missing() -> None:
    frame = np.full((8, 8, 3), [32, 96, 160], dtype=np.uint8)
    bundle = {
        "protocol": "dreamwalker-sim2real-capture/v1",
        "type": "route-capture-bundle",
        "fragmentLabel": "Residency Route Pose",
        "route": [
            {
                "position": [3.0, 0.5, 0.0],
                "yawDegrees": 15.0,
            }
        ],
        "captures": [
            {
                "index": 0,
                "label": "gt:1",
                "response": _render_response(frame, position=[30.0, 0.0, 0.0]),
            }
        ],
    }

    normalized = normalize_route_capture_bundle(bundle)

    assert normalized["fragmentLabel"] == "Residency Route Pose"
    assert normalized["captures"][0]["pose"]["position"] == [3.0, 0.5, 0.0]
    assert normalized["captures"][0]["pose"]["yawDegrees"] == 15.0
