"""Tests for the headless splat render server."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import numpy as np
import pytest
import gs_sim2real.robotics.gsplat_render_server as gsplat_render_server

from gs_sim2real.robotics.gsplat_render_server import (
    CameraPose,
    HeadlessSplatRenderer,
    PendingRenderQueryStore,
    RenderFrameBundle,
    RenderBackendSelection,
    RenderQueryHandler,
    build_view_matrix_world_to_camera,
    build_query_event_error_input,
    build_render_query_response,
    compute_camera_intrinsics,
    encode_rgb_to_jpeg,
    parse_render_query_request,
    resolve_render_backend,
    resolve_connection_source_id,
    resolve_query_endpoint,
    resolve_query_transport_selection,
    yaw_to_quaternion,
)


def write_test_ply(path: Path, rows: list[tuple[float, ...]]) -> None:
    """Write a small binary 3DGS-style PLY fixture."""
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(rows)}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        for row in rows:
            file.write(struct.pack("<17f", *row))


def test_compute_camera_intrinsics_returns_expected_center() -> None:
    """Vertical FOV intrinsics should center the projection at image midpoint."""
    fx, fy, cx, cy = compute_camera_intrinsics(640, 480, 60.0)
    assert fy > 0.0
    assert fx > fy
    assert cx == 320.0
    assert cy == 240.0


def test_encode_rgb_to_jpeg_returns_binary_data() -> None:
    """JPEG encoding should produce a non-empty binary blob."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6, 0] = 255
    jpeg = encode_rgb_to_jpeg(image)
    assert isinstance(jpeg, bytes)
    assert len(jpeg) > 32


def test_build_view_matrix_world_to_camera_encodes_translation() -> None:
    """A translated identity pose should produce the matching world-to-camera offset."""
    viewmat = build_view_matrix_world_to_camera(
        CameraPose(position=(1.0, 2.0, 3.0), orientation=yaw_to_quaternion(0.0))
    )
    np.testing.assert_allclose(viewmat[:3, :3], np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(viewmat[:3, 3], np.array([-1.0, -2.0, -3.0], dtype=np.float32))


def test_resolve_render_backend_prefers_gsplat_when_ready() -> None:
    """Auto mode should select gsplat only when every prerequisite is present."""
    selection = resolve_render_backend(
        "auto",
        has_gaussian_splat=True,
        gsplat_available=True,
        cuda_available=True,
    )
    assert selection == RenderBackendSelection(
        "gsplat",
        "auto-selected because gsplat, CUDA, and Gaussian PLY parameters are available",
    )


def test_resolve_render_backend_falls_back_to_simple_without_gsplat() -> None:
    """Auto mode should fall back cleanly when gsplat is unavailable."""
    selection = resolve_render_backend(
        "auto",
        has_gaussian_splat=True,
        gsplat_available=False,
        cuda_available=True,
    )
    assert selection == RenderBackendSelection(
        "simple",
        "fallback because the optional `gsplat` package is not installed",
    )


def test_resolve_query_endpoint_flips_default_endpoint_to_selected_transport() -> None:
    """Transport resolution should not keep the other transport's default endpoint."""
    assert resolve_query_endpoint("ws", "tcp://127.0.0.1:5588") == "ws://127.0.0.1:8781/sim2real"
    assert resolve_query_endpoint("zmq", "ws://127.0.0.1:8781/sim2real") == "tcp://127.0.0.1:5588"


def test_resolve_query_transport_selection_prefers_ws_for_query_pose_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-side auto transport should default to websocket queries for browser-driven workloads."""

    def fake_find_spec(name: str) -> object | None:
        return object() if name in {"zmq", "websockets"} else None

    monkeypatch.setattr(gsplat_render_server.importlib.util, "find_spec", fake_find_spec)

    selection = resolve_query_transport_selection(
        "auto",
        pose_source="query",
        query_endpoint="tcp://127.0.0.1:5588",
    )

    assert selection.transport == "ws"
    assert selection.endpoint == "ws://127.0.0.1:8781/sim2real"
    assert "browser-facing clients" in selection.reason


def test_resolve_query_transport_selection_disables_transport_for_static_pose_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side auto transport should stay off when the workload is not interactive."""

    def fake_find_spec(name: str) -> object | None:
        return object() if name in {"zmq", "websockets"} else None

    monkeypatch.setattr(gsplat_render_server.importlib.util, "find_spec", fake_find_spec)

    selection = resolve_query_transport_selection(
        "auto",
        pose_source="static",
        query_endpoint="ws://127.0.0.1:8781/sim2real",
    )

    assert selection.transport == "none"
    assert selection.endpoint == ""
    assert "does not request interactive queries" in selection.reason


def test_resolve_connection_source_id_prefers_remote_address() -> None:
    class FakeWebSocket:
        remote_address = ("127.0.0.1", 50123)

    source_id = resolve_connection_source_id(
        transport="ws",
        endpoint="ws://127.0.0.1:8781/sim2real",
        connection_serial=9,
        websocket=FakeWebSocket(),
    )

    assert source_id == "ws-127.0.0.1-50123"


def test_build_query_event_error_input_maps_canonical_queue_drop_codes() -> None:
    error_input = build_query_event_error_input(
        "queue_dropped",
        reason="evicted lower-priority queued work in favor of an interactive request",
        transport="ws",
        request_type="localization-image-benchmark",
    )

    assert error_input.error == (
        "query dropped from queue: evicted lower-priority queued work in favor of an interactive request"
    )
    assert error_input.error_type == "RuntimeError"
    assert error_input.error_code == "query_queue_dropped"


def test_resolve_query_response_timeout_seconds_scales_benchmark_workload() -> None:
    timeout_seconds = gsplat_render_server.resolve_query_response_timeout_seconds(
        {
            "type": "localization-image-benchmark",
            "groundTruthBundle": {"type": "route-capture-bundle", "captures": [{} for _ in range(12)]},
            "estimate": {"type": "localization-estimate", "poses": []},
            "responseTimeoutSeconds": 45,
        },
        transport="ws",
    )

    assert timeout_seconds == 96.0


def test_pending_render_query_store_evicts_background_work_for_interactive_render() -> None:
    store = PendingRenderQueryStore(transport="ws", queue_policy="interactive_first", max_pending=2)

    pending_benchmark, _ = store.enqueue(
        {
            "type": "localization-image-benchmark",
            "groundTruthBundle": {"type": "route-capture-bundle", "captures": [{} for _ in range(16)]},
            "estimate": {"type": "localization-estimate", "poses": []},
        }
    )
    pending_render_1, _ = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        }
    )
    pending_render_2, reason = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [1.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        }
    )

    assert pending_benchmark is not None
    assert pending_render_1 is not None
    assert pending_render_2 is not None
    assert "interactive request" in reason
    assert pending_benchmark.response_ready.is_set()
    assert pending_benchmark.response is not None
    assert pending_benchmark.response["type"] == "error"
    assert pending_benchmark.response["error"].startswith("query dropped from queue:")

    first = store.dispatch_next()
    second = store.dispatch_next()
    third = store.dispatch_next()

    assert first is pending_render_1
    assert second is pending_render_2
    assert third is None


def test_pending_render_query_store_cancels_source_backlog() -> None:
    store = PendingRenderQueryStore(
        transport="ws",
        queue_policy="interactive_first",
        cancellation_policy="cancel_source_backlog",
        max_pending=4,
    )

    pending_render, _ = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        },
        source_id="socket-a",
    )
    pending_benchmark, _ = store.enqueue(
        {
            "type": "localization-image-benchmark",
            "groundTruthBundle": {"type": "route-capture-bundle", "captures": [{} for _ in range(8)]},
            "estimate": {"type": "localization-estimate", "poses": []},
        },
        source_id="socket-a",
    )
    pending_other, _ = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [1.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        },
        source_id="socket-b",
    )

    assert store.cancel_source("socket-a", event="connection_closed") is True
    assert pending_render is not None and pending_render.response_ready.is_set()
    assert pending_benchmark is not None and pending_benchmark.response_ready.is_set()
    assert pending_other is not None and not pending_other.response_ready.is_set()
    assert pending_render.response is not None
    assert pending_render.response["error"].startswith("query canceled:")

    dispatched = store.dispatch_next()
    assert dispatched is pending_other


def test_pending_render_query_store_coalesces_latest_render_per_source() -> None:
    store = PendingRenderQueryStore(
        transport="ws",
        queue_policy="interactive_first",
        coalescing_policy="latest_render_per_source",
        max_pending=4,
    )

    pending_first, _ = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        },
        source_id="socket-a",
    )
    pending_second, reason = store.enqueue(
        {
            "type": "render",
            "pose": {"position": [1.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        },
        source_id="socket-a",
    )

    assert pending_first is not None
    assert pending_second is not None
    assert "same source" in reason
    assert pending_first.response_ready.is_set()
    assert pending_first.response is not None
    assert pending_first.response["type"] == "error"
    assert pending_first.response["error"].startswith("query dropped from queue:")

    dispatched = store.dispatch_next()
    assert dispatched is pending_second


def test_resolve_render_backend_rejects_explicit_gsplat_without_gaussian_params() -> None:
    """Explicit gsplat mode should fail fast for plain point clouds."""
    try:
        resolve_render_backend(
            "gsplat",
            has_gaussian_splat=False,
            gsplat_available=True,
            cuda_available=True,
        )
    except RuntimeError as error:
        assert "Gaussian parameters" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError for missing Gaussian parameters")


def test_parse_render_query_request_applies_defaults_and_overrides() -> None:
    """Query payloads should inherit defaults and allow per-request overrides."""
    request = parse_render_query_request(
        {
            "type": "render",
            "pose": {
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "width": 320,
            "pointRadius": 4,
        },
        default_width=640,
        default_height=480,
        default_fov_degrees=60.0,
        default_near_clip=0.05,
        default_far_clip=50.0,
        default_point_radius=1,
    )
    assert request["pose"] == CameraPose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert request["width"] == 320
    assert request["height"] == 480
    assert request["fov_degrees"] == 60.0
    assert request["near_clip"] == 0.05
    assert request["far_clip"] == 50.0
    assert request["point_radius"] == 4


def test_parse_render_query_request_rejects_bad_pose() -> None:
    """Malformed query payloads should fail clearly."""
    with pytest.raises(ValueError, match="pose.orientation"):
        parse_render_query_request(
            {
                "pose": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 1.0, 0.0],
                }
            },
            default_width=640,
            default_height=480,
            default_fov_degrees=60.0,
            default_near_clip=0.05,
            default_far_clip=50.0,
            default_point_radius=1,
        )


def test_build_render_query_response_encodes_payloads() -> None:
    """Render query responses should carry base64 RGB/depth payloads and camera info."""
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 1] = 128
    depth = np.array([[1.5, 2.0, 2.5], [3.0, 3.5, 4.0]], dtype=np.float32)
    rgb_jpeg = encode_rgb_to_jpeg(rgb, quality=80)
    frame = RenderFrameBundle(
        pose=CameraPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        width=3,
        height=2,
        fov_degrees=60.0,
        near_clip=0.05,
        far_clip=20.0,
        point_radius=1,
        rgb=rgb,
        depth=depth,
        rgb_jpeg=rgb_jpeg,
    )
    response = build_render_query_response(frame, frame_id="dreamwalker_map")
    assert response["type"] == "render-result"
    assert response["cameraInfo"]["frameId"] == "dreamwalker_map"
    assert base64.b64decode(response["colorJpegBase64"]) == rgb_jpeg
    assert base64.b64decode(response["depthBase64"]) == np.asarray(depth, dtype="<f4").tobytes()


def test_render_query_handler_renders_and_invokes_publish_callback(tmp_path: Path) -> None:
    """The query handler should render one frame and expose the encoded response."""
    ply_path = tmp_path / "fixture.ply"
    write_test_ply(
        ply_path,
        [
            (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -1.77, -1.77, 1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 5.0, 0.0, 0.0, 0.0, -1.77, 1.77, -1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    published: list[RenderFrameBundle] = []
    handler = RenderQueryHandler(
        HeadlessSplatRenderer(ply_path, backend="simple"),
        frame_id="dreamwalker_map",
        default_width=64,
        default_height=48,
        default_fov_degrees=60.0,
        default_near_clip=0.05,
        default_far_clip=20.0,
        default_point_radius=1,
        jpeg_quality=85,
        publish_callback=published.append,
    )

    response = handler.handle_request(
        {
            "type": "render",
            "pose": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    assert response["type"] == "render-result"
    assert response["width"] == 64
    assert response["height"] == 48
    assert len(published) == 1
    assert published[0].depth.shape == (48, 64)


def test_render_query_handler_accepts_alias_friendly_render_wrappers(tmp_path: Path) -> None:
    """Production handler should accept wrapper + alias render payloads through the core importer."""
    ply_path = tmp_path / "fixture.ply"
    write_test_ply(
        ply_path,
        [
            (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -1.77, -1.77, 1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    handler = RenderQueryHandler(
        HeadlessSplatRenderer(ply_path, backend="simple"),
        frame_id="dreamwalker_map",
        default_width=64,
        default_height=48,
        default_fov_degrees=60.0,
        default_near_clip=0.05,
        default_far_clip=20.0,
        default_point_radius=1,
        jpeg_quality=85,
    )

    response = handler.handle_request(
        {
            "requestType": "render",
            "request": {
                "cameraPose": {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion": [0.0, 0.0, 0.0, 1.0],
                },
                "imageWidth": 80,
                "imageHeight": 60,
                "fov": 55.0,
                "radius": 2,
            },
        }
    )

    assert response["type"] == "render-result"
    assert response["width"] == 80
    assert response["height"] == 60
    assert response["fovDegrees"] == 55.0


def test_render_query_handler_benchmarks_localization_images_without_publishing(tmp_path: Path) -> None:
    """Image benchmark requests should render in-process and return a report payload."""
    ply_path = tmp_path / "fixture.ply"
    write_test_ply(
        ply_path,
        [
            (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -1.77, -1.77, 1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 5.0, 0.0, 0.0, 0.0, -1.77, 1.77, -1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )
    published: list[RenderFrameBundle] = []
    handler = RenderQueryHandler(
        HeadlessSplatRenderer(ply_path, backend="simple"),
        frame_id="dreamwalker_map",
        default_width=64,
        default_height=48,
        default_fov_degrees=60.0,
        default_near_clip=0.05,
        default_far_clip=20.0,
        default_point_radius=1,
        jpeg_quality=85,
        query_endpoint="ws://127.0.0.1:8781/sim2real",
        publish_callback=published.append,
    )

    render_payload = {
        "type": "render",
        "pose": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    }
    ground_truth_response = build_render_query_response(
        handler.render_request(render_payload, publish=False),
        frame_id="dreamwalker_map",
    )

    report = handler.handle_request(
        {
            "type": "localization-image-benchmark",
            "groundTruthBundle": {
                "type": "route-capture-bundle",
                "fragmentId": "residency",
                "fragmentLabel": "Residency GT",
                "capturedAt": "2026-04-02T00:00:00Z",
                "request": {
                    "width": 64,
                    "height": 48,
                    "fovDegrees": 60.0,
                    "nearClip": 0.05,
                    "farClip": 20.0,
                    "pointRadius": 1,
                },
                "captures": [
                    {
                        "index": 0,
                        "label": "gt:start",
                        "capturedAt": "2026-04-02T00:00:00Z",
                        "relativeTimeSeconds": 0,
                        "pose": {
                            "position": [0.0, 0.0, 0.0],
                            "yawDegrees": 0.0,
                        },
                        "response": ground_truth_response,
                    }
                ],
            },
            "estimate": {
                "type": "localization-estimate",
                "label": "Perfect Estimate",
                "sourceType": "poses",
                "poses": [
                    {
                        "position": [0.0, 0.0, 0.0],
                        "yawDegrees": 0.0,
                        "timestampSeconds": 0,
                    }
                ],
            },
            "alignment": "index",
            "metrics": ["psnr", "ssim"],
            "responseTimeoutSeconds": 45,
        }
    )

    assert report["type"] == "localization-image-benchmark-report"
    assert report["endpoint"] == "ws://127.0.0.1:8781/sim2real"
    assert report["matching"]["matchedCount"] == 1
    assert report["metrics"]["summary"]["psnr"]["mean"] > 0.0
    assert report["metrics"]["highlights"]["psnr"]["frameIndex"] == 0
    assert len(published) == 0


def test_headless_renderer_projects_points_into_rgb_and_depth(tmp_path: Path) -> None:
    """The nearest point should win both RGB color and depth at the center pixel."""
    ply_path = tmp_path / "fixture.ply"
    write_test_ply(
        ply_path,
        [
            # Blue-ish point closer to the camera at image center.
            (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, -1.77, -1.77, 1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            # Red point farther away at the same projected pixel.
            (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.77, -1.77, -1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            # Green point off to the right to prove multiple pixels can be filled.
            (1.0, 0.0, 5.0, 0.0, 0.0, 0.0, -1.77, 1.77, -1.77, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    renderer = HeadlessSplatRenderer(ply_path, backend="simple")
    rgb, depth = renderer.render_rgbd(
        CameraPose(position=(0.0, 0.0, 0.0), orientation=yaw_to_quaternion(0.0)),
        width=64,
        height=48,
        fov_degrees=60.0,
        near_clip=0.05,
        far_clip=20.0,
        point_radius=1,
    )

    center_pixel = rgb[24, 32]
    right_pixel = rgb[24, 43]

    assert rgb.shape == (48, 64, 3)
    assert depth.shape == (48, 64)
    assert depth[24, 32] == np.float32(3.0)
    assert center_pixel[2] > center_pixel[0]
    assert right_pixel[1] > right_pixel[0]
    assert np.count_nonzero(depth < 20.0) >= 2
