"""Tests for the sim2real render query client."""

from __future__ import annotations

import argparse
import base64
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import gs_sim2real.robotics.render_query_client as render_query_client

from gs_sim2real.robotics.gsplat_render_server import (
    WebSocketRenderQueryServer,
    ZmqRenderQueryServer,
    build_query_ready_payload,
)
from gs_sim2real.robotics.render_query_client import (
    build_render_query_payload,
    decode_render_query_response,
    save_render_query_outputs,
    send_render_query,
)


def _sample_response(*, width: int = 2, height: int = 2) -> dict[str, object]:
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")[:height, :width]
    return {
        "protocol": "dreamwalker-sim2real-query/v1",
        "type": "render-result",
        "frameId": "dreamwalker_map",
        "width": width,
        "height": height,
        "cameraInfo": {
            "frameId": "dreamwalker_map",
            "width": width,
            "height": height,
        },
        "colorJpegBase64": base64.b64encode(b"jpeg-bytes").decode("ascii"),
        "depthBase64": base64.b64encode(depth.tobytes()).decode("ascii"),
    }


def _allocate_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{probe.getsockname()[1]}"


def _allocate_ws_endpoint(path: str = "/sim2real") -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return f"ws://127.0.0.1:{probe.getsockname()[1]}{path}"


def test_build_render_query_payload_defaults_to_identity_orientation() -> None:
    args = argparse.Namespace(
        position=[1.0, 2.0, 3.0],
        orientation=None,
        yaw_degrees=None,
        width=320,
        height=240,
        fov_degrees=75.0,
        near_clip=0.1,
        far_clip=20.0,
        point_radius=2,
    )

    payload = build_render_query_payload(args)

    assert payload == {
        "type": "render",
        "pose": {
            "position": [1.0, 2.0, 3.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        "width": 320,
        "height": 240,
        "fovDegrees": 75.0,
        "nearClip": 0.1,
        "farClip": 20.0,
        "pointRadius": 2,
    }


def test_build_render_query_payload_uses_yaw_degrees() -> None:
    args = argparse.Namespace(
        position=[0.0, 0.0, 0.0],
        orientation=None,
        yaw_degrees=90.0,
        width=640,
        height=480,
        fov_degrees=60.0,
        near_clip=0.05,
        far_clip=50.0,
        point_radius=1,
    )

    payload = build_render_query_payload(args)

    assert payload["pose"]["orientation"] == pytest.approx([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])


def test_decode_render_query_response_decodes_binary_fields() -> None:
    result = decode_render_query_response(_sample_response())

    assert result.width == 2
    assert result.height == 2
    assert result.color_jpeg == b"jpeg-bytes"
    assert result.camera_info["frameId"] == "dreamwalker_map"
    assert np.allclose(result.depth, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_save_render_query_outputs_writes_requested_files(tmp_path: Path) -> None:
    result = decode_render_query_response(_sample_response())

    written = save_render_query_outputs(
        result,
        jpeg_out=tmp_path / "frame.jpg",
        depth_out=tmp_path / "depth.npy",
        camera_info_out=tmp_path / "camera_info.json",
        response_out=tmp_path / "response.json",
    )

    assert written == {
        "jpeg": str(tmp_path / "frame.jpg"),
        "depth": str(tmp_path / "depth.npy"),
        "cameraInfo": str(tmp_path / "camera_info.json"),
        "response": str(tmp_path / "response.json"),
    }
    assert (tmp_path / "frame.jpg").read_bytes() == b"jpeg-bytes"
    assert np.allclose(np.load(tmp_path / "depth.npy"), result.depth)
    assert '"frameId": "dreamwalker_map"' in (tmp_path / "camera_info.json").read_text(encoding="utf-8")
    assert '"type": "render-result"' in (tmp_path / "response.json").read_text(encoding="utf-8")


def test_send_render_query_round_trip_with_zmq() -> None:
    pytest.importorskip("zmq")

    endpoint = _allocate_tcp_endpoint()
    received: dict[str, object] = {}
    handled = threading.Event()

    def handle_request(payload: dict[str, object]) -> dict[str, object]:
        received["payload"] = payload
        handled.set()
        return _sample_response()

    server = ZmqRenderQueryServer(endpoint, handle_request)

    def serve() -> None:
        deadline = time.monotonic() + 5.0
        while not handled.is_set() and time.monotonic() < deadline:
            if server.poll_once():
                continue
            time.sleep(0.01)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        response = send_render_query(
            endpoint,
            {
                "type": "render",
                "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            },
            timeout_ms=1000,
        )
    finally:
        thread.join(timeout=5.0)
        server.close()

    assert handled.is_set()
    assert received["payload"] == {
        "type": "render",
        "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
    }
    assert response["type"] == "render-result"


def test_send_render_query_round_trip_with_websocket() -> None:
    pytest.importorskip("websockets")

    endpoint = _allocate_ws_endpoint()
    received: dict[str, object] = {}
    handled = threading.Event()

    def handle_request(payload: dict[str, object]) -> dict[str, object]:
        received["payload"] = payload
        handled.set()
        return _sample_response(width=4, height=3)

    server = WebSocketRenderQueryServer(
        endpoint,
        handle_request,
        ready_payload=build_query_ready_payload(
            transport="ws",
            endpoint=endpoint,
            frame_id="dreamwalker_map",
            renderer="simple",
            renderer_reason="test renderer",
            width=4,
            height=3,
            fov_degrees=60.0,
            near_clip=0.05,
            far_clip=50.0,
            point_radius=1,
        ),
    )

    def serve() -> None:
        deadline = time.monotonic() + 5.0
        while not handled.is_set() and time.monotonic() < deadline:
            if server.poll_once():
                continue
            time.sleep(0.01)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        response = send_render_query(
            endpoint,
            {
                "type": "render",
                "pose": {"position": [1.0, 2.0, 3.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
            },
            timeout_ms=1000,
        )
    finally:
        thread.join(timeout=5.0)
        server.close()

    assert handled.is_set()
    assert received["payload"] == {
        "type": "render",
        "pose": {"position": [1.0, 2.0, 3.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
    }
    assert response["type"] == "render-result"
    assert response["width"] == 4
    assert response["height"] == 3


def test_send_render_query_retries_websocket_timeouts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_send(endpoint: str, payload: dict[str, object], *, timeout_ms: int) -> dict[str, object]:
        attempts.append(timeout_ms)
        if len(attempts) == 1:
            raise RuntimeError(f"timed out waiting for render query response from {endpoint}")
        return _sample_response()

    monkeypatch.setattr(render_query_client, "_send_render_query_ws", fake_send)
    monkeypatch.setattr(render_query_client.time, "sleep", lambda _seconds: None)

    response = render_query_client.send_render_query(
        "ws://127.0.0.1:8781/sim2real",
        {
            "type": "render",
            "pose": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        },
        timeout_ms=10_000,
    )

    assert response["type"] == "render-result"
    assert attempts == [4_925, 4_925]
