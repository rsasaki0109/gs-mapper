"""Client helpers for the sim2real headless render query transport."""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from gs_sim2real.core.query_timeout_policy import (
    build_query_timeout_policy_request,
    resolve_query_timeout_plan,
)

from .gsplat_render_server import yaw_to_quaternion


@dataclass(frozen=True)
class DecodedRenderQueryResult:
    """Decoded payload returned by the render query server."""

    response: dict[str, Any]
    width: int
    height: int
    color_jpeg: bytes
    depth: np.ndarray
    camera_info: dict[str, Any]


def build_render_query_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build a render query payload from parsed CLI args."""
    orientation = args.orientation
    if args.yaw_degrees is not None:
        orientation = yaw_to_quaternion(math.radians(float(args.yaw_degrees)))
    elif orientation is None:
        orientation = (0.0, 0.0, 0.0, 1.0)

    return {
        "type": "render",
        "pose": {
            "position": [float(value) for value in args.position],
            "orientation": [float(value) for value in orientation],
        },
        "width": int(args.width),
        "height": int(args.height),
        "fovDegrees": float(args.fov_degrees),
        "nearClip": float(args.near_clip),
        "farClip": float(args.far_clip),
        "pointRadius": int(args.point_radius),
    }


def send_render_query(endpoint: str, payload: dict[str, Any], *, timeout_ms: int) -> dict[str, Any]:
    """Send one render query over the endpoint's transport and return the JSON response."""
    scheme = urlparse(endpoint).scheme.lower()
    if scheme not in {"ws", "wss", "tcp"}:
        raise RuntimeError(f"unsupported render query endpoint scheme: {scheme or 'missing'}")
    timeout_request = build_query_timeout_policy_request(
        payload,
        transport=scheme,
        explicit_client_timeout_ms=timeout_ms,
        allow_retry=True,
    )
    timeout_plan = resolve_query_timeout_plan(timeout_request)
    last_error: Exception | None = None
    for attempt_index in range(timeout_plan.max_attempts):
        try:
            if scheme in {"ws", "wss"}:
                return _send_render_query_ws(endpoint, payload, timeout_ms=timeout_plan.attempt_timeout_ms)
            return _send_render_query_zmq(endpoint, payload, timeout_ms=timeout_plan.attempt_timeout_ms)
        except RuntimeError as error:
            last_error = error
            if not _is_retryable_timeout_error(error) or attempt_index + 1 >= timeout_plan.max_attempts:
                raise
            if timeout_plan.retry_backoff_ms > 0:
                time.sleep(float(timeout_plan.retry_backoff_ms) / 1000.0)
    if last_error is not None:
        raise last_error
    raise RuntimeError("render query failed without producing a transport error")


def _send_render_query_zmq(endpoint: str, payload: dict[str, Any], *, timeout_ms: int) -> dict[str, Any]:
    """Send one render query over ZeroMQ and return the JSON response."""
    try:
        import zmq
    except ModuleNotFoundError as exc:
        raise RuntimeError("sim2real-query requires the optional `pyzmq` package") from exc

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.linger = 0
    socket.rcvtimeo = int(timeout_ms)
    socket.sndtimeo = int(timeout_ms)
    socket.connect(endpoint)

    try:
        socket.send_json(payload)
        response = socket.recv_json()
    except zmq.error.Again as exc:
        raise RuntimeError(f"timed out waiting for render query response from {endpoint}") from exc
    finally:
        socket.close(linger=0)

    if not isinstance(response, dict):
        raise RuntimeError("render query response must be a JSON object")

    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error") or "render query failed"))

    return response


def _send_render_query_ws(endpoint: str, payload: dict[str, Any], *, timeout_ms: int) -> dict[str, Any]:
    """Send one render query over WebSocket and return the JSON response."""
    try:
        from websockets.sync.client import connect
    except ModuleNotFoundError as exc:
        raise RuntimeError("sim2real-query requires the optional `websockets` package") from exc

    timeout_seconds = max(float(timeout_ms) / 1000.0, 0.001)
    with connect(endpoint, open_timeout=timeout_seconds, close_timeout=1, max_size=16 * 1024 * 1024) as socket:
        socket.send(json.dumps(payload))
        response = _receive_ws_json(socket, endpoint=endpoint, timeout_seconds=timeout_seconds)
        if response.get("type") == "query-ready":
            response = _receive_ws_json(socket, endpoint=endpoint, timeout_seconds=timeout_seconds)

    if not isinstance(response, dict):
        raise RuntimeError("render query response must be a JSON object")

    if response.get("type") == "error":
        raise RuntimeError(str(response.get("error") or "render query failed"))

    return response


def decode_render_query_response(response: dict[str, Any]) -> DecodedRenderQueryResult:
    """Decode binary fields from a render query response."""
    if not isinstance(response, dict):
        raise ValueError("render query response must be a JSON object")
    if response.get("type") != "render-result":
        raise ValueError("render query response type must be 'render-result'")

    width = _normalize_positive_int(response.get("width"), "width")
    height = _normalize_positive_int(response.get("height"), "height")
    camera_info = response.get("cameraInfo")
    if not isinstance(camera_info, dict):
        raise ValueError("render query response must include cameraInfo")

    color_b64 = response.get("colorJpegBase64")
    if not isinstance(color_b64, str) or not color_b64:
        raise ValueError("render query response must include colorJpegBase64")
    depth_b64 = response.get("depthBase64")
    if not isinstance(depth_b64, str) or not depth_b64:
        raise ValueError("render query response must include depthBase64")

    color_jpeg = base64.b64decode(color_b64, validate=True)
    depth_bytes = base64.b64decode(depth_b64, validate=True)
    expected_depth_bytes = width * height * 4
    if len(depth_bytes) != expected_depth_bytes:
        raise ValueError(f"depth payload size mismatch: expected {expected_depth_bytes} bytes, got {len(depth_bytes)}")
    depth = np.frombuffer(depth_bytes, dtype="<f4").reshape(height, width).copy()

    return DecodedRenderQueryResult(
        response=response,
        width=width,
        height=height,
        color_jpeg=color_jpeg,
        depth=depth,
        camera_info=camera_info,
    )


def save_render_query_outputs(
    result: DecodedRenderQueryResult,
    *,
    jpeg_out: str | Path | None = None,
    depth_out: str | Path | None = None,
    camera_info_out: str | Path | None = None,
    response_out: str | Path | None = None,
) -> dict[str, str]:
    """Persist selected render query artifacts and return the written paths."""
    written: dict[str, str] = {}

    if jpeg_out is not None:
        path = Path(jpeg_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.color_jpeg)
        written["jpeg"] = str(path)

    if depth_out is not None:
        path = Path(depth_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, result.depth)
        written["depth"] = str(path)

    if camera_info_out is not None:
        path = Path(camera_info_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.camera_info, indent=2) + "\n", encoding="utf-8")
        written["cameraInfo"] = str(path)

    if response_out is not None:
        path = Path(response_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.response, indent=2) + "\n", encoding="utf-8")
        written["response"] = str(path)

    return written


def build_render_query_summary(
    result: DecodedRenderQueryResult,
    *,
    endpoint: str,
    outputs: dict[str, str],
) -> dict[str, Any]:
    """Build a compact summary suitable for CLI stdout."""
    return {
        "type": "render-result-summary",
        "endpoint": endpoint,
        "frameId": result.response.get("frameId"),
        "width": result.width,
        "height": result.height,
        "colorBytes": len(result.color_jpeg),
        "depthBytes": int(result.depth.nbytes),
        "cameraInfoFrameId": result.camera_info.get("frameId"),
        "outputs": outputs,
    }


def run_cli(args: argparse.Namespace) -> None:
    """Execute the render query client from parsed CLI args."""
    response = send_render_query(args.endpoint, build_render_query_payload(args), timeout_ms=args.timeout_ms)
    result = decode_render_query_response(response)
    outputs = save_render_query_outputs(
        result,
        jpeg_out=args.jpeg_out,
        depth_out=args.depth_out,
        camera_info_out=args.camera_info_out,
        response_out=args.response_out,
    )
    print(json.dumps(build_render_query_summary(result, endpoint=args.endpoint, outputs=outputs), indent=2))


def _normalize_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be an integer")
    normalized = int(value)
    if float(normalized) != float(value) or normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


def _receive_ws_json(socket: Any, *, endpoint: str, timeout_seconds: float) -> dict[str, Any]:
    try:
        raw_message = socket.recv(timeout=timeout_seconds)
    except TimeoutError as exc:
        raise RuntimeError(f"timed out waiting for render query response from {endpoint}") from exc

    try:
        response = json.loads(raw_message if isinstance(raw_message, str) else raw_message.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"render query response from {endpoint} was not valid JSON") from exc

    if not isinstance(response, dict):
        raise RuntimeError("render query response must be a JSON object")
    return response


def _is_retryable_timeout_error(error: RuntimeError) -> bool:
    return "timed out waiting for render query response" in str(error)
