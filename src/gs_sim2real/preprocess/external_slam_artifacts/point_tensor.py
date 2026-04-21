"""Convert dense point tensor artifacts into flat point-cloud arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


POINT_TENSOR_SUFFIXES = (".npz", ".pt", ".pth")
_POINT_KEYS = ("points", "world_points", "pts3d", "pointcloud", "point_cloud")
_CONFIDENCE_KEYS = ("conf", "confidence", "scores")
_COLOR_KEYS = ("images", "colors", "rgb")
_NESTED_OUTPUT_KEYS = ("pi3_sequence", "sequence", "output", "outputs", "predictions", "result", "results")


def is_point_tensor_artifact(path: Path) -> bool:
    return path.suffix.lower() in POINT_TENSOR_SUFFIXES


def materialize_point_tensor_cloud(
    pointcloud_path: str | Path,
    output_dir: str | Path,
    *,
    max_points: int = 100000,
    confidence_threshold: float = 0.2,
) -> Path:
    """Flatten a dense model point tensor into a loadable ``.npy`` point cloud."""

    pointcloud_path = Path(pointcloud_path)
    output_dir = Path(output_dir)
    data = _load_point_tensor_artifact(pointcloud_path)
    points = _extract_points(data, role=str(pointcloud_path))
    confidence = _extract_optional_confidence(data, len(points))
    colors = _extract_optional_colors(data, len(points))

    mask = np.all(np.isfinite(points), axis=1)
    if confidence is not None:
        mask &= confidence >= confidence_threshold
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) > max_points:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    if colors is not None:
        cloud = np.hstack([points, colors])
    else:
        cloud = points

    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = output_dir / f"{pointcloud_path.stem}_pointcloud.npy"
    np.save(cloud_path, cloud.astype(np.float32, copy=False))
    return cloud_path


def inspect_point_tensor_artifact(pointcloud_path: str | Path) -> dict[str, Any]:
    """Read point tensor metadata without writing a flattened cloud."""

    pointcloud_path = Path(pointcloud_path)
    data = _load_point_tensor_artifact(pointcloud_path)
    points = _extract_points(data, role=str(pointcloud_path))
    confidence = _extract_optional_confidence(data, len(points))
    colors = _extract_optional_colors(data, len(points))
    return {
        "pointCount": int(len(points)),
        "finitePointCount": int(np.sum(np.all(np.isfinite(points), axis=1))),
        "pointShape": list(points.shape),
        "hasConfidence": confidence is not None,
        "hasColor": colors is not None,
    }


def _load_point_tensor_artifact(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix in (".pt", ".pth"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is normally available in this project.
            raise ImportError(f"Reading {path.suffix} point artifacts requires PyTorch to be installed.") from exc
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported point tensor artifact: {path}")


def _extract_points(data: Any, *, role: str) -> np.ndarray:
    raw = _find_named_value(data, _POINT_KEYS)
    if raw is None:
        raise ValueError(f"Could not find point tensor in {role}")
    points = _to_numpy(raw).astype(np.float32, copy=False)
    if points.ndim < 2 or points.shape[-1] < 3:
        raise ValueError(f"Expected point tensor ending in XYZ channels, got {points.shape}: {role}")
    return points.reshape(-1, points.shape[-1])[:, :3]


def _extract_optional_confidence(data: Any, point_count: int) -> np.ndarray | None:
    raw = _find_named_value(data, _CONFIDENCE_KEYS)
    if raw is None:
        return None
    confidence = _to_numpy(raw).astype(np.float32, copy=False).reshape(-1)
    if len(confidence) != point_count:
        return None
    if np.nanmin(confidence) < 0.0 or np.nanmax(confidence) > 1.0:
        confidence = 1.0 / (1.0 + np.exp(-confidence))
    return confidence


def _extract_optional_colors(data: Any, point_count: int) -> np.ndarray | None:
    raw = _find_named_value(data, _COLOR_KEYS)
    if raw is None:
        return None
    colors = _to_numpy(raw).astype(np.float32, copy=False)
    if colors.ndim < 2 or colors.shape[-1] < 3:
        return None
    colors = colors.reshape(-1, colors.shape[-1])[:, :3]
    if len(colors) != point_count:
        return None
    if np.nanmax(colors) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0.0, 255.0)


def _find_named_value(
    data: Any, keys: tuple[str, ...], *, _depth: int = 0, _seen: set[int] | None = None
) -> Any | None:
    if _seen is None:
        _seen = set()
    if _depth > 3 or _is_array_like(data):
        return None
    obj_id = id(data)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    direct = _find_named_value_shallow(data, keys)
    if direct is not None:
        return direct

    for child in _iter_named_children(data):
        nested = _find_named_value(child, keys, _depth=_depth + 1, _seen=_seen)
        if nested is not None:
            return nested
    return None


def _find_named_value_shallow(data: Any, keys: tuple[str, ...]) -> Any | None:
    if hasattr(data, "files"):
        for key in keys:
            if key in data.files:
                return data[key]
        return None
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        return None
    for key in keys:
        if hasattr(data, key):
            return getattr(data, key)
    return None


def _iter_named_children(data: Any) -> list[Any]:
    if isinstance(data, dict):
        return [data[key] for key in _NESTED_OUTPUT_KEYS if key in data]
    return [getattr(data, key) for key in _NESTED_OUTPUT_KEYS if hasattr(data, key)]


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or hasattr(value, "detach")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
