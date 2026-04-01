"""PLY to web format conversion for browser-based rendering.

Converts trained Gaussian Splat PLY files to JSON or compact binary
formats that can be rendered in the browser using the existing Three.js
viewer on GitHub Pages.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_scene_id(value: str) -> str:
    text = str(value or "").strip().lower()
    normalized: list[str] = []
    last_was_dash = False
    for char in text:
        if char.isalnum():
            normalized.append(char)
            last_was_dash = False
        elif not last_was_dash:
            normalized.append("-")
            last_was_dash = True
    scene_id = "".join(normalized).strip("-")
    return scene_id or "scene"


def _load_web_point_data(ply_path: str, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    from gs_sim2real.viewer.web_viewer import load_ply

    ply_data = load_ply(ply_path)
    positions = np.asarray(ply_data.positions, dtype=np.float32)
    colors = np.asarray(ply_data.colors, dtype=np.float32)
    n = len(positions)

    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        indices.sort()
        positions = positions[indices]
        colors = colors[indices]

    return positions, colors


def _compute_bounds(positions: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }


def _estimate_camera(bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    minimum = np.asarray(bounds["min"], dtype=np.float32)
    maximum = np.asarray(bounds["max"], dtype=np.float32)
    center = (minimum + maximum) * 0.5
    extents = np.maximum(maximum - minimum, 1e-3)
    radius = float(max(np.linalg.norm(extents), extents.max()) * 0.9)
    position = center + np.array([radius * 1.4, radius * 0.75, radius * 1.4], dtype=np.float32)
    return {
        "position": position.astype(np.float32).tolist(),
        "target": center.astype(np.float32).tolist(),
        "up": [0.0, 1.0, 0.0],
    }


def points_to_scene_bundle(
    positions: np.ndarray,
    colors: np.ndarray,
    output_dir: str,
    *,
    asset_format: str = "binary",
    scene_id: str = "scene",
    label: str = "Scene",
    description: str = "",
    camera: dict[str, list[float]] | None = None,
) -> str:
    """Write positions/colors directly as a static web scene bundle."""
    normalized_asset_format = str(asset_format or "binary").strip().lower()
    if normalized_asset_format not in {"json", "binary"}:
        raise ValueError("asset_format must be one of: json, binary")

    positions_array = np.asarray(positions, dtype=np.float32)
    colors_array = np.asarray(colors, dtype=np.float32)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError("positions must be an array of shape (N, 3)")
    if colors_array.ndim != 2 or colors_array.shape[1] != 3:
        raise ValueError("colors must be an array of shape (N, 3)")
    if len(positions_array) != len(colors_array):
        raise ValueError("positions and colors must contain the same number of points")
    if len(positions_array) == 0:
        raise ValueError("scene bundle requires at least one point")

    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    resolved_scene_id = _sanitize_scene_id(scene_id)
    resolved_label = str(label or "Scene").strip() or "Scene"
    asset_name = (
        f"{resolved_scene_id}.points.json" if normalized_asset_format == "json" else f"{resolved_scene_id}.points.bin"
    )
    asset_path = bundle_dir / asset_name
    if normalized_asset_format == "json":
        _write_json_asset(asset_path, positions_array, colors_array)
    else:
        _write_binary_asset(asset_path, positions_array, colors_array)

    bounds = _compute_bounds(positions_array)
    manifest = {
        "version": "gs-sim2real-web-scene/v1",
        "type": "web-scene-manifest",
        "sceneId": resolved_scene_id,
        "label": resolved_label,
        "description": str(description or ""),
        "asset": {
            "href": asset_name,
            "format": normalized_asset_format,
        },
        "count": int(len(positions_array)),
        "bounds": bounds,
        "camera": camera or _estimate_camera(bounds),
    }
    manifest_path = bundle_dir / "scene.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Exported scene bundle to %s", manifest_path)
    return str(manifest_path)


def _write_json_asset(output_path: str | Path, positions: np.ndarray, colors: np.ndarray) -> str:
    bounds = _compute_bounds(positions)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "positions": positions.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "count": int(len(positions)),
        "bounds": bounds,
    }
    with open(out, "w", encoding="utf-8") as file:
        json.dump(data, file)
    return str(out)


def _write_binary_asset(output_path: str | Path, positions: np.ndarray, colors: np.ndarray) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bounds = np.concatenate([positions.min(axis=0), positions.max(axis=0)]).astype(np.float32)
    with open(out, "wb") as file:
        file.write(struct.pack("<I", int(len(positions))))
        file.write(bounds.tobytes())
        combined = np.hstack([positions, colors]).astype(np.float32)
        file.write(combined.tobytes())
    return str(out)


def ply_to_json(ply_path: str, output_path: str, max_points: int = 100000) -> str:
    """Convert PLY point cloud to JSON format for web viewer.

    Outputs a JSON file with:
    {
        "positions": [x1,y1,z1, x2,y2,z2, ...],
        "colors": [r1,g1,b1, r2,g2,b2, ...],  // normalized 0-1
        "count": N,
        "bounds": {"min": [x,y,z], "max": [x,y,z]}
    }

    Args:
        ply_path: Path to the input PLY file.
        output_path: Path to the output JSON file.
        max_points: Maximum number of points to include (subsampled if exceeded).

    Returns:
        Path to the written output file as a string.
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    result = _write_json_asset(output_path, positions, colors)
    logger.info("Exported %d points to %s", len(positions), result)
    return result


def ply_to_binary(ply_path: str, output_path: str, max_points: int = 100000) -> str:
    """Convert PLY to compact binary format for faster web loading.

    Binary format:
    - 4 bytes: uint32 num_points
    - 24 bytes: float32[6] bounds (min_x, min_y, min_z, max_x, max_y, max_z)
    - num_points * 24 bytes: float32[6] per point (x, y, z, r, g, b)

    Args:
        ply_path: Path to the input PLY file.
        output_path: Path to the output binary file.
        max_points: Maximum number of points to include (subsampled if exceeded).

    Returns:
        Path to the written output file as a string.
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    result = _write_binary_asset(output_path, positions, colors)
    size_kb = Path(result).stat().st_size / 1024
    logger.info("Exported %d points to %s (%.1f KB)", len(positions), result, size_kb)
    return result


def ply_to_scene_bundle(
    ply_path: str,
    output_dir: str,
    *,
    asset_format: str = "binary",
    scene_id: str | None = None,
    label: str | None = None,
    description: str = "",
    max_points: int = 100000,
) -> str:
    """Export a self-contained scene bundle for static hosting on GitHub Pages.

    The output directory contains:
    - ``scene.json``: metadata + relative asset pointer
    - ``<scene-id>.points.json`` or ``<scene-id>.points.bin``: point data
    """
    positions, colors = _load_web_point_data(ply_path, max_points)
    return points_to_scene_bundle(
        positions,
        colors,
        output_dir,
        asset_format=asset_format,
        scene_id=scene_id or Path(ply_path).stem,
        label=label or Path(ply_path).stem.replace("_", " ").replace("-", " "),
        description=description,
    )
