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
    from nerf_gs_playground.viewer.web_viewer import load_ply

    ply_data = load_ply(ply_path)
    positions = ply_data.positions  # (N, 3)
    colors = ply_data.colors  # (N, 3)

    n = len(positions)

    # Subsample if too many points
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        indices.sort()
        positions = positions[indices]
        colors = colors[indices]
        n = max_points

    # Compute bounds
    bounds = {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "positions": positions.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "count": n,
        "bounds": bounds,
    }

    with open(out, "w") as f:
        json.dump(data, f)

    logger.info("Exported %d points to %s", n, out)
    return str(out)


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
    from nerf_gs_playground.viewer.web_viewer import load_ply

    ply_data = load_ply(ply_path)
    positions = ply_data.positions  # (N, 3)
    colors = ply_data.colors  # (N, 3)

    n = len(positions)
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        indices.sort()
        positions = positions[indices]
        colors = colors[indices]
        n = max_points

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bounds = np.concatenate([positions.min(axis=0), positions.max(axis=0)])

    with open(out, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(bounds.astype(np.float32).tobytes())
        combined = np.hstack([positions, colors]).astype(np.float32)
        f.write(combined.tobytes())

    logger.info("Exported %d points to %s (%.1f KB)", n, out, out.stat().st_size / 1024)
    return str(out)
