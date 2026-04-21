"""Project LiDAR point clouds onto camera image planes to create depth maps.

Provides utilities to:
- Load LiDAR point clouds (.npy, .pcd, .ply)
- Project 3D points to 2D camera coordinates using calibration
- Generate per-frame sparse depth maps for depth-supervised 3DGS training
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def project_lidar_to_image(
    points_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Project a 3D LiDAR point cloud onto a camera image plane.

    Args:
        points_3d: (N, 3+) array of 3D points in LiDAR frame. Extra channels are ignored.
        extrinsic: (4, 4) LiDAR-to-camera extrinsic matrix.
        intrinsic: (3, 3) camera intrinsic matrix.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        (H, W) sparse depth map with zero for pixels without LiDAR hits.
    """
    points_3d = points_3d[:, :3]
    N = points_3d.shape[0]
    if N == 0:
        return np.zeros((height, width), dtype=np.float32)

    # Transform to camera frame
    ones = np.ones((N, 1), dtype=np.float64)
    pts_hom = np.hstack([points_3d, ones])  # (N, 4)
    pts_cam = (extrinsic @ pts_hom.T).T  # (N, 4)
    pts_cam = pts_cam[:, :3]  # (N, 3)

    # Filter points behind camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]

    if pts_cam.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)

    # Project to image plane
    pts_2d = (intrinsic @ pts_cam.T).T  # (N, 3)
    px = pts_2d[:, 0] / pts_2d[:, 2]
    py = pts_2d[:, 1] / pts_2d[:, 2]
    depth = pts_cam[:, 2]

    # Filter to valid pixel coordinates
    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[in_bounds].astype(np.int32)
    py = py[in_bounds].astype(np.int32)
    depth = depth[in_bounds]

    # Build depth map (use closest point for each pixel)
    depth_map = np.zeros((height, width), dtype=np.float32)
    for i in range(len(px)):
        x, y, d = px[i], py[i], depth[i]
        if depth_map[y, x] == 0 or d < depth_map[y, x]:
            depth_map[y, x] = d

    return depth_map


def fill_sparse_depth(sparse_depth: np.ndarray, method: str = "nearest") -> np.ndarray:
    """Fill sparse depth map to create a denser supervision signal.

    Args:
        sparse_depth: (H, W) sparse depth map (zero = no data).
        method: Interpolation method ('nearest' or 'none').

    Returns:
        (H, W) filled depth map.
    """
    if method == "none":
        return sparse_depth.copy()

    from scipy.interpolate import NearestNDInterpolator

    valid = sparse_depth > 0
    if valid.sum() == 0:
        return sparse_depth.copy()

    ys, xs = np.nonzero(valid)
    values = sparse_depth[valid]

    interp = NearestNDInterpolator(list(zip(ys, xs)), values)

    H, W = sparse_depth.shape
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    filled = interp(grid_y, grid_x).astype(np.float32)
    return filled


def load_pointcloud(path: Path | str) -> np.ndarray:
    """Load a 3D point cloud from file.

    Args:
        path: Path to point cloud file (.npy, .ply, or .pcd).

    Returns:
        (N, 3) xyz points, or (N, 6) xyz + rgb when color is present.
    """
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
        if data.ndim == 2 and data.shape[1] >= 3:
            if data.shape[1] >= 6:
                out = np.empty((data.shape[0], 6), dtype=np.float64)
                out[:, :3] = data[:, :3].astype(np.float64)
                out[:, 3:6] = data[:, 3:6].astype(np.float64)
                return out
            return data[:, :3].astype(np.float64)
        raise ValueError(f"Expected (N, 3+) array, got shape {data.shape}")

    if path.suffix == ".ply":
        return _load_ply_points(path)

    if path.suffix == ".pcd":
        return _load_pcd_points(path)

    raise ValueError(f"Unsupported point cloud format: {path.suffix}")


_PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _ply_numpy_dtype(type_name: str, endian: str) -> np.dtype:
    dtype = _PLY_SCALAR_DTYPES.get(type_name)
    if dtype is None:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return np.dtype(dtype if dtype.endswith("1") else f"{endian}{dtype}")


def _load_ply_points(path: Path) -> np.ndarray:
    """Load xyz or xyzrgb from a PLY file (ASCII or binary)."""

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        fmt = "ascii"
        num_vertices = 0
        vertex_properties: list[tuple[str, str]] = []
        in_vertex_element = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
                continue
            if parts[0] == "element":
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    num_vertices = int(parts[2])
                continue
            if in_vertex_element and parts[0] == "property":
                if len(parts) >= 2 and parts[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                if len(parts) >= 3:
                    vertex_properties.append((parts[2], parts[1]))

        if num_vertices == 0:
            return np.zeros((0, 3), dtype=np.float64)

        prop_names = [name for name, _ in vertex_properties]
        try:
            xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError("PLY vertex element must include x, y, z properties") from exc
        rgb_indices = [prop_names.index(axis) for axis in ("red", "green", "blue") if axis in prop_names]
        has_rgb = len(rgb_indices) == 3

        if fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            vertex_dtype = np.dtype(
                [(name, _ply_numpy_dtype(type_name, endian)) for name, type_name in vertex_properties]
            )
            data = f.read(num_vertices * vertex_dtype.itemsize)
            if len(data) < num_vertices * vertex_dtype.itemsize:
                raise ValueError(f"PLY vertex payload ended early: expected {num_vertices} records")
            records = np.frombuffer(data, dtype=vertex_dtype, count=num_vertices)
            points = np.zeros((num_vertices, 6 if has_rgb else 3), dtype=np.float64)
            for out_idx, name in enumerate(("x", "y", "z")):
                points[:, out_idx] = records[name].astype(np.float64)
            if has_rgb:
                for out_idx, name in enumerate(("red", "green", "blue"), start=3):
                    points[:, out_idx] = records[name].astype(np.float64)
            return points

        if fmt != "ascii":
            raise ValueError(f"Unsupported PLY format: {fmt}")

        points = np.zeros((num_vertices, 6 if has_rgb else 3), dtype=np.float64)
        for row in range(num_vertices):
            line = f.readline().decode("ascii").strip()
            vals = line.split()
            if len(vals) < len(vertex_properties):
                raise ValueError(f"PLY vertex row {row} has too few values")
            for out_idx, prop_idx in enumerate(xyz_indices):
                points[row, out_idx] = float(vals[prop_idx])
            if has_rgb:
                for out_idx, prop_idx in enumerate(rgb_indices, start=3):
                    points[row, out_idx] = float(vals[prop_idx])
        return points


def _load_pcd_points(path: Path) -> np.ndarray:
    """Load xyz from a PCD file (ASCII only)."""
    with open(path) as f:
        header_done = False
        points = []
        for line in f:
            line = line.strip()
            if not header_done:
                if line == "DATA ascii":
                    header_done = True
                continue
            vals = line.split()
            if len(vals) >= 3:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.array(points, dtype=np.float64) if points else np.zeros((0, 3), dtype=np.float64)


def generate_depth_maps(
    pointcloud_path: Path | str,
    image_dir: Path | str,
    output_dir: Path | str,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    fill_method: str = "none",
) -> int:
    """Generate depth maps for all images by projecting a LiDAR point cloud.

    Args:
        pointcloud_path: Path to point cloud file or directory of per-frame .npy files.
        image_dir: Directory of images (used for frame name matching).
        output_dir: Directory to save depth maps as .npy files.
        extrinsic: (4, 4) LiDAR-to-camera extrinsic.
        intrinsic: (3, 3) camera intrinsic matrix.
        height: Image height.
        width: Image width.
        fill_method: Depth filling method ('none' or 'nearest').

    Returns:
        Number of depth maps generated.
    """
    pointcloud_path = Path(pointcloud_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )

    if pointcloud_path.is_dir():
        # Per-frame point clouds
        count = 0
        for img_path in image_files:
            pc_path = pointcloud_path / f"{img_path.stem}.npy"
            if not pc_path.exists():
                continue
            pts = np.load(pc_path).astype(np.float64)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                pts = pts[:, :3]
            else:
                continue
            depth = project_lidar_to_image(pts, extrinsic, intrinsic, height, width)
            if fill_method != "none":
                depth = fill_sparse_depth(depth, method=fill_method)
            np.save(output_dir / f"{img_path.stem}.npy", depth)
            count += 1
        logger.info("Generated %d per-frame depth maps", count)
        return count
    else:
        # Single merged point cloud projected to all frames
        pts = load_pointcloud(pointcloud_path)
        depth = project_lidar_to_image(pts, extrinsic, intrinsic, height, width)
        if fill_method != "none":
            depth = fill_sparse_depth(depth, method=fill_method)
        # Save same depth for all frames
        count = 0
        for img_path in image_files:
            np.save(output_dir / f"{img_path.stem}.npy", depth)
            count += 1
        logger.info("Generated %d depth maps from merged point cloud", count)
        return count
