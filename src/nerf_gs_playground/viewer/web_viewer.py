"""Viser-based web viewer for 3D Gaussian Splatting models.

This module provides an interactive web-based viewer for visualizing
trained 3DGS point clouds (.ply files) in the browser using viser.

Features:
- Load and render .ply Gaussian splat files
- Interactive camera controls (orbit, pan, zoom)
- Support for both point cloud PLY and Gaussian splat PLY formats
- Fallback to matplotlib if viser is not installed

Reference: viser (https://github.com/nerfstudio-project/viser)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PLYData:
    """Container for loaded PLY data."""

    positions: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) in [0, 1]
    normals: np.ndarray | None = None  # (N, 3) or None
    scales: np.ndarray | None = None  # (N, 3) or None
    rotations: np.ndarray | None = None  # (N, 4) or None
    opacities: np.ndarray | None = None  # (N,) or None
    sh_coeffs: np.ndarray | None = None  # (N, C, 3) or None
    is_gaussian_splat: bool = False


class GaussianViewer:
    """Web-based 3D viewer for Gaussian splat and point cloud PLY files."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize the viewer.

        Args:
            host: Host address to bind the viewer server.
            port: Port number for the viewer server.
        """
        self.host = host
        self.port = port

    def view_ply(self, ply_path: Path | str) -> None:
        """Load and display a .ply point cloud / Gaussian splat file.

        Args:
            ply_path: Path to the .ply file.

        Raises:
            FileNotFoundError: If the .ply file does not exist.
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        data = load_ply(ply_path)
        print(f"Loaded PLY: {len(data.positions)} points")
        if data.is_gaussian_splat:
            print("  Format: Gaussian Splat (with scales, rotations, SH coefficients)")
        else:
            print("  Format: Point cloud")

        try:
            self._view_viser(data, str(ply_path.name))
        except ImportError:
            print("viser not installed, falling back to matplotlib viewer.")
            self._view_matplotlib(data, str(ply_path.name))

    def view_colmap(self, sparse_dir: Path | str) -> None:
        """Visualize a COLMAP sparse reconstruction.

        Loads points3D.txt/bin and camera positions from a COLMAP sparse model.

        Args:
            sparse_dir: Path to COLMAP sparse reconstruction directory.
        """
        sparse_dir = Path(sparse_dir)
        if not sparse_dir.exists():
            raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

        # Load points3D
        points3d_txt = sparse_dir / "points3D.txt"
        points3d_bin = sparse_dir / "points3D.bin"

        positions = []
        colors = []

        if points3d_txt.exists():
            with open(points3d_txt) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    colors.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
        elif points3d_bin.exists():
            with open(points3d_bin, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_points):
                    struct.unpack("<Q", f.read(8))  # point_id
                    x, y, z = struct.unpack("<3d", f.read(24))
                    r, g, b = struct.unpack("<3B", f.read(3))
                    struct.unpack("<d", f.read(8))  # error
                    track_len = struct.unpack("<Q", f.read(8))[0]
                    f.read(track_len * 8)
                    positions.append([x, y, z])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
        else:
            raise FileNotFoundError(f"No points3D file found in {sparse_dir}")

        data = PLYData(
            positions=np.array(positions, dtype=np.float32),
            colors=np.array(colors, dtype=np.float32),
        )

        print(f"Loaded COLMAP sparse model: {len(positions)} points")

        try:
            self._view_viser(data, "COLMAP Sparse Model")
        except ImportError:
            print("viser not installed, falling back to matplotlib viewer.")
            self._view_matplotlib(data, "COLMAP Sparse Model")

    def _view_viser(self, data: PLYData, title: str) -> None:
        """Launch viser web viewer.

        Args:
            data: PLY data to visualize.
            title: Title for the viewer.

        Raises:
            ImportError: If viser is not installed.
        """
        import viser

        server = viser.ViserServer(host=self.host, port=self.port)

        # Add point cloud
        colors_uint8 = (np.clip(data.colors, 0, 1) * 255).astype(np.uint8)

        server.scene.add_point_cloud(
            name="/point_cloud",
            points=data.positions,
            colors=colors_uint8,
            point_size=0.005,
        )

        # Add coordinate frame at origin
        server.scene.add_frame(
            name="/world_frame",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
            axes_length=0.5,
            axes_radius=0.01,
        )

        url = f"http://localhost:{self.port}" if self.host == "0.0.0.0" else f"http://{self.host}:{self.port}"
        print(f"\n{'='*60}")
        print(f"  3D Viewer: {title}")
        print(f"  Points: {len(data.positions):,}")
        print(f"  Open in browser: {url}")
        print(f"{'='*60}")
        print("  Press Ctrl+C to stop the viewer.\n")

        try:
            while True:
                import time
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nViewer stopped.")

    def _view_matplotlib(self, data: PLYData, title: str) -> None:
        """Fallback viewer using matplotlib 3D scatter plot.

        Args:
            data: PLY data to visualize.
            title: Title for the plot.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Subsample if too many points
        max_display = 50000
        if len(data.positions) > max_display:
            indices = np.random.choice(len(data.positions), max_display, replace=False)
            positions = data.positions[indices]
            colors = data.colors[indices]
            print(f"Subsampled to {max_display} points for display")
        else:
            positions = data.positions
            colors = data.colors

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            s=0.5,
            alpha=0.5,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title} ({len(data.positions):,} points)")

        plt.tight_layout()
        print("Displaying matplotlib 3D scatter plot...")
        plt.show()


def load_ply(path: Path | str) -> PLYData:
    """Load a PLY file and extract points, colors, and Gaussian parameters.

    Implements a simple PLY parser using numpy. Supports both ASCII and
    binary_little_endian formats.

    Args:
        path: Path to the PLY file.

    Returns:
        PLYData containing positions, colors, and optional Gaussian parameters.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Extract format, element count, and properties
        fmt = "ascii"
        num_vertices = 0
        properties: list[tuple[str, str]] = []

        for line in header_lines:
            if line.startswith("format"):
                parts = line.split()
                fmt = parts[1]
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        # Map property names to indices
        prop_indices = {name: i for i, (name, _) in enumerate(properties)}

        # Read data
        if fmt == "ascii":
            data = _read_ply_ascii(f, num_vertices, len(properties))
        elif fmt == "binary_little_endian":
            data = _read_ply_binary_le(f, num_vertices, properties)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    # Extract positions
    x_idx = prop_indices.get("x", 0)
    y_idx = prop_indices.get("y", 1)
    z_idx = prop_indices.get("z", 2)
    positions = np.column_stack([data[:, x_idx], data[:, y_idx], data[:, z_idx]])

    # Extract colors
    colors = None
    if "red" in prop_indices:
        r = data[:, prop_indices["red"]]
        g = data[:, prop_indices["green"]]
        b = data[:, prop_indices["blue"]]
        # Normalize to [0, 1] if values > 1
        max_val = max(r.max(), g.max(), b.max(), 1.0)
        if max_val > 1.0:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        colors = np.column_stack([r, g, b])
    elif "f_dc_0" in prop_indices:
        # Extract colors from SH DC coefficients
        C0 = 0.28209479177387814
        dc_r = data[:, prop_indices["f_dc_0"]] * C0 + 0.5
        dc_g = data[:, prop_indices["f_dc_1"]] * C0 + 0.5
        dc_b = data[:, prop_indices["f_dc_2"]] * C0 + 0.5
        colors = np.clip(np.column_stack([dc_r, dc_g, dc_b]), 0, 1)

    if colors is None:
        colors = np.ones((num_vertices, 3), dtype=np.float32) * 0.5

    # Extract normals
    normals = None
    if "nx" in prop_indices:
        normals = np.column_stack([
            data[:, prop_indices["nx"]],
            data[:, prop_indices["ny"]],
            data[:, prop_indices["nz"]],
        ])

    # Check if this is a Gaussian splat file
    is_gs = "scale_0" in prop_indices and "rot_0" in prop_indices

    scales = None
    rotations = None
    opacities = None

    if is_gs:
        scales = np.column_stack([
            data[:, prop_indices["scale_0"]],
            data[:, prop_indices["scale_1"]],
            data[:, prop_indices["scale_2"]],
        ])
        rotations = np.column_stack([
            data[:, prop_indices["rot_0"]],
            data[:, prop_indices["rot_1"]],
            data[:, prop_indices["rot_2"]],
            data[:, prop_indices["rot_3"]],
        ])
        if "opacity" in prop_indices:
            opacities = data[:, prop_indices["opacity"]]

    return PLYData(
        positions=positions.astype(np.float32),
        colors=colors.astype(np.float32),
        normals=normals,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        is_gaussian_splat=is_gs,
    )


def _read_ply_ascii(f, num_vertices: int, num_properties: int) -> np.ndarray:
    """Read ASCII PLY vertex data."""
    data = np.zeros((num_vertices, num_properties), dtype=np.float32)
    for i in range(num_vertices):
        line = f.readline().decode("ascii").strip()
        values = line.split()
        for j, v in enumerate(values):
            data[i, j] = float(v)
    return data


def _read_ply_binary_le(
    f, num_vertices: int, properties: list[tuple[str, str]]
) -> np.ndarray:
    """Read binary little-endian PLY vertex data."""
    # Map PLY types to struct format and numpy dtype
    type_map = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uint": ("I", 4),
        "short": ("h", 2),
        "ushort": ("H", 2),
        "char": ("b", 1),
        "uchar": ("B", 1),
        "int8": ("b", 1),
        "uint8": ("B", 1),
        "int16": ("h", 2),
        "uint16": ("H", 2),
        "int32": ("i", 4),
        "uint32": ("I", 4),
        "float32": ("f", 4),
        "float64": ("d", 8),
    }

    # Build struct format string
    fmt_chars = []
    for _, ptype in properties:
        if ptype in type_map:
            fmt_chars.append(type_map[ptype][0])
        else:
            raise ValueError(f"Unsupported PLY property type: {ptype}")

    fmt_str = "<" + "".join(fmt_chars)
    row_size = struct.calcsize(fmt_str)

    data = np.zeros((num_vertices, len(properties)), dtype=np.float32)
    for i in range(num_vertices):
        row_bytes = f.read(row_size)
        if len(row_bytes) < row_size:
            logger.warning("Unexpected end of PLY file at vertex %d", i)
            break
        values = struct.unpack(fmt_str, row_bytes)
        data[i] = [float(v) for v in values]

    return data


def launch_viewer(
    ply_path: Path | str,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Launch a viser web viewer for a 3DGS point cloud.

    Opens an interactive 3D viewer in the browser at http://{host}:{port}.

    Args:
        ply_path: Path to the .ply file containing the Gaussian splat data.
        host: Host address to bind the viewer server.
        port: Port number for the viewer server.

    Raises:
        FileNotFoundError: If the .ply file does not exist.
    """
    viewer = GaussianViewer(host=host, port=port)
    viewer.view_ply(ply_path)
