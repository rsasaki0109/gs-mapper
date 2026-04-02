"""Tests for PLY loading in the viewer module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.viewer.web_viewer import PLYData, load_ply


def _write_ascii_ply(path: Path, positions: list[list[float]], colors: list[list[int]]) -> Path:
    """Write a minimal ASCII PLY file with positions and RGB colors.

    Args:
        path: Output file path.
        positions: List of [x, y, z] coordinates.
        colors: List of [r, g, b] values (0-255).

    Returns:
        Path to the written PLY file.
    """
    n = len(positions)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = []
    for pos, col in zip(positions, colors):
        lines.append(f"{pos[0]} {pos[1]} {pos[2]} {col[0]} {col[1]} {col[2]}")

    path.write_text(header + "\n".join(lines) + "\n")
    return path


class TestLoadPly:
    """Tests for load_ply."""

    def test_load_ply_ascii(self, tmp_path: Path) -> None:
        """Load a minimal ASCII PLY file and verify positions and colors."""
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)

        data = load_ply(ply_path)

        assert isinstance(data, PLYData)
        assert data.positions.shape == (3, 3)
        assert data.colors.shape == (3, 3)
        # Check position values
        np.testing.assert_allclose(data.positions[0], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(data.positions[2], [7.0, 8.0, 9.0], atol=1e-5)
        # Colors should be normalized to [0, 1]
        np.testing.assert_allclose(data.colors[0], [1.0, 0.0, 0.0], atol=1e-2)
        np.testing.assert_allclose(data.colors[2], [0.0, 0.0, 1.0], atol=1e-2)
        assert data.is_gaussian_splat is False

    def test_load_ply_single_point(self, tmp_path: Path) -> None:
        """Load a PLY with a single point."""
        ply_path = _write_ascii_ply(tmp_path / "single.ply", [[0.0, 0.0, 0.0]], [[128, 128, 128]])
        data = load_ply(ply_path)

        assert data.positions.shape == (1, 3)
        assert data.colors.shape == (1, 3)

    def test_load_ply_nonexistent(self) -> None:
        """load_ply raises an error for a nonexistent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_ply(Path("/nonexistent/model.ply"))

    def test_load_ply_no_colors(self, tmp_path: Path) -> None:
        """PLY without color properties gets default grey colors."""
        ply_path = tmp_path / "no_color.ply"
        header = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        data_lines = "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        ply_path.write_text(header + data_lines)

        data = load_ply(ply_path)
        assert data.positions.shape == (2, 3)
        # Default color should be 0.5 grey
        np.testing.assert_allclose(data.colors, 0.5, atol=1e-5)
