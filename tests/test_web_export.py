"""Tests for PLY to web format conversion."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from gs_sim2real.viewer.web_export import ply_to_binary, ply_to_json, ply_to_scene_bundle


def _write_ascii_ply(path: Path, positions: list[list[float]], colors: list[list[int]]) -> Path:
    """Write a minimal ASCII PLY file with positions and RGB colors."""
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


class TestPlyToJson:
    """Tests for ply_to_json conversion."""

    def test_creates_valid_json(self, tmp_path: Path) -> None:
        """ply_to_json creates a valid JSON file with correct structure."""
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "out.json"

        result = ply_to_json(str(ply_path), str(out_path))

        assert Path(result).exists()
        with open(result) as f:
            data = json.load(f)

        assert "positions" in data
        assert "colors" in data
        assert "count" in data
        assert "bounds" in data
        assert data["count"] == 3
        assert len(data["positions"]) == 9  # 3 points * 3 coords
        assert len(data["colors"]) == 9
        assert "min" in data["bounds"]
        assert "max" in data["bounds"]
        assert len(data["bounds"]["min"]) == 3
        assert len(data["bounds"]["max"]) == 3

    def test_subsampling(self, tmp_path: Path) -> None:
        """ply_to_json subsamples when max_points is smaller than point count."""
        positions = [[float(i), float(i), float(i)] for i in range(20)]
        colors = [[128, 128, 128]] * 20
        ply_path = _write_ascii_ply(tmp_path / "many.ply", positions, colors)
        out_path = tmp_path / "sub.json"

        result = ply_to_json(str(ply_path), str(out_path), max_points=5)

        with open(result) as f:
            data = json.load(f)

        assert data["count"] == 5
        assert len(data["positions"]) == 15  # 5 * 3
        assert len(data["colors"]) == 15

    def test_bounds_correct(self, tmp_path: Path) -> None:
        """Bounds min/max match the actual point positions."""
        positions = [[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]
        colors = [[255, 255, 255], [0, 0, 0]]
        ply_path = _write_ascii_ply(tmp_path / "bounds.ply", positions, colors)
        out_path = tmp_path / "bounds.json"

        ply_to_json(str(ply_path), str(out_path))

        with open(out_path) as f:
            data = json.load(f)

        np.testing.assert_allclose(data["bounds"]["min"], [0.0, -1.0, -5.0], atol=1e-5)
        np.testing.assert_allclose(data["bounds"]["max"], [3.0, 4.0, 2.0], atol=1e-5)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """ply_to_json creates parent directories for the output path."""
        positions = [[1.0, 2.0, 3.0]]
        colors = [[128, 128, 128]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "nested" / "dir" / "out.json"

        result = ply_to_json(str(ply_path), str(out_path))
        assert Path(result).exists()


class TestPlyToBinary:
    """Tests for ply_to_binary conversion."""

    def test_creates_binary_with_correct_header(self, tmp_path: Path) -> None:
        """ply_to_binary creates a binary file with correct num_points and bounds."""
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        colors = [[255, 0, 0], [0, 255, 0]]
        ply_path = _write_ascii_ply(tmp_path / "test.ply", positions, colors)
        out_path = tmp_path / "out.bin"

        result = ply_to_binary(str(ply_path), str(out_path))
        assert Path(result).exists()

        with open(result, "rb") as f:
            # Read num_points
            (num_points,) = struct.unpack("<I", f.read(4))
            assert num_points == 2

            # Read bounds (6 floats: min_x, min_y, min_z, max_x, max_y, max_z)
            bounds = struct.unpack("<6f", f.read(24))
            np.testing.assert_allclose(bounds[:3], [1.0, 2.0, 3.0], atol=1e-5)
            np.testing.assert_allclose(bounds[3:], [4.0, 5.0, 6.0], atol=1e-5)

            # Read point data: 2 points * 6 floats each
            point_data = struct.unpack(f"<{2 * 6}f", f.read(2 * 24))
            # First point: x, y, z, r, g, b
            np.testing.assert_allclose(point_data[0:3], [1.0, 2.0, 3.0], atol=1e-5)
            # Colors are normalized 0-1
            np.testing.assert_allclose(point_data[3], 1.0, atol=1e-2)  # red=255 -> 1.0

    def test_file_size(self, tmp_path: Path) -> None:
        """Binary file has expected size: 4 + 24 + N*24 bytes."""
        n = 10
        positions = [[float(i), float(i), float(i)] for i in range(n)]
        colors = [[128, 128, 128]] * n
        ply_path = _write_ascii_ply(tmp_path / "size.ply", positions, colors)
        out_path = tmp_path / "size.bin"

        result = ply_to_binary(str(ply_path), str(out_path))
        expected_size = 4 + 24 + n * 24
        assert Path(result).stat().st_size == expected_size

    def test_subsampling(self, tmp_path: Path) -> None:
        """ply_to_binary subsamples when max_points is smaller than point count."""
        positions = [[float(i), float(i), float(i)] for i in range(20)]
        colors = [[128, 128, 128]] * 20
        ply_path = _write_ascii_ply(tmp_path / "many.ply", positions, colors)
        out_path = tmp_path / "sub.bin"

        result = ply_to_binary(str(ply_path), str(out_path), max_points=5)

        with open(result, "rb") as f:
            (num_points,) = struct.unpack("<I", f.read(4))
            assert num_points == 5

        expected_size = 4 + 24 + 5 * 24
        assert Path(result).stat().st_size == expected_size


class TestPlyToSceneBundle:
    """Tests for GitHub Pages scene bundle export."""

    def test_creates_scene_manifest_and_binary_asset(self, tmp_path: Path) -> None:
        positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 2.5, 1.0]]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ply_path = _write_ascii_ply(tmp_path / "bundle.ply", positions, colors)
        output_dir = tmp_path / "bundle"

        manifest_path = ply_to_scene_bundle(
            str(ply_path),
            str(output_dir),
            asset_format="binary",
            scene_id="demo-room",
            label="Demo Room",
            description="Tiny binary demo scene",
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["type"] == "web-scene-manifest"
        assert manifest["sceneId"] == "demo-room"
        assert manifest["label"] == "Demo Room"
        assert manifest["asset"]["format"] == "binary"
        assert manifest["asset"]["href"] == "demo-room.points.bin"
        assert (output_dir / "demo-room.points.bin").exists()
        assert manifest["camera"]["target"] == [4.0, 3.5, 3.5]

    def test_creates_scene_manifest_and_json_asset(self, tmp_path: Path) -> None:
        positions = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        colors = [[255, 255, 255], [0, 0, 0]]
        ply_path = _write_ascii_ply(tmp_path / "bundle-json.ply", positions, colors)
        output_dir = tmp_path / "bundle-json"

        manifest_path = ply_to_scene_bundle(
            str(ply_path),
            str(output_dir),
            asset_format="json",
            scene_id="json-scene",
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        asset = json.loads((output_dir / "json-scene.points.json").read_text(encoding="utf-8"))
        assert manifest["asset"]["format"] == "json"
        assert asset["count"] == 2
