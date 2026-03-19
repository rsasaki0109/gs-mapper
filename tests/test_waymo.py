"""Tests for Waymo Open Dataset loader."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nerf_gs_playground.datasets.waymo import WaymoLoader


class TestRotationToQuaternion:
    """Tests for _rotation_to_quaternion static method."""

    def test_identity_matrix(self):
        """Identity rotation should produce quaternion (1, 0, 0, 0)."""
        R = np.eye(3)
        qw, qx, qy, qz = WaymoLoader._rotation_to_quaternion(R)
        assert pytest.approx(qw, abs=1e-6) == 1.0
        assert pytest.approx(qx, abs=1e-6) == 0.0
        assert pytest.approx(qy, abs=1e-6) == 0.0
        assert pytest.approx(qz, abs=1e-6) == 0.0

    def test_90_degree_rotation_z(self):
        """90-degree rotation around Z axis."""
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        qw, qx, qy, qz = WaymoLoader._rotation_to_quaternion(R)
        # Expected: cos(45deg), 0, 0, sin(45deg)
        expected_qw = np.cos(np.pi / 4)
        expected_qz = np.sin(np.pi / 4)
        assert pytest.approx(qw, abs=1e-6) == expected_qw
        assert pytest.approx(qx, abs=1e-6) == 0.0
        assert pytest.approx(qy, abs=1e-6) == 0.0
        assert pytest.approx(qz, abs=1e-6) == expected_qz

    def test_90_degree_rotation_x(self):
        """90-degree rotation around X axis."""
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        qw, qx, qy, qz = WaymoLoader._rotation_to_quaternion(R)
        expected_qw = np.cos(np.pi / 4)
        expected_qx = np.sin(np.pi / 4)
        assert pytest.approx(qw, abs=1e-6) == expected_qw
        assert pytest.approx(qx, abs=1e-6) == expected_qx
        assert pytest.approx(qy, abs=1e-6) == 0.0
        assert pytest.approx(qz, abs=1e-6) == 0.0

    def test_180_degree_rotation_z(self):
        """180-degree rotation around Z axis (trace < 0 branch)."""
        R = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        qw, qx, qy, qz = WaymoLoader._rotation_to_quaternion(R)
        # 180 deg around Z: qw=0, qx=0, qy=0, qz=1
        assert pytest.approx(abs(qw), abs=1e-6) == 0.0
        assert pytest.approx(abs(qx), abs=1e-6) == 0.0
        assert pytest.approx(abs(qy), abs=1e-6) == 0.0
        assert pytest.approx(abs(qz), abs=1e-6) == 1.0

    def test_quaternion_is_unit(self):
        """Output quaternion should be unit length for any valid rotation."""
        # Random rotation via Rodrigues
        axis = np.array([1.0, 2.0, 3.0])
        axis = axis / np.linalg.norm(axis)
        angle = 1.23
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        qw, qx, qy, qz = WaymoLoader._rotation_to_quaternion(R)
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert pytest.approx(norm, abs=1e-6) == 1.0


class TestLoadPreExtracted:
    """Tests for _load_pre_extracted method."""

    def test_finds_jpg_images(self, tmp_path):
        """Should find pre-extracted JPG images."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(5):
            (images_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8dummy")

        loader = WaymoLoader(data_dir=str(tmp_path))
        result = loader._load_pre_extracted(tmp_path, max_frames=10)
        assert result == str(images_dir)

    def test_finds_png_images(self, tmp_path):
        """Should find pre-extracted PNG images."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for i in range(3):
            (images_dir / f"frame_{i:04d}.png").write_bytes(b"\x89PNGdummy")

        loader = WaymoLoader(data_dir=str(tmp_path))
        result = loader._load_pre_extracted(tmp_path, max_frames=10)
        assert result == str(images_dir)

    def test_falls_back_to_data_dir(self, tmp_path):
        """Should fall back to data_dir when images/ subdir does not exist."""
        # Put images directly in data_dir
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8dummy")

        loader = WaymoLoader(data_dir=str(tmp_path))
        result = loader._load_pre_extracted(tmp_path, max_frames=10)
        assert result == str(tmp_path)

    def test_raises_on_empty_directory(self, tmp_path):
        """Should raise FileNotFoundError when no images are found."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        loader = WaymoLoader(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No images found"):
            loader._load_pre_extracted(tmp_path, max_frames=10)


class TestToColmapFormat:
    """Tests for to_colmap_format method."""

    def test_creates_colmap_files(self, tmp_path):
        """Should create cameras.txt, images.txt, and points3D.txt."""
        # Create synthetic camera_params.json
        pose = np.eye(4).tolist()
        params = {
            "camera": "FRONT",
            "num_frames": 2,
            "intrinsics": [[2000.0, 2000.0, 960.0, 640.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "poses": [pose, pose],
        }
        params_path = tmp_path / "camera_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f)

        output_dir = tmp_path / "output"
        loader = WaymoLoader(data_dir=str(tmp_path))
        sparse_dir = loader.to_colmap_format(str(params_path), str(output_dir))

        sparse_path = output_dir / "sparse" / "0"
        assert sparse_path.exists()
        assert (sparse_path / "cameras.txt").exists()
        assert (sparse_path / "images.txt").exists()
        assert (sparse_path / "points3D.txt").exists()
        assert sparse_dir == str(sparse_path)

    def test_cameras_txt_content(self, tmp_path):
        """cameras.txt should contain correct PINHOLE parameters."""
        params = {
            "camera": "FRONT",
            "num_frames": 1,
            "intrinsics": [[2000.0, 2000.0, 960.0, 640.0]],
            "poses": [np.eye(4).tolist()],
        }
        params_path = tmp_path / "camera_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f)

        output_dir = tmp_path / "output"
        loader = WaymoLoader(data_dir=str(tmp_path))
        loader.to_colmap_format(str(params_path), str(output_dir))

        cameras_txt = (output_dir / "sparse" / "0" / "cameras.txt").read_text()
        assert "PINHOLE" in cameras_txt
        assert "1920" in cameras_txt  # w = cx * 2
        assert "1280" in cameras_txt  # h = cy * 2
        assert "2000.0" in cameras_txt

    def test_images_txt_has_correct_count(self, tmp_path):
        """images.txt should have an entry per pose."""
        num_poses = 5
        params = {
            "camera": "FRONT",
            "num_frames": num_poses,
            "intrinsics": [[2000.0, 2000.0, 960.0, 640.0]],
            "poses": [np.eye(4).tolist() for _ in range(num_poses)],
        }
        params_path = tmp_path / "camera_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f)

        output_dir = tmp_path / "output"
        loader = WaymoLoader(data_dir=str(tmp_path))
        loader.to_colmap_format(str(params_path), str(output_dir))

        images_txt = (output_dir / "sparse" / "0" / "images.txt").read_text()
        # Each pose produces two lines (data + empty), plus header
        image_lines = [line for line in images_txt.strip().split("\n") if line and not line.startswith("#")]
        assert len(image_lines) == num_poses

    def test_default_intrinsics_when_empty(self, tmp_path):
        """Should use default intrinsics when intrinsics list is empty."""
        params = {
            "camera": "FRONT",
            "num_frames": 1,
            "intrinsics": [],
            "poses": [np.eye(4).tolist()],
        }
        params_path = tmp_path / "camera_params.json"
        with open(params_path, "w") as f:
            json.dump(params, f)

        output_dir = tmp_path / "output"
        loader = WaymoLoader(data_dir=str(tmp_path))
        loader.to_colmap_format(str(params_path), str(output_dir))

        cameras_txt = (output_dir / "sparse" / "0" / "cameras.txt").read_text()
        # Default: fx=1000, fy=1000, cx=960, cy=640
        assert "1920" in cameras_txt
        assert "1280" in cameras_txt
