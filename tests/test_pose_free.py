"""Tests for pose-free preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _create_test_images(image_dir: Path, num_images: int = 3, size: tuple[int, int] = (64, 48)) -> list[Path]:
    """Create synthetic test images.

    Args:
        image_dir: Directory where images will be written.
        num_images: Number of images to create.
        size: (width, height) of each image.

    Returns:
        List of paths to the created images.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(num_images):
        img = np.full((size[1], size[0], 3), fill_value=(i * 50) % 256, dtype=np.uint8)
        path = image_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


class TestPoseFreeProcessor:
    """Tests for PoseFreeProcessor."""

    @pytest.fixture(autouse=True)
    def _check_deps(self) -> None:
        pytest.importorskip("cv2")
        pytest.importorskip("numpy")

    def test_simple_init_creates_colmap_files(self, tmp_path: Path) -> None:
        """Simple initialization creates cameras.txt, images.txt, and points3D.txt."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=4)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        sparse_dir = Path(sparse_path)
        assert (sparse_dir / "cameras.txt").exists()
        assert (sparse_dir / "images.txt").exists()
        assert (sparse_dir / "points3D.txt").exists()

    def test_simple_init_cameras_txt_content(self, tmp_path: Path) -> None:
        """cameras.txt contains a valid PINHOLE camera entry."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3, size=(80, 60))

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        cameras_txt = Path(sparse_path) / "cameras.txt"
        content = cameras_txt.read_text()
        # Should contain PINHOLE and correct dimensions
        lines = [line for line in content.strip().split("\n") if not line.startswith("#")]
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[1] == "PINHOLE"
        assert parts[2] == "80"
        assert parts[3] == "60"

    def test_simple_init_images_txt_entries(self, tmp_path: Path) -> None:
        """images.txt contains one entry per input image."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=5)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        images_txt = Path(sparse_path) / "images.txt"
        content = images_txt.read_text()
        # Non-comment, non-empty lines should alternate: image line, empty 2D points line
        lines = [line for line in content.split("\n") if not line.startswith("#")]
        image_lines = [line for line in lines if line.strip()]
        assert len(image_lines) == 5

    def test_too_few_images_raises_value_error(self, tmp_path: Path) -> None:
        """Providing fewer than 2 images raises ValueError."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=1)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")

        with pytest.raises(ValueError, match="Need at least 2 images"):
            processor.estimate_poses(str(image_dir), str(output_dir))

    def test_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        """An empty image directory raises ValueError."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        image_dir.mkdir()

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")

        with pytest.raises(ValueError, match="Need at least 2 images"):
            processor.estimate_poses(str(image_dir), str(output_dir))

    def test_output_directory_structure(self, tmp_path: Path) -> None:
        """Output has sparse/0/ and images/ directories."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")
        processor.estimate_poses(str(image_dir), str(output_dir))

        assert (output_dir / "sparse" / "0").is_dir()
        assert (output_dir / "images").is_dir()
        # Images should be copied
        copied = list((output_dir / "images").glob("*.png"))
        assert len(copied) == 3

    def test_points3d_txt_has_points(self, tmp_path: Path) -> None:
        """points3D.txt contains 1000 initial points."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=2)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        points_txt = Path(sparse_path) / "points3D.txt"
        content = points_txt.read_text()
        data_lines = [line for line in content.strip().split("\n") if not line.startswith("#")]
        assert len(data_lines) == 1000

    def test_run_pose_free_convenience_function(self, tmp_path: Path) -> None:
        """The run_pose_free convenience function works correctly."""
        from nerf_gs_playground.preprocess.pose_free import run_pose_free

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3)

        output_dir = tmp_path / "output"
        result = run_pose_free(str(image_dir), str(output_dir), method="simple")

        assert Path(result).exists()
        assert (Path(result) / "cameras.txt").exists()

    def test_dust3r_falls_back_to_simple(self, tmp_path: Path) -> None:
        """DUSt3R method falls back to simple init when dust3r is not installed."""
        from nerf_gs_playground.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="dust3r")
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        # Should still produce valid output via fallback
        assert (Path(sparse_path) / "cameras.txt").exists()
