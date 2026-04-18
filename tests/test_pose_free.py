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
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

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
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

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
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

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
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=1)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")

        with pytest.raises(ValueError, match="Need at least 2 images"):
            processor.estimate_poses(str(image_dir), str(output_dir))

    def test_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        """An empty image directory raises ValueError."""
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        image_dir.mkdir()

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(method="simple")

        with pytest.raises(ValueError, match="Need at least 2 images"):
            processor.estimate_poses(str(image_dir), str(output_dir))

    def test_output_directory_structure(self, tmp_path: Path) -> None:
        """Output has sparse/0/ and images/ directories."""
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

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
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

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
        from gs_sim2real.preprocess.pose_free import run_pose_free

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3)

        output_dir = tmp_path / "output"
        result = run_pose_free(str(image_dir), str(output_dir), method="simple")

        assert Path(result).exists()
        assert (Path(result) / "cameras.txt").exists()

    def test_dust3r_falls_back_to_simple(self, tmp_path: Path) -> None:
        """DUSt3R method falls back to simple init when the clone is missing."""
        from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

        image_dir = tmp_path / "images"
        _create_test_images(image_dir, num_images=3)

        output_dir = tmp_path / "output"
        processor = PoseFreeProcessor(
            method="dust3r",
            dust3r_root=tmp_path / "no_such_dust3r",
            checkpoint=tmp_path / "no_such.pth",
        )
        sparse_path = processor.estimate_poses(str(image_dir), str(output_dir))

        # Should still produce valid output via fallback
        assert (Path(sparse_path) / "cameras.txt").exists()


def _create_color_images(image_dir: Path, num_images: int, size: tuple[int, int] = (32, 24)) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(num_images):
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img[..., 0] = (i * 20) % 256
        img[..., 1] = 128
        img[..., 2] = 255 - ((i * 20) % 256)
        path = image_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


def test_write_colmap_sparse_round_trip(tmp_path: Path) -> None:
    """write_colmap_sparse produces files the gsplat trainer can load."""
    from gs_sim2real.preprocess.pose_free import write_colmap_sparse

    image_dir = tmp_path / "imgs"
    images = _create_color_images(image_dir, num_images=3, size=(32, 24))

    # Two frames translated along X, unit rotation.
    poses = np.stack(
        [
            np.eye(4),
            np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]], dtype=np.float32),
            np.array([[1, 0, 0, 1.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    focals = np.array([[28.0], [28.0], [28.0]], dtype=np.float32)
    # Fake per-view points: 5 random points per view, matching RGB in 0..1.
    rng = np.random.default_rng(0)
    pts3d_per_view = [rng.uniform(-1, 1, size=(5, 3)).astype(np.float32) for _ in range(3)]
    rgb_per_view = [rng.uniform(0, 1, size=(5, 3)).astype(np.float32) for _ in range(3)]
    # DUSt3R working shapes == original shapes here so focal scaling is identity.
    dust3r_shapes = [(24, 32)] * 3

    out_dir = tmp_path / "dust3r_out"
    sparse_dir = write_colmap_sparse(
        out_dir,
        image_paths=images,
        poses=poses,
        focals=focals,
        pts3d_per_view=pts3d_per_view,
        rgb_per_view=rgb_per_view,
        dust3r_shapes=dust3r_shapes,
        max_points=0,
    )
    sparse_dir = Path(sparse_dir)

    assert (sparse_dir / "cameras.txt").exists()
    assert (sparse_dir / "images.txt").exists()
    assert (sparse_dir / "points3D.txt").exists()
    # Images copied.
    assert len(list((out_dir / "images").glob("*.png"))) == 3

    # cameras.txt: 3 PINHOLE lines with matching width/height.
    cam_lines = [
        line for line in (sparse_dir / "cameras.txt").read_text().splitlines() if line and not line.startswith("#")
    ]
    assert len(cam_lines) == 3
    for line in cam_lines:
        parts = line.split()
        assert parts[1] == "PINHOLE"
        assert parts[2] == "32"
        assert parts[3] == "24"

    # points3D.txt: 15 points total (3 views x 5 points each).
    pt_lines = [
        line for line in (sparse_dir / "points3D.txt").read_text().splitlines() if line and not line.startswith("#")
    ]
    assert len(pt_lines) == 15
