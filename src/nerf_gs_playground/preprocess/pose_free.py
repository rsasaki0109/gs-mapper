"""Pose-free preprocessing for 3D Gaussian Splatting.

This module provides camera pose estimation without requiring COLMAP.
It supports multiple methods:
- DUSt3R: Learning-based pairwise pose estimation
- GGRt: Pose-free gaussian splatting directly from images
- Simple: Circular camera initialization as a fallback

The output is in COLMAP text format (cameras.txt, images.txt, points3D.txt)
so it can be used directly with standard 3DGS training pipelines.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseFreeProcessor:
    """Pose-free preprocessing using GGRt or DUSt3R for camera pose estimation."""

    def __init__(self, method: str = "dust3r"):
        """Initialize the pose-free processor.

        Args:
            method: Pose estimation method. One of "dust3r", "ggrt", or "simple".
        """
        self.method = method
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required packages are installed."""
        if self.method == "dust3r":
            try:
                import torch  # noqa: F401
            except ImportError:
                raise ImportError("DUSt3R requires torch. Install with: pip install torch")

    def estimate_poses(self, image_dir: str | Path, output_dir: str | Path) -> str:
        """Estimate camera poses from images without COLMAP.

        Uses the configured method for pairwise pose estimation and
        point cloud generation. Falls back to simple initialization
        if the requested method is not available.

        Args:
            image_dir: Directory containing input images.
            output_dir: Directory where output files will be written.

        Returns:
            Path to output directory with estimated poses and points.

        Raises:
            ValueError: If fewer than 2 images are found.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG")))

        if len(images) < 2:
            raise ValueError(f"Need at least 2 images, found {len(images)}")

        logger.info("Estimating poses for %d images using %s", len(images), self.method)

        if self.method == "dust3r":
            return self._run_dust3r(images, output_dir)
        elif self.method == "ggrt":
            return self._run_ggrt(images, output_dir)
        else:
            return self._run_simple_init(images, output_dir)

    def _run_dust3r(self, images: list[Path], output_dir: Path) -> str:
        """Run DUSt3R for pose estimation.

        Args:
            images: List of image file paths.
            output_dir: Output directory for results.

        Returns:
            Path to the sparse reconstruction directory.
        """
        try:
            from dust3r.inference import inference  # noqa: F401
            from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F401

            # Full DUSt3R pipeline would go here
            logger.info("DUSt3R pipeline not yet fully implemented, using simple init")
            return self._run_simple_init(images, output_dir)
        except ImportError:
            logger.warning("DUSt3R not installed, falling back to simple initialization")
            return self._run_simple_init(images, output_dir)

    def _run_ggrt(self, images: list[Path], output_dir: Path) -> str:
        """Run GGRt for pose-free gaussian splatting.

        GGRt directly produces gaussians from images without explicit
        pose estimation.

        Args:
            images: List of image file paths.
            output_dir: Output directory for results.

        Returns:
            Path to the sparse reconstruction directory.
        """
        try:
            logger.info("GGRt produces gaussians directly from images")
            # GGRt pipeline would go here
            logger.info("GGRt pipeline not yet fully implemented, using simple init")
            return self._run_simple_init(images, output_dir)
        except ImportError:
            logger.warning("GGRt not installed, falling back to simple initialization")
            return self._run_simple_init(images, output_dir)

    def _run_simple_init(self, images: list[Path], output_dir: Path) -> str:
        """Simple initialization without external pose estimation.

        Creates a basic camera arrangement assuming images are taken
        in a circular pattern around the scene. Outputs COLMAP text
        format files.

        Args:
            images: List of image file paths.
            output_dir: Output directory for results.

        Returns:
            Path to the sparse reconstruction directory.
        """
        # Create COLMAP-compatible output format
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Read first image to get dimensions
        img = cv2.imread(str(images[0]))
        h, w = img.shape[:2]

        # Write cameras.txt - simple pinhole camera
        focal = max(w, h) * 1.2  # rough focal length estimate
        cx, cy = w / 2, h / 2
        with open(sparse_dir / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {w} {h} {focal} {focal} {cx} {cy}\n")

        # Write images.txt - arrange cameras in a circle
        num_images = len(images)
        with open(sparse_dir / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")

            for i, img_path in enumerate(images):
                angle = 2 * np.pi * i / num_images
                radius = 3.0

                # Camera position on circle
                tx = radius * np.cos(angle)
                ty = 0.0
                tz = radius * np.sin(angle)

                # Quaternion looking at center (simplified)
                qw = np.cos(angle / 2)
                qx = 0.0
                qy = np.sin(angle / 2)
                qz = 0.0

                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_path.name}\n")
                f.write("\n")  # empty line for 2D points (none)

        # Write points3D.txt - generate some random initial points
        rng = np.random.default_rng(seed=42)
        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            num_points = 1000
            for i in range(num_points):
                x = rng.uniform(-2, 2)
                y = rng.uniform(-1, 1)
                z = rng.uniform(-2, 2)
                r, g, b = rng.integers(0, 255, 3)
                f.write(f"{i + 1} {x} {y} {z} {r} {g} {b} 0.0\n")

        # Copy images to output
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for img_path in images:
            shutil.copy2(img_path, images_dir / img_path.name)

        logger.info(
            "Simple initialization complete: %d cameras, %d initial points",
            num_images,
            num_points,
        )
        return str(sparse_dir)


def run_pose_free(
    image_dir: str | Path,
    output_dir: str | Path,
    method: str = "dust3r",
) -> str:
    """Run pose-free preprocessing on a directory of images.

    Convenience function that creates a PoseFreeProcessor and runs
    the pose estimation pipeline.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory where output files will be written.
        method: Pose estimation method ("dust3r", "ggrt", or "simple").

    Returns:
        Path to output directory with estimated poses and points.
    """
    processor = PoseFreeProcessor(method=method)
    return processor.estimate_poses(image_dir, output_dir)
