"""COLMAP-based Structure-from-Motion preprocessing.

This module wraps COLMAP to perform feature extraction, feature matching,
and sparse reconstruction from a set of input images. The resulting camera
poses and sparse point cloud are exported in a format compatible with
3D Gaussian Splatting training.

Pipeline steps:
1. Feature extraction (SIFT)
2. Feature matching (exhaustive or sequential)
3. Sparse reconstruction (incremental mapper)
4. Undistortion and export to COLMAP text format

Requirements:
- COLMAP must be installed and accessible on PATH.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class COLMAPProcessor:
    """Wrapper for the COLMAP Structure-from-Motion pipeline."""

    def __init__(self, colmap_path: str = "colmap"):
        """Initialize the COLMAP processor.

        Args:
            colmap_path: Path to the COLMAP executable.

        Raises:
            FileNotFoundError: If COLMAP is not found.
        """
        self.colmap_path = colmap_path
        self._check_colmap_installed()

    def _check_colmap_installed(self) -> None:
        """Check if COLMAP is available on the system.

        Raises:
            FileNotFoundError: If COLMAP is not found.
        """
        if shutil.which(self.colmap_path) is None:
            raise FileNotFoundError(
                f"COLMAP not found at '{self.colmap_path}'. "
                "Please install COLMAP: https://colmap.github.io/install.html\n"
                "  Ubuntu: sudo apt install colmap\n"
                "  macOS: brew install colmap\n"
                "  Or build from source: https://github.com/colmap/colmap"
            )

    def _run_command(self, cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
        """Run a COLMAP command with error handling.

        Args:
            cmd: Command and arguments to execute.
            desc: Description of the step for logging.

        Returns:
            CompletedProcess result.

        Raises:
            subprocess.CalledProcessError: If the command fails.
        """
        cmd_str = " ".join(cmd)
        logger.info("Running: %s", cmd_str)
        if desc:
            print(f"  [{desc}] {cmd_str}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error("COLMAP command failed: %s\nstderr: %s", cmd_str, e.stderr)
            raise

    def run_sfm(
        self,
        image_dir: Path | str,
        output_dir: Path | str,
        camera_model: str = "OPENCV",
        use_gpu: bool = True,
        matching: str = "exhaustive",
    ) -> Path:
        """Run the full COLMAP SfM pipeline.

        Steps:
        1. Feature extraction (SIFT)
        2. Feature matching (exhaustive or sequential)
        3. Incremental mapping (sparse reconstruction)
        4. Image undistortion

        Args:
            image_dir: Directory containing input images.
            output_dir: Directory where COLMAP outputs will be written.
            camera_model: Camera model for feature extraction.
            use_gpu: Whether to use GPU acceleration.
            matching: Matching strategy ("exhaustive" or "sequential").

        Returns:
            Path to the sparse reconstruction output directory.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Count images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        print(f"Found {len(images)} images in {image_dir}")

        # Create output directories
        database_path = output_dir / "database.db"
        sparse_dir = output_dir / "sparse"
        undistorted_dir = output_dir / "undistorted"

        output_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        gpu_flag = "1" if use_gpu else "0"

        # Step 1: Feature extraction
        print("\nStep 1/4: Feature extraction...")
        self._run_command(
            [
                self.colmap_path,
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--ImageReader.camera_model",
                camera_model,
                "--ImageReader.single_camera",
                "1",
                "--SiftExtraction.use_gpu",
                gpu_flag,
            ],
            desc="Feature extraction",
        )

        # Step 2: Feature matching
        print("\nStep 2/4: Feature matching...")
        if matching == "exhaustive":
            self._run_command(
                [
                    self.colmap_path,
                    "exhaustive_matcher",
                    "--database_path",
                    str(database_path),
                    "--SiftMatching.use_gpu",
                    gpu_flag,
                ],
                desc="Exhaustive matching",
            )
        elif matching == "sequential":
            self._run_command(
                [
                    self.colmap_path,
                    "sequential_matcher",
                    "--database_path",
                    str(database_path),
                    "--SiftMatching.use_gpu",
                    gpu_flag,
                ],
                desc="Sequential matching",
            )
        else:
            raise ValueError(f"Unknown matching strategy: {matching}. Use 'exhaustive' or 'sequential'.")

        # Step 3: Sparse reconstruction (mapper)
        print("\nStep 3/4: Sparse reconstruction...")
        self._run_command(
            [
                self.colmap_path,
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse_dir),
            ],
            desc="Mapper",
        )

        # Find the reconstruction directory (usually sparse/0)
        recon_dirs = sorted(sparse_dir.iterdir())
        if not recon_dirs:
            raise RuntimeError("COLMAP mapper produced no reconstruction. Check your images.")
        recon_dir = recon_dirs[0]
        print(f"Reconstruction found at: {recon_dir}")

        # Step 4: Image undistortion
        print("\nStep 4/4: Image undistortion...")
        undistorted_dir.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [
                self.colmap_path,
                "image_undistorter",
                "--image_path",
                str(image_dir),
                "--input_path",
                str(recon_dir),
                "--output_path",
                str(undistorted_dir),
                "--output_type",
                "COLMAP",
            ],
            desc="Undistortion",
        )

        print(f"\nCOLMAP pipeline complete. Output at: {output_dir}")
        print(f"  Sparse model: {recon_dir}")
        print(f"  Undistorted: {undistorted_dir}")

        return recon_dir

    def export_text(self, sparse_dir: Path | str, output_dir: Path | str) -> Path:
        """Export COLMAP binary model to text format.

        Args:
            sparse_dir: Path to COLMAP sparse reconstruction (binary format).
            output_dir: Directory where text format files will be written.

        Returns:
            Path to the text format output directory.
        """
        sparse_dir = Path(sparse_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._run_command(
            [
                self.colmap_path,
                "model_converter",
                "--input_path",
                str(sparse_dir),
                "--output_path",
                str(output_dir),
                "--output_type",
                "TXT",
            ],
            desc="Model export to TXT",
        )
        return output_dir


def run_colmap(
    image_dir: Path | str,
    output_dir: Path | str,
    matching: str = "exhaustive",
    gpu_index: int = 0,
    use_gpu: bool = True,
) -> Path:
    """Run the full COLMAP SfM pipeline on a directory of images.

    Convenience function that creates a COLMAPProcessor and runs the pipeline.

    Args:
        image_dir: Directory containing input images.
        output_dir: Directory where COLMAP outputs will be written.
        matching: Matching strategy, one of "exhaustive" or "sequential".
        gpu_index: GPU index for COLMAP feature extraction (unused, reserved).
        use_gpu: Whether to use GPU for feature extraction and matching.

    Returns:
        Path to the sparse reconstruction output directory.

    Raises:
        FileNotFoundError: If COLMAP is not found on PATH.
        subprocess.CalledProcessError: If a COLMAP step fails.
    """
    processor = COLMAPProcessor()
    return processor.run_sfm(
        image_dir=image_dir,
        output_dir=output_dir,
        use_gpu=use_gpu,
        matching=matching,
    )
