"""nerfstudio-based 3D Gaussian Splatting training wrapper.

This module wraps nerfstudio's splatfacto method to train 3D Gaussian
Splatting models. It handles:
- Converting data to nerfstudio's expected format
- Launching nerfstudio training with configurable hyperparameters
- Exporting the trained model as a .ply point cloud

Reference: nerfstudio (https://github.com/nerfstudio-project/nerfstudio)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NerfstudioTrainer:
    """Wrapper for nerfstudio's splatfacto training method."""

    def __init__(self) -> None:
        """Initialize the trainer and check nerfstudio installation."""
        self._check_nerfstudio_installed()

    def _check_nerfstudio_installed(self) -> None:
        """Check if nerfstudio CLI tools are available.

        Raises:
            ImportError: If nerfstudio is not installed.
        """
        if shutil.which("ns-train") is None:
            raise ImportError(
                "nerfstudio is not installed or not on PATH. "
                "Install it with: pip install nerfstudio\n"
                "See: https://docs.nerf.studio/quickstart/installation.html"
            )

    def process_data(
        self,
        data_dir: Path | str,
        output_dir: Path | str,
    ) -> Path:
        """Run nerfstudio data processing on images.

        Calls ``ns-process-data images`` to run COLMAP and prepare data
        in nerfstudio's expected format.

        Args:
            data_dir: Directory containing input images.
            output_dir: Directory where processed data will be written.

        Returns:
            Path to the processed data directory.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ns-process-data",
            "images",
            "--data",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]

        print(f"Running nerfstudio data processing: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=False,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nerfstudio data processing failed: {e}")

        print(f"Data processed to: {output_dir}")
        return output_dir

    def train(
        self,
        data_dir: Path | str,
        output_dir: Path | str,
        method: str = "splatfacto",
        num_iterations: int = 30000,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Train a 3DGS model using nerfstudio's splatfacto method.

        Args:
            data_dir: Directory containing processed data (images + transforms.json).
            output_dir: Directory where nerfstudio outputs will be saved.
            method: Nerfstudio method name (default: "splatfacto").
            num_iterations: Number of training iterations.
            config: Optional dictionary of additional training parameters.

        Returns:
            Path to the nerfstudio output directory.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Check for transforms.json
        transforms_path = data_dir / "transforms.json"
        if not transforms_path.exists():
            print("No transforms.json found. Running data processing first...")
            self.process_data(data_dir, data_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ns-train",
            method,
            "--data",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--max-num-iterations",
            str(num_iterations),
        ]

        # Add any additional config overrides
        if config:
            for key, value in config.items():
                cmd.extend([f"--{key}", str(value)])

        print("Starting nerfstudio training:")
        print(f"  Method: {method}")
        print(f"  Data: {data_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output and parse progress
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"  [ns] {line}")

            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"nerfstudio training failed with return code {process.returncode}")

        except FileNotFoundError:
            raise ImportError("ns-train command not found. Is nerfstudio installed and on PATH?")

        print(f"\nNerfstudio training complete. Output at: {output_dir}")
        return output_dir

    def export_ply(
        self,
        config_path: Path | str,
        output_path: Path | str,
    ) -> Path:
        """Export a trained nerfstudio model to PLY format.

        Args:
            config_path: Path to the nerfstudio training config.yml.
            output_path: Path for the output .ply file.

        Returns:
            Path to the exported .ply file.
        """
        config_path = Path(config_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(output_path.parent),
        ]

        print(f"Exporting model to PLY: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nerfstudio export failed: {e}")

        # Find the exported PLY file
        ply_files = list(output_path.parent.glob("*.ply"))
        if ply_files:
            result = ply_files[0]
            print(f"Exported PLY: {result}")
            return result
        else:
            raise FileNotFoundError(f"No PLY file found after export in {output_path.parent}")


def train_nerfstudio(
    data_dir: Path | str,
    output_dir: Path | str,
    method: str = "splatfacto",
    config: dict[str, Any] | None = None,
) -> Path:
    """Train a 3DGS model using nerfstudio's splatfacto method.

    Convenience function that creates a NerfstudioTrainer and runs training.

    Args:
        data_dir: Directory containing preprocessed data (images + transforms.json).
        output_dir: Directory where nerfstudio outputs will be saved.
        method: Nerfstudio method name (default: "splatfacto").
        config: Optional dictionary of training hyperparameters.

    Returns:
        Path to the nerfstudio output directory.

    Raises:
        ImportError: If nerfstudio is not installed.
        FileNotFoundError: If the data directory is missing required files.
    """
    trainer = NerfstudioTrainer()
    return trainer.train(
        data_dir=data_dir,
        output_dir=output_dir,
        method=method,
        config=config,
    )
