"""Waymo Open Dataset loader for 3DGS reconstruction.

Provides utilities to extract camera frames from Waymo tfrecord files
and convert camera parameters to COLMAP text format for downstream
3D Gaussian Splatting training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class WaymoLoader:
    """Load and process Waymo Open Dataset for 3DGS reconstruction."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def extract_frames(
        self,
        output_dir: str,
        camera: str = "FRONT",
        max_frames: int = 100,
        every_n: int = 1,
    ) -> str:
        """Extract camera frames from Waymo tfrecord files.

        Args:
            output_dir: Where to save extracted images.
            camera: Camera name (FRONT, FRONT_LEFT, FRONT_RIGHT,
                SIDE_LEFT, SIDE_RIGHT).
            max_frames: Maximum frames to extract.
            every_n: Extract every N-th frame.

        Returns:
            Path to output image directory.
        """
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        try:
            import tensorflow as tf
            from waymo_open_dataset import dataset_pb2
        except ImportError:
            logger.warning(
                "Waymo Open Dataset SDK not installed. Install with: pip install waymo-open-dataset-tf-2-12-0"
            )
            logger.info("Falling back to image-based loading (looking for pre-extracted images)")
            return self._load_pre_extracted(output_path, max_frames)

        camera_name_map = {
            "FRONT": dataset_pb2.CameraName.FRONT,
            "FRONT_LEFT": dataset_pb2.CameraName.FRONT_LEFT,
            "FRONT_RIGHT": dataset_pb2.CameraName.FRONT_RIGHT,
            "SIDE_LEFT": dataset_pb2.CameraName.SIDE_LEFT,
            "SIDE_RIGHT": dataset_pb2.CameraName.SIDE_RIGHT,
        }

        if camera not in camera_name_map:
            raise ValueError(f"Unknown camera: {camera}. Choose from: {list(camera_name_map.keys())}")

        camera_id = camera_name_map[camera]

        # Find tfrecord files
        tfrecords = sorted(self.data_dir.glob("*.tfrecord"))
        if not tfrecords:
            raise FileNotFoundError(f"No .tfrecord files found in {self.data_dir}")

        frame_count = 0
        poses = []
        intrinsics_list = []

        for tfrecord in tfrecords:
            dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type="")

            for i, raw_record in enumerate(dataset):
                if i % every_n != 0:
                    continue
                if frame_count >= max_frames:
                    break

                frame = dataset_pb2.Frame()
                frame.ParseFromString(raw_record.numpy())

                for image in frame.images:
                    if image.name == camera_id:
                        # Decode image
                        img_array = tf.image.decode_jpeg(image.image).numpy()

                        # Save image
                        import cv2

                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        img_path = images_dir / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(img_path), img_bgr)

                        # Extract camera parameters
                        for calibration in frame.context.camera_calibrations:
                            if calibration.name == camera_id:
                                intrinsics = list(calibration.intrinsic)
                                extrinsic = np.array(calibration.extrinsic.transform).reshape(4, 4)
                                poses.append(extrinsic.tolist())
                                intrinsics_list.append(intrinsics)
                                break

                        frame_count += 1
                        break

            if frame_count >= max_frames:
                break

        # Save camera parameters
        params = {
            "camera": camera,
            "num_frames": frame_count,
            "intrinsics": intrinsics_list,
            "poses": poses,
        }
        with open(output_path / "camera_params.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info("Extracted %d frames from camera %s", frame_count, camera)
        return str(images_dir)

    def _load_pre_extracted(self, output_dir: Path, max_frames: int) -> str:
        """Load pre-extracted images from directory."""
        images_dir = output_dir / "images"
        if not images_dir.exists():
            images_dir = self.data_dir

        images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        if not images:
            raise FileNotFoundError(
                f"No images found in {images_dir}. Either provide .tfrecord files or pre-extracted images."
            )

        logger.info("Found %d pre-extracted images", len(images))
        return str(images_dir)

    def to_colmap_format(self, camera_params_path: str, output_dir: str) -> str:
        """Convert Waymo camera parameters to COLMAP text format.

        Args:
            camera_params_path: Path to camera_params.json from extract_frames().
            output_dir: Where to write COLMAP text files.

        Returns:
            Path to sparse directory.
        """
        with open(camera_params_path) as f:
            params = json.load(f)

        sparse_dir = Path(output_dir) / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Write cameras.txt
        # Waymo intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        intrinsics = params["intrinsics"][0] if params["intrinsics"] else [1000, 1000, 960, 640]
        fx, fy, cx, cy = intrinsics[:4]
        w, h = int(cx * 2), int(cy * 2)

        with open(sparse_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

        # Write images.txt
        with open(sparse_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, pose in enumerate(params["poses"]):
                pose_mat = np.array(pose)
                # Convert to quaternion
                R = pose_mat[:3, :3]
                t = pose_mat[:3, 3]
                qw, qx, qy, qz = self._rotation_to_quaternion(R)
                name = f"frame_{i:06d}.jpg"
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n")
                f.write("\n")

        # Write empty points3D.txt (will be filled by SfM or training)
        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list\n")

        logger.info("Converted %d poses to COLMAP format at %s", len(params["poses"]), sparse_dir)
        return str(sparse_dir)

    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return float(w), float(x), float(y), float(z)
