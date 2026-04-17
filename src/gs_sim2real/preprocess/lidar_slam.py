"""LiDAR SLAM trajectory import for 3DGS preprocessing.

Imports camera trajectories from external LiDAR SLAM systems
(KISS-ICP, LIO-SAM, ORB-SLAM3, etc.) and converts them to
COLMAP text format for downstream 3DGS training.

Supported trajectory formats:
- TUM: ``timestamp tx ty tz qx qy qz qw``
- KITTI: 12 values per line (3x4 row-major matrix)
- NMEA: mixed ``GGA`` / ``RMC`` sentences converted to ENU poses
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LiDARSLAMProcessor:
    """Import external SLAM trajectories and convert to COLMAP format."""

    def import_trajectory(
        self,
        trajectory_path: str | Path,
        image_dir: str | Path,
        output_dir: str | Path,
        trajectory_format: str = "tum",
        pointcloud_path: str | Path | None = None,
        lidar_to_camera: np.ndarray | None = None,
        max_points: int = 100000,
        pinhole_calib_path: str | Path | None = None,
    ) -> str:
        """Import a SLAM trajectory and optional point cloud.

        Args:
            trajectory_path: Path to trajectory file (TUM, KITTI, or NMEA format).
            image_dir: Directory of images, named with timestamps or sequential.
            output_dir: Where to write COLMAP output.
            trajectory_format: 'tum', 'kitti', or 'nmea'.
            pointcloud_path: Optional path to a merged point cloud (.ply/.npy/.pcd).
            lidar_to_camera: (4, 4) extrinsic from LiDAR to camera frame.
                If None, identity is assumed (camera = LiDAR frame).
            max_points: Maximum number of 3D points to write.
            pinhole_calib_path: Optional JSON from :meth:`MCDLoader.extract_camera_info` with
                keys ``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy`` (PINHOLE).

        Returns:
            Path to sparse reconstruction directory.
        """
        trajectory_path = Path(trajectory_path)
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load trajectory
        if trajectory_format == "tum":
            timestamps, poses = self._load_tum_trajectory(trajectory_path)
        elif trajectory_format == "kitti":
            timestamps, poses = self._load_kitti_trajectory(trajectory_path)
        elif trajectory_format == "nmea":
            timestamps, poses = self._load_nmea_trajectory(trajectory_path)
        else:
            raise ValueError(f"Unknown trajectory format: {trajectory_format}. Use 'tum', 'kitti', or 'nmea'.")

        logger.info("Loaded %d poses from %s trajectory", len(poses), trajectory_format)

        # Find images and align to trajectory
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if not images:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Align images to poses
        aligned = self._align_images_to_poses(images, timestamps, poses)
        logger.info("Aligned %d images to trajectory poses", len(aligned))

        if len(aligned) < 2:
            raise ValueError(f"Need at least 2 aligned image-pose pairs, got {len(aligned)}")

        # Apply LiDAR-to-camera transform if provided
        if lidar_to_camera is not None:
            for entry in aligned:
                entry["pose"] = entry["pose"] @ np.linalg.inv(lidar_to_camera)

        pinhole: tuple[float, float, float, float, int, int] | None = None
        if pinhole_calib_path is not None:
            calib_path = Path(pinhole_calib_path)
            with open(calib_path) as f:
                calib = json.load(f)
            pinhole = (
                float(calib["fx"]),
                float(calib["fy"]),
                float(calib["cx"]),
                float(calib["cy"]),
                int(calib["width"]),
                int(calib["height"]),
            )

        # Read first image for dimensions (or from calibration file)
        if pinhole is not None:
            w, h = pinhole[4], pinhole[5]
        else:
            img = cv2.imread(str(aligned[0]["image"]))
            h, w = img.shape[:2]

        # Estimate focal length when calibration JSON is not provided
        focal = max(w, h) * 1.2

        # Load point cloud if provided
        points3d = None
        if pointcloud_path is not None:
            from gs_sim2real.preprocess.depth_from_lidar import load_pointcloud

            pointcloud_path = Path(pointcloud_path)
            if pointcloud_path.exists():
                pts = load_pointcloud(pointcloud_path)
                xyz = pts[:, :3]
                rgb = pts[:, 3:6] if pts.shape[1] >= 6 else None
                if lidar_to_camera is not None:
                    ones = np.ones((len(xyz), 1))
                    pts_hom = np.hstack([xyz, ones])
                    xyz = (lidar_to_camera @ pts_hom.T).T[:, :3]
                # Subsample
                if len(xyz) > max_points:
                    rng = np.random.default_rng(seed=42)
                    indices = rng.choice(len(xyz), max_points, replace=False)
                    xyz = xyz[indices]
                    if rgb is not None:
                        rgb = rgb[indices]
                points3d = np.hstack([xyz, rgb]) if rgb is not None else xyz
                logger.info("Loaded %d points from point cloud", len(points3d))

        # Write COLMAP format
        return self._write_colmap(aligned, focal, w, h, points3d, output_dir, pinhole=pinhole)

    def import_multicam_vehicle_trajectory(
        self,
        trajectory_path: str | Path,
        images_root: str | Path,
        output_dir: str | Path,
        cameras: list[dict],
        pointcloud_path: str | Path | None = None,
        max_points: int = 100000,
        hybrid_tf: Any | None = None,
        base_frame: str = "base_link",
    ) -> str:
        """Build one COLMAP model from a **vehicle** TUM (ENU + identity) and per-camera TF extrinsics.

        Each ``cameras`` entry should have:
        - ``subdir``: folder name under ``images_root`` (e.g. sanitized ROS topic)
        - ``camera_id``: COLMAP camera id (1-based)
        - ``camera_frame``: optional TF frame id (required for ``hybrid_tf`` per-image lookup)
        - ``T_base_cam``: optional (4, 4) with ``p_base = T @ p_cam``; identity if missing
        - ``pinhole``: optional ``(fx, fy, cx, cy, width, height)``

        If ``hybrid_tf`` is set (see :class:`~gs_sim2real.datasets.ros_tf.HybridTfLookup`), each view uses
        ``T_base_cam`` at the image timestamp; otherwise a constant ``T_base_cam`` per camera is used.
        """
        trajectory_path = Path(trajectory_path)
        images_root = Path(images_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamps, poses = self._load_tum_trajectory(trajectory_path)
        if len(poses) < 2:
            raise ValueError("Vehicle trajectory must have at least 2 poses")

        ts_csv = images_root / "image_timestamps.csv"
        if not ts_csv.is_file():
            raise FileNotFoundError(f"Multi-camera GNSS seed requires {ts_csv} (re-run with MCD extract timestamps).")

        mapping: dict[str, float] = {}
        with open(ts_csv, newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["filename"].strip()] = float(row["timestamp_ns"]) * 1e-9

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        aligned: list[dict] = []

        for cam in cameras:
            subdir = str(cam["subdir"])
            camera_id = int(cam["camera_id"])
            camera_frame = str(cam.get("camera_frame") or "")
            T_const = cam.get("T_base_cam")
            if T_const is not None:
                T_const = np.asarray(T_const, dtype=np.float64)

            folder = images_root / subdir
            if not folder.is_dir():
                logger.warning("Missing image folder for camera %s: %s", camera_id, folder)
                continue

            cam_images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
            for img in cam_images:
                rel = img.relative_to(images_root).as_posix()
                if rel not in mapping:
                    logger.warning("Missing timestamp for %s in image_timestamps.csv", rel)
                    continue
                img_ts = mapping[rel]
                diffs = [abs(ts - img_ts) for ts in timestamps]
                min_idx = int(np.argmin(diffs))
                if diffs[min_idx] >= 0.5:
                    continue
                pose_veh = poses[min_idx].copy()
                stamp_ns = int(round(img_ts * 1e9))

                if hybrid_tf is not None and camera_frame:
                    T_bc = hybrid_tf.lookup(base_frame, camera_frame, stamp_ns)
                    if T_bc is None:
                        T_bc = T_const if T_const is not None else np.eye(4, dtype=np.float64)
                elif T_const is not None:
                    T_bc = T_const
                else:
                    T_bc = np.eye(4, dtype=np.float64)

                pose_cam = pose_veh @ T_bc
                aligned.append(
                    {
                        "image": img,
                        "pose": pose_cam,
                        "camera_id": camera_id,
                        "rel_name": rel,
                    }
                )

        aligned.sort(key=lambda e: e["rel_name"])
        if len(aligned) < 2:
            raise ValueError(f"Need at least 2 aligned views across cameras, got {len(aligned)}")

        pinholes: dict[int, tuple[float, float, float, float, int, int]] = {}
        for cam in cameras:
            cid = int(cam["camera_id"])
            ph = cam.get("pinhole")
            if ph is not None:
                pinholes[cid] = ph

        points3d = None
        if pointcloud_path is not None:
            from gs_sim2real.preprocess.depth_from_lidar import load_pointcloud

            pc_path = Path(pointcloud_path)
            if pc_path.exists():
                pts = load_pointcloud(pc_path)
                if len(pts) > max_points:
                    rng = np.random.default_rng(seed=42)
                    idx = rng.choice(len(pts), max_points, replace=False)
                    pts = pts[idx]
                points3d = pts
                logger.info("Loaded %d points from point cloud", len(points3d))

        return self._write_colmap_multiview(aligned, pinholes, points3d, output_dir)

    def _load_tum_trajectory(self, path: Path) -> tuple[list[float], list[np.ndarray]]:
        """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw.

        Returns:
            Tuple of (timestamps, list of 4x4 pose matrices).
        """
        timestamps = []
        poses = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                R = self._quat_to_rotation(qw, qx, qy, qz)
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = [tx, ty, tz]

                timestamps.append(ts)
                poses.append(pose)

        return timestamps, poses

    def _load_nmea_trajectory(self, path: Path) -> tuple[list[float], list[np.ndarray]]:
        """Load an NMEA trajectory and convert GNSS fixes into ENU poses."""
        fixes_by_time: dict[float, dict[str, float]] = {}
        dates_by_time: dict[float, tuple[int, int, int]] = {}
        courses_by_time: dict[float, float] = {}

        with open(path, encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or not line.startswith("$"):
                    continue

                fields = self._split_nmea_sentence(line)
                if not fields:
                    continue

                sentence_type = fields[0][-3:]
                if sentence_type == "GGA":
                    parsed = self._parse_nmea_gga(fields)
                    if parsed is None:
                        continue
                    fixes_by_time[parsed["seconds"]] = parsed
                elif sentence_type == "RMC":
                    parsed = self._parse_nmea_rmc(fields)
                    if parsed is None:
                        continue
                    if "lat" in parsed and "lon" in parsed:
                        fixes_by_time.setdefault(parsed["seconds"], {}).update(
                            {
                                "seconds": float(parsed["seconds"]),
                                "lat": float(parsed["lat"]),
                                "lon": float(parsed["lon"]),
                                "alt": float(parsed.get("alt", 0.0)),
                            }
                        )
                    if "course_deg" in parsed:
                        courses_by_time[float(parsed["seconds"])] = float(parsed["course_deg"])
                    if "date" in parsed:
                        dates_by_time[float(parsed["seconds"])] = parsed["date"]  # type: ignore[assignment]

        ordered_times = sorted(fixes_by_time)
        if not ordered_times:
            raise ValueError(f"No valid NMEA fixes found in {path}")

        ref_fix = fixes_by_time[ordered_times[0]]
        ref_lat = ref_fix["lat"]
        ref_lon = ref_fix["lon"]
        ref_alt = ref_fix.get("alt", 0.0)

        timestamps: list[float] = []
        poses: list[np.ndarray] = []
        latest_yaw: float | None = None
        unix_origin: float | None = None
        second_origin = ordered_times[0]

        for seconds in ordered_times:
            fix = fixes_by_time[seconds]
            east, north, up = self._wgs84_to_enu(
                lat=fix["lat"],
                lon=fix["lon"],
                alt=fix.get("alt", 0.0),
                ref_lat=ref_lat,
                ref_lon=ref_lon,
                ref_alt=ref_alt,
            )

            course_deg = courses_by_time.get(seconds)
            if course_deg is not None:
                latest_yaw = np.deg2rad(90.0 - course_deg)

            yaw = 0.0 if latest_yaw is None else latest_yaw
            pose = np.eye(4)
            pose[:3, :3] = self._yaw_to_rotation(yaw)
            pose[:3, 3] = [east, north, up]
            poses.append(pose)

            if seconds in dates_by_time:
                year, month, day = dates_by_time[seconds]
                dt = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(seconds=float(seconds))
                timestamp = dt.timestamp()
                if unix_origin is None:
                    unix_origin = timestamp - (seconds - second_origin)
            elif unix_origin is not None:
                timestamp = unix_origin + (seconds - second_origin)
            else:
                timestamp = seconds - second_origin

            timestamps.append(timestamp)

        return timestamps, poses

    def _load_kitti_trajectory(self, path: Path) -> tuple[list[float], list[np.ndarray]]:
        """Load KITTI-format trajectory: 12 values per line (3x4 row-major).

        Returns:
            Tuple of (timestamps as sequential indices, list of 4x4 pose matrices).
        """
        timestamps = []
        poses = []

        with open(path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [float(x) for x in line.split()]
                if len(parts) < 12:
                    continue
                pose = np.eye(4)
                pose[:3, :] = np.array(parts[:12]).reshape(3, 4)
                timestamps.append(float(idx))
                poses.append(pose)

        return timestamps, poses

    def _align_images_to_poses(
        self,
        images: list[Path],
        timestamps: list[float],
        poses: list[np.ndarray],
    ) -> list[dict]:
        """Align images to trajectory poses.

        If ``image_timestamps.csv`` exists (from MCD extraction), matches by bag timestamp.
        Otherwise tries timestamp-based matching (image stem as float),
        then falls back to sequential matching.
        """
        image_dir = images[0].parent if images else Path(".")
        for p in images[:1]:
            for ancestor in [p] + list(p.parents):
                ts_csv = ancestor / "image_timestamps.csv"
                if ts_csv.is_file():
                    image_dir = ancestor
                    break

        ts_path = image_dir / "image_timestamps.csv"
        if ts_path.is_file():
            mapping: dict[str, float] = {}
            with open(ts_path, newline="") as f:
                for row in csv.DictReader(f):
                    fn = row["filename"].strip()
                    mapping[fn] = float(row["timestamp_ns"]) * 1e-9
            aligned: list[dict] = []
            for img in sorted(images):
                rel = img.relative_to(image_dir).as_posix()
                if rel not in mapping:
                    logger.warning("Missing timestamp for %s in image_timestamps.csv", rel)
                    continue
                img_ts = mapping[rel]
                diffs = [abs(ts - img_ts) for ts in timestamps]
                min_idx = int(np.argmin(diffs))
                if diffs[min_idx] < 0.5:
                    aligned.append({"image": img, "pose": poses[min_idx].copy()})
            if len(aligned) >= 2:
                return aligned
            logger.warning("image_timestamps.csv found but <2 poses within 0.5s; trying other alignment methods")

        # Try timestamp matching
        try:
            img_timestamps = [(float(img.stem), img) for img in images]
            aligned = []
            for img_ts, img_path in img_timestamps:
                # Find closest trajectory timestamp
                diffs = [abs(ts - img_ts) for ts in timestamps]
                min_idx = int(np.argmin(diffs))
                if diffs[min_idx] < 0.1:  # within 100ms
                    aligned.append({"image": img_path, "pose": poses[min_idx].copy()})
            if len(aligned) >= 2:
                return aligned
        except (ValueError, TypeError):
            pass

        # Fall back to sequential matching
        n = min(len(images), len(poses))
        return [{"image": images[i], "pose": poses[i].copy()} for i in range(n)]

    @staticmethod
    def _split_nmea_sentence(line: str) -> list[str]:
        """Split an NMEA sentence, dropping the leading '$' and checksum."""
        payload = line[1:] if line.startswith("$") else line
        payload = payload.split("*", 1)[0]
        return payload.split(",")

    @staticmethod
    def _parse_nmea_time(value: str) -> float | None:
        """Parse HHMMSS(.sss) into seconds from midnight."""
        if not value:
            return None
        try:
            raw = float(value)
        except ValueError:
            return None
        hours = int(raw // 10000)
        minutes = int((raw - hours * 10000) // 100)
        seconds = raw - hours * 10000 - minutes * 100
        return hours * 3600.0 + minutes * 60.0 + seconds

    @staticmethod
    def _parse_nmea_coordinate(value: str, hemisphere: str) -> float | None:
        """Parse an NMEA latitude or longitude field into decimal degrees."""
        if not value or hemisphere not in {"N", "S", "E", "W"}:
            return None
        try:
            raw = float(value)
        except ValueError:
            return None

        degrees = int(raw // 100)
        minutes = raw - degrees * 100
        decimal = degrees + minutes / 60.0
        if hemisphere in {"S", "W"}:
            decimal *= -1.0
        return decimal

    def _parse_nmea_gga(self, fields: list[str]) -> dict[str, float] | None:
        """Parse a GGA sentence into a GNSS fix."""
        if len(fields) < 10:
            return None

        seconds = self._parse_nmea_time(fields[1])
        lat = self._parse_nmea_coordinate(fields[2], fields[3])
        lon = self._parse_nmea_coordinate(fields[4], fields[5])
        fix_quality = fields[6]
        if seconds is None or lat is None or lon is None or fix_quality in {"", "0"}:
            return None

        try:
            alt = float(fields[9]) if fields[9] else 0.0
        except ValueError:
            alt = 0.0

        return {
            "seconds": seconds,
            "lat": lat,
            "lon": lon,
            "alt": alt,
        }

    def _parse_nmea_rmc(self, fields: list[str]) -> dict[str, float | tuple[int, int, int]] | None:
        """Parse an RMC sentence into timestamp, position, and course data."""
        if len(fields) < 10 or fields[2] != "A":
            return None

        seconds = self._parse_nmea_time(fields[1])
        lat = self._parse_nmea_coordinate(fields[3], fields[4])
        lon = self._parse_nmea_coordinate(fields[5], fields[6])
        if seconds is None or lat is None or lon is None:
            return None

        result: dict[str, float | tuple[int, int, int]] = {
            "seconds": seconds,
            "lat": lat,
            "lon": lon,
        }
        if fields[8]:
            try:
                result["course_deg"] = float(fields[8])
            except ValueError:
                pass
        date = self._parse_nmea_date(fields[9])
        if date is not None:
            result["date"] = date
        return result

    @staticmethod
    def _parse_nmea_date(value: str) -> tuple[int, int, int] | None:
        """Parse DDMYMY into (year, month, day)."""
        if len(value) != 6 or not value.isdigit():
            return None
        day = int(value[:2])
        month = int(value[2:4])
        year_2digit = int(value[4:6])
        year = 2000 + year_2digit if year_2digit < 80 else 1900 + year_2digit
        return year, month, day

    @staticmethod
    def _wgs84_to_enu(
        lat: float,
        lon: float,
        alt: float,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
    ) -> tuple[float, float, float]:
        """Convert WGS84 coordinates to local ENU coordinates."""
        x, y, z = LiDARSLAMProcessor._wgs84_to_ecef(lat, lon, alt)
        x0, y0, z0 = LiDARSLAMProcessor._wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

        lat0 = np.deg2rad(ref_lat)
        lon0 = np.deg2rad(ref_lon)
        dx = x - x0
        dy = y - y0
        dz = z - z0

        east = -np.sin(lon0) * dx + np.cos(lon0) * dy
        north = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
        up = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz
        return float(east), float(north), float(up)

    @staticmethod
    def _wgs84_to_ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
        """Convert WGS84 geodetic coordinates to ECEF."""
        a = 6378137.0
        e_sq = 6.69437999014e-3
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        N = a / np.sqrt(1.0 - e_sq * sin_lat * sin_lat)
        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1.0 - e_sq) + alt) * sin_lat
        return float(x), float(y), float(z)

    @staticmethod
    def _yaw_to_rotation(yaw_rad: float) -> np.ndarray:
        """Convert a planar yaw angle into a 3x3 rotation matrix."""
        c = np.cos(yaw_rad)
        s = np.sin(yaw_rad)
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def _write_colmap(
        self,
        aligned: list[dict],
        focal: float,
        width: int,
        height: int,
        points3d: np.ndarray | None,
        output_dir: Path,
        pinhole: tuple[float, float, float, float, int, int] | None = None,
    ) -> str:
        """Write aligned data as COLMAP text format."""
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        if pinhole is not None:
            fx, fy, cx, cy, width, height = pinhole
        else:
            fx = fy = focal
            cx, cy = width / 2.0, height / 2.0

        # cameras.txt
        with open(sparse_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

        # images.txt: convert camera-to-world → world-to-camera
        with open(sparse_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, entry in enumerate(aligned):
                c2w = entry["pose"]
                R_c2w = c2w[:3, :3]
                t_c2w = c2w[:3, 3]
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                qw, qx, qy, qz = self._rotation_to_quaternion(R_w2c)
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {entry['image'].name}\n")
                f.write("\n")

        # points3D.txt
        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list\n")
            if points3d is not None:
                has_color = points3d.ndim == 2 and points3d.shape[1] >= 6
                for i, pt in enumerate(points3d):
                    if has_color:
                        r, g, b = int(pt[3]), int(pt[4]), int(pt[5])
                    else:
                        r = g = b = 128
                    f.write(f"{i + 1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.0\n")
            else:
                # Generate sparse random points as fallback
                rng = np.random.default_rng(seed=42)
                centers = np.array([e["pose"][:3, 3] for e in aligned])
                center = centers.mean(axis=0)
                extent = max(np.linalg.norm(centers - center, axis=1).max(), 1.0)
                for i in range(1000):
                    pt = center + rng.uniform(-extent, extent, size=3)
                    f.write(f"{i + 1} {pt[0]} {pt[1]} {pt[2]} 128 128 128 0.0\n")

        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        for entry in aligned:
            dst = images_dir / entry["image"].name
            if Path(entry["image"]).resolve() == dst.resolve():
                continue
            shutil.copy2(entry["image"], dst)

        logger.info(
            "LiDAR SLAM import complete: %d cameras written to %s",
            len(aligned),
            sparse_dir,
        )
        return str(sparse_dir)

    def _write_colmap_multiview(
        self,
        aligned: list[dict],
        pinholes: dict[int, tuple[float, float, float, float, int, int]],
        points3d: np.ndarray | None,
        output_dir: Path,
    ) -> str:
        """Write multi-camera COLMAP text (PINHOLE per camera id, nested image paths)."""
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        cam_ids = sorted({int(e["camera_id"]) for e in aligned})
        for cid in cam_ids:
            if cid not in pinholes:
                e0 = next(x for x in aligned if int(x["camera_id"]) == cid)
                img = cv2.imread(str(e0["image"]))
                if img is None:
                    raise FileNotFoundError(f"Cannot read image for camera {cid}: {e0['image']}")
                h, w = img.shape[:2]
                focal = float(max(w, h) * 1.2)
                pinholes[cid] = (focal, focal, w / 2.0, h / 2.0, w, h)

        with open(sparse_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            for cid in cam_ids:
                fx, fy, cx, cy, w, h = pinholes[cid]
                f.write(f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

        with open(sparse_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            for i, entry in enumerate(aligned):
                c2w = entry["pose"]
                R_c2w = c2w[:3, :3]
                t_c2w = c2w[:3, 3]
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                qw, qx, qy, qz = self._rotation_to_quaternion(R_w2c)
                cid = int(entry["camera_id"])
                name = entry["rel_name"]
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {cid} {name}\n\n")

        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list\n")
            if points3d is not None:
                has_color = points3d.ndim == 2 and points3d.shape[1] >= 6
                for i, pt in enumerate(points3d):
                    if has_color:
                        r, g, b = int(pt[3]), int(pt[4]), int(pt[5])
                    else:
                        r = g = b = 128
                    f.write(f"{i + 1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.0\n")
            else:
                rng = np.random.default_rng(seed=42)
                centers = np.array([e["pose"][:3, 3] for e in aligned])
                center = centers.mean(axis=0)
                extent = max(np.linalg.norm(centers - center, axis=1).max(), 1.0)
                for i in range(1000):
                    pt = center + rng.uniform(-extent, extent, size=3)
                    f.write(f"{i + 1} {pt[0]} {pt[1]} {pt[2]} 128 128 128 0.0\n")

        images_out = output_dir / "images"
        images_out.mkdir(exist_ok=True)
        for entry in aligned:
            rel = Path(entry["rel_name"])
            dest = images_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if Path(entry["image"]).resolve() == dest.resolve():
                continue
            shutil.copy2(entry["image"], dest)

        logger.info("Multi-view LiDAR SLAM import: %d images -> %s", len(aligned), sparse_dir)
        return str(sparse_dir)

    @staticmethod
    def _quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return float(qw), float(qx), float(qy), float(qz)


def import_lidar_slam(
    trajectory_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    trajectory_format: str = "tum",
    pointcloud_path: str | Path | None = None,
    lidar_to_camera: np.ndarray | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> str:
    """Convenience function to import a LiDAR SLAM trajectory.

    Args:
        trajectory_path: Path to trajectory file.
        image_dir: Directory of images.
        output_dir: Output directory.
        trajectory_format: 'tum', 'kitti', or 'nmea'.
        pointcloud_path: Optional merged point cloud.
        lidar_to_camera: Optional (4, 4) extrinsic calibration.
        pinhole_calib_path: Optional PINHOLE JSON (e.g. from MCD CameraInfo export).

    Returns:
        Path to sparse reconstruction directory.
    """
    processor = LiDARSLAMProcessor()
    return processor.import_trajectory(
        trajectory_path=trajectory_path,
        image_dir=image_dir,
        output_dir=output_dir,
        trajectory_format=trajectory_format,
        pointcloud_path=pointcloud_path,
        lidar_to_camera=lidar_to_camera,
        pinhole_calib_path=pinhole_calib_path,
    )


def import_multicam_vehicle_trajectory(
    trajectory_path: str | Path,
    images_root: str | Path,
    output_dir: str | Path,
    cameras: list[dict],
    pointcloud_path: str | Path | None = None,
    hybrid_tf: Any | None = None,
    base_frame: str = "base_link",
) -> str:
    """Convenience wrapper for :meth:`LiDARSLAMProcessor.import_multicam_vehicle_trajectory`."""
    return LiDARSLAMProcessor().import_multicam_vehicle_trajectory(
        trajectory_path=trajectory_path,
        images_root=images_root,
        output_dir=output_dir,
        cameras=cameras,
        pointcloud_path=pointcloud_path,
        hybrid_tf=hybrid_tf,
        base_frame=base_frame,
    )
