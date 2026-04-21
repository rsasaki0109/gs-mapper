"""MCD rosbag preprocessing orchestration.

This module keeps MCD-specific extraction, GNSS pose seeding, calibration
fallbacks, and LiDAR seed/depth setup out of the top-level CLI.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gs_sim2real.datasets.ros_tf import HybridTfLookup, StaticTfMap, load_static_calibration_yaml, merge_static_tf_maps

if TYPE_CHECKING:
    from gs_sim2real.datasets.mcd import MCDLoader


PinholeTuple = tuple[float, float, float, float, int, int]


def _mcd_loader_cls():
    from gs_sim2real.datasets import mcd as mcd_module

    return mcd_module.MCDLoader


def parse_topic_arg(value: str | None) -> str | list[str] | None:
    """Parse a CLI topic argument, allowing comma-separated topic lists."""
    if value is None:
        return None
    topics = [topic.strip() for topic in value.split(",") if topic.strip()]
    if not topics:
        return None
    if len(topics) == 1:
        return topics[0]
    return topics


def topic_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def pinhole_tuple_from_json(path: Path) -> PinholeTuple:
    """Load PINHOLE intrinsics tuple from MCD CameraInfo JSON."""
    with open(path) as f:
        c = json.load(f)
    return (
        float(c["fx"]),
        float(c["fy"]),
        float(c["cx"]),
        float(c["cy"]),
        int(c["width"]),
        int(c["height"]),
    )


@dataclass(slots=True)
class MCDPreprocessOptions:
    image_topic: str | None = None
    lidar_topic: str | None = None
    imu_topic: str | None = None
    max_frames: int | None = None
    every_n: int = 1
    extract_lidar: bool = False
    extract_imu: bool = False
    matching: str = "exhaustive"
    no_gpu: bool = False
    colmap_path: str = "colmap"
    run_colmap: bool = True
    pointcloud: str | None = None
    gnss_topic: str | None = None
    mcd_seed_poses_from_gnss: bool = False
    mcd_static_calibration: str | None = None
    mcd_base_frame: str = "base_link"
    mcd_camera_frame: str | None = None
    mcd_lidar_frame: str | None = None
    mcd_include_tf_dynamic: bool = False
    mcd_disable_tf_extrinsics: bool = False
    mcd_tf_use_image_stamps: bool = False
    mcd_skip_lidar_seed: bool = False
    mcd_skip_lidar_colorize: bool = False
    mcd_export_depth: bool = False
    mcd_gnss_antenna_offset_enu: tuple[float, float, float] | None = None
    mcd_gnss_antenna_offset_base: tuple[float, float, float] | None = None
    mcd_reference_origin: str | None = None
    mcd_reference_bag: str | None = None
    mcd_skip_imu_orientation: bool = False
    mcd_imu_csv: str | None = None
    mcd_flatten_gnss_altitude: bool = False
    mcd_start_offset_sec: float = 0.0

    @classmethod
    def from_namespace(cls, args: Any) -> "MCDPreprocessOptions":
        def tuple3(name: str) -> tuple[float, float, float] | None:
            value = getattr(args, name, None)
            if not value:
                return None
            return (float(value[0]), float(value[1]), float(value[2]))

        return cls(
            image_topic=getattr(args, "image_topic", None),
            lidar_topic=getattr(args, "lidar_topic", None),
            imu_topic=getattr(args, "imu_topic", None),
            max_frames=getattr(args, "max_frames", None),
            every_n=getattr(args, "every_n", 1),
            extract_lidar=getattr(args, "extract_lidar", False),
            extract_imu=getattr(args, "extract_imu", False),
            matching=getattr(args, "matching", "exhaustive"),
            no_gpu=getattr(args, "no_gpu", False),
            colmap_path=getattr(args, "colmap_path", "colmap"),
            pointcloud=getattr(args, "pointcloud", None),
            gnss_topic=getattr(args, "gnss_topic", None),
            mcd_seed_poses_from_gnss=getattr(args, "mcd_seed_poses_from_gnss", False),
            mcd_static_calibration=getattr(args, "mcd_static_calibration", None),
            mcd_base_frame=(getattr(args, "mcd_base_frame", None) or "base_link").strip(),
            mcd_camera_frame=getattr(args, "mcd_camera_frame", None),
            mcd_lidar_frame=getattr(args, "mcd_lidar_frame", None),
            mcd_include_tf_dynamic=getattr(args, "mcd_include_tf_dynamic", False),
            mcd_disable_tf_extrinsics=getattr(args, "mcd_disable_tf_extrinsics", False),
            mcd_tf_use_image_stamps=getattr(args, "mcd_tf_use_image_stamps", False),
            mcd_skip_lidar_seed=getattr(args, "mcd_skip_lidar_seed", False),
            mcd_skip_lidar_colorize=getattr(args, "mcd_skip_lidar_colorize", False),
            mcd_export_depth=getattr(args, "mcd_export_depth", False),
            mcd_gnss_antenna_offset_enu=tuple3("mcd_gnss_antenna_offset_enu"),
            mcd_gnss_antenna_offset_base=tuple3("mcd_gnss_antenna_offset_base"),
            mcd_reference_origin=getattr(args, "mcd_reference_origin", None),
            mcd_reference_bag=getattr(args, "mcd_reference_bag", None),
            mcd_skip_imu_orientation=getattr(args, "mcd_skip_imu_orientation", False),
            mcd_imu_csv=getattr(args, "mcd_imu_csv", None),
            mcd_flatten_gnss_altitude=getattr(args, "mcd_flatten_gnss_altitude", False),
            mcd_start_offset_sec=getattr(args, "mcd_start_offset_sec", 0.0) or 0.0,
        )

    @property
    def parsed_image_topics(self) -> str | list[str] | None:
        return parse_topic_arg(self.image_topic)

    @property
    def image_topic_list(self) -> list[str]:
        return topic_list(self.parsed_image_topics)

    @property
    def start_offset_sec(self) -> float:
        return max(0.0, float(self.mcd_start_offset_sec or 0.0))

    @property
    def base_frame(self) -> str:
        return (self.mcd_base_frame or "base_link").strip()

    @property
    def camera_frame(self) -> str:
        return (self.mcd_camera_frame or "").strip()

    @property
    def lidar_frame(self) -> str:
        return (self.mcd_lidar_frame or "").strip()


@dataclass(slots=True)
class MCDCameraCalibration:
    path: Path | None
    frame_id: str
    pinhole: PinholeTuple | None


class MCDCalibrationResolver:
    """Resolve MCD CameraInfo JSONs and optional official calibration YAML."""

    def __init__(self, loader: "MCDLoader", colmap_dir: Path, options: MCDPreprocessOptions) -> None:
        self.loader = loader
        self.colmap_dir = colmap_dir
        self.options = options
        self._yaml_body: dict[str, Any] | None = None

    def load_static_tf(self) -> StaticTfMap | None:
        raw = self.options.mcd_static_calibration
        if not raw:
            return None
        path = str(raw).strip()
        if not path:
            return None
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--mcd-static-calibration: not a file: {p}")
        return load_static_calibration_yaml(p, base_frame=self.options.base_frame)

    def extract_camera_infos(self, topics: list[str]) -> list[MCDCameraCalibration]:
        if not topics:
            return []
        extracted = [Path(p) for p in self.loader.extract_camera_info(self.colmap_dir, image_topics=topics)]
        for p in extracted:
            print(f"MCD camera calibration: {p}")

        calibrations: list[MCDCameraCalibration] = []
        for idx, topic in enumerate(topics):
            path = extracted[idx] if idx < len(extracted) else None
            if path is None:
                path = self.write_pinhole_from_calibration_yaml(topic)
            calibrations.append(self._read_camera_calibration(path))
        return calibrations

    def write_pinhole_from_calibration_yaml(self, image_topic: str) -> Path | None:
        entry_name, entry = self._find_yaml_camera_entry(image_topic)
        if entry is None:
            return None
        intr = entry.get("intrinsics")
        res = entry.get("resolution")
        if not intr or len(intr) < 4 or not res or len(res) < 2:
            return None

        label = self.loader._sanitize_topic_name(image_topic)
        calib_dir = self.colmap_dir / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        out_path = calib_dir / f"{label}.json"
        payload = {
            "width": int(res[0]),
            "height": int(res[1]),
            "fx": float(intr[0]),
            "fy": float(intr[1]),
            "cx": float(intr[2]),
            "cy": float(intr[3]),
            "frame_id": entry_name,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"MCD: PINHOLE calibration from YAML -> {out_path}")
        return out_path

    def _read_camera_calibration(self, path: Path | None) -> MCDCameraCalibration:
        if path is None:
            return MCDCameraCalibration(path=None, frame_id="", pinhole=None)
        with open(path) as f:
            frame_id = str(json.load(f).get("frame_id") or "").strip()
        return MCDCameraCalibration(path=path, frame_id=frame_id, pinhole=pinhole_tuple_from_json(path))

    def _find_yaml_camera_entry(self, image_topic: str) -> tuple[str, dict[str, Any] | None]:
        body = self._load_yaml_body()
        if body is None:
            return "", None
        child = self.options.camera_frame
        if child and child in body and isinstance(body[child], dict):
            return child, body[child]
        for name, ent in body.items():
            if not isinstance(ent, dict):
                continue
            rt = str(ent.get("rostopic") or "").strip().rstrip("/")
            it = str(image_topic).strip().rstrip("/")
            if rt == it:
                return str(name), ent
        return child, None

    def _load_yaml_body(self) -> dict[str, Any] | None:
        if self._yaml_body is not None:
            return self._yaml_body
        raw = self.options.mcd_static_calibration
        if not raw:
            return None
        ypath = Path(str(raw).strip()).expanduser()
        if not ypath.is_file():
            return None
        import yaml

        doc = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        body = doc.get("body") if isinstance(doc, dict) else None
        self._yaml_body = body if isinstance(body, dict) else None
        return self._yaml_body


class MCDPreprocessor:
    """Extract MCD sensor data and prepare COLMAP-compatible sparse inputs."""

    def __init__(self, source_dir: Path, colmap_dir: Path, options: MCDPreprocessOptions) -> None:
        self.source_dir = source_dir
        self.colmap_dir = colmap_dir
        self.options = options
        self.loader = _mcd_loader_cls()(data_dir=str(source_dir))

    def run(self) -> str:
        image_topics = self.options.parsed_image_topics
        images_out = self.loader.extract_frames(
            output_dir=str(self.colmap_dir),
            image_topic=image_topics,
            max_frames=self.options.max_frames,
            every_n=self.options.every_n,
            save_image_timestamps=self.options.mcd_seed_poses_from_gnss,
            start_offset_sec=self.options.start_offset_sec,
        )
        print(f"MCD frames available at: {images_out}")

        if self.options.extract_lidar:
            lidar_dir = self.loader.extract_lidar(
                output_dir=str(self.colmap_dir),
                lidar_topic=self.options.lidar_topic,
                max_frames=self.options.max_frames,
                every_n=self.options.every_n,
                save_timestamps=True,
                start_offset_sec=self.options.start_offset_sec,
            )
            print(f"MCD LiDAR extracted to: {lidar_dir}")

        if self.options.extract_imu:
            imu_path = self.loader.extract_imu(output_dir=str(self.colmap_dir), imu_topic=self.options.imu_topic)
            print(f"MCD IMU extracted to: {imu_path}")

        if self.options.mcd_seed_poses_from_gnss:
            return MCDGnssSparseImporter(self.loader, self.colmap_dir, images_out, self.options).run()

        if not self.options.run_colmap:
            return images_out

        from gs_sim2real.preprocess.colmap import run_colmap

        return run_colmap(
            image_dir=images_out,
            output_dir=self.colmap_dir,
            matching=self.options.matching,
            use_gpu=not self.options.no_gpu,
            colmap_path=self.options.colmap_path,
            single_camera_per_folder=isinstance(image_topics, list),
        )


class MCDGnssSparseImporter:
    """Build a COLMAP sparse model from GNSS poses, camera calibration, and LiDAR seed data."""

    def __init__(
        self,
        loader: "MCDLoader",
        colmap_dir: Path,
        images_out: str,
        options: MCDPreprocessOptions,
    ) -> None:
        self.loader = loader
        self.colmap_dir = colmap_dir
        self.images_out = images_out
        self.options = options
        self.calibration = MCDCalibrationResolver(loader, colmap_dir, options)
        self.static_calib_tf: StaticTfMap | None = None
        self.tf_map: StaticTfMap | None = None

    def run(self) -> str:
        topics = self.options.image_topic_list
        calibrations = self.calibration.extract_camera_infos(topics)
        self.static_calib_tf = self.calibration.load_static_tf()
        self.tf_map = self._build_tf_map()
        if len(topics) > 1:
            return self._run_multicamera(topics, calibrations)
        return self._run_single_camera(topics, calibrations)

    def _build_tf_map(self) -> StaticTfMap:
        tf_map = self.loader.build_tf_map(include_dynamic_tf=self.options.mcd_include_tf_dynamic)
        if self.static_calib_tf is not None:
            n_yaml = len(self.static_calib_tf)
            tf_map = merge_static_tf_maps(self.static_calib_tf, tf_map)
            print(f"MCD: merged calibration YAML ({n_yaml} edges) with bag TF -> {len(tf_map)} static edges")
        return tf_map

    def _run_multicamera(self, topics: list[str], calibrations: list[MCDCameraCalibration]) -> str:
        cameras = [self._camera_config_for_topic(topic, i, calibrations[i]) for i, topic in enumerate(topics)]
        print(
            f"MCD multi-camera GNSS seed: {len(cameras)} cameras "
            f"(tf_edges={len(self.tf_map or [])}, dynamic_tf={self.options.mcd_include_tf_dynamic})"
        )

        tum_path = self._extract_gnss_trajectory(vehicle_frame_only=True)
        ref_origin = self._resolve_reference_origin()
        if ref_origin is not None:
            print(f"MCD vehicle GNSS trajectory (TUM, shared origin {ref_origin}): {tum_path}")
        else:
            print(f"MCD vehicle GNSS trajectory (TUM): {tum_path}")

        pointcloud_path: str | Path | None = self.options.pointcloud
        if not pointcloud_path:
            seeded = self._lidar_world_seed(tum_path)
            if seeded is not None:
                pointcloud_path = seeded

        hybrid_tf = (
            self._hybrid_tf()
            if self.options.mcd_tf_use_image_stamps and not self.options.mcd_disable_tf_extrinsics
            else None
        )
        if hybrid_tf is not None:
            print("MCD: per-image TF extrinsics (HybridTfLookup: /tf_static topology + /tf samples)")

        pointcloud_path = self._maybe_colorize_and_export_depth(pointcloud_path, tum_path, cameras, hybrid_tf)
        from gs_sim2real.preprocess.lidar_slam import import_multicam_vehicle_trajectory

        sparse_dir = import_multicam_vehicle_trajectory(
            trajectory_path=tum_path,
            images_root=self.images_out,
            output_dir=str(self.colmap_dir),
            cameras=cameras,
            pointcloud_path=pointcloud_path,
            hybrid_tf=hybrid_tf,
            base_frame=self.options.base_frame,
        )
        print(f"MCD GNSS-seeded COLMAP sparse model at: {sparse_dir}")
        return sparse_dir

    def _run_single_camera(self, topics: list[str], calibrations: list[MCDCameraCalibration]) -> str:
        calib = calibrations[0] if calibrations else MCDCameraCalibration(path=None, frame_id="", pinhole=None)
        camera_frame = self._single_camera_frame(calib)
        T_base_cam = self._lookup_single_camera_extrinsics(camera_frame)

        pointcloud_path: str | Path | None = self.options.pointcloud
        lidar_tum_path = None
        if not pointcloud_path and not self.options.mcd_skip_lidar_seed:
            lidar_tum_path = self._extract_lidar_side_vehicle_tum()

        tum_path = self._extract_gnss_trajectory(vehicle_frame_only=False, T_base_cam=T_base_cam)
        print(f"MCD GNSS trajectory (TUM): {tum_path}")

        if lidar_tum_path is not None:
            seeded = self._lidar_world_seed(lidar_tum_path)
            if seeded is not None:
                pointcloud_path = seeded

        if pointcloud_path and len(topics) == 1:
            cam_mono = {
                "subdir": "",
                "camera_id": 1,
                "camera_frame": camera_frame,
                "T_base_cam": T_base_cam,
                "pinhole": calib.pinhole,
            }
            hybrid_tf = (
                self._hybrid_tf()
                if self.options.mcd_tf_use_image_stamps and not self.options.mcd_disable_tf_extrinsics
                else None
            )
            if hybrid_tf is not None:
                print("MCD: per-image TF (HybridTfLookup) for single-camera depth/colorize")
            pointcloud_path = self._maybe_colorize_and_export_depth(pointcloud_path, tum_path, [cam_mono], hybrid_tf)

        from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

        sparse_dir = import_lidar_slam(
            trajectory_path=tum_path,
            image_dir=self.images_out,
            output_dir=self.colmap_dir,
            trajectory_format="tum",
            pointcloud_path=pointcloud_path,
            pinhole_calib_path=str(calib.path) if calib.path else None,
        )
        print(f"MCD GNSS-seeded COLMAP sparse model at: {sparse_dir}")
        return sparse_dir

    def _camera_config_for_topic(
        self,
        topic: str,
        index: int,
        calibration: MCDCameraCalibration,
    ) -> dict[str, Any]:
        frame_id = calibration.frame_id or self.options.camera_frame
        T_base_cam = None
        if not self.options.mcd_disable_tf_extrinsics and frame_id and self.tf_map is not None and len(self.tf_map) > 0:
            T_base_cam = self.tf_map.lookup(self.options.base_frame, frame_id)
            if T_base_cam is None:
                print(
                    f"Warning: no TF path {self.options.base_frame!r} -> {frame_id!r} for topic {topic}; "
                    "using identity extrinsics for this camera.",
                    file=sys.stderr,
                )
        return {
            "subdir": self.loader._sanitize_topic_name(topic),
            "camera_id": index + 1,
            "camera_frame": frame_id,
            "T_base_cam": T_base_cam,
            "pinhole": calibration.pinhole,
        }

    def _single_camera_frame(self, calibration: MCDCameraCalibration) -> str:
        return self.options.camera_frame or calibration.frame_id

    def _lookup_single_camera_extrinsics(self, camera_frame: str):
        if self.options.mcd_disable_tf_extrinsics:
            return None
        if camera_frame and self.tf_map is not None and len(self.tf_map) > 0:
            T_base_cam = self.tf_map.lookup(self.options.base_frame, camera_frame)
            if T_base_cam is not None:
                src = "/tf_static + /tf" if self.options.mcd_include_tf_dynamic else "/tf_static"
                print(f"MCD TF extrinsics: {self.options.base_frame} <- {camera_frame} ({src})")
                return T_base_cam
            print(
                f"Warning: no TF path from {self.options.base_frame!r} to {camera_frame!r}; "
                "using GNSS translation-only trajectory (vehicle frame).",
                file=sys.stderr,
            )
        elif self.tf_map is not None and len(self.tf_map) == 0:
            print("Warning: no TF in bag; using GNSS without camera extrinsics.", file=sys.stderr)
        elif not camera_frame:
            print(
                "Warning: no camera frame_id (--mcd-camera-frame or CameraInfo); "
                "using GNSS without TF camera extrinsics.",
                file=sys.stderr,
            )
        return None

    def _extract_gnss_trajectory(self, *, vehicle_frame_only: bool, T_base_cam=None) -> str:
        return self.loader.extract_navsat_trajectory(
            self.colmap_dir,
            gnss_topic=self.options.gnss_topic,
            max_poses=None,
            T_base_cam=T_base_cam,
            vehicle_frame_only=vehicle_frame_only,
            antenna_offset_enu=self.options.mcd_gnss_antenna_offset_enu,
            antenna_offset_base=self.options.mcd_gnss_antenna_offset_base,
            reference_origin=self._resolve_reference_origin(),
            imu_csv_path=self._resolve_imu_csv(),
            flatten_altitude=self.options.mcd_flatten_gnss_altitude,
            start_offset_sec=self.options.start_offset_sec,
        )

    def _extract_lidar_side_vehicle_tum(self) -> str | None:
        try:
            vehicle_tum = self._extract_gnss_trajectory(vehicle_frame_only=True)
            vehicle_path = Path(vehicle_tum)
            lidar_side = vehicle_path.with_name("gnss_trajectory_vehicle.tum")
            vehicle_path.replace(lidar_side)
            return str(lidar_side)
        except Exception as exc:
            print(
                f"Warning: could not extract vehicle-frame TUM for LiDAR seed ({exc}); skipping LiDAR seed.",
                file=sys.stderr,
            )
            return None

    def _resolve_imu_csv(self) -> str | None:
        if self.options.mcd_skip_imu_orientation:
            return None
        if self.options.mcd_imu_csv:
            p = Path(self.options.mcd_imu_csv)
            return str(p) if p.is_file() else None
        if self.options.extract_imu:
            default_path = self.colmap_dir / "imu.csv"
            if default_path.is_file():
                return str(default_path)
        return None

    def _resolve_reference_origin(self) -> tuple[float, float, float] | None:
        explicit = self.options.mcd_reference_origin
        if explicit:
            parts = [p.strip() for p in str(explicit).split(",") if p.strip()]
            if len(parts) != 3:
                raise ValueError(f"--mcd-reference-origin must be 'lat,lon,alt', got {explicit!r}")
            return float(parts[0]), float(parts[1]), float(parts[2])

        ref_bag = self.options.mcd_reference_bag
        if ref_bag:
            pose_dir = Path(ref_bag) / "pose"
            if not (pose_dir / "origin_wgs84.json").is_file():
                raise FileNotFoundError(
                    f"--mcd-reference-bag={ref_bag!r} has no pose/origin_wgs84.json; "
                    "preprocess that bag first (so it writes its GNSS origin)."
                )
            origin = _mcd_loader_cls().load_navsat_origin(pose_dir)
            assert origin is not None
            return origin
        return None

    def _hybrid_tf(self) -> HybridTfLookup:
        static_topo = self.loader.build_tf_map(include_dynamic_tf=False)
        if self.static_calib_tf is not None:
            static_topo = merge_static_tf_maps(self.static_calib_tf, static_topo)
        dyn = self.loader.collect_tf_dynamic_edges()
        return HybridTfLookup(static_topo, dyn if len(dyn) > 0 else None)

    def _lidar_world_seed(self, lidar_tum_path: str) -> str | None:
        if self.options.mcd_skip_lidar_seed:
            return None
        lidar_dir = self.colmap_dir / "lidar"
        if not lidar_dir.is_dir() or not any(lidar_dir.glob("frame_*.npy")):
            return None

        T_base_lidar = None
        if self.options.lidar_frame and self.tf_map is not None and len(self.tf_map) > 0:
            T_base_lidar = self.tf_map.lookup(self.options.base_frame, self.options.lidar_frame)
            if T_base_lidar is None:
                print(
                    f"Warning: no TF path {self.options.base_frame!r} -> {self.options.lidar_frame!r}; "
                    "using identity T_base_lidar for LiDAR seed.",
                    file=sys.stderr,
                )

        merged_npy = self.colmap_dir / "lidar_world.npy"
        try:
            out_path = self.loader.merge_lidar_frames_to_world(
                lidar_dir=lidar_dir,
                trajectory_path=lidar_tum_path,
                output_path=merged_npy,
                T_base_lidar=T_base_lidar,
            )
        except Exception as exc:
            print(f"Warning: MCD LiDAR world merge failed ({exc}); falling back to random seed.", file=sys.stderr)
            return None

        try:
            count = int(np.load(out_path).shape[0])
        except Exception:
            count = -1
        print(f"MCD LiDAR world seed: {count} points -> {out_path}")
        return out_path

    def _maybe_colorize_and_export_depth(
        self,
        pointcloud_path: str | Path | None,
        trajectory_path: str,
        cameras: list[dict],
        hybrid_tf: HybridTfLookup | None,
    ) -> str | Path | None:
        if not pointcloud_path:
            return pointcloud_path
        out_path: str | Path = pointcloud_path
        if not self.options.mcd_skip_lidar_colorize:
            out_path = self._colorize_seed(Path(out_path), trajectory_path, cameras, hybrid_tf)
        if self.options.mcd_export_depth:
            self._export_depth_maps(Path(out_path), trajectory_path, cameras, hybrid_tf)
        return out_path

    def _colorize_seed(
        self,
        xyz_npy: Path,
        trajectory_path: str,
        cameras: list[dict],
        hybrid_tf: HybridTfLookup | None,
    ) -> Path:
        try:
            pts = np.load(xyz_npy)
        except Exception as exc:
            print(f"Warning: colorize_seed could not load {xyz_npy} ({exc})", file=sys.stderr)
            return xyz_npy
        xyz = pts[:, :3].astype(np.float32)
        try:
            rgb = self.loader.colorize_lidar_world_from_images(
                lidar_world_xyz=xyz,
                images_root=self.images_out,
                trajectory_path=trajectory_path,
                cameras=cameras,
                hybrid_tf=hybrid_tf,
                base_frame=self.options.base_frame,
            )
        except Exception as exc:
            print(f"Warning: LiDAR colorize failed ({exc}); keeping grey seed", file=sys.stderr)
            return xyz_npy

        colored = np.hstack([xyz.astype(np.float32), rgb.astype(np.float32)])
        out_npy = xyz_npy.with_name("lidar_world_rgb.npy")
        np.save(out_npy, colored)
        covered = int((rgb.sum(axis=1) != 128 * 3).sum())
        print(f"MCD LiDAR colorized seed: {covered}/{len(xyz)} points with image RGB -> {out_npy}")
        return out_npy

    def _export_depth_maps(
        self,
        xyz_npy: Path,
        trajectory_path: str,
        cameras: list[dict],
        hybrid_tf: HybridTfLookup | None,
    ) -> None:
        try:
            pts = np.load(xyz_npy)
        except Exception as exc:
            print(f"Warning: depth export could not load {xyz_npy} ({exc})", file=sys.stderr)
            return
        xyz = pts[:, :3].astype(np.float32)
        depth_dir = self.colmap_dir / "depth"
        try:
            written = self.loader.export_lidar_depth_per_image(
                lidar_world_xyz=xyz,
                images_root=self.images_out,
                trajectory_path=trajectory_path,
                cameras=cameras,
                output_dir=depth_dir,
                hybrid_tf=hybrid_tf,
                base_frame=self.options.base_frame,
            )
            print(f"MCD per-image LiDAR depth: {written} maps -> {depth_dir}")
        except Exception as exc:
            print(f"Warning: per-image depth export failed ({exc})", file=sys.stderr)


def run_mcd_preprocess_to_colmap(
    source_dir: Path,
    colmap_dir: Path,
    options: MCDPreprocessOptions,
) -> str:
    return MCDPreprocessor(source_dir, colmap_dir, options).run()
