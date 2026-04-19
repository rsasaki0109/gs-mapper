"""Command-line interface for GS Mapper.

Provides subcommands for the full 3DGS pipeline:
- download: Download datasets from supported sources
- preprocess: Run COLMAP or frame extraction on raw data
- train: Train a 3DGS model using gsplat or nerfstudio
- view: Launch the web viewer for a trained model
- run: Run the full pipeline end-to-end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="gs-mapper",
        description="Large-scale 3D Gaussian Splatting mapper for robotics and driving datasets",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>", help="Available commands")

    # download
    dl = subparsers.add_parser("download", help="Download a dataset")
    dl.add_argument("--dataset", default=None, help="Dataset name (e.g. covla, mcd, autoware_leo_drive_bagN)")
    dl.add_argument("--output", default=None, help="Output directory (default: data/)")
    dl.add_argument("--max-samples", type=int, default=None, help="Max samples to download")
    dl.add_argument("--sample-images", action="store_true", help="Download sample images for quick testing")

    # preprocess
    pp = subparsers.add_parser("preprocess", help="Preprocess images with COLMAP or frame extraction")
    pp.add_argument("--images", required=True, help="Input image directory or video file")
    pp.add_argument("--output", default="outputs/colmap", help="Output directory")
    pp.add_argument(
        "--method",
        choices=["colmap", "frames", "pose-free", "dust3r", "simple", "waymo", "mcd", "lidar-slam"],
        default="colmap",
        help="Preprocessing method (default: colmap). "
        "'pose-free' and 'dust3r' use DUSt3R for pose estimation; "
        "'simple' uses circular camera initialization; "
        "'waymo' extracts frames from Waymo tfrecord files; "
        "'mcd' extracts images and optional sensors from MCD rosbags; "
        "'lidar-slam' imports an external SLAM trajectory.",
    )
    pp.add_argument("--fps", type=float, default=2.0, help="FPS for frame extraction (default: 2)")
    pp.add_argument("--max-frames", type=int, default=100, help="Max frames to extract (default: 100)")
    pp.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy (default: exhaustive)",
    )
    pp.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    pp.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP")
    pp.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera to extract (default: FRONT)",
    )
    pp.add_argument("--every-n", type=int, default=1, help="Extract every N-th frame for Waymo/MCD (default: 1)")
    pp.add_argument("--image-topic", default=None, help="ROS image topic for MCD preprocessing")
    pp.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for MCD preprocessing")
    pp.add_argument("--imu-topic", default=None, help="ROS IMU topic for MCD preprocessing")
    pp.add_argument(
        "--list-topics",
        action="store_true",
        help="For MCD preprocessing, print bag topics with inferred roles and exit",
    )
    pp.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For Waymo preprocessing, also project TOP lidar into per-frame depth maps",
    )
    pp.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For Waymo preprocessing, also generate per-frame dynamic-object masks from camera labels",
    )
    pp.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For MCD preprocessing, also export PointCloud2 frames to lidar/*.npy",
    )
    pp.add_argument(
        "--extract-imu",
        action="store_true",
        help="For MCD preprocessing, also export IMU measurements to imu.csv",
    )
    pp.add_argument(
        "--gnss-topic",
        default=None,
        help="For MCD preprocessing, NavSatFix topic used with --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    pp.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For MCD preprocessing, write COLMAP sparse from GNSS (NavSatFix) + images; single --image-topic only",
    )
    pp.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    pp.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    pp.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    pp.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    pp.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    pp.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna position in base_link (x forward, y left, z up, metres); "
            "subtract using per-sample heading from GNSS motion (do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    pp.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    pp.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    pp.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    pp.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    pp.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy (for depth_loss_weight > 0)",
    )
    pp.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    pp.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    pp.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion "
        "(falls back to the motion-inferred yaw if the column is constant).",
    )
    pp.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    pp.add_argument("--trajectory", default=None, help="SLAM trajectory file (for lidar-slam method)")
    pp.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format (default: tum)",
    )
    pp.add_argument(
        "--nmea-time-offset-sec",
        type=float,
        default=0.0,
        help="Fixed seconds added to NMEA-derived timestamps to realign against a drifted logger clock.",
    )
    pp.add_argument("--pointcloud", default=None, help="Point cloud file for lidar-slam (.ply/.npy/.pcd)")

    # train
    tr = subparsers.add_parser("train", help="Train a 3DGS model")
    tr.add_argument("--data", required=True, help="Preprocessed data directory")
    tr.add_argument("--output", default="outputs/train", help="Training output directory")
    tr.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    tr.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    tr.add_argument("--config", default=None, help="Path to training config YAML override")
    tr.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    # view
    vw = subparsers.add_parser("view", help="Launch the web viewer")
    vw.add_argument("--model", required=True, help="Path to the .ply file or COLMAP sparse dir")
    vw.add_argument("--host", default="0.0.0.0", help="Viewer host (default: 0.0.0.0)")
    vw.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    vw.add_argument("--colmap", action="store_true", help="View COLMAP sparse model instead of PLY")

    # export
    ex = subparsers.add_parser("export", help="Export PLY to web-friendly format")
    ex.add_argument("--model", required=True, help="Path to the .ply file")
    ex.add_argument(
        "--format",
        choices=["json", "binary", "scene-bundle", "splat"],
        default="json",
        help="Output format (default: json). 'splat' writes the antimatter15/splat 32-byte-per-gauss "
        "binary that docs/splat.html renders directly.",
    )
    ex.add_argument("--output", required=True, help="Output file path")
    ex.add_argument("--max-points", type=int, default=100000, help="Max points to export (default: 100000)")
    ex.add_argument(
        "--bundle-asset-format",
        choices=["json", "binary"],
        default="binary",
        help="Asset format used inside --format scene-bundle (default: binary)",
    )
    ex.add_argument("--scene-id", default=None, help="Optional scene id for --format scene-bundle")
    ex.add_argument("--label", default=None, help="Optional scene label for --format scene-bundle")
    ex.add_argument("--description", default="", help="Optional scene description for --format scene-bundle")
    ex.add_argument(
        "--splat-normalize-extent",
        type=float,
        default=None,
        help="For --format splat: rescale the scene so its max-axis extent equals this value (meters). "
        "17.0 matches docs/splat.html's default camera. Leave unset to keep original world scale.",
    )
    ex.add_argument(
        "--splat-min-opacity",
        type=float,
        default=0.0,
        help="For --format splat: drop gaussians below this sigmoid(opacity) threshold (default: 0, no filter).",
    )
    ex.add_argument(
        "--splat-max-scale",
        type=float,
        default=None,
        help="For --format splat: drop gaussians whose max exp(log_scale) exceeds this (meters).",
    )

    # photos-to-splat (one-shot: image dir -> DUSt3R -> gsplat -> .splat)
    p2s = subparsers.add_parser(
        "photos-to-splat",
        help="One-shot: a folder of JPG/PNG -> DUSt3R pose-free -> gsplat train -> .splat file",
    )
    p2s.add_argument("--images", required=True, help="Directory of input images (jpg/png)")
    p2s.add_argument(
        "--output", default="outputs/photos_splat", help="Root output directory (default: outputs/photos_splat)"
    )
    p2s.add_argument(
        "--preprocess",
        choices=["dust3r", "mast3r", "simple"],
        default="dust3r",
        help="Pose-estimation backend. 'mast3r' uses naver/mast3r (newer, metric-aware). "
        "'simple' is a non-metric circular fallback for smoke tests.",
    )
    p2s.add_argument(
        "--num-frames", type=int, default=20, help="DUSt3R/MAST3R frame cap (0 = all). Default 20 fits a 16 GB GPU."
    )
    p2s.add_argument(
        "--scene-graph", default="complete", help="DUSt3R/MAST3R pair graph (complete / swin-N / oneref-K)"
    )
    p2s.add_argument("--dust3r-checkpoint", default=None, help="DUSt3R checkpoint .pth path")
    p2s.add_argument(
        "--dust3r-root", default=None, help="Local clone of naver/dust3r (default: DUST3R_PATH env or /tmp/dust3r)"
    )
    p2s.add_argument("--mast3r-checkpoint", default=None, help="MAST3R checkpoint .pth path")
    p2s.add_argument(
        "--mast3r-root", default=None, help="Local clone of naver/mast3r (default: MAST3R_PATH env or /tmp/mast3r)"
    )
    p2s.add_argument("--mast3r-subsample", type=int, default=8, help="MAST3R pointcloud subsample stride")
    p2s.add_argument("--align-iters", type=int, default=300, help="DUSt3R global alignment iterations")
    p2s.add_argument("--iterations", type=int, default=3000, help="gsplat training iterations")
    p2s.add_argument("--config", default=None, help="Training config YAML override")
    p2s.add_argument(
        "--splat-max-points", type=int, default=400000, help="Max gaussians in .splat output (default: 400k)"
    )
    p2s.add_argument(
        "--splat-normalize-extent",
        type=float,
        default=17.0,
        help="Rescale so the scene max-axis extent matches this (matches docs/splat.html defaults).",
    )
    p2s.add_argument("--splat-min-opacity", type=float, default=0.02, help="Drop gaussians below this opacity")
    p2s.add_argument("--splat-max-scale", type=float, default=2.0, help="Drop gaussians above this scale (meters)")
    p2s.add_argument("--skip-data-check", action="store_true", help="Skip COLMAP sparse preflight before training")

    # benchmark
    bm = subparsers.add_parser("benchmark", help="Benchmark training backends")
    bm.add_argument("--data", required=True, help="Data directory for training")
    bm.add_argument("--iterations", type=int, default=1000, help="Number of training iterations (default: 1000)")
    bm.add_argument("--output", default="outputs/benchmark", help="Benchmark output directory")
    bm.add_argument("--dataset-name", default="default", help="Dataset name label (default: default)")
    bm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio", "both"],
        default="both",
        help="Backend to benchmark (default: both)",
    )
    bm.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat benchmark (not recommended)",
    )

    # run (full pipeline)
    rn = subparsers.add_parser("run", help="Run the full pipeline end-to-end")
    rn.add_argument("--images", required=True, help="Input image directory")
    rn.add_argument("--output", default="outputs", help="Root output directory")
    rn.add_argument(
        "--max-frames", type=int, default=100, help="Max frames to extract for dataset-specific preprocessors"
    )
    rn.add_argument(
        "--every-n", type=int, default=1, help="Extract every N-th frame for dataset-specific preprocessors"
    )
    rn.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    rn.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy for COLMAP-based preprocessors (default: exhaustive)",
    )
    rn.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP-based preprocessing")
    rn.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    rn.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    rn.add_argument("--config", default=None, help="Path to training config YAML override")
    rn.add_argument(
        "--preprocess-method",
        choices=["colmap", "pose-free", "dust3r", "simple", "waymo", "mcd", "lidar-slam"],
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    rn.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera for --preprocess-method waymo (default: FRONT)",
    )
    rn.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For --preprocess-method waymo, also project TOP lidar into per-frame depth maps",
    )
    rn.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For --preprocess-method waymo, also generate per-frame dynamic-object masks from camera labels",
    )
    rn.add_argument("--image-topic", default=None, help="ROS image topic for --preprocess-method mcd")
    rn.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for --preprocess-method mcd")
    rn.add_argument("--imu-topic", default=None, help="ROS IMU topic for --preprocess-method mcd")
    rn.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For --preprocess-method mcd, also export PointCloud2 frames to lidar/*.npy",
    )
    rn.add_argument(
        "--extract-imu",
        action="store_true",
        help="For --preprocess-method mcd, also export IMU measurements to imu.csv",
    )
    rn.add_argument(
        "--gnss-topic",
        default=None,
        help="For --preprocess-method mcd, NavSatFix topic for --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    rn.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For --preprocess-method mcd, COLMAP sparse from GNSS + images; single --image-topic only",
    )
    rn.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    rn.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    rn.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    rn.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    rn.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    rn.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna in base_link (x forward, y left, z up, metres); heading from GNSS motion "
            "(do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    rn.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    rn.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    rn.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    rn.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    rn.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy",
    )
    rn.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    rn.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    rn.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion.",
    )
    rn.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    rn.add_argument("--trajectory", default=None, help="Trajectory file for --preprocess-method lidar-slam")
    rn.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format for --preprocess-method lidar-slam (default: tum)",
    )
    rn.add_argument("--pointcloud", default=None, help="Point cloud file for --preprocess-method lidar-slam")
    rn.add_argument("--skip-preprocess", action="store_true", help="Skip COLMAP preprocessing")
    rn.add_argument("--no-viewer", action="store_true", help="Skip launching the viewer")
    rn.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    rn.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    # demo (end-to-end: images -> splat -> DreamWalker teleop)
    dm = subparsers.add_parser("demo", help="End-to-end demo: images -> 3DGS -> DreamWalker robot teleop")
    dm.add_argument("--images", default=None, help="Input image directory or video file")
    dm.add_argument("--ply", default=None, help="Skip training, stage an existing PLY file directly")
    dm.add_argument("--output", default="outputs", help="Root output directory (default: outputs)")
    dm.add_argument(
        "--max-frames", type=int, default=100, help="Max frames to extract for dataset-specific preprocessors"
    )
    dm.add_argument(
        "--every-n", type=int, default=1, help="Extract every N-th frame for dataset-specific preprocessors"
    )
    dm.add_argument("--colmap-path", default="colmap", help="Path to the COLMAP executable (default: colmap)")
    dm.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy for COLMAP-based preprocessors (default: exhaustive)",
    )
    dm.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP-based preprocessing")
    dm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    dm.add_argument("--iterations", type=int, default=1000, help="Training iterations (default: 1000)")
    dm.add_argument("--config", default=None, help="Path to training config YAML override")
    dm.add_argument(
        "--preprocess-method",
        choices=["colmap", "pose-free", "dust3r", "simple", "waymo", "mcd", "lidar-slam"],
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    dm.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera for --preprocess-method waymo (default: FRONT)",
    )
    dm.add_argument(
        "--extract-lidar-depth",
        action="store_true",
        help="For --preprocess-method waymo, also project TOP lidar into per-frame depth maps",
    )
    dm.add_argument(
        "--extract-dynamic-masks",
        action="store_true",
        help="For --preprocess-method waymo, also generate per-frame dynamic-object masks from camera labels",
    )
    dm.add_argument("--image-topic", default=None, help="ROS image topic for --preprocess-method mcd")
    dm.add_argument("--lidar-topic", default=None, help="ROS PointCloud2 topic for --preprocess-method mcd")
    dm.add_argument("--imu-topic", default=None, help="ROS IMU topic for --preprocess-method mcd")
    dm.add_argument(
        "--extract-lidar",
        action="store_true",
        help="For --preprocess-method mcd, also export PointCloud2 frames to lidar/*.npy",
    )
    dm.add_argument(
        "--extract-imu",
        action="store_true",
        help="For --preprocess-method mcd, also export IMU measurements to imu.csv",
    )
    dm.add_argument(
        "--gnss-topic",
        default=None,
        help="For --preprocess-method mcd, NavSatFix topic for --mcd-seed-poses-from-gnss (default: try /gnss/fix)",
    )
    dm.add_argument(
        "--mcd-seed-poses-from-gnss",
        action="store_true",
        help="For --preprocess-method mcd, COLMAP sparse from GNSS + images; single --image-topic only",
    )
    dm.add_argument(
        "--mcd-base-frame",
        default="base_link",
        help="For MCD GNSS seeding, parent frame for /tf_static lookup (default: base_link)",
    )
    dm.add_argument(
        "--mcd-camera-frame",
        default=None,
        help="For MCD GNSS seeding, camera frame id (default: CameraInfo header.frame_id)",
    )
    dm.add_argument(
        "--mcd-disable-tf-extrinsics",
        action="store_true",
        help="For MCD GNSS seeding, ignore /tf_static (GNSS-only trajectory at vehicle frame)",
    )
    dm.add_argument(
        "--mcd-include-tf-dynamic",
        action="store_true",
        help="For MCD GNSS seeding, merge /tf into TF map after /tf_static (slower on large bags)",
    )
    dm.add_argument(
        "--mcd-gnss-antenna-offset-enu",
        nargs=3,
        type=float,
        default=None,
        metavar=("E", "N", "U"),
        help="Subtract (East, North, Up) meters from each NavSat fix in ENU (approx. base vs antenna)",
    )
    dm.add_argument(
        "--mcd-gnss-antenna-offset-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Antenna in base_link (x forward, y left, z up, metres); heading from GNSS motion "
            "(do not combine with --mcd-gnss-antenna-offset-enu)"
        ),
    )
    dm.add_argument(
        "--mcd-tf-use-image-stamps",
        action="store_true",
        help="Multi-camera GNSS seed: resolve TF at each image time (/tf + /tf_static topology)",
    )
    dm.add_argument(
        "--mcd-lidar-frame",
        default="",
        help="For MCD GNSS seeding, LiDAR frame id under base_link (empty = identity T_base_lidar)",
    )
    dm.add_argument(
        "--mcd-skip-lidar-seed",
        action="store_true",
        help="For MCD GNSS seeding, skip merging LiDAR frames to world as points3D seed",
    )
    dm.add_argument(
        "--mcd-skip-lidar-colorize",
        action="store_true",
        help="Skip the image->LiDAR RGB projection that seeds points3D.txt with real colors",
    )
    dm.add_argument(
        "--mcd-export-depth",
        action="store_true",
        help="Project the world LiDAR cloud into each training image as sparse depth .npy",
    )
    dm.add_argument(
        "--mcd-reference-origin",
        default="",
        help="Share the ENU origin across bags. 'lat,lon,alt' in WGS84 degrees/metres.",
    )
    dm.add_argument(
        "--mcd-reference-bag",
        default="",
        help="Use the ENU origin recorded under <path>/pose/origin_wgs84.json from a previously preprocessed bag.",
    )
    dm.add_argument(
        "--mcd-imu-csv",
        default="",
        help="Path to an imu.csv with orientation_* columns. Interpolated into each TUM row as the base_link quaternion.",
    )
    dm.add_argument(
        "--mcd-skip-imu-orientation",
        action="store_true",
        help="Ignore any imu.csv and keep the default motion-inferred yaw.",
    )
    dm.add_argument("--trajectory", default=None, help="Trajectory file for --preprocess-method lidar-slam")
    dm.add_argument(
        "--trajectory-format",
        choices=["tum", "kitti", "nmea"],
        default="tum",
        help="Trajectory format for --preprocess-method lidar-slam (default: tum)",
    )
    dm.add_argument("--pointcloud", default=None, help="Point cloud file for --preprocess-method lidar-slam")
    dm.add_argument("--fragment", default="residency", help="DreamWalker fragment name (default: residency)")
    dm.add_argument("--no-launch", action="store_true", help="Skip launching the Vite dev server")
    dm.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip COLMAP sparse preflight before gsplat training (not recommended)",
    )

    # robotics ROS2 node
    rb = subparsers.add_parser("robotics-node", help="Launch the DreamWalker ROS2 bridge node scaffold")
    rb.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    rb.add_argument("--node-name", default="dreamwalker_bridge_node", help="ROS2 node name")
    rb.add_argument("--frame-id", default="dreamwalker_map", help="Expected map frame id")
    rb.add_argument("--log-period", type=float, default=2.0, help="Summary log period in seconds")
    rb.add_argument("--zones-file", default=None, help="Optional semantic zone JSON file")
    rb.add_argument("--costmap-period", type=float, default=10.0, help="Costmap republish period in seconds")
    rb.add_argument("--request-state-on-start", action="store_true", help="Publish request_state on startup")
    rb.add_argument(
        "--enable-image-relay",
        action="store_true",
        help="Subscribe to camera relay topics and log received frames",
    )
    rb.add_argument(
        "--demo-teleop",
        choices=["forward", "backward", "turn-left", "turn-right"],
        default=None,
        help="Publish one teleop command on startup",
    )
    rb.add_argument(
        "--demo-camera",
        choices=["front", "chase", "top"],
        default=None,
        help="Publish one camera command on startup",
    )
    rb.add_argument(
        "--demo-waypoint",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Publish one waypoint command on startup",
    )
    rb.add_argument(
        "--demo-pose2d",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA_RAD"),
        default=None,
        help="Publish one Pose2D command on startup",
    )

    # headless PLY render server
    sr = subparsers.add_parser(
        "sim2real-server",
        help="Headless PLY renderer that publishes RGB + depth to DreamWalker ROS2 topics",
    )
    sr.add_argument("--ply", required=True, help="Path to the trained PLY point cloud")
    sr.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    sr.add_argument("--node-name", default="dreamwalker_sim2real_server", help="ROS2 node name")
    sr.add_argument("--frame-id", default="dreamwalker_map", help="Camera frame id for published messages")
    sr.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sr.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sr.add_argument("--fps", type=float, default=5.0, help="Publish rate in Hz")
    sr.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sr.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sr.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sr.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sr.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for camera output")
    sr.add_argument(
        "--renderer",
        choices=["auto", "simple", "gsplat"],
        default="auto",
        help="Rasterization backend. auto uses gsplat only when CUDA and Gaussian PLY parameters are available",
    )
    sr.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of points to load from the PLY for rendering",
    )
    sr.add_argument(
        "--pose-source",
        choices=["static", "robot_pose_stamped", "robot_pose2d", "query"],
        default="robot_pose_stamped",
        help="Source of camera poses used for rendering",
    )
    sr.add_argument(
        "--static-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Static camera position used when --pose-source static",
    )
    sr.add_argument(
        "--static-orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=(0.0, 0.0, 0.0, 1.0),
        help="Static camera orientation quaternion used when --pose-source static",
    )
    sr.add_argument(
        "--pose2d-z",
        type=float,
        default=0.0,
        help="Z position to use when pose source is robot_pose2d",
    )
    sr.add_argument(
        "--query-transport",
        choices=["auto", "none", "zmq", "ws"],
        default="none",
        help="Optional request-response transport for ad-hoc render queries",
    )
    sr.add_argument(
        "--query-endpoint",
        default="tcp://127.0.0.1:5588",
        help=(
            "Bind endpoint for the query transport when enabled. "
            "Defaults: tcp://127.0.0.1:5588 for zmq, ws://127.0.0.1:8781/sim2real for ws"
        ),
    )
    sr.add_argument(
        "--query-poll-period",
        type=float,
        default=0.01,
        help="Polling period in seconds for the query transport",
    )
    sr.add_argument("--run-once", action="store_true", help="Publish one frame and exit")

    # sim2real query client
    sq = subparsers.add_parser(
        "sim2real-query",
        help="Send one pose-based render query to a sim2real headless render server",
    )
    sq.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sq.add_argument("--timeout-ms", type=int, default=10000, help="Request timeout in milliseconds")
    sq.add_argument(
        "--position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="World-space camera position",
    )
    orientation = sq.add_mutually_exclusive_group()
    orientation.add_argument(
        "--orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help="World-space camera orientation quaternion",
    )
    orientation.add_argument(
        "--yaw-degrees",
        type=float,
        default=None,
        help="Planar yaw in degrees, converted to a quaternion around +Z",
    )
    sq.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sq.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sq.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sq.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sq.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sq.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sq.add_argument("--jpeg-out", default=None, help="Optional path for the returned JPEG frame")
    sq.add_argument("--depth-out", default=None, help="Optional path for the returned depth .npy file")
    sq.add_argument("--camera-info-out", default=None, help="Optional path for the returned cameraInfo JSON")
    sq.add_argument("--response-out", default=None, help="Optional path for the full raw response JSON")

    # sim2real localization image benchmark
    sb = subparsers.add_parser(
        "sim2real-benchmark-images",
        help="Render estimate poses through sim2real-server and compare RGB frames against captured ground truth",
    )
    sb.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sb.add_argument(
        "--run",
        default=None,
        help="Optional localization-run-snapshot JSON exported from the web panel",
    )
    sb.add_argument(
        "--ground-truth",
        default=None,
        help="Path to a route-capture-bundle JSON file. Required unless --run already embeds it",
    )
    sb.add_argument(
        "--estimate",
        default=None,
        help="Path to a localization estimate JSON or TUM/ORB-SLAM text trajectory. Required unless --run embeds it",
    )
    sb.add_argument(
        "--alignment",
        choices=["auto", "index", "timestamp"],
        default="auto",
        help="Pose matching mode before rendering estimate frames",
    )
    sb.add_argument(
        "--metrics",
        nargs="+",
        choices=["psnr", "ssim", "lpips"],
        default=["psnr", "ssim", "lpips"],
        help="Image metrics to compute for each matched frame",
    )
    sb.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone used when --metrics includes lpips",
    )
    sb.add_argument(
        "--device",
        default="cpu",
        help="Torch device for LPIPS. Use auto to prefer CUDA when available",
    )
    sb.add_argument("--timeout-ms", type=int, default=10000, help="Per-frame render timeout in milliseconds")
    sb.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of matched frames")
    sb.add_argument("--output", default=None, help="Optional path for the full benchmark report JSON")

    # experiment labs — specs drive a nested `experiment` subparser plus
    # hidden top-level aliases for back-compat.
    experiment_specs: list[tuple[str, str, str]] = [
        (
            "localization-alignment",
            "experiment-localization-alignment",
            "Compare multiple localization alignment strategies and optionally refresh experiment docs",
        ),
        (
            "render-backend-selection",
            "experiment-render-backend-selection",
            "Compare render backend-selection policies and optionally refresh experiment docs",
        ),
        (
            "localization-import",
            "experiment-localization-import",
            "Compare localization estimate import policies and optionally refresh experiment docs",
        ),
        (
            "query-transport-selection",
            "experiment-query-transport-selection",
            "Compare query transport policies and optionally refresh experiment docs",
        ),
        (
            "query-request-import",
            "experiment-query-request-import",
            "Compare query request import policies and optionally refresh experiment docs",
        ),
        (
            "live-localization-stream-import",
            "experiment-live-localization-stream-import",
            "Compare live localization stream import policies and optionally refresh experiment docs",
        ),
        (
            "route-capture-import",
            "experiment-route-capture-import",
            "Compare route capture bundle import policies and optionally refresh experiment docs",
        ),
        (
            "sim2real-websocket-protocol",
            "experiment-sim2real-websocket-protocol",
            "Compare sim2real websocket message protocol policies and optionally refresh experiment docs",
        ),
        (
            "localization-review-bundle-import",
            "experiment-localization-review-bundle-import",
            "Compare localization review bundle import policies and optionally refresh experiment docs",
        ),
        (
            "query-cancellation-policy",
            "experiment-query-cancellation-policy",
            "Compare query cancellation policies and optionally refresh experiment docs",
        ),
        (
            "query-coalescing-policy",
            "experiment-query-coalescing-policy",
            "Compare query dedupe/coalescing policies and optionally refresh experiment docs",
        ),
        (
            "query-error-mapping",
            "experiment-query-error-mapping",
            "Compare query error mapping policies and optionally refresh experiment docs",
        ),
        (
            "query-queue-policy",
            "experiment-query-queue-policy",
            "Compare query queue policies and optionally refresh experiment docs",
        ),
        (
            "query-source-identity",
            "experiment-query-source-identity",
            "Compare query source identity policies and optionally refresh experiment docs",
        ),
        (
            "query-timeout-policy",
            "experiment-query-timeout-policy",
            "Compare query timeout policies and optionally refresh experiment docs",
        ),
        (
            "query-response-build",
            "experiment-query-response-build",
            "Compare query response build policies and optionally refresh experiment docs",
        ),
    ]

    def _add_experiment_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repetitions", type=int, default=200, help="Runtime benchmark repetitions per fixture")
        p.add_argument(
            "--write-docs",
            action="store_true",
            help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
        )
        p.add_argument(
            "--docs-dir",
            default="docs",
            help="Directory where experiment process docs are written when --write-docs is set",
        )
        p.add_argument("--output", default=None, help="Optional path for the full experiment report JSON")

    exp_parent = subparsers.add_parser(
        "experiment",
        help="Experiment labs: A/B strategies and refresh docs/experiments.md",
    )
    exp_sub = exp_parent.add_subparsers(dest="experiment_command", metavar="<lab>", help="Available experiment labs")

    for short, _legacy, help_text in experiment_specs:
        nested = exp_sub.add_parser(short, help=help_text)
        _add_experiment_flags(nested)
    # Back-compat for the flat `experiment-foo` aliases is handled via
    # argv rewriting in `main()` so the main --help stays focused on core tools.

    return parser


LEGACY_EXPERIMENT_ALIASES: dict[str, tuple[str, str]] = {
    "experiment-localization-alignment": ("experiment", "localization-alignment"),
    "experiment-render-backend-selection": ("experiment", "render-backend-selection"),
    "experiment-localization-import": ("experiment", "localization-import"),
    "experiment-query-transport-selection": ("experiment", "query-transport-selection"),
    "experiment-query-request-import": ("experiment", "query-request-import"),
    "experiment-live-localization-stream-import": ("experiment", "live-localization-stream-import"),
    "experiment-route-capture-import": ("experiment", "route-capture-import"),
    "experiment-sim2real-websocket-protocol": ("experiment", "sim2real-websocket-protocol"),
    "experiment-localization-review-bundle-import": ("experiment", "localization-review-bundle-import"),
    "experiment-query-cancellation-policy": ("experiment", "query-cancellation-policy"),
    "experiment-query-coalescing-policy": ("experiment", "query-coalescing-policy"),
    "experiment-query-error-mapping": ("experiment", "query-error-mapping"),
    "experiment-query-queue-policy": ("experiment", "query-queue-policy"),
    "experiment-query-source-identity": ("experiment", "query-source-identity"),
    "experiment-query-timeout-policy": ("experiment", "query-timeout-policy"),
    "experiment-query-response-build": ("experiment", "query-response-build"),
}


def _rewrite_legacy_experiment_argv(argv: list[str]) -> list[str]:
    """Rewrite `gs-mapper experiment-foo ...` -> `gs-mapper experiment foo ...`.

    Keeps old scripts + the READMEs from PR #67 working while the main
    --help listing stays focused on core tools.
    """
    if not argv:
        return argv
    legacy = argv[0]
    mapped = LEGACY_EXPERIMENT_ALIASES.get(legacy)
    if mapped is None:
        return argv
    import warnings

    warnings.warn(
        f"`gs-mapper {legacy}` is deprecated; use `gs-mapper {mapped[0]} {mapped[1]}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return [mapped[0], mapped[1], *argv[1:]]


def cmd_download(args: argparse.Namespace) -> None:
    """Handle the download subcommand."""
    if args.sample_images:
        from gs_sim2real.common.download import download_sample_images

        output_dir = Path(args.output) if args.output else Path("data/sample")
        download_sample_images(output_dir)
        return

    if args.dataset is None:
        print("Error: --dataset is required (unless using --sample-images).")
        sys.exit(1)

    from gs_sim2real.common.download import download_dataset

    output_dir = Path(args.output) if args.output else None
    download_dataset(
        name=args.dataset,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Handle the preprocess subcommand."""
    images_path = Path(args.images)
    output_dir = Path(args.output)

    if args.method == "waymo":
        from gs_sim2real.datasets.waymo import WaymoLoader

        loader = WaymoLoader(data_dir=str(images_path))
        images_out = loader.extract_frames(
            output_dir=str(output_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        # Convert to COLMAP format if camera_params.json exists
        params_path = output_dir / "camera_params.json"
        if params_path.exists():
            sparse_dir = loader.to_colmap_format(
                camera_params_path=str(params_path),
                output_dir=str(output_dir),
            )
            print(f"Waymo frames extracted to: {images_out}")
            print(f"COLMAP sparse model at: {sparse_dir}")
        else:
            print(f"Waymo frames loaded from: {images_out}")
        if args.extract_lidar_depth:
            depth_dir = loader.extract_lidar_depth(
                output_dir=str(output_dir),
                camera=args.camera,
                max_frames=args.max_frames,
                every_n=args.every_n,
            )
            print(f"Waymo LiDAR depth extracted to: {depth_dir}")
        if args.extract_dynamic_masks:
            masks_dir = loader.extract_dynamic_masks(
                output_dir=str(output_dir),
                camera=args.camera,
                max_frames=args.max_frames,
                every_n=args.every_n,
            )
            print(f"Waymo dynamic masks extracted to: {masks_dir}")
        return

    if args.method == "mcd":
        from gs_sim2real.datasets.mcd import MCDLoader

        loader = MCDLoader(data_dir=str(images_path))
        if args.list_topics:
            topics = loader.list_topics()
            if not topics:
                print(f"No rosbag topics found under: {images_path}")
                return
            print("MCD rosbag topics:")
            for topic in topics:
                preferred = " default" if topic["is_preferred_default"] else ""
                print(f"  [{topic['role']}] {topic['topic']} ({topic['msgtype']}, {topic['msgcount']} msgs){preferred}")
            return
        image_topics = _parse_topic_arg(args.image_topic)
        seed_gnss = getattr(args, "mcd_seed_poses_from_gnss", False)
        images_out = loader.extract_frames(
            output_dir=str(output_dir),
            image_topic=image_topics,
            max_frames=args.max_frames,
            every_n=args.every_n,
            save_image_timestamps=seed_gnss,
        )
        print(f"MCD frames available at: {images_out}")
        if args.extract_lidar:
            lidar_dir = loader.extract_lidar(
                output_dir=str(output_dir),
                lidar_topic=args.lidar_topic,
                max_frames=args.max_frames,
                every_n=args.every_n,
                save_timestamps=getattr(args, "mcd_seed_poses_from_gnss", False),
            )
            print(f"MCD LiDAR extracted to: {lidar_dir}")
        if args.extract_imu:
            imu_path = loader.extract_imu(
                output_dir=str(output_dir),
                imu_topic=args.imu_topic,
            )
            print(f"MCD IMU extracted to: {imu_path}")
        if getattr(args, "mcd_seed_poses_from_gnss", False):
            _mcd_gnss_sparse_import(loader, Path(output_dir), images_out, args)
        return

    if args.method == "frames":
        from gs_sim2real.preprocess.extract_frames import (
            extract_frames,
            extract_frames_from_dir,
        )

        if images_path.is_file():
            extract_frames(
                video_path=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        elif images_path.is_dir():
            extract_frames_from_dir(
                input_dir=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        else:
            print(f"Error: '{images_path}' is not a file or directory.")
            sys.exit(1)
    elif args.method == "lidar-slam":
        from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

        if not args.trajectory:
            print("Error: --trajectory is required for lidar-slam method.")
            sys.exit(1)
        sparse_dir = import_lidar_slam(
            trajectory_path=args.trajectory,
            image_dir=images_path,
            output_dir=output_dir,
            trajectory_format=args.trajectory_format,
            pointcloud_path=args.pointcloud,
            nmea_time_offset_sec=getattr(args, "nmea_time_offset_sec", 0.0),
        )
        print(f"LiDAR SLAM import complete: {sparse_dir}")
    elif args.method in ("pose-free", "dust3r", "simple"):
        from gs_sim2real.preprocess.pose_free import run_pose_free

        # Map CLI method names to PoseFreeProcessor methods
        method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
        run_pose_free(
            image_dir=images_path,
            output_dir=output_dir,
            method=method_map[args.method],
        )
    else:
        from gs_sim2real.preprocess.colmap import run_colmap

        run_colmap(
            image_dir=images_path,
            output_dir=output_dir,
            matching=args.matching,
            use_gpu=not args.no_gpu,
            colmap_path=args.colmap_path,
        )


def _preflight_gsplat_train_data(data_dir: Path, skip: bool) -> None:
    """Fail fast if COLMAP sparse model is missing or incomplete (gsplat)."""
    if skip:
        return
    from gs_sim2real.preprocess.colmap_ready import require_colmap_sparse_model

    require_colmap_sparse_model(data_dir)


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the train subcommand."""
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Load config override if provided
    config = None
    if args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)

    if args.method == "gsplat":
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        _preflight_gsplat_train_data(data_dir, getattr(args, "skip_data_check", False))
        ply_path = train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
            num_iterations=args.iterations,
        )
        print(f"\nTrained model saved to: {ply_path}")
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        output = train_nerfstudio(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
        )
        print(f"\nNerfstudio output at: {output}")


def cmd_view(args: argparse.Namespace) -> None:
    """Handle the view subcommand."""
    from gs_sim2real.viewer.web_viewer import GaussianViewer

    viewer = GaussianViewer(host=args.host, port=args.port)

    if args.colmap:
        viewer.view_colmap(args.model)
    else:
        viewer.view_ply(args.model)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export subcommand."""
    if args.format == "json":
        from gs_sim2real.viewer.web_export import ply_to_json

        result = ply_to_json(args.model, args.output, max_points=args.max_points)
    elif args.format == "binary":
        from gs_sim2real.viewer.web_export import ply_to_binary

        result = ply_to_binary(args.model, args.output, max_points=args.max_points)
    elif args.format == "splat":
        from gs_sim2real.viewer.web_export import ply_to_splat

        result = ply_to_splat(
            args.model,
            args.output,
            max_points=args.max_points,
            normalize_target_extent=args.splat_normalize_extent,
            min_opacity=args.splat_min_opacity,
            max_scale=args.splat_max_scale,
        )
    else:
        from gs_sim2real.viewer.web_export import ply_to_scene_bundle

        result = ply_to_scene_bundle(
            args.model,
            args.output,
            asset_format=args.bundle_asset_format,
            scene_id=args.scene_id,
            label=args.label,
            description=args.description,
            max_points=args.max_points,
        )

    print(f"Exported to: {result}")


def cmd_photos_to_splat(args: argparse.Namespace) -> None:
    """Handle the photos-to-splat subcommand.

    One-shot pipeline: image directory -> pose-free sparse (DUSt3R by default)
    -> gsplat training -> antimatter15 .splat binary written to ``<output>/<name>.splat``.
    The .splat can be dropped into ``docs/assets/...`` or served through the
    Pages ``splat.html?url=...`` URL directly.
    """
    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor
    from gs_sim2real.train.gsplat_trainer import train_gsplat
    from gs_sim2real.viewer.web_export import ply_to_splat

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Error: --images must be a directory (got {images_dir})")
        sys.exit(2)

    output_dir = Path(args.output)
    sparse_dir = output_dir / "sparse_input"
    train_dir = output_dir / "train"
    splat_path = output_dir / f"{images_dir.name}.splat"

    print("=" * 60)
    print(f"Step 1/3: Pose-free preprocess ({args.preprocess})")
    print("=" * 60)
    processor_kwargs: dict = {
        "method": args.preprocess,
        "num_frames": args.num_frames,
        "scene_graph": args.scene_graph,
        "align_iters": args.align_iters,
        "mast3r_subsample": args.mast3r_subsample,
    }
    if args.preprocess == "mast3r":
        if args.mast3r_checkpoint:
            processor_kwargs["checkpoint"] = Path(args.mast3r_checkpoint)
        if args.mast3r_root:
            processor_kwargs["mast3r_root"] = Path(args.mast3r_root)
    else:
        if args.dust3r_checkpoint:
            processor_kwargs["checkpoint"] = Path(args.dust3r_checkpoint)
        if args.dust3r_root:
            processor_kwargs["dust3r_root"] = Path(args.dust3r_root)
    processor = PoseFreeProcessor(**processor_kwargs)
    processor.estimate_poses(images_dir, sparse_dir)

    print("\n" + "=" * 60)
    print(f"Step 2/3: gsplat training ({args.iterations} iterations)")
    print("=" * 60)
    config = None
    if args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)
    _preflight_gsplat_train_data(sparse_dir, getattr(args, "skip_data_check", False))
    ply_path = train_gsplat(
        data_dir=sparse_dir,
        output_dir=train_dir,
        config=config,
        num_iterations=args.iterations,
    )

    print("\n" + "=" * 60)
    print("Step 3/3: Exporting to antimatter15 .splat format")
    print("=" * 60)
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    ply_to_splat(
        ply_path,
        splat_path,
        max_points=args.splat_max_points,
        normalize_target_extent=args.splat_normalize_extent,
        min_opacity=args.splat_min_opacity,
        max_scale=args.splat_max_scale,
    )
    print(f"\nDone. Open locally: docs/splat.html?url={splat_path}")
    print(f"Splat file: {splat_path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the benchmark subcommand."""
    from gs_sim2real.benchmark import Benchmark

    bench = Benchmark(data_dir=args.data, output_dir=args.output)

    if args.method in ("gsplat", "both"):
        print("Running gsplat benchmark...")
        bench.run_gsplat(
            num_iterations=args.iterations,
            dataset_name=args.dataset_name,
            skip_data_check=getattr(args, "skip_data_check", False),
        )

    if args.method in ("nerfstudio", "both"):
        print("Running nerfstudio benchmark...")
        bench.run_nerfstudio(num_iterations=args.iterations, dataset_name=args.dataset_name)

    print("\n" + bench.compare())
    bench.save_results()
    print(f"\nResults saved to: {Path(args.output) / 'benchmark_results.json'}")


def _run_waymo_preprocess(
    source_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
):
    """Extract Waymo frames and prepare COLMAP-format inputs for training."""
    from gs_sim2real.datasets.waymo import WaymoLoader
    from gs_sim2real.preprocess.colmap import run_colmap

    loader = WaymoLoader(data_dir=str(source_dir))
    images_out = loader.extract_frames(
        output_dir=str(colmap_dir),
        camera=args.camera,
        max_frames=args.max_frames,
        every_n=args.every_n,
    )

    params_path = colmap_dir / "camera_params.json"
    if params_path.exists():
        sparse_dir = loader.to_colmap_format(
            camera_params_path=str(params_path),
            output_dir=str(colmap_dir),
        )
        print(f"Waymo frames extracted to: {images_out}")
        print(f"COLMAP sparse model at: {sparse_dir}")
    else:
        sparse_dir = run_colmap(
            image_dir=images_out,
            output_dir=colmap_dir,
            matching=getattr(args, "matching", "exhaustive"),
            use_gpu=not getattr(args, "no_gpu", False),
            colmap_path=getattr(args, "colmap_path", "colmap"),
        )
        print(f"Waymo frames loaded from: {images_out}")
        print(f"COLMAP sparse model at: {sparse_dir}")

    if getattr(args, "extract_lidar_depth", False):
        depth_dir = loader.extract_lidar_depth(
            output_dir=str(colmap_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        print(f"Waymo LiDAR depth extracted to: {depth_dir}")

    if getattr(args, "extract_dynamic_masks", False):
        masks_dir = loader.extract_dynamic_masks(
            output_dir=str(colmap_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        print(f"Waymo dynamic masks extracted to: {masks_dir}")

    return sparse_dir


def _parse_topic_arg(value: str | None) -> str | list[str] | None:
    """Parse a CLI topic argument, allowing comma-separated topic lists."""
    if value is None:
        return None
    topics = [topic.strip() for topic in value.split(",") if topic.strip()]
    if not topics:
        return None
    if len(topics) == 1:
        return topics[0]
    return topics


def _pinhole_tuple_from_json(path: Path) -> tuple[float, float, float, float, int, int]:
    """Load PINHOLE intrinsics tuple from MCD CameraInfo JSON."""
    import json

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


def _mcd_antenna_offset_enu(args: argparse.Namespace) -> tuple[float, float, float] | None:
    """Return ``(E,N,U)`` antenna offset or None."""
    v = getattr(args, "mcd_gnss_antenna_offset_enu", None)
    if not v:
        return None
    return (float(v[0]), float(v[1]), float(v[2]))


def _mcd_antenna_offset_base(args: argparse.Namespace) -> tuple[float, float, float] | None:
    """Return ``(x,y,z)`` base_link antenna offset or None."""
    v = getattr(args, "mcd_gnss_antenna_offset_base", None)
    if not v:
        return None
    return (float(v[0]), float(v[1]), float(v[2]))


def _resolve_mcd_imu_csv(args, colmap_dir: Path) -> str | None:
    """Return the path to an IMU CSV that extract_navsat_trajectory can interpolate."""
    if getattr(args, "mcd_skip_imu_orientation", False):
        return None
    explicit = getattr(args, "mcd_imu_csv", None)
    if explicit:
        p = Path(explicit)
        return str(p) if p.is_file() else None
    if getattr(args, "extract_imu", False):
        default_path = Path(colmap_dir) / "imu.csv"
        if default_path.is_file():
            return str(default_path)
    return None


def _resolve_mcd_reference_origin(args) -> tuple[float, float, float] | None:
    """Return (lat, lon, alt) for ENU origin sharing across bags, or None."""
    explicit = getattr(args, "mcd_reference_origin", None)
    if explicit:
        parts = [p.strip() for p in str(explicit).split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"--mcd-reference-origin must be 'lat,lon,alt', got {explicit!r}")
        return float(parts[0]), float(parts[1]), float(parts[2])

    ref_bag = getattr(args, "mcd_reference_bag", None)
    if ref_bag:
        from gs_sim2real.datasets.mcd import MCDLoader

        pose_dir = Path(ref_bag) / "pose"
        if not (pose_dir / "origin_wgs84.json").is_file():
            raise FileNotFoundError(
                f"--mcd-reference-bag={ref_bag!r} has no pose/origin_wgs84.json; "
                "preprocess that bag first (so it writes its GNSS origin)."
            )
        origin = MCDLoader.load_navsat_origin(pose_dir)
        assert origin is not None
        return origin
    return None


def _mcd_export_depth_maps(
    loader,
    xyz_npy: Path,
    colmap_dir: Path,
    images_root: str,
    trajectory_path: str,
    cameras: list[dict],
    hybrid_tf,
    base_frame: str,
) -> None:
    """Project the world LiDAR cloud into each training image and save sparse depth .npy."""
    import numpy as np

    try:
        pts = np.load(xyz_npy)
    except Exception as exc:
        print(f"Warning: depth export could not load {xyz_npy} ({exc})", file=sys.stderr)
        return
    xyz = pts[:, :3].astype(np.float32)
    depth_dir = colmap_dir / "depth"
    try:
        written = loader.export_lidar_depth_per_image(
            lidar_world_xyz=xyz,
            images_root=images_root,
            trajectory_path=trajectory_path,
            cameras=cameras,
            output_dir=depth_dir,
            hybrid_tf=hybrid_tf,
            base_frame=base_frame,
        )
        print(f"MCD per-image LiDAR depth: {written} maps -> {depth_dir}")
    except Exception as exc:
        print(f"Warning: per-image depth export failed ({exc})", file=sys.stderr)


def _mcd_colorize_seed(
    loader,
    xyz_npy: Path,
    images_root: str,
    trajectory_path: str,
    cameras: list[dict],
    hybrid_tf,
    base_frame: str,
) -> Path:
    """Project images onto a world-frame LiDAR cloud and save an Nx6 .npy."""
    import numpy as np

    try:
        pts = np.load(xyz_npy)
    except Exception as exc:
        print(f"Warning: colorize_seed could not load {xyz_npy} ({exc})", file=sys.stderr)
        return xyz_npy
    xyz = pts[:, :3].astype(np.float32)
    try:
        rgb = loader.colorize_lidar_world_from_images(
            lidar_world_xyz=xyz,
            images_root=images_root,
            trajectory_path=trajectory_path,
            cameras=cameras,
            hybrid_tf=hybrid_tf,
            base_frame=base_frame,
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


def _mcd_lidar_world_seed(
    loader,
    colmap_dir: Path,
    lidar_tum_path: str,
    tf_map,
    base_frame: str,
    args: argparse.Namespace,
) -> str | None:
    """Merge per-frame LiDAR NPYs into a world-frame ``.npy``; return path or None on failure/skip."""
    if getattr(args, "mcd_skip_lidar_seed", False):
        return None
    lidar_dir = colmap_dir / "lidar"
    if not lidar_dir.is_dir():
        return None
    if not any(lidar_dir.glob("frame_*.npy")):
        return None

    lidar_frame = (getattr(args, "mcd_lidar_frame", "") or "").strip()
    T_base_lidar = None
    if lidar_frame and tf_map is not None and len(tf_map) > 0:
        T_base_lidar = tf_map.lookup(base_frame, lidar_frame)
        if T_base_lidar is None:
            print(
                f"Warning: no TF path {base_frame!r} -> {lidar_frame!r}; using identity T_base_lidar for LiDAR seed.",
                file=sys.stderr,
            )

    merged_npy = colmap_dir / "lidar_world.npy"
    try:
        out_path = loader.merge_lidar_frames_to_world(
            lidar_dir=lidar_dir,
            trajectory_path=lidar_tum_path,
            output_path=merged_npy,
            T_base_lidar=T_base_lidar,
        )
    except Exception as exc:
        print(f"Warning: MCD LiDAR world merge failed ({exc}); falling back to random seed.", file=sys.stderr)
        return None

    try:
        import numpy as np

        count = int(np.load(out_path).shape[0])
    except Exception:
        count = -1
    print(f"MCD LiDAR world seed: {count} points -> {out_path}")
    return out_path


def _mcd_gnss_sparse_import(
    loader,
    colmap_dir: Path,
    images_out: str,
    args: argparse.Namespace,
) -> str:
    """Build COLMAP sparse from NavSatFix TUM + CameraInfo JSON (+ optional TF)."""
    import json

    from gs_sim2real.datasets.ros_tf import HybridTfLookup
    from gs_sim2real.preprocess.lidar_slam import import_lidar_slam, import_multicam_vehicle_trajectory

    image_topics = _parse_topic_arg(getattr(args, "image_topic", None))
    it_list: list[str] = []
    if isinstance(image_topics, list):
        it_list = image_topics
    elif image_topics is not None:
        it_list = [image_topics]

    include_tf_dynamic = getattr(args, "mcd_include_tf_dynamic", False)
    disable_tf = getattr(args, "mcd_disable_tf_extrinsics", False)
    base_frame = (getattr(args, "mcd_base_frame", None) or "base_link").strip()

    calib_files: list[str] = []
    if it_list:
        calib_files = loader.extract_camera_info(colmap_dir, image_topics=it_list)
        for p in calib_files:
            print(f"MCD camera calibration: {p}")

    tf_map = loader.build_tf_map(include_dynamic_tf=include_tf_dynamic)

    # --- Multi-camera: vehicle TUM + per-camera TF + multiview COLMAP ---
    if len(it_list) > 1:
        cameras: list[dict] = []
        for i, topic in enumerate(it_list):
            label = type(loader)._sanitize_topic_name(topic)
            calib_path = calib_files[i] if i < len(calib_files) else None
            frame_id = ""
            if calib_path:
                with open(calib_path) as f:
                    frame_id = str(json.load(f).get("frame_id") or "").strip()
            if not frame_id:
                frame_id = (getattr(args, "mcd_camera_frame", None) or "").strip()
            T_base_cam = None
            if not disable_tf and frame_id and len(tf_map) > 0:
                T_base_cam = tf_map.lookup(base_frame, frame_id)
                if T_base_cam is None:
                    print(
                        f"Warning: no TF path {base_frame!r} -> {frame_id!r} for topic {topic}; "
                        "using identity extrinsics for this camera.",
                        file=sys.stderr,
                    )
            pinhole = _pinhole_tuple_from_json(Path(calib_path)) if calib_path else None
            cameras.append(
                {
                    "subdir": label,
                    "camera_id": i + 1,
                    "camera_frame": frame_id,
                    "T_base_cam": T_base_cam,
                    "pinhole": pinhole,
                }
            )
        print(
            f"MCD multi-camera GNSS seed: {len(cameras)} cameras "
            f"(tf_edges={len(tf_map)}, dynamic_tf={include_tf_dynamic})"
        )

        ref_origin = _resolve_mcd_reference_origin(args)
        imu_csv_for_gnss = _resolve_mcd_imu_csv(args, colmap_dir)
        tum_path = loader.extract_navsat_trajectory(
            colmap_dir,
            gnss_topic=getattr(args, "gnss_topic", None),
            max_poses=None,
            T_base_cam=None,
            vehicle_frame_only=True,
            antenna_offset_enu=_mcd_antenna_offset_enu(args),
            antenna_offset_base=_mcd_antenna_offset_base(args),
            reference_origin=ref_origin,
            imu_csv_path=imu_csv_for_gnss,
        )
        if ref_origin is not None:
            print(f"MCD vehicle GNSS trajectory (TUM, shared origin {ref_origin}): {tum_path}")
        else:
            print(f"MCD vehicle GNSS trajectory (TUM): {tum_path}")

        pointcloud_path: str | Path | None = getattr(args, "pointcloud", None)
        if not pointcloud_path:
            seeded = _mcd_lidar_world_seed(loader, colmap_dir, tum_path, tf_map, base_frame, args)
            if seeded is not None:
                pointcloud_path = seeded

        hybrid_tf = None
        use_stamp_tf = getattr(args, "mcd_tf_use_image_stamps", False) and not disable_tf
        if use_stamp_tf:
            static_topo = loader.build_tf_map(include_dynamic_tf=False)
            dyn = loader.collect_tf_dynamic_edges()
            hybrid_tf = HybridTfLookup(static_topo, dyn if len(dyn) > 0 else None)
            print("MCD: per-image TF extrinsics (HybridTfLookup: /tf_static topology + /tf samples)")

        if pointcloud_path and not getattr(args, "mcd_skip_lidar_colorize", False):
            pointcloud_path = _mcd_colorize_seed(
                loader,
                Path(pointcloud_path),
                images_out,
                tum_path,
                cameras,
                hybrid_tf,
                base_frame,
            )

        if getattr(args, "mcd_export_depth", False) and pointcloud_path:
            _mcd_export_depth_maps(
                loader,
                Path(pointcloud_path),
                colmap_dir,
                images_out,
                tum_path,
                cameras,
                hybrid_tf,
                base_frame,
            )

        sparse_dir = import_multicam_vehicle_trajectory(
            trajectory_path=tum_path,
            images_root=images_out,
            output_dir=str(colmap_dir),
            cameras=cameras,
            pointcloud_path=pointcloud_path,
            hybrid_tf=hybrid_tf,
            base_frame=base_frame,
        )
        print(f"MCD GNSS-seeded COLMAP sparse model at: {sparse_dir}")
        return sparse_dir

    # --- Single camera ---
    calib_path: str | None = calib_files[0] if calib_files else None

    camera_frame = (getattr(args, "mcd_camera_frame", None) or "").strip()
    if not camera_frame and calib_files:
        with open(calib_files[0]) as f:
            camera_frame = str(json.load(f).get("frame_id") or "").strip()

    T_base_cam = None
    if not disable_tf:
        if camera_frame and len(tf_map) > 0:
            T_base_cam = tf_map.lookup(base_frame, camera_frame)
            if T_base_cam is not None:
                src = "/tf_static + /tf" if include_tf_dynamic else "/tf_static"
                print(f"MCD TF extrinsics: {base_frame} <- {camera_frame} ({src})")
            else:
                print(
                    f"Warning: no TF path from {base_frame!r} to {camera_frame!r}; "
                    "using GNSS translation-only trajectory (vehicle frame).",
                    file=sys.stderr,
                )
        elif len(tf_map) == 0:
            print("Warning: no TF in bag; using GNSS without camera extrinsics.", file=sys.stderr)
        elif not camera_frame:
            print(
                "Warning: no camera frame_id (--mcd-camera-frame or CameraInfo); "
                "using GNSS without TF camera extrinsics.",
                file=sys.stderr,
            )

    ref_origin_single = _resolve_mcd_reference_origin(args)
    imu_csv_for_single = _resolve_mcd_imu_csv(args, colmap_dir)
    pointcloud_path: str | Path | None = getattr(args, "pointcloud", None)
    lidar_tum_path: str | None = None
    if not pointcloud_path and not getattr(args, "mcd_skip_lidar_seed", False):
        try:
            vehicle_tum = loader.extract_navsat_trajectory(
                colmap_dir,
                gnss_topic=getattr(args, "gnss_topic", None),
                max_poses=None,
                T_base_cam=None,
                vehicle_frame_only=True,
                antenna_offset_enu=_mcd_antenna_offset_enu(args),
                antenna_offset_base=_mcd_antenna_offset_base(args),
                reference_origin=ref_origin_single,
                imu_csv_path=imu_csv_for_single,
            )
            vehicle_path = Path(vehicle_tum)
            lidar_side = vehicle_path.with_name("gnss_trajectory_vehicle.tum")
            vehicle_path.replace(lidar_side)
            lidar_tum_path = str(lidar_side)
        except Exception as exc:
            print(
                f"Warning: could not extract vehicle-frame TUM for LiDAR seed ({exc}); skipping LiDAR seed.",
                file=sys.stderr,
            )
            lidar_tum_path = None

    tum_path = loader.extract_navsat_trajectory(
        colmap_dir,
        gnss_topic=getattr(args, "gnss_topic", None),
        max_poses=None,
        T_base_cam=T_base_cam,
        vehicle_frame_only=False,
        antenna_offset_enu=_mcd_antenna_offset_enu(args),
        antenna_offset_base=_mcd_antenna_offset_base(args),
        reference_origin=ref_origin_single,
        imu_csv_path=imu_csv_for_single,
    )
    print(f"MCD GNSS trajectory (TUM): {tum_path}")

    if lidar_tum_path is not None:
        seeded = _mcd_lidar_world_seed(loader, colmap_dir, lidar_tum_path, tf_map, base_frame, args)
        if seeded is not None:
            pointcloud_path = seeded

    sparse_dir = import_lidar_slam(
        trajectory_path=tum_path,
        image_dir=images_out,
        output_dir=colmap_dir,
        trajectory_format="tum",
        pointcloud_path=pointcloud_path,
        pinhole_calib_path=calib_path,
    )
    print(f"MCD GNSS-seeded COLMAP sparse model at: {sparse_dir}")
    return sparse_dir


def _run_mcd_preprocess_to_colmap(
    source_dir: Path,
    colmap_dir: Path,
    args: argparse.Namespace,
):
    """Extract MCD bag data to images and run COLMAP on the extracted frames."""
    from gs_sim2real.datasets.mcd import MCDLoader
    from gs_sim2real.preprocess.colmap import run_colmap

    loader = MCDLoader(data_dir=str(source_dir))
    image_topics = _parse_topic_arg(getattr(args, "image_topic", None))
    seed_gnss = getattr(args, "mcd_seed_poses_from_gnss", False)
    images_out = loader.extract_frames(
        output_dir=str(colmap_dir),
        image_topic=image_topics,
        max_frames=args.max_frames,
        every_n=args.every_n,
        save_image_timestamps=seed_gnss,
    )
    print(f"MCD frames available at: {images_out}")

    if getattr(args, "extract_lidar", False):
        lidar_dir = loader.extract_lidar(
            output_dir=str(colmap_dir),
            lidar_topic=getattr(args, "lidar_topic", None),
            max_frames=args.max_frames,
            every_n=args.every_n,
            save_timestamps=True,
        )
        print(f"MCD LiDAR extracted to: {lidar_dir}")

    if getattr(args, "extract_imu", False):
        imu_path = loader.extract_imu(
            output_dir=str(colmap_dir),
            imu_topic=getattr(args, "imu_topic", None),
        )
        print(f"MCD IMU extracted to: {imu_path}")

    if getattr(args, "mcd_seed_poses_from_gnss", False):
        return _mcd_gnss_sparse_import(loader, colmap_dir, images_out, args)

    return run_colmap(
        image_dir=images_out,
        output_dir=colmap_dir,
        matching=getattr(args, "matching", "exhaustive"),
        use_gpu=not getattr(args, "no_gpu", False),
        colmap_path=getattr(args, "colmap_path", "colmap"),
        single_camera_per_folder=isinstance(image_topics, list),
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the run subcommand (full pipeline)."""
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    colmap_dir = output_dir / "colmap"
    train_dir = output_dir / "train"
    config = None

    if args.config:
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)

    # Step 1: Preprocess
    if not args.skip_preprocess:
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method == "lidar-slam":
            from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

            trajectory = getattr(args, "trajectory", None)
            if not trajectory:
                print("Error: --trajectory is required for lidar-slam method.")
                sys.exit(1)
            import_lidar_slam(
                trajectory_path=trajectory,
                image_dir=images_dir,
                output_dir=colmap_dir,
                trajectory_format=getattr(args, "trajectory_format", "tum"),
                pointcloud_path=getattr(args, "pointcloud", None),
            )
        elif preprocess_method == "waymo":
            _run_waymo_preprocess(images_dir, colmap_dir, args)
        elif preprocess_method == "mcd":
            _run_mcd_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method in ("pose-free", "dust3r", "simple"):
            from gs_sim2real.preprocess.pose_free import run_pose_free

            method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                method=method_map[preprocess_method],
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap

            run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
                matching=args.matching,
                use_gpu=not args.no_gpu,
                colmap_path=args.colmap_path,
            )
    else:
        print("Skipping preprocessing (--skip-preprocess)")

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Training")
    print("=" * 60)

    ply_path = None
    if args.method == "gsplat":
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        _preflight_gsplat_train_data(colmap_dir, getattr(args, "skip_data_check", False))
        ply_path = train_gsplat(
            data_dir=colmap_dir,
            output_dir=train_dir,
            config=config,
            num_iterations=args.iterations,
        )
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        train_nerfstudio(
            data_dir=colmap_dir,
            output_dir=train_dir,
        )

    # Step 3: View
    if not args.no_viewer and ply_path is not None:
        print("\n" + "=" * 60)
        print("Step 3: Viewer")
        print("=" * 60)
        from gs_sim2real.viewer.web_viewer import launch_viewer

        launch_viewer(ply_path, port=args.port)

    print("\nPipeline complete!")


def cmd_demo(args: argparse.Namespace) -> None:
    """Handle the demo subcommand (images -> splat -> DreamWalker teleop)."""
    import subprocess

    ply_path = None

    if args.ply:
        # Use an existing PLY directly
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"Error: PLY file not found: {ply_path}")
            sys.exit(1)
        print(f"Using existing PLY: {ply_path}")
    elif args.images:
        images_dir = Path(args.images)
        output_dir = Path(args.output)
        colmap_dir = output_dir / "colmap"
        train_dir = output_dir / "train"
        config = None

        if args.config:
            from gs_sim2real.common.config import load_config

            config = load_config(args.config)

        # Step 1: Preprocess
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1/3: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method == "lidar-slam":
            from gs_sim2real.preprocess.lidar_slam import import_lidar_slam as _slam_import

            trajectory = getattr(args, "trajectory", None)
            if not trajectory:
                print("Error: --trajectory is required for lidar-slam method.")
                sys.exit(1)
            _slam_import(
                trajectory_path=trajectory,
                image_dir=images_dir,
                output_dir=colmap_dir,
                trajectory_format=getattr(args, "trajectory_format", "tum"),
                pointcloud_path=getattr(args, "pointcloud", None),
            )
        elif preprocess_method == "waymo":
            _run_waymo_preprocess(images_dir, colmap_dir, args)
        elif preprocess_method == "mcd":
            _run_mcd_preprocess_to_colmap(images_dir, colmap_dir, args)
        elif preprocess_method in ("pose-free", "dust3r", "simple"):
            from gs_sim2real.preprocess.pose_free import run_pose_free

            method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                method=method_map[preprocess_method],
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap as _run_colmap

            _run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
                matching=args.matching,
                use_gpu=not args.no_gpu,
                colmap_path=args.colmap_path,
            )

        # Step 2: Train
        print("\n" + "=" * 60)
        print("Step 2/3: Training")
        print("=" * 60)

        if args.method == "gsplat":
            from gs_sim2real.train.gsplat_trainer import train_gsplat

            _preflight_gsplat_train_data(colmap_dir, getattr(args, "skip_data_check", False))
            ply_path = train_gsplat(
                data_dir=colmap_dir,
                output_dir=train_dir,
                config=config,
                num_iterations=args.iterations,
            )
        else:
            from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

            train_nerfstudio(data_dir=colmap_dir, output_dir=train_dir)
            ply_path = train_dir / "point_cloud.ply"
    else:
        print("Error: --images or --ply is required.")
        sys.exit(1)

    # Step 3: Stage into DreamWalker
    print("\n" + "=" * 60)
    print("Step 3/3: Staging for DreamWalker")
    print("=" * 60)

    from gs_sim2real.demo.stage_for_dreamwalker import stage_ply

    result = stage_ply(ply_path, fragment=args.fragment)
    print(f"Splat staged: {result['splat_dest']}")
    print(f"Manifest updated: {result['manifest']}")
    print(f"Launch URL: {result['launch_url']}")

    # Launch Vite dev server
    if not args.no_launch:
        dreamwalker_dir = Path(result["manifest"]).parent.parent.parent
        print(f"\nStarting DreamWalker dev server in {dreamwalker_dir} ...")
        print("Open your browser at:", result["launch_url"])
        print("Controls: WASD = move, Mouse = look, R = toggle robot mode")
        subprocess.run(["npm", "run", "dev"], cwd=dreamwalker_dir)
    else:
        print("\nTo launch manually:")
        print(f"  cd {Path(result['manifest']).parent.parent.parent}")
        print("  npm run dev")
        print(f"  Open: {result['launch_url']}")


def cmd_robotics_node(args: argparse.Namespace) -> None:
    """Handle the robotics-node subcommand."""
    from gs_sim2real.robotics.ros2_bridge_node import run_cli

    run_cli(args)


def cmd_sim2real_server(args: argparse.Namespace) -> None:
    """Handle the sim2real-server subcommand."""
    from gs_sim2real.robotics.gsplat_render_server import run_cli

    run_cli(args)


def cmd_sim2real_query(args: argparse.Namespace) -> None:
    """Handle the sim2real-query subcommand."""
    from gs_sim2real.robotics.render_query_client import run_cli

    run_cli(args)


def cmd_sim2real_benchmark_images(args: argparse.Namespace) -> None:
    """Handle the sim2real-benchmark-images subcommand."""
    from gs_sim2real.robotics.localization_image_benchmark import run_cli

    run_cli(args)


def cmd_experiment(args: argparse.Namespace) -> None:
    """Handle the nested `experiment` subcommand by deferring to the legacy handler."""
    handler_map = {
        "localization-alignment": cmd_experiment_localization_alignment,
        "render-backend-selection": cmd_experiment_render_backend_selection,
        "localization-import": cmd_experiment_localization_import,
        "query-transport-selection": cmd_experiment_query_transport_selection,
        "query-request-import": cmd_experiment_query_request_import,
        "live-localization-stream-import": cmd_experiment_live_localization_stream_import,
        "route-capture-import": cmd_experiment_route_capture_import,
        "sim2real-websocket-protocol": cmd_experiment_sim2real_websocket_protocol,
        "localization-review-bundle-import": cmd_experiment_localization_review_bundle_import,
        "query-cancellation-policy": cmd_experiment_query_cancellation_policy,
        "query-coalescing-policy": cmd_experiment_query_coalescing_policy,
        "query-error-mapping": cmd_experiment_query_error_mapping,
        "query-queue-policy": cmd_experiment_query_queue_policy,
        "query-source-identity": cmd_experiment_query_source_identity,
        "query-timeout-policy": cmd_experiment_query_timeout_policy,
        "query-response-build": cmd_experiment_query_response_build,
    }
    subcmd = getattr(args, "experiment_command", None)
    if subcmd is None:
        print("Error: specify an experiment lab. Run `gs-mapper experiment --help`.", file=sys.stderr)
        sys.exit(2)
    handler = handler_map.get(subcmd)
    if handler is None:
        print(f"Unknown experiment lab: {subcmd}", file=sys.stderr)
        sys.exit(1)
    handler(args)


def cmd_experiment_localization_alignment(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-alignment subcommand."""
    from gs_sim2real.experiments.localization_alignment_lab import run_cli

    run_cli(args)


def cmd_experiment_render_backend_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-render-backend-selection subcommand."""
    from gs_sim2real.experiments.render_backend_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-import subcommand."""
    from gs_sim2real.experiments.localization_estimate_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_transport_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-query-transport-selection subcommand."""
    from gs_sim2real.experiments.query_transport_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_query_request_import(args: argparse.Namespace) -> None:
    """Handle the experiment-query-request-import subcommand."""
    from gs_sim2real.experiments.query_request_import_lab import run_cli

    run_cli(args)


def cmd_experiment_live_localization_stream_import(args: argparse.Namespace) -> None:
    """Handle the experiment-live-localization-stream-import subcommand."""
    from gs_sim2real.experiments.live_localization_stream_import_lab import run_cli

    run_cli(args)


def cmd_experiment_route_capture_import(args: argparse.Namespace) -> None:
    """Handle the experiment-route-capture-import subcommand."""
    from gs_sim2real.experiments.route_capture_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_sim2real_websocket_protocol(args: argparse.Namespace) -> None:
    """Handle the experiment-sim2real-websocket-protocol subcommand."""
    from gs_sim2real.experiments.sim2real_websocket_protocol_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_review_bundle_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-review-bundle-import subcommand."""
    from gs_sim2real.experiments.localization_review_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_cancellation_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-cancellation-policy subcommand."""
    from gs_sim2real.experiments.query_cancellation_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_coalescing_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-coalescing-policy subcommand."""
    from gs_sim2real.experiments.query_coalescing_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_error_mapping(args: argparse.Namespace) -> None:
    """Handle the experiment-query-error-mapping subcommand."""
    from gs_sim2real.experiments.query_error_mapping_lab import run_cli

    run_cli(args)


def cmd_experiment_query_queue_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-queue-policy subcommand."""
    from gs_sim2real.experiments.query_queue_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_source_identity(args: argparse.Namespace) -> None:
    """Handle the experiment-query-source-identity subcommand."""
    from gs_sim2real.experiments.query_source_identity_lab import run_cli

    run_cli(args)


def cmd_experiment_query_timeout_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-timeout-policy subcommand."""
    from gs_sim2real.experiments.query_timeout_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_response_build(args: argparse.Namespace) -> None:
    """Handle the experiment-query-response-build subcommand."""
    from gs_sim2real.experiments.query_response_build_lab import run_cli

    run_cli(args)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the GS Mapper CLI."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    rewritten = _rewrite_legacy_experiment_argv(raw_argv)
    args = parser.parse_args(rewritten)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "view": cmd_view,
        "export": cmd_export,
        "photos-to-splat": cmd_photos_to_splat,
        "benchmark": cmd_benchmark,
        "run": cmd_run,
        "demo": cmd_demo,
        "robotics-node": cmd_robotics_node,
        "sim2real-server": cmd_sim2real_server,
        "sim2real-query": cmd_sim2real_query,
        "sim2real-benchmark-images": cmd_sim2real_benchmark_images,
        "experiment": cmd_experiment,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
