"""Known external visual SLAM artifact naming conventions."""

from __future__ import annotations

from dataclasses import dataclass


_POSE_TENSOR_TRAJECTORY_CANDIDATES = (
    "camera_poses.npz",
    "camera_poses.npy",
    "camera_poses.pt",
    "camera_poses.pth",
    "poses.npz",
    "poses.npy",
    "poses.pt",
    "poses.pth",
    "predictions.npz",
    "predictions.pt",
    "predictions.pth",
    "prediction.npz",
    "result.npz",
    "results.npz",
    "output.npz",
    "outputs.npz",
)

_DEFAULT_TRAJECTORY_CANDIDATES = (
    "poses.txt",
    "trajectory.txt",
    "traj.txt",
    "output_txt.txt",
    "*.tum",
    "*.traj",
    *_POSE_TENSOR_TRAJECTORY_CANDIDATES,
    "*.txt",
)

_DEFAULT_POINTCLOUD_CANDIDATES = (
    "points.ply",
    "pointcloud.ply",
    "map.ply",
    "reconstruction.ply",
    "*_points.pcd",
    "*.ply",
    "*.npy",
    "*.pcd",
)


@dataclass(frozen=True, slots=True)
class ExternalSLAMProfile:
    """Known artifact conventions for an external SLAM/reconstruction system."""

    key: str
    display_name: str
    default_trajectory_format: str
    trajectory_candidates: tuple[str, ...]
    pointcloud_candidates: tuple[str, ...]


PROFILES: dict[str, ExternalSLAMProfile] = {
    "generic": ExternalSLAMProfile(
        key="generic",
        display_name="external SLAM",
        default_trajectory_format="tum",
        trajectory_candidates=_DEFAULT_TRAJECTORY_CANDIDATES,
        pointcloud_candidates=_DEFAULT_POINTCLOUD_CANDIDATES,
    ),
    "mast3r-slam": ExternalSLAMProfile(
        key="mast3r-slam",
        display_name="MASt3R-SLAM",
        default_trajectory_format="tum",
        trajectory_candidates=("poses.txt", "trajectory.txt", "*.tum", "*.txt"),
        pointcloud_candidates=("reconstruction.ply", "map.ply", "*.ply"),
    ),
    "vggt-slam": ExternalSLAMProfile(
        key="vggt-slam",
        display_name="VGGT-SLAM 2.0",
        default_trajectory_format="tum",
        trajectory_candidates=("poses.txt", "trajectory.txt", "*.tum", "*.txt"),
        pointcloud_candidates=("*_points.pcd", "points.pcd", "*.ply", "*.pcd"),
    ),
    "loger": ExternalSLAMProfile(
        key="loger",
        display_name="LoGeR",
        default_trajectory_format="tum",
        trajectory_candidates=(
            "trajectory.txt",
            "pred_traj.txt",
            "poses.txt",
            "output_txt.txt",
            *_POSE_TENSOR_TRAJECTORY_CANDIDATES,
            "results.pth",
            "output.pt",
            "output.pth",
            "result.pt",
            "result.pth",
            "*.tum",
            "*.txt",
            "*.pt",
            "*.pth",
        ),
        pointcloud_candidates=(
            "points.npy",
            "points3d.npy",
            "points3d.npz",
            "pointcloud.npy",
            "pointcloud.npz",
            "points.ply",
            "predictions.pt",
            "predictions.npz",
            "results.pt",
            "results.npz",
            "output.pt",
            "output.npz",
            "result.pt",
            "result.npz",
            "*.ply",
            "*.npy",
            "*.pcd",
            "*.pt",
            "*.pth",
        ),
    ),
    "pi3": ExternalSLAMProfile(
        key="pi3",
        display_name="Pi3/Pi3X",
        default_trajectory_format="tum",
        trajectory_candidates=(
            "poses.txt",
            "trajectory.txt",
            *_POSE_TENSOR_TRAJECTORY_CANDIDATES,
            "*.tum",
            "*.txt",
        ),
        pointcloud_candidates=(
            "result.ply",
            "points.npy",
            "points3d.npy",
            "points3d.npz",
            "pointcloud.npy",
            "pointcloud.npz",
            "points.ply",
            "points.npz",
            "predictions.pt",
            "predictions.npz",
            "prediction.npz",
            "result.npz",
            "outputs.npz",
            "*.ply",
            "*.npy",
            "*.pt",
            "*.pth",
        ),
    ),
}

ALIASES = {
    "mast3r": "mast3r-slam",
    "mast3rslam": "mast3r-slam",
    "mast3r_slam": "mast3r-slam",
    "vggt": "vggt-slam",
    "vggt-slam-2": "vggt-slam",
    "vggt-slam-2.0": "vggt-slam",
    "vggt_slam": "vggt-slam",
    "vggt_slam_2": "vggt-slam",
    "vggt_slam_2.0": "vggt-slam",
    "pi3x": "pi3",
}

SYSTEM_CHOICES = tuple(PROFILES)


def normalize_system(system: str | None) -> str:
    """Normalize external system names while keeping CLI choices stable."""

    if not system:
        return "generic"
    key = system.strip().lower().replace(" ", "-")
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        supported = ", ".join(SYSTEM_CHOICES)
        raise ValueError(f"Unsupported external SLAM system '{system}'. Supported: {supported}")
    return key
