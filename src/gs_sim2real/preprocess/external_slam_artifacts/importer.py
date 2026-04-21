"""Orchestrate external visual SLAM artifact import into COLMAP sparse text."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.preprocess.external_slam_artifacts.pose_tensor import (
    is_pose_tensor_artifact,
    materialize_pose_tensor_trajectory,
)
from gs_sim2real.preprocess.external_slam_artifacts.resolver import resolve_external_slam_artifacts


def import_external_slam(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
    nmea_time_offset_sec: float = 0.0,
) -> str:
    """Convert external visual SLAM artifacts into a COLMAP sparse directory."""

    artifacts = resolve_external_slam_artifacts(
        system=system,
        artifact_dir=artifact_dir,
        trajectory_path=trajectory_path,
        trajectory_format=trajectory_format,
        pointcloud_path=pointcloud_path,
        pinhole_calib_path=pinhole_calib_path,
    )

    trajectory_for_import = artifacts.trajectory_path
    trajectory_format_for_import = artifacts.trajectory_format
    if is_pose_tensor_artifact(trajectory_for_import):
        trajectory_for_import = materialize_pose_tensor_trajectory(
            trajectory_for_import,
            Path(output_dir) / "external_slam",
        )
        trajectory_format_for_import = "tum"

    from gs_sim2real.preprocess.lidar_slam import import_lidar_slam

    return import_lidar_slam(
        trajectory_path=trajectory_for_import,
        image_dir=image_dir,
        output_dir=output_dir,
        trajectory_format=trajectory_format_for_import,
        pointcloud_path=artifacts.pointcloud_path,
        pinhole_calib_path=artifacts.pinhole_calib_path,
        nmea_time_offset_sec=nmea_time_offset_sec,
    )
