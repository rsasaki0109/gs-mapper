"""Tests for external visual SLAM artifact import boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gs_sim2real.preprocess.external_slam import (
    ExternalSLAMManifestGatePolicy,
    build_external_slam_artifact_manifest,
    evaluate_external_slam_manifest_gate,
    import_external_slam,
    materialize_pose_tensor_trajectory,
    normalize_system,
    render_external_slam_artifact_manifest_json,
    render_external_slam_artifact_manifest_text,
    render_external_slam_manifest_gate_text,
    resolve_external_slam_artifacts,
)


def _write_dummy_images(image_dir: Path, count: int = 2) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        cv2.imwrite(str(image_dir / f"frame_{idx:06d}.jpg"), np.zeros((32, 48, 3), dtype=np.uint8))


def test_normalize_system_accepts_common_aliases() -> None:
    assert normalize_system("MASt3R") == "mast3r-slam"
    assert normalize_system("VGGT-SLAM-2.0") == "vggt-slam"
    assert normalize_system("Pi3X") == "pi3"


def test_resolve_loger_artifacts_from_output_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "loger_out"
    nested = artifact_dir / "sequence_a"
    nested.mkdir(parents=True)
    trajectory = nested / "trajectory.txt"
    pointcloud = nested / "points.ply"
    trajectory.write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")
    pointcloud.write_text("ply\n")

    artifacts = resolve_external_slam_artifacts(system="loger", artifact_dir=artifact_dir)

    assert artifacts.system == "loger"
    assert artifacts.trajectory_path == trajectory
    assert artifacts.trajectory_format == "tum"
    assert artifacts.pointcloud_path == pointcloud


def test_relative_explicit_paths_resolve_under_artifact_dir(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "vggt_out"
    artifact_dir.mkdir()
    trajectory = artifact_dir / "custom_poses.txt"
    trajectory.write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")

    artifacts = resolve_external_slam_artifacts(
        system="vggt-slam",
        artifact_dir=artifact_dir,
        trajectory_path="custom_poses.txt",
    )

    assert artifacts.trajectory_path == trajectory


def test_pi3_pointcloud_only_still_requires_trajectory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "pi3_out"
    artifact_dir.mkdir()
    (artifact_dir / "result.ply").write_text("ply\n")

    with pytest.raises(FileNotFoundError, match="Pi3/Pi3X trajectory"):
        resolve_external_slam_artifacts(system="pi3", artifact_dir=artifact_dir)


def test_resolve_pi3_camera_pose_tensor_from_output_directory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "pi3_out"
    artifact_dir.mkdir()
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    np.save(artifact_dir / "camera_poses.npy", poses)
    (artifact_dir / "result.ply").write_text("ply\n")

    artifacts = resolve_external_slam_artifacts(system="pi3", artifact_dir=artifact_dir)

    assert artifacts.trajectory_path == artifact_dir / "camera_poses.npy"
    assert artifacts.pointcloud_path == artifact_dir / "result.ply"


def test_build_external_slam_manifest_marks_tensor_materialization(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir, count=2)
    artifact_dir = tmp_path / "pi3_out"
    artifact_dir.mkdir()
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    np.savez(artifact_dir / "camera_poses.npz", camera_poses=poses)
    np.savez(
        artifact_dir / "points.npz",
        points=np.zeros((2, 2, 3), dtype=np.float32),
        conf=np.ones((2, 2), dtype=np.float32),
        images=np.ones((2, 2, 3), dtype=np.float32),
    )

    manifest = build_external_slam_artifact_manifest(
        image_dir=image_dir,
        system="pi3",
        artifact_dir=artifact_dir,
        trajectory_path="camera_poses.npz",
        pointcloud_path="points.npz",
    )
    text = render_external_slam_artifact_manifest_text(manifest)
    payload = json.loads(render_external_slam_artifact_manifest_json(manifest))

    assert manifest["type"] == "external-slam-artifact-manifest"
    assert manifest["system"] == "pi3"
    assert manifest["displayName"] == "Pi3/Pi3X"
    assert manifest["trajectory"]["materialization"] == "pose_tensor_to_tum"
    assert manifest["pointcloud"]["materialization"] == "point_tensor_to_npy"
    assert manifest["trajectory"]["poseCount"] == 2
    assert manifest["pointcloud"]["pointCount"] == 4
    assert manifest["images"]["imageCount"] == 2
    assert manifest["alignment"]["status"] == "ok"
    assert manifest["alignment"]["alignedFrameCount"] == 2
    assert manifest["trajectory"]["bytes"] > 0
    assert "External SLAM artifacts: Pi3/Pi3X (pi3)" in text
    assert "materialization=pose_tensor_to_tum" in text
    assert "2 aligned" in text
    assert payload["pointcloud"]["path"].endswith("points.npz")


def test_build_external_slam_manifest_flags_count_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir, count=3)
    artifact_dir = tmp_path / "mast3r_out"
    artifact_dir.mkdir()
    (artifact_dir / "poses.txt").write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")

    manifest = build_external_slam_artifact_manifest(
        image_dir=image_dir,
        system="mast3r-slam",
        artifact_dir=artifact_dir,
    )

    assert manifest["trajectory"]["poseCount"] == 2
    assert manifest["images"]["imageCount"] == 3
    assert manifest["alignment"]["status"] == "warning"
    assert manifest["alignment"]["alignedFrameCount"] == 2
    assert manifest["alignment"]["droppedImageCount"] == 1
    assert manifest["alignment"]["unusedPoseCount"] == 0


def test_external_slam_manifest_gate_flags_dropped_images_and_point_count(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir, count=3)
    artifact_dir = tmp_path / "pi3_out"
    artifact_dir.mkdir()
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    np.savez(artifact_dir / "camera_poses.npz", camera_poses=poses)
    np.savez(artifact_dir / "points.npz", points=np.zeros((2, 2, 3), dtype=np.float32))
    manifest = build_external_slam_artifact_manifest(
        image_dir=image_dir,
        system="pi3",
        artifact_dir=artifact_dir,
        trajectory_path="camera_poses.npz",
        pointcloud_path="points.npz",
    )

    strict_gate = evaluate_external_slam_manifest_gate(
        manifest,
        ExternalSLAMManifestGatePolicy(require_pointcloud=True, min_point_count=5),
    )
    failed = {check["name"] for check in strict_gate["checks"] if not check["passed"]}
    loose_gate = evaluate_external_slam_manifest_gate(
        manifest,
        ExternalSLAMManifestGatePolicy(
            allow_dropped_images=True,
            require_pointcloud=True,
            min_point_count=4,
        ),
    )

    assert strict_gate["passed"] is False
    assert failed == {"dropped_images", "point_count"}
    assert "External SLAM manifest gate: fail" in render_external_slam_manifest_gate_text(strict_gate)
    assert loose_gate["passed"] is True


def test_materialize_npz_camera_poses_to_tum(tmp_path: Path) -> None:
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.5
    pose_file = tmp_path / "camera_poses.npz"
    np.savez(pose_file, camera_poses=poses, timestamps=np.array([10.0, 11.0]))

    tum_path = materialize_pose_tensor_trajectory(pose_file, tmp_path / "converted")

    lines = [line for line in tum_path.read_text().splitlines() if line and not line.startswith("#")]
    assert (
        lines[0] == "10.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000"
    )
    assert lines[1].startswith("11.000000000 1.500000000 0.000000000 0.000000000")


def test_materialize_torch_nested_pi3_sequence_to_tum(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 2, 3] = 2.0
    pose_file = tmp_path / "predictions.pt"
    torch.save(
        {
            "pi3_sequence": SimpleNamespace(
                camera_poses=poses,
                timestamps=torch.tensor([3.0, 4.0]),
            )
        },
        pose_file,
    )

    tum_path = materialize_pose_tensor_trajectory(pose_file, tmp_path / "converted")

    lines = [line for line in tum_path.read_text().splitlines() if line and not line.startswith("#")]
    assert lines[0].startswith("3.000000000 0.000000000 0.000000000 0.000000000")
    assert lines[1].startswith("4.000000000 0.000000000 0.000000000 2.000000000")


def test_import_external_slam_writes_colmap_from_npz_pose_container(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir)
    artifact_dir = tmp_path / "pi3_out"
    artifact_dir.mkdir()
    poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
    poses[1, 0, 3] = 1.0
    np.savez(artifact_dir / "camera_poses.npz", camera_poses=poses)
    output_dir = tmp_path / "colmap"

    sparse_dir = import_external_slam(
        image_dir=image_dir,
        output_dir=output_dir,
        system="pi3",
        artifact_dir=artifact_dir,
    )

    assert Path(sparse_dir) == output_dir / "sparse" / "0"
    assert (output_dir / "external_slam" / "camera_poses_trajectory.tum").exists()
    images_txt = (output_dir / "sparse" / "0" / "images.txt").read_text()
    assert "frame_000000.jpg" in images_txt
    assert "frame_000001.jpg" in images_txt


def test_import_external_slam_writes_colmap_from_loger_pt_container(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir)
    artifact_dir = tmp_path / "loger_out"
    artifact_dir.mkdir()
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 1.0
    points = torch.tensor(
        [
            [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]],
            [[[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]], [[0.0, 1.0, 3.0], [1.0, 1.0, 3.0]]],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "camera_poses": poses,
            "points": points,
            "conf": torch.ones(2, 2, 2, 1),
            "images": torch.ones(2, 2, 2, 3),
        },
        artifact_dir / "predictions.pt",
    )
    output_dir = tmp_path / "colmap"

    sparse_dir = import_external_slam(
        image_dir=image_dir,
        output_dir=output_dir,
        system="loger",
        artifact_dir=artifact_dir,
    )

    assert Path(sparse_dir) == output_dir / "sparse" / "0"
    assert (output_dir / "external_slam" / "predictions_trajectory.tum").exists()
    assert (output_dir / "external_slam" / "predictions_pointcloud.npy").exists()
    points3d = (output_dir / "sparse" / "0" / "points3D.txt").read_text()
    assert "255 255 255" in points3d


def test_import_external_slam_writes_colmap_from_tum_trajectory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_dummy_images(image_dir)
    artifact_dir = tmp_path / "mast3r_slam_out"
    artifact_dir.mkdir()
    (artifact_dir / "poses.txt").write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")
    output_dir = tmp_path / "colmap"

    sparse_dir = import_external_slam(
        image_dir=image_dir,
        output_dir=output_dir,
        system="mast3r-slam",
        artifact_dir=artifact_dir,
    )

    assert Path(sparse_dir) == output_dir / "sparse" / "0"
    images_txt = (output_dir / "sparse" / "0" / "images.txt").read_text()
    assert "frame_000000.jpg" in images_txt
    assert "frame_000001.jpg" in images_txt
