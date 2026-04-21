"""Convert pose tensor artifacts from feed-forward geometry models to TUM text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


POSE_TENSOR_SUFFIXES = (".npy", ".npz", ".pt", ".pth")
_POSE_MATRIX_KEYS = (
    "camera_poses",
    "camera_pose",
    "poses",
    "pose",
    "pred_poses",
    "pred_pose",
    "Twc",
    "T_wc",
    "c2w",
    "cam2world",
    "camera_to_world",
)
_TIMESTAMP_KEYS = ("timestamps", "timestamp", "times", "frame_ids", "indices")
_NESTED_OUTPUT_KEYS = ("pi3_sequence", "sequence", "output", "outputs", "predictions", "result", "results")
_RECORD_SEQUENCE_KEYS = ("pred", "preds", "frames", "views")


def is_pose_tensor_artifact(path: Path) -> bool:
    return path.suffix.lower() in POSE_TENSOR_SUFFIXES


def materialize_pose_tensor_trajectory(trajectory_path: str | Path, output_dir: str | Path) -> Path:
    """Convert a pose tensor artifact into a TUM trajectory file.

    Pi3/Pi3X and LoGeR expose camera-to-world matrices in their Python outputs,
    but their demos often save ``.ply`` / ``.pt`` / ``.npz`` rather than a TUM
    file. Keep that conversion here so downstream training still consumes the
    same simple text trajectory boundary.
    """

    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    poses, timestamps = _load_pose_tensor_artifact(trajectory_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tum_path = output_dir / f"{trajectory_path.stem}_trajectory.tum"
    _write_tum_trajectory(tum_path, poses, timestamps)
    return tum_path


def _load_pose_tensor_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            poses = _extract_pose_matrices(data, role=str(path))
            timestamps = _extract_timestamps(data, len(poses))
            return poses, timestamps
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        data = data.item() if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object else data
        poses = _extract_pose_matrices(data, role=str(path))
        timestamps = _extract_timestamps(data, len(poses))
        return poses, timestamps
    if suffix in (".pt", ".pth"):
        data = _torch_load_cpu(path)
        poses = _extract_pose_matrices(data, role=str(path))
        timestamps = _extract_timestamps(data, len(poses))
        return poses, timestamps
    raise ValueError(f"Unsupported pose tensor artifact: {path}")


def _torch_load_cpu(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is normally available in this project.
        raise ImportError(f"Reading {path.suffix} pose artifacts requires PyTorch to be installed.") from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_pose_matrices(data: Any, *, role: str) -> np.ndarray:
    raw = _find_named_value(data, _POSE_MATRIX_KEYS)
    if raw is None:
        raw = _extract_pose_sequence_from_records(data)
    if raw is None:
        raw = data
    poses = _to_numpy(raw)
    poses = _normalize_pose_matrix_shape(poses, role=role)
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"Pose tensor contains non-finite values: {role}")
    return poses


def _extract_timestamps(data: Any, pose_count: int) -> np.ndarray:
    raw = _find_named_value(data, _TIMESTAMP_KEYS)
    if raw is None:
        return np.arange(pose_count, dtype=np.float64)
    timestamps = _to_numpy(raw).astype(np.float64).reshape(-1)
    if len(timestamps) != pose_count:
        raise ValueError(f"Timestamp count {len(timestamps)} does not match pose count {pose_count}")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Pose tensor timestamps contain non-finite values")
    return timestamps


def _find_named_value(
    data: Any, keys: tuple[str, ...], *, _depth: int = 0, _seen: set[int] | None = None
) -> Any | None:
    if _seen is None:
        _seen = set()
    if _depth > 3 or _is_array_like(data):
        return None
    obj_id = id(data)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    direct = _find_named_value_shallow(data, keys)
    if direct is not None:
        return direct

    for child in _iter_named_children(data):
        nested = _find_named_value(child, keys, _depth=_depth + 1, _seen=_seen)
        if nested is not None:
            return nested
    return None


def _find_named_value_shallow(data: Any, keys: tuple[str, ...]) -> Any | None:
    if hasattr(data, "files"):
        for key in keys:
            if key in data.files:
                return data[key]
        return None
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        return None
    for key in keys:
        if hasattr(data, key):
            return getattr(data, key)
    return None


def _iter_named_children(data: Any) -> list[Any]:
    children: list[Any] = []
    if isinstance(data, dict):
        children.extend(data[key] for key in _NESTED_OUTPUT_KEYS if key in data)
    else:
        for key in _NESTED_OUTPUT_KEYS:
            if hasattr(data, key):
                children.append(getattr(data, key))
    return children


def _extract_pose_sequence_from_records(data: Any) -> Any | None:
    records = None
    if isinstance(data, dict):
        for key in _RECORD_SEQUENCE_KEYS:
            value = data.get(key)
            if isinstance(value, list | tuple):
                records = value
                break
    elif isinstance(data, list | tuple):
        records = data
    if not records:
        return None

    matrices = []
    for record in records:
        raw = _find_named_value_shallow(record, ("camera_pose", "pose", "Twc", "T_wc", "c2w"))
        if raw is None:
            continue
        pose = _normalize_pose_matrix_shape(_to_numpy(raw), role="record pose")
        if len(pose) != 1:
            return None
        matrices.append(pose[0])
    if not matrices:
        return None
    return np.stack(matrices, axis=0)


def _is_array_like(value: Any) -> bool:
    return isinstance(value, np.ndarray) or hasattr(value, "detach")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_pose_matrix_shape(poses: np.ndarray, *, role: str) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    while poses.ndim > 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim == 4 and poses.shape[1] == 1 and poses.shape[2:] in ((4, 4), (3, 4)):
        poses = poses[:, 0]
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, ...]
    if poses.ndim != 3:
        raise ValueError(f"Expected pose tensor with shape (N, 4, 4) or (N, 3, 4), got {poses.shape}: {role}")
    if poses.shape[1:] == (3, 4):
        bottom = np.zeros((poses.shape[0], 1, 4), dtype=np.float64)
        bottom[:, 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=1)
    if poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose tensor with shape (N, 4, 4) or (N, 3, 4), got {poses.shape}: {role}")
    return poses


def _write_tum_trajectory(path: Path, poses: np.ndarray, timestamps: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, pose in zip(timestamps, poses):
            qw, qx, qy, qz = _rotation_to_quaternion(pose[:3, :3])
            tx, ty, tz = pose[:3, 3]
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[2, 1] - rotation[1, 2]) * scale
        qy = (rotation[0, 2] - rotation[2, 0]) * scale
        qz = (rotation[1, 0] - rotation[0, 1]) * scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("Rotation matrix produced a zero-length quaternion")
    return qw / norm, qx / norm, qy / norm, qz / norm
