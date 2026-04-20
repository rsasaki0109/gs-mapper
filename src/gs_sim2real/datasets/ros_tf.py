"""ROS TF helpers: static transform graphs and ``geometry_msgs`` conversion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalize_frame_id(frame_id: str) -> str:
    """Strip whitespace and a leading slash (ROS sometimes uses ``/map`` style)."""
    return frame_id.strip().lstrip("/")


def geometry_transform_to_matrix(transform: Any) -> np.ndarray:
    """Convert ``geometry_msgs/Transform`` to a 4x4.

    ROS convention: ``p_parent = R @ p_child + t`` (same as ``T @ p_child`` in homogeneous coords).
    """
    qx = float(transform.rotation.x)
    qy = float(transform.rotation.y)
    qz = float(transform.rotation.z)
    qw = float(transform.rotation.w)
    R = quat_wxyz_to_rotation_matrix(qw, qx, qy, qz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = float(transform.translation.x)
    T[1, 3] = float(transform.translation.y)
    T[2, 3] = float(transform.translation.z)
    return T


def quat_wxyz_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


class StaticTfMap:
    """Directed tree of static transforms: ``p_parent = T @ p_child``."""

    def __init__(self) -> None:
        self._child_to_edge: dict[str, tuple[str, np.ndarray]] = {}

    def get_parent_and_transform(self, child: str) -> tuple[str, np.ndarray] | None:
        """Return ``(parent, T_parent_from_child)`` for a child frame, if registered."""
        child = normalize_frame_id(child)
        edge = self._child_to_edge.get(child)
        if edge is None:
            return None
        return edge[0], edge[1]

    def add(self, parent: str, child: str, T_parent_from_child: np.ndarray) -> None:
        """Register one edge (latest message wins if duplicate child)."""
        self._child_to_edge[normalize_frame_id(child)] = (
            normalize_frame_id(parent),
            T_parent_from_child.astype(np.float64),
        )

    def lookup(self, target_parent: str, child: str) -> np.ndarray | None:
        """Return ``T_target_parent_from_child`` such that ``p_target = T @ p_child``."""
        target_parent = normalize_frame_id(target_parent)
        child = normalize_frame_id(child)
        if child == target_parent:
            return np.eye(4, dtype=np.float64)

        cur = child
        T_accum = np.eye(4, dtype=np.float64)
        visited: set[str] = set()

        while cur != target_parent:
            if cur in visited:
                logger.warning("TF cycle detected at frame %s", cur)
                return None
            visited.add(cur)

            edge = self._child_to_edge.get(cur)
            if edge is None:
                return None
            parent, T_parent_from_child = edge
            T_accum = T_parent_from_child @ T_accum
            cur = parent

        return T_accum

    def __len__(self) -> int:
        return len(self._child_to_edge)


class TimestampedTfEdges:
    """Time-stamped ``(parent, child)`` transforms from ``/tf`` (nearest-neighbor lookup)."""

    def __init__(self) -> None:
        self._by_edge: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}

    def add(self, stamp_ns: int, parent: str, child: str, T_parent_from_child: np.ndarray) -> None:
        p = normalize_frame_id(parent)
        c = normalize_frame_id(child)
        key = (p, c)
        self._by_edge.setdefault(key, []).append((int(stamp_ns), T_parent_from_child.astype(np.float64)))

    def finalize(self) -> None:
        """Sort each edge list by stamp for binary search."""
        for key in self._by_edge:
            self._by_edge[key].sort(key=lambda x: x[0])

    def nearest(self, parent: str, child: str, stamp_ns: int, max_delta_ns: int = 500_000_000) -> np.ndarray | None:
        """Return ``T`` at the closest stamp; None if missing or farther than ``max_delta_ns``."""
        key = (normalize_frame_id(parent), normalize_frame_id(child))
        lst = self._by_edge.get(key)
        if not lst:
            return None
        stamps = np.array([x[0] for x in lst], dtype=np.int64)
        idx = int(np.argmin(np.abs(stamps - stamp_ns)))
        if abs(int(stamps[idx]) - stamp_ns) > max_delta_ns:
            return None
        return lst[idx][1]

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_edge.values())


class HybridTfLookup:
    """Walk a :class:`StaticTfMap` topology; override each edge with ``/tf`` samples when available."""

    def __init__(self, static_map: StaticTfMap, dynamic_edges: TimestampedTfEdges | None):
        self.static = static_map
        self.dynamic = dynamic_edges

    def lookup(self, target_parent: str, child: str, stamp_ns: int) -> np.ndarray | None:
        """``T_target_parent_from_child`` at time ``stamp_ns`` (nanoseconds)."""
        target_parent = normalize_frame_id(target_parent)
        child = normalize_frame_id(child)
        if child == target_parent:
            return np.eye(4, dtype=np.float64)

        cur = child
        T_accum = np.eye(4, dtype=np.float64)
        visited: set[str] = set()

        while cur != target_parent:
            if cur in visited:
                logger.warning("TF cycle detected at frame %s", cur)
                return None
            visited.add(cur)

            edge = self.static.get_parent_and_transform(cur)
            if edge is None:
                return None
            parent, T_static = edge
            T_use = T_static
            if self.dynamic is not None:
                T_dyn = self.dynamic.nearest(parent, cur, stamp_ns)
                if T_dyn is not None:
                    T_use = T_dyn
            T_accum = T_use @ T_accum
            cur = parent

        return T_accum


def load_static_calibration_yaml(
    path: str | Path,
    *,
    base_frame: str = "body",
) -> StaticTfMap:
    """Load an MCDVIRAL-style sensor calibration YAML into a :class:`StaticTfMap`.

    MCDVIRAL distributes its handheld / ATV calibration as ``body: { <sensor>:
    { T: [[…4×4…]], intrinsics, distortion_*, rostopic, timeshift_cam_imu } }``
    where ``T`` is the 4×4 transform that takes a point expressed in the sensor
    frame to the body frame (``p_body = T @ p_sensor``). Each ``<sensor>`` entry
    becomes one edge in the returned tree with the sensor as the child and
    ``base_frame`` as the parent. The caller can then pass that tree anywhere a
    bag-derived ``StaticTfMap`` would normally plug in (e.g. MCD preprocess).

    The ``base_frame`` argument lets you re-label the YAML's parent key so the
    returned map matches whatever frame name the caller is already using
    downstream (MCD preprocess defaults to ``base_link``).

    Sensors whose ``T`` cannot be parsed as a 4×4 float matrix are skipped with
    a warning rather than failing the whole load.
    """
    import yaml

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        raise ValueError(f"calibration YAML {p} has no top-level mapping")
    body = doc.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"calibration YAML {p} has no 'body:' mapping (expected MCDVIRAL handheld/ATV format)")

    tf_map = StaticTfMap()
    for sensor_name, entry in body.items():
        if not isinstance(entry, dict):
            continue
        T_raw = entry.get("T")
        if T_raw is None:
            continue
        try:
            T = np.asarray(T_raw, dtype=np.float64)
        except (TypeError, ValueError):
            logger.warning("calibration YAML: sensor %r has non-numeric T; skipping", sensor_name)
            continue
        if T.shape != (4, 4):
            logger.warning(
                "calibration YAML: sensor %r T must be 4×4, got %s; skipping",
                sensor_name,
                T.shape,
            )
            continue
        tf_map.add(base_frame, str(sensor_name), T)

    return tf_map


def merge_static_tf_maps(*maps: StaticTfMap) -> StaticTfMap:
    """Combine several :class:`StaticTfMap`\\ s into one; later maps win on child-key collisions."""
    out = StaticTfMap()
    for m in maps:
        if m is None:
            continue
        for child, (parent, T) in m._child_to_edge.items():  # noqa: SLF001 — internal merge
            out.add(parent, child, T)
    return out
