"""Artifact-only integration boundary for external visual SLAM front-ends."""

from __future__ import annotations

from gs_sim2real.preprocess.external_slam_artifacts.importer import import_external_slam
from gs_sim2real.preprocess.external_slam_artifacts.pose_tensor import materialize_pose_tensor_trajectory
from gs_sim2real.preprocess.external_slam_artifacts.profiles import (
    ALIASES,
    PROFILES,
    SYSTEM_CHOICES,
    ExternalSLAMProfile,
    normalize_system,
)
from gs_sim2real.preprocess.external_slam_artifacts.resolver import (
    ExternalSLAMArtifacts,
    resolve_external_slam_artifacts,
)

__all__ = [
    "ALIASES",
    "PROFILES",
    "SYSTEM_CHOICES",
    "ExternalSLAMArtifacts",
    "ExternalSLAMProfile",
    "import_external_slam",
    "materialize_pose_tensor_trajectory",
    "normalize_system",
    "resolve_external_slam_artifacts",
]
