"""Compatibility facade for external visual SLAM artifact imports.

The implementation is split under ``external_slam_artifacts`` so adding one
more front-end convention does not mix path discovery, pose tensor decoding, and
COLMAP import orchestration in a single module. Keep this facade stable for
existing callers and tests.
"""

from __future__ import annotations

from gs_sim2real.preprocess.external_slam_artifacts import (
    ALIASES,
    PROFILES,
    SYSTEM_CHOICES,
    ExternalSLAMArtifacts,
    ExternalSLAMProfile,
    import_external_slam,
    materialize_pose_tensor_trajectory,
    normalize_system,
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
