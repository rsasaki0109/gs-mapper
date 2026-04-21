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
    ExternalSLAMManifestGatePolicy,
    ExternalSLAMProfile,
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

__all__ = [
    "ALIASES",
    "PROFILES",
    "SYSTEM_CHOICES",
    "ExternalSLAMArtifacts",
    "ExternalSLAMManifestGatePolicy",
    "ExternalSLAMProfile",
    "build_external_slam_artifact_manifest",
    "evaluate_external_slam_manifest_gate",
    "import_external_slam",
    "materialize_pose_tensor_trajectory",
    "normalize_system",
    "render_external_slam_artifact_manifest_json",
    "render_external_slam_artifact_manifest_text",
    "render_external_slam_manifest_gate_text",
    "resolve_external_slam_artifacts",
]
