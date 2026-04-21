"""Quality gates for external SLAM dry-run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalSLAMManifestGatePolicy:
    """Thresholds for deciding whether a dry-run manifest is safe to import."""

    min_aligned_frames: int = 2
    allow_dropped_images: bool = False
    allow_unused_poses: bool = True
    require_pointcloud: bool = False
    min_point_count: int = 0
    require_known_alignment: bool = True


def evaluate_external_slam_manifest_gate(
    manifest: dict[str, Any],
    policy: ExternalSLAMManifestGatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate a resolved external SLAM manifest against an import gate."""

    active_policy = policy or ExternalSLAMManifestGatePolicy()
    checks: list[dict[str, Any]] = []
    alignment = manifest.get("alignment") or {}
    pointcloud = manifest.get("pointcloud")

    if active_policy.require_known_alignment:
        _append_check(
            checks,
            "alignment_known",
            alignment.get("status") != "unknown",
            observed=alignment.get("status"),
            expected="known",
        )
    _append_min_check(
        checks,
        "aligned_frames",
        alignment.get("alignedFrameCount"),
        active_policy.min_aligned_frames,
    )
    if not active_policy.allow_dropped_images:
        _append_check(
            checks,
            "dropped_images",
            alignment.get("droppedImageCount") == 0,
            observed=alignment.get("droppedImageCount"),
            expected=0,
        )
    if not active_policy.allow_unused_poses:
        _append_check(
            checks,
            "unused_poses",
            alignment.get("unusedPoseCount") == 0,
            observed=alignment.get("unusedPoseCount"),
            expected=0,
        )
    if active_policy.require_pointcloud:
        _append_check(
            checks,
            "pointcloud",
            pointcloud is not None,
            observed="present" if pointcloud is not None else "missing",
            expected="present",
        )
    if active_policy.min_point_count > 0:
        _append_min_check(
            checks,
            "point_count",
            pointcloud.get("pointCount") if pointcloud else None,
            active_policy.min_point_count,
        )

    return {
        "type": "external-slam-manifest-gate-report",
        "policy": asdict(active_policy),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def render_external_slam_manifest_gate_text(gate_report: dict[str, Any]) -> str:
    """Render a dry-run gate report as compact CLI text."""

    failed = [check["name"] for check in gate_report["checks"] if not check["passed"]]
    lines = [
        f"External SLAM manifest gate: {'pass' if gate_report['passed'] else 'fail'}",
        f"- failed checks: {', '.join(failed) if failed else 'none'}",
    ]
    return "\n".join(lines) + "\n"


def _append_min_check(checks: list[dict[str, Any]], name: str, actual: Any, minimum: int) -> None:
    _append_check(
        checks,
        name,
        actual is not None and int(actual) >= minimum,
        observed=actual,
        expected=f">= {minimum}",
    )


def _append_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


__all__ = [
    "ExternalSLAMManifestGatePolicy",
    "evaluate_external_slam_manifest_gate",
    "render_external_slam_manifest_gate_text",
]
