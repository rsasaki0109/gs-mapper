"""Collect external SLAM import dry-run manifest summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gs_sim2real.experiments.external_slam_import_plan import ExternalSLAMImportPlan, ExternalSLAMImportRunPlan


def collect_external_slam_import_preflight_results(plan: ExternalSLAMImportPlan) -> dict[str, Any]:
    """Collect saved dry-run manifests for an external SLAM import plan."""

    run_reports = [collect_external_slam_import_preflight_run_result(run) for run in plan.runs]
    manifest_count = sum(1 for run in run_reports if run["manifestExists"])
    passed_count = sum(1 for run in run_reports if run["gatePassed"] is True)
    failed_count = sum(1 for run in run_reports if run["gatePassed"] is False)
    invalid_count = sum(1 for run in run_reports if run["manifestExists"] and not run["manifestValid"])
    return {
        "type": "external-slam-import-preflight-report",
        "context": plan.context.__dict__,
        "runCount": len(run_reports),
        "manifestCount": manifest_count,
        "passedCount": passed_count,
        "failedCount": failed_count,
        "missingCount": len(run_reports) - manifest_count,
        "invalidCount": invalid_count,
        "runs": run_reports,
    }


def collect_external_slam_import_preflight_run_result(run: ExternalSLAMImportRunPlan) -> dict[str, Any]:
    """Collect one saved external SLAM dry-run manifest."""

    manifest_path = Path(run.manifest_path)
    manifest, manifest_error = _read_manifest_json(manifest_path)
    manifest_exists = manifest_path.is_file()
    manifest_valid = manifest is not None
    images = _nested_dict(manifest, "images")
    trajectory = _nested_dict(manifest, "trajectory")
    alignment = _nested_dict(manifest, "alignment")
    pointcloud = _nested_dict(manifest, "pointcloud")
    gate = _nested_dict(manifest, "gate")
    missing = _missing_manifest_fields(manifest_exists, manifest_valid, gate)

    return {
        "name": run.profile.name,
        "label": run.profile.label,
        "system": run.profile.system,
        "artifactDir": run.artifact_dir,
        "outputDir": run.output_dir,
        "manifestPath": run.manifest_path,
        "manifestExists": manifest_exists,
        "manifestValid": manifest_valid,
        "manifestError": manifest_error,
        "gatePassed": _optional_bool(gate.get("passed")),
        "imageCount": _optional_int(images.get("imageCount")),
        "poseCount": _optional_int(trajectory.get("poseCount")),
        "alignedFrameCount": _optional_int(alignment.get("alignedFrameCount")),
        "droppedImageCount": _optional_int(alignment.get("droppedImageCount")),
        "unusedPoseCount": _optional_int(alignment.get("unusedPoseCount")),
        "pointCount": _optional_int(pointcloud.get("pointCount")),
        "missing": missing,
    }


def render_external_slam_import_report_json(report: dict[str, Any]) -> str:
    """Render collected external SLAM preflight results as stable JSON."""

    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_external_slam_import_report_markdown(report: dict[str, Any]) -> str:
    """Render collected external SLAM preflight results as a compact table."""

    lines = [
        "# External SLAM Import Preflight Results",
        "",
        (
            f"Runs: {report['passedCount']}/{report['runCount']} gates passed, "
            f"{report['manifestCount']}/{report['runCount']} manifests present"
        ),
        "",
        "| Run | System | Gate | Images | Poses | Aligned | Dropped | Unused | Points | Missing |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        missing = ", ".join(run["missing"]) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    run["system"],
                    _format_gate_status(run),
                    _format_optional_int(run["imageCount"]),
                    _format_optional_int(run["poseCount"]),
                    _format_optional_int(run["alignedFrameCount"]),
                    _format_optional_int(run["droppedImageCount"]),
                    _format_optional_int(run["unusedPoseCount"]),
                    _format_optional_int(run["pointCount"]),
                    missing,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _read_manifest_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(data, dict):
        return None, "manifest JSON root is not an object"
    return data, None


def _nested_dict(manifest: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if manifest is None:
        return {}
    value = manifest.get(key)
    return value if isinstance(value, dict) else {}


def _missing_manifest_fields(manifest_exists: bool, manifest_valid: bool, gate: dict[str, Any]) -> list[str]:
    if not manifest_exists:
        return ["manifest"]
    if not manifest_valid:
        return ["manifest_json"]
    if not gate:
        return ["gate"]
    return []


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _format_gate_status(run: dict[str, Any]) -> str:
    if not run["manifestExists"]:
        return "n/a"
    if not run["manifestValid"]:
        return "invalid"
    if run["gatePassed"] is True:
        return "pass"
    if run["gatePassed"] is False:
        return "fail"
    return "n/a"


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


__all__ = [
    "collect_external_slam_import_preflight_results",
    "collect_external_slam_import_preflight_run_result",
    "render_external_slam_import_report_json",
    "render_external_slam_import_report_markdown",
]
