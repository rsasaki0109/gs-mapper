"""Tests for external SLAM import dry-run planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from gs_sim2real.experiments.external_slam_import_collect import (
    collect_external_slam_import_preflight_results,
    render_external_slam_import_report_json,
    render_external_slam_import_report_markdown,
)
from gs_sim2real.experiments.external_slam_import_plan import (
    ExternalSLAMImportPlanContext,
    build_external_slam_import_plan,
    render_plan_json,
    render_plan_markdown,
    render_plan_shell,
)


def test_external_slam_import_plan_contains_all_default_frontends() -> None:
    plan = build_external_slam_import_plan(
        ExternalSLAMImportPlanContext(
            image_dir="data/images",
            artifact_root="outputs/slam",
            output_root="outputs/imports",
            min_aligned_frames=3,
            min_point_count=10,
        )
    )

    names = [run.profile.name for run in plan.runs]
    commands = [list(run.dry_run_command) for run in plan.runs]

    assert names == ["bag6_mast3r_slam", "bag6_vggt_slam_2", "bag6_pi3", "bag6_loger"]
    assert [run.profile.system for run in plan.runs] == ["mast3r-slam", "vggt-slam", "pi3", "loger"]
    for command in commands:
        assert "--external-slam-dry-run" in command
        assert "--external-slam-fail-on-dry-run-gate" in command
        assert command[command.index("--images") + 1] == "data/images"
        assert command[command.index("--external-slam-min-aligned-frames") + 1] == "3"
        assert command[command.index("--external-slam-min-point-count") + 1] == "10"
        assert "--external-slam-require-pointcloud" in command
    assert commands[0][commands[0].index("--external-slam-output") + 1] == "outputs/slam/mast3r-slam"
    assert commands[1][commands[1].index("--external-slam-output") + 1] == "outputs/slam/vggt-slam"
    assert plan.runs[0].manifest_path == "outputs/imports/bag6_mast3r_slam/manifest.json"


def test_external_slam_import_plan_renders_json_markdown_and_shell() -> None:
    plan = build_external_slam_import_plan(
        ExternalSLAMImportPlanContext(image_dir="images", artifact_root="artifacts", output_root="imports")
    )

    payload = json.loads(render_plan_json(plan))
    markdown = render_plan_markdown(plan)
    shell = render_plan_shell(plan)

    assert payload["type"] == "external-slam-import-plan"
    assert payload["runs"][0]["system"] == "mast3r-slam"
    assert payload["runs"][0]["manifestPath"] == "imports/bag6_mast3r_slam/manifest.json"
    assert "External SLAM Import Preflight Plan" in markdown
    assert "Bag6 VGGT-SLAM 2.0" in markdown
    assert "`imports/bag6_mast3r_slam/manifest.json`" in markdown
    assert "PYTHONPATH=src python3 -m gs_sim2real.cli preprocess" in shell
    assert "--external-slam-fail-on-dry-run-gate" in shell
    assert "mkdir -p imports/bag6_mast3r_slam" in shell
    assert "> imports/bag6_mast3r_slam/manifest.json || status=$?" in shell
    assert 'exit "$status"' in shell


def test_plan_external_slam_imports_script_can_emit_single_profile_json() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_external_slam_imports.py",
            "--format",
            "json",
            "--profile",
            "bag6_pi3",
            "--images",
            "data/pi3/images",
            "--artifact-root",
            "outputs/pi3_candidates",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert [run["name"] for run in payload["runs"]] == ["bag6_pi3"]
    command = payload["runs"][0]["dryRunCommand"]
    assert command[command.index("--external-slam-system") + 1] == "pi3"
    assert command[command.index("--external-slam-output") + 1] == "outputs/pi3_candidates/pi3"
    assert payload["runs"][0]["manifestPath"] == "outputs/external_slam_imports/bag6_pi3/manifest.json"


def test_collect_external_slam_import_preflight_results_reads_saved_manifests(tmp_path) -> None:
    plan = build_external_slam_import_plan(ExternalSLAMImportPlanContext(output_root=str(tmp_path / "imports")))
    manifest_path = Path(plan.runs[0].manifest_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "type": "external-slam-artifact-manifest",
                "system": "mast3r-slam",
                "images": {
                    "path": "/tmp/images",
                    "exists": True,
                    "isDirectory": True,
                    "imageCount": 3,
                },
                "trajectory": {"poseCount": 3},
                "resolution": {
                    "trajectory": {
                        "selectedPath": "/tmp/artifacts/mast3r/camera_poses.npz",
                        "reason": "selected_candidate",
                    },
                    "pointcloud": {
                        "selectedPath": "/tmp/artifacts/mast3r/pointcloud.npz",
                        "reason": "selected_candidate",
                    },
                },
                "alignment": {
                    "alignedFrameCount": 3,
                    "droppedImageCount": 0,
                    "unusedPoseCount": 0,
                },
                "pointcloud": {"pointCount": 123},
                "gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    report = collect_external_slam_import_preflight_results(plan)
    markdown = render_external_slam_import_report_markdown(report)
    payload = json.loads(render_external_slam_import_report_json(report))
    first = report["runs"][0]

    assert report["passedCount"] == 1
    assert report["missingCount"] == 3
    assert first["gatePassed"] is True
    assert first["imagePath"] == "/tmp/images"
    assert first["imageExists"] is True
    assert first["imageIsDirectory"] is True
    assert first["imageStatus"] == "ok"
    assert first["imageCount"] == 3
    assert first["poseCount"] == 3
    assert first["alignedFrameCount"] == 3
    assert first["pointCount"] == 123
    assert first["trajectorySelectedPath"] == "/tmp/artifacts/mast3r/camera_poses.npz"
    assert first["pointcloudSelectedPath"] == "/tmp/artifacts/mast3r/pointcloud.npz"
    assert first["trajectoryResolutionReason"] == "selected_candidate"
    assert first["missing"] == []
    assert "# External SLAM Import Preflight Results" in markdown
    assert "1/4 gates passed" in markdown
    assert "| Bag6 MASt3R-SLAM | mast3r-slam | pass | ok | 3 |" in markdown
    assert "camera_poses.npz" in markdown
    assert "pointcloud.npz" in markdown
    assert "/tmp/artifacts/mast3r/camera_poses.npz" not in markdown
    assert payload["type"] == "external-slam-import-preflight-report"
    assert payload["runs"][0]["trajectorySelectedPath"] == "/tmp/artifacts/mast3r/camera_poses.npz"


def test_collect_external_slam_import_preflight_results_reads_error_manifests(tmp_path) -> None:
    plan = build_external_slam_import_plan(ExternalSLAMImportPlanContext(output_root=str(tmp_path / "imports")))
    manifest_path = Path(plan.runs[0].manifest_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "type": "external-slam-artifact-manifest",
                "system": "mast3r-slam",
                "displayName": "MASt3R-SLAM",
                "images": {
                    "path": "data/outdoor/bag6/images",
                    "exists": False,
                    "isDirectory": False,
                    "imageCount": None,
                },
                "resolution": {
                    "trajectory": {
                        "selectedPath": None,
                        "reason": "no_candidate_match",
                    },
                    "pointcloud": {
                        "selectedPath": None,
                        "reason": "no_candidate_match",
                    },
                },
                "trajectory": None,
                "pointcloud": None,
                "alignment": {"status": "unknown"},
                "ready": False,
                "error": {
                    "type": "FileNotFoundError",
                    "message": "Could not find MASt3R-SLAM trajectory",
                },
            }
        ),
        encoding="utf-8",
    )

    report = collect_external_slam_import_preflight_results(plan)
    markdown = render_external_slam_import_report_markdown(report)
    first = report["runs"][0]

    assert report["manifestCount"] == 1
    assert report["passedCount"] == 0
    assert report["errorCount"] == 1
    assert report["imageMissingCount"] == 1
    assert first["ready"] is False
    assert first["imagePath"] == "data/outdoor/bag6/images"
    assert first["imageStatus"] == "missing"
    assert first["errorType"] == "FileNotFoundError"
    assert first["errorMessage"] == "Could not find MASt3R-SLAM trajectory"
    assert first["trajectoryResolutionReason"] == "no_candidate_match"
    assert first["missing"] == ["error"]
    assert "| Bag6 MASt3R-SLAM | mast3r-slam | error | missing | n/a |" in markdown
    assert "no_candidate_match" in markdown
    assert "Could not find MASt3R-SLAM trajectory" in markdown


def test_collect_external_slam_imports_script_can_emit_missing_json(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_external_slam_imports.py",
            "--format",
            "json",
            "--profile",
            "bag6_pi3",
            "--output-root",
            str(tmp_path / "missing"),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["type"] == "external-slam-import-preflight-report"
    assert payload["missingCount"] == 1
    assert payload["runs"][0]["name"] == "bag6_pi3"
    assert payload["runs"][0]["missing"] == ["manifest"]
