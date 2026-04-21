"""Tests for external SLAM import dry-run planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys

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


def test_external_slam_import_plan_renders_json_markdown_and_shell() -> None:
    plan = build_external_slam_import_plan(
        ExternalSLAMImportPlanContext(image_dir="images", artifact_root="artifacts", output_root="imports")
    )

    payload = json.loads(render_plan_json(plan))
    markdown = render_plan_markdown(plan)
    shell = render_plan_shell(plan)

    assert payload["type"] == "external-slam-import-plan"
    assert payload["runs"][0]["system"] == "mast3r-slam"
    assert "External SLAM Import Preflight Plan" in markdown
    assert "Bag6 VGGT-SLAM 2.0" in markdown
    assert "PYTHONPATH=src python3 -m gs_sim2real.cli preprocess" in shell
    assert "--external-slam-fail-on-dry-run-gate" in shell


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
