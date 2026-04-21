"""Tests for reproducible MCD quality-run planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from gs_sim2real.experiments.mcd_quality_plan import (
    MCDQualityPlanContext,
    build_mcd_quality_plan,
    render_plan_json,
    render_plan_markdown,
    render_plan_shell,
)


def test_mcd_quality_plan_contains_baseline_and_quality_candidates() -> None:
    plan = build_mcd_quality_plan()

    names = [run.profile.name for run in plan.runs]

    assert names == [
        "ntu_day02_single_400_depth_long",
        "ntu_day02_single_800_ba",
        "ntu_day02_multi_3cam_300each_ba",
    ]
    assert plan.preflight_command[:3] == ("python3", "scripts/check_mcd_gnss.py", "data/mcd/ntu_day_02")
    assert "--flatten-altitude" in plan.preflight_command
    assert "--start-offset-sec" in plan.preflight_command


def test_single_camera_command_reproduces_known_good_trimmed_baseline() -> None:
    plan = build_mcd_quality_plan()
    baseline = plan.runs[0]

    command = list(baseline.preprocess_command)

    assert command[command.index("--image-topic") + 1] == "/d455b/color/image_raw"
    assert command[command.index("--mcd-camera-frame") + 1] == "d455b_color"
    assert command[command.index("--max-frames") + 1] == "400"
    assert command[command.index("--every-n") + 1] == "14"
    assert command[command.index("--mcd-start-offset-sec") + 1] == "35"
    assert baseline.train_command[baseline.train_command.index("--config") + 1] == "configs/training_depth_long.yaml"
    assert baseline.export_command[baseline.export_command.index("--max-points") + 1] == "400000"


def test_multicamera_profile_uses_topic_list_and_yaml_frames() -> None:
    plan = build_mcd_quality_plan()
    multi = plan.runs[2]
    command = list(multi.preprocess_command)

    assert multi.profile.requires_full_folder is True
    assert command[command.index("--image-topic") + 1] == (
        "/d455b/color/image_raw,/d455t/color/image_raw,/d435i/color/image_raw"
    )
    assert "--mcd-camera-frame" not in command
    assert multi.train_command[multi.train_command.index("--config") + 1] == "configs/training_ba.yaml"


def test_mcd_quality_plan_renders_json_markdown_and_shell() -> None:
    plan = build_mcd_quality_plan(
        MCDQualityPlanContext(output_root="outputs/q", asset_dir="outputs/q/assets", pythonpath="src")
    )

    payload = json.loads(render_plan_json(plan))
    markdown = render_plan_markdown(plan)
    shell = render_plan_shell(plan)

    assert payload["runs"][0]["preprocessDir"] == "outputs/q/ntu_day02_single_400_depth_long/preprocess"
    assert "MCD Quality Run Plan" in markdown
    assert "Single D455B 800 BA" in markdown
    assert "Three-Camera 300 Each BA" in markdown
    assert "PYTHONPATH=src python3 -m gs_sim2real.cli preprocess" in shell
    assert "outputs/q/assets/ntu_day02_single_400_depth_long.splat" in shell


def test_plan_mcd_quality_runs_script_can_emit_single_profile_json() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_mcd_quality_runs.py",
            "--format",
            "json",
            "--profile",
            "ntu_day02_single_800_ba",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert [run["name"] for run in payload["runs"]] == ["ntu_day02_single_800_ba"]
    assert payload["runs"][0]["iterations"] == 50000
