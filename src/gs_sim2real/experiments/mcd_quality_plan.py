"""Quality-run planning helpers for supervised MCD sessions."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MCDQualityPlanContext:
    """Paths and common sensor settings for MCD quality runs."""

    session_dir: str = "data/mcd/ntu_day_02"
    output_root: str = "outputs/mcd_quality"
    calibration_path: str = "data/mcd/calibration_atv.yaml"
    asset_dir: str = "outputs/mcd_quality/assets"
    python_executable: str = "python3"
    pythonpath: str = "src"
    gnss_topic: str = "/vn200/GPS"
    imu_topic: str = "/vn200/imu"
    lidar_topic: str = "/os_cloud_node/points"
    lidar_frame: str = "os_sensor"
    start_offset_sec: float = 35.0


@dataclass(frozen=True)
class MCDQualityRunProfile:
    """One supervised MCD quality candidate."""

    name: str
    label: str
    intent: str
    image_topics: tuple[str, ...]
    max_frames: int
    every_n: int
    iterations: int
    config_path: str
    camera_frame: str | None = None
    include_dynamic_tf: bool = False
    requires_full_folder: bool = False
    export_max_points: int = 400_000
    splat_normalize_extent: float = 17.0
    splat_min_opacity: float = 0.02
    splat_max_scale: float = 2.0


@dataclass(frozen=True)
class MCDQualityRunPlan:
    """Concrete commands and expected artifacts for one quality run."""

    profile: MCDQualityRunProfile
    preprocess_dir: str
    train_dir: str
    export_path: str
    preprocess_command: tuple[str, ...]
    train_command: tuple[str, ...]
    export_command: tuple[str, ...]
    expected_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class MCDQualityPlan:
    """A reproducible quality-run matrix."""

    context: MCDQualityPlanContext
    preflight_command: tuple[str, ...]
    runs: tuple[MCDQualityRunPlan, ...]


def default_mcd_quality_profiles() -> tuple[MCDQualityRunProfile, ...]:
    """Return the default ntu_day_02 quality-push matrix."""
    return (
        MCDQualityRunProfile(
            name="ntu_day02_single_400_depth_long",
            label="Single D455B 400 Depth Long",
            intent="Reproduce the production MCD supervised baseline before changing quality knobs.",
            image_topics=("/d455b/color/image_raw",),
            camera_frame="d455b_color",
            max_frames=400,
            every_n=14,
            iterations=30_000,
            config_path="configs/training_depth_long.yaml",
        ),
        MCDQualityRunProfile(
            name="ntu_day02_single_800_ba",
            label="Single D455B 800 BA",
            intent="Double temporal coverage and enable depth + appearance + pose refinement on the known-good camera.",
            image_topics=("/d455b/color/image_raw",),
            camera_frame="d455b_color",
            max_frames=800,
            every_n=7,
            iterations=50_000,
            config_path="configs/training_ba.yaml",
        ),
        MCDQualityRunProfile(
            name="ntu_day02_multi_3cam_300each_ba",
            label="Three-Camera 300 Each BA",
            intent="Use the official ATV calibration to compare multi-camera coverage against the single-camera baseline.",
            image_topics=(
                "/d455b/color/image_raw",
                "/d455t/color/image_raw",
                "/d435i/color/image_raw",
            ),
            max_frames=300,
            every_n=14,
            iterations=50_000,
            config_path="configs/training_ba.yaml",
            requires_full_folder=True,
        ),
    )


def build_mcd_quality_plan(
    context: MCDQualityPlanContext | None = None,
    *,
    profiles: Iterable[MCDQualityRunProfile] | None = None,
) -> MCDQualityPlan:
    """Build concrete commands for an MCD quality-run matrix."""
    ctx = context or MCDQualityPlanContext()
    selected_profiles = tuple(profiles or default_mcd_quality_profiles())
    preflight_command = (
        ctx.python_executable,
        "scripts/check_mcd_gnss.py",
        ctx.session_dir,
        "--gnss-topic",
        ctx.gnss_topic,
        "--flatten-altitude",
        "--start-offset-sec",
        _format_number(ctx.start_offset_sec),
    )
    runs = tuple(_build_run_plan(ctx, profile) for profile in selected_profiles)
    return MCDQualityPlan(context=ctx, preflight_command=preflight_command, runs=runs)


def _build_run_plan(ctx: MCDQualityPlanContext, profile: MCDQualityRunProfile) -> MCDQualityRunPlan:
    run_root = Path(ctx.output_root) / profile.name
    preprocess_dir = run_root / "preprocess"
    train_dir = run_root / "train"
    export_path = Path(ctx.asset_dir) / f"{profile.name}.splat"

    preprocess_parts: list[str] = [
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "preprocess",
        "--images",
        ctx.session_dir,
        "--output",
        str(preprocess_dir),
        "--method",
        "mcd",
        "--image-topic",
        ",".join(profile.image_topics),
        "--gnss-topic",
        ctx.gnss_topic,
        "--mcd-static-calibration",
        ctx.calibration_path,
        "--mcd-seed-poses-from-gnss",
        "--mcd-flatten-gnss-altitude",
        "--mcd-start-offset-sec",
        _format_number(ctx.start_offset_sec),
        "--mcd-tf-use-image-stamps",
        "--lidar-topic",
        ctx.lidar_topic,
        "--mcd-lidar-frame",
        ctx.lidar_frame,
        "--imu-topic",
        ctx.imu_topic,
        "--extract-lidar",
        "--extract-imu",
        "--mcd-export-depth",
        "--max-frames",
        str(profile.max_frames),
        "--every-n",
        str(profile.every_n),
        "--matching",
        "sequential",
        "--no-gpu",
    ]
    if profile.camera_frame:
        preprocess_parts.extend(["--mcd-camera-frame", profile.camera_frame])
    if profile.include_dynamic_tf:
        preprocess_parts.append("--mcd-include-tf-dynamic")

    train_command = (
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "train",
        "--data",
        str(preprocess_dir),
        "--output",
        str(train_dir),
        "--method",
        "gsplat",
        "--iterations",
        str(profile.iterations),
        "--config",
        profile.config_path,
    )
    export_command = (
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "export",
        "--model",
        str(train_dir / "point_cloud.ply"),
        "--format",
        "splat",
        "--output",
        str(export_path),
        "--max-points",
        str(profile.export_max_points),
        "--splat-normalize-extent",
        _format_number(profile.splat_normalize_extent),
        "--splat-min-opacity",
        _format_number(profile.splat_min_opacity),
        "--splat-max-scale",
        _format_number(profile.splat_max_scale),
    )

    expected_artifacts = (
        str(preprocess_dir / "images" / "image_timestamps.csv"),
        str(preprocess_dir / "pose" / "origin_wgs84.json"),
        str(preprocess_dir / "lidar_world_rgb.npy"),
        str(preprocess_dir / "depth"),
        str(preprocess_dir / "sparse" / "0" / "cameras.txt"),
        str(preprocess_dir / "sparse" / "0" / "images.txt"),
        str(preprocess_dir / "sparse" / "0" / "points3D.txt"),
        str(train_dir / "point_cloud.ply"),
        str(export_path),
    )
    return MCDQualityRunPlan(
        profile=profile,
        preprocess_dir=str(preprocess_dir),
        train_dir=str(train_dir),
        export_path=str(export_path),
        preprocess_command=tuple(preprocess_parts),
        train_command=train_command,
        export_command=export_command,
        expected_artifacts=expected_artifacts,
    )


def plan_to_dict(plan: MCDQualityPlan) -> dict[str, Any]:
    """Convert a quality plan into JSON-serializable data."""
    return {
        "context": plan.context.__dict__,
        "preflightCommand": list(plan.preflight_command),
        "runs": [
            {
                "name": run.profile.name,
                "label": run.profile.label,
                "intent": run.profile.intent,
                "requiresFullFolder": run.profile.requires_full_folder,
                "imageTopics": list(run.profile.image_topics),
                "maxFrames": run.profile.max_frames,
                "everyN": run.profile.every_n,
                "iterations": run.profile.iterations,
                "configPath": run.profile.config_path,
                "preprocessDir": run.preprocess_dir,
                "trainDir": run.train_dir,
                "exportPath": run.export_path,
                "preprocessCommand": list(run.preprocess_command),
                "trainCommand": list(run.train_command),
                "exportCommand": list(run.export_command),
                "expectedArtifacts": list(run.expected_artifacts),
            }
            for run in plan.runs
        ],
    }


def render_plan_json(plan: MCDQualityPlan) -> str:
    """Render the plan as stable JSON."""
    return json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n"


def render_plan_markdown(plan: MCDQualityPlan) -> str:
    """Render a quality plan as a compact runbook."""
    lines = [
        "# MCD Quality Run Plan",
        "",
        "## Shared Preflight",
        "",
        "```bash",
        render_shell_command(plan.preflight_command),
        "```",
        "",
        "## Runs",
        "",
    ]
    for run in plan.runs:
        lines.extend(
            [
                f"### {run.profile.label}",
                "",
                run.profile.intent,
                "",
                f"- profile: `{run.profile.name}`",
                f"- frames: `{run.profile.max_frames}` every `{run.profile.every_n}`",
                f"- config: `{run.profile.config_path}`",
                f"- requires full folder: `{str(run.profile.requires_full_folder).lower()}`",
                "",
                "```bash",
                render_shell_command(run.preprocess_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.train_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.export_command, pythonpath=plan.context.pythonpath),
                "```",
                "",
                "Expected artifacts:",
            ]
        )
        lines.extend(f"- `{artifact}`" for artifact in run.expected_artifacts)
        lines.append("")
    return "\n".join(lines)


def render_plan_shell(plan: MCDQualityPlan) -> str:
    """Render all commands as a shell runbook."""
    lines = ["set -euo pipefail", "", "# Shared GNSS preflight", render_shell_command(plan.preflight_command), ""]
    for run in plan.runs:
        lines.extend(
            [
                f"# {run.profile.label}",
                render_shell_command(run.preprocess_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.train_command, pythonpath=plan.context.pythonpath),
                render_shell_command(run.export_command, pythonpath=plan.context.pythonpath),
                "",
            ]
        )
    return "\n".join(lines)


def render_shell_command(command: Iterable[str], *, pythonpath: str | None = None) -> str:
    """Quote a command for bash copy/paste."""
    parts = [shlex.quote(str(part)) for part in command]
    if pythonpath:
        parts.insert(0, f"PYTHONPATH={shlex.quote(str(pythonpath))}")
    return " ".join(parts)


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


__all__ = [
    "MCDQualityPlan",
    "MCDQualityPlanContext",
    "MCDQualityRunPlan",
    "MCDQualityRunProfile",
    "build_mcd_quality_plan",
    "default_mcd_quality_profiles",
    "plan_to_dict",
    "render_plan_json",
    "render_plan_markdown",
    "render_plan_shell",
    "render_shell_command",
]
