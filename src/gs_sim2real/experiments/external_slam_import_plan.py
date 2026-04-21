"""Planning helpers for external SLAM import dry-run gates."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ExternalSLAMImportPlanContext:
    """Shared paths and CLI defaults for external SLAM import preflights."""

    image_dir: str = "data/outdoor/bag6/images"
    output_root: str = "outputs/external_slam_imports"
    artifact_root: str = "outputs/external_slam"
    python_executable: str = "python3"
    pythonpath: str = "src"
    manifest_format: str = "json"
    min_aligned_frames: int = 2
    min_point_count: int = 0
    allow_dropped_images: bool = False
    require_pointcloud: bool = False


@dataclass(frozen=True)
class ExternalSLAMImportProfile:
    """One external SLAM front-end artifact convention."""

    name: str
    system: str
    label: str
    intent: str
    artifact_subdir: str
    trajectory_path: str | None = None
    pointcloud_path: str | None = None
    pinhole_calib_path: str | None = None
    require_pointcloud: bool | None = None
    min_point_count: int | None = None
    allow_dropped_images: bool | None = None


@dataclass(frozen=True)
class ExternalSLAMImportRunPlan:
    """Concrete dry-run command for one external SLAM import preflight."""

    profile: ExternalSLAMImportProfile
    output_dir: str
    artifact_dir: str
    dry_run_command: tuple[str, ...]


@dataclass(frozen=True)
class ExternalSLAMImportPlan:
    """A reproducible external SLAM import preflight matrix."""

    context: ExternalSLAMImportPlanContext
    runs: tuple[ExternalSLAMImportRunPlan, ...]


def default_external_slam_import_profiles() -> tuple[ExternalSLAMImportProfile, ...]:
    """Return the default visual-SLAM import candidates."""

    return (
        ExternalSLAMImportProfile(
            name="bag6_mast3r_slam",
            system="mast3r-slam",
            label="Bag6 MASt3R-SLAM",
            intent="Validate MASt3R-SLAM trajectory and reconstruction artifacts before COLMAP import.",
            artifact_subdir="mast3r-slam",
            require_pointcloud=True,
        ),
        ExternalSLAMImportProfile(
            name="bag6_vggt_slam_2",
            system="vggt-slam",
            label="Bag6 VGGT-SLAM 2.0",
            intent="Validate VGGT-SLAM 2.0 trajectory and point cloud artifacts before COLMAP import.",
            artifact_subdir="vggt-slam",
            require_pointcloud=True,
        ),
        ExternalSLAMImportProfile(
            name="bag6_pi3",
            system="pi3",
            label="Bag6 Pi3/Pi3X",
            intent="Validate Pi3/Pi3X tensor exports before materializing TUM and point cloud files.",
            artifact_subdir="pi3",
            require_pointcloud=True,
        ),
        ExternalSLAMImportProfile(
            name="bag6_loger",
            system="loger",
            label="Bag6 LoGeR",
            intent="Validate LoGeR tensor or trajectory exports before materializing import artifacts.",
            artifact_subdir="loger",
            require_pointcloud=True,
        ),
    )


def build_external_slam_import_plan(
    context: ExternalSLAMImportPlanContext | None = None,
    *,
    profiles: Iterable[ExternalSLAMImportProfile] | None = None,
) -> ExternalSLAMImportPlan:
    """Build dry-run commands for all selected external SLAM import candidates."""

    ctx = context or ExternalSLAMImportPlanContext()
    selected_profiles = tuple(profiles or default_external_slam_import_profiles())
    runs = tuple(_build_run_plan(ctx, profile) for profile in selected_profiles)
    return ExternalSLAMImportPlan(context=ctx, runs=runs)


def _build_run_plan(
    ctx: ExternalSLAMImportPlanContext,
    profile: ExternalSLAMImportProfile,
) -> ExternalSLAMImportRunPlan:
    output_dir = Path(ctx.output_root) / profile.name
    artifact_dir = Path(ctx.artifact_root) / profile.artifact_subdir
    require_pointcloud = ctx.require_pointcloud if profile.require_pointcloud is None else profile.require_pointcloud
    min_point_count = ctx.min_point_count if profile.min_point_count is None else profile.min_point_count
    allow_dropped = ctx.allow_dropped_images if profile.allow_dropped_images is None else profile.allow_dropped_images
    command: list[str] = [
        ctx.python_executable,
        "-m",
        "gs_sim2real.cli",
        "preprocess",
        "--images",
        ctx.image_dir,
        "--output",
        str(output_dir),
        "--method",
        "external-slam",
        "--external-slam-system",
        profile.system,
        "--external-slam-output",
        str(artifact_dir),
        "--external-slam-dry-run",
        "--external-slam-manifest-format",
        ctx.manifest_format,
        "--external-slam-fail-on-dry-run-gate",
        "--external-slam-min-aligned-frames",
        str(ctx.min_aligned_frames),
    ]
    if allow_dropped:
        command.append("--external-slam-allow-dropped-images")
    if require_pointcloud:
        command.append("--external-slam-require-pointcloud")
    if min_point_count > 0:
        command.extend(["--external-slam-min-point-count", str(min_point_count)])
    if profile.trajectory_path:
        command.extend(["--trajectory", profile.trajectory_path])
    if profile.pointcloud_path:
        command.extend(["--pointcloud", profile.pointcloud_path])
    if profile.pinhole_calib_path:
        command.extend(["--pinhole-calib", profile.pinhole_calib_path])
    return ExternalSLAMImportRunPlan(
        profile=profile,
        output_dir=str(output_dir),
        artifact_dir=str(artifact_dir),
        dry_run_command=tuple(command),
    )


def plan_to_dict(plan: ExternalSLAMImportPlan) -> dict[str, Any]:
    """Convert an external SLAM import plan into JSON-serializable data."""

    return {
        "type": "external-slam-import-plan",
        "context": plan.context.__dict__,
        "runs": [
            {
                "name": run.profile.name,
                "system": run.profile.system,
                "label": run.profile.label,
                "intent": run.profile.intent,
                "artifactDir": run.artifact_dir,
                "outputDir": run.output_dir,
                "dryRunCommand": list(run.dry_run_command),
            }
            for run in plan.runs
        ],
    }


def render_plan_json(plan: ExternalSLAMImportPlan) -> str:
    """Render the import preflight plan as stable JSON."""

    return json.dumps(plan_to_dict(plan), indent=2, sort_keys=True) + "\n"


def render_plan_markdown(plan: ExternalSLAMImportPlan) -> str:
    """Render the import preflight plan as a compact runbook."""

    lines = [
        "# External SLAM Import Preflight Plan",
        "",
        "| Run | System | Artifact Dir | Gate Command |",
        "| --- | --- | --- | --- |",
    ]
    for run in plan.runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run.profile.label,
                    run.profile.system,
                    f"`{run.artifact_dir}`",
                    f"`{render_shell_command(run.dry_run_command, pythonpath=plan.context.pythonpath)}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_plan_shell(plan: ExternalSLAMImportPlan) -> str:
    """Render all dry-run gate commands as a shell runbook."""

    lines = ["set -euo pipefail", ""]
    for run in plan.runs:
        lines.extend(
            [
                f"# {run.profile.label}",
                render_shell_command(run.dry_run_command, pythonpath=plan.context.pythonpath),
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


__all__ = [
    "ExternalSLAMImportPlan",
    "ExternalSLAMImportPlanContext",
    "ExternalSLAMImportProfile",
    "ExternalSLAMImportRunPlan",
    "build_external_slam_import_plan",
    "default_external_slam_import_profiles",
    "plan_to_dict",
    "render_plan_json",
    "render_plan_markdown",
    "render_plan_shell",
    "render_shell_command",
]
