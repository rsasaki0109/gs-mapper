#!/usr/bin/env python3
"""Print reproducible external SLAM import dry-run gate commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.experiments.external_slam_import_plan import (  # noqa: E402
    ExternalSLAMImportPlanContext,
    build_external_slam_import_plan,
    default_external_slam_import_profiles,
    render_plan_json,
    render_plan_markdown,
    render_plan_shell,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/outdoor/bag6/images", help="Image directory to validate")
    parser.add_argument(
        "--artifact-root",
        default="outputs/external_slam",
        help="Root directory containing one subdirectory per external SLAM profile",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/external_slam_imports",
        help="Root directory for planned import outputs",
    )
    parser.add_argument("--python", default="python3", help="Python executable used in generated commands")
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH used in generated gs_sim2real commands")
    parser.add_argument("--manifest-format", choices=["text", "json"], default="json")
    parser.add_argument("--min-aligned-frames", type=int, default=2)
    parser.add_argument("--min-point-count", type=int, default=0)
    parser.add_argument("--allow-dropped-images", action="store_true")
    parser.add_argument("--require-pointcloud", action="store_true")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_external_slam_import_profiles()],
        help="Only include the named profile. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=["markdown", "json", "shell"], default="markdown")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    context = ExternalSLAMImportPlanContext(
        image_dir=args.images,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        python_executable=args.python,
        pythonpath=args.pythonpath,
        manifest_format=args.manifest_format,
        min_aligned_frames=args.min_aligned_frames,
        min_point_count=args.min_point_count,
        allow_dropped_images=args.allow_dropped_images,
        require_pointcloud=args.require_pointcloud,
    )
    profiles = default_external_slam_import_profiles()
    if args.profile:
        wanted = set(args.profile)
        profiles = tuple(profile for profile in profiles if profile.name in wanted)
    plan = build_external_slam_import_plan(context, profiles=profiles)

    if args.format == "json":
        print(render_plan_json(plan), end="")
    elif args.format == "shell":
        print(render_plan_shell(plan))
    else:
        print(render_plan_markdown(plan), end="")


if __name__ == "__main__":
    main()
