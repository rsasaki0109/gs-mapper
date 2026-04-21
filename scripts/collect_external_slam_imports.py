#!/usr/bin/env python3
"""Collect saved external SLAM import dry-run manifest summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.experiments.external_slam_import_collect import (  # noqa: E402
    collect_external_slam_import_preflight_results,
    render_external_slam_import_report_json,
    render_external_slam_import_report_markdown,
)
from gs_sim2real.experiments.external_slam_import_plan import (  # noqa: E402
    ExternalSLAMImportPlanContext,
    build_external_slam_import_plan,
    default_external_slam_import_profiles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/outdoor/bag6/images", help="Image directory used by the plan")
    parser.add_argument(
        "--artifact-root",
        default="outputs/external_slam",
        help="Root directory containing one subdirectory per external SLAM profile",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/external_slam_imports",
        help="Root directory containing saved dry-run manifests",
    )
    parser.add_argument("--python", default="python3", help="Python executable recorded in the plan context")
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH recorded in the plan context")
    parser.add_argument("--manifest-format", choices=["text", "json"], default="json")
    parser.add_argument("--min-aligned-frames", type=int, default=2)
    parser.add_argument("--min-point-count", type=int, default=0)
    parser.add_argument("--allow-dropped-images", action="store_true")
    parser.add_argument("--require-pointcloud", action="store_true")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_external_slam_import_profiles()],
        help="Only collect the named profile. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", default=None, help="Optional path to write the rendered report")
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
    report = collect_external_slam_import_preflight_results(build_external_slam_import_plan(context, profiles=profiles))

    rendered = (
        render_external_slam_import_report_json(report)
        if args.format == "json"
        else render_external_slam_import_report_markdown(report)
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
