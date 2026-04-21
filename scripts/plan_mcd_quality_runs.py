#!/usr/bin/env python3
"""Print reproducible MCD supervised quality-run commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.experiments.mcd_quality_plan import (  # noqa: E402
    MCDQualityPlanContext,
    build_mcd_quality_plan,
    default_mcd_quality_profiles,
    render_plan_json,
    render_plan_markdown,
    render_plan_shell,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default="data/mcd/ntu_day_02", help="MCD session directory")
    parser.add_argument("--output-root", default="outputs/mcd_quality", help="Root directory for planned outputs")
    parser.add_argument(
        "--calibration",
        default="data/mcd/calibration_atv.yaml",
        help="Official MCD ATV calibration YAML path",
    )
    parser.add_argument(
        "--asset-dir", default="outputs/mcd_quality/assets", help="Directory for planned .splat exports"
    )
    parser.add_argument("--python", default="python3", help="Python executable used in generated commands")
    parser.add_argument("--pythonpath", default="src", help="PYTHONPATH used in generated gs_sim2real commands")
    parser.add_argument("--start-offset-sec", type=float, default=35.0, help="GNSS/image/LiDAR warm-up trim")
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.name for profile in default_mcd_quality_profiles()],
        help="Only include the named profile. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=["markdown", "json", "shell"], default="markdown")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    context = MCDQualityPlanContext(
        session_dir=args.session_dir,
        output_root=args.output_root,
        calibration_path=args.calibration,
        asset_dir=args.asset_dir,
        python_executable=args.python,
        pythonpath=args.pythonpath,
        start_offset_sec=args.start_offset_sec,
    )
    profiles = default_mcd_quality_profiles()
    if args.profile:
        wanted = set(args.profile)
        profiles = tuple(profile for profile in profiles if profile.name in wanted)
    plan = build_mcd_quality_plan(context, profiles=profiles)

    if args.format == "json":
        print(render_plan_json(plan), end="")
    elif args.format == "shell":
        print(render_plan_shell(plan))
    else:
        print(render_plan_markdown(plan))


if __name__ == "__main__":
    main()
