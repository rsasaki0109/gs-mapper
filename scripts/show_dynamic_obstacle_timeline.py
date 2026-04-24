#!/usr/bin/env python3
"""Print a DynamicObstacleTimeline JSON as a Markdown or stable-JSON summary.

Useful when a scenario author wants to eyeball a reactive-obstacle timeline
without opening the raw JSON — the Markdown form highlights which obstacles
are static waypoint vs chase vs flee, plus each obstacle's speed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gs_sim2real.sim import (  # noqa: E402
    load_route_policy_dynamic_obstacle_timeline_json,
    render_route_policy_dynamic_obstacle_timeline_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "timeline",
        type=Path,
        help="Path to a DynamicObstacleTimeline JSON artifact",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format: Markdown summary (default) or stable JSON round-trip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Prints to stdout when omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timeline = load_route_policy_dynamic_obstacle_timeline_json(args.timeline)
    if args.format == "markdown":
        rendered = render_route_policy_dynamic_obstacle_timeline_markdown(timeline)
    else:
        rendered = json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
