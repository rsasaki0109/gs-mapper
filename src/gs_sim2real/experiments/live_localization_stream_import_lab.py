"""Bridge experiment lab for web live-localization stream import policies."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tool_path() -> Path:
    return _repo_root() / "apps" / "dreamwalker-web" / "tools" / "live-localization-stream-import-lab.mjs"


def build_live_localization_stream_import_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Run the web-side lab and parse its JSON report."""
    node_bin = shutil.which("node")
    if not node_bin:
        raise RuntimeError("node is required to run the live localization stream import experiment lab")

    result = subprocess.run(
        [
            node_bin,
            str(_tool_path()),
            "--json",
            "--repetitions",
            str(int(repetitions)),
        ],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def build_live_localization_stream_import_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the JS report into the shared docs section shape."""
    comparison_rows = []
    for policy in report["policies"]:
        comparison_rows.append(
            [
                policy["label"],
                policy["tier"],
                policy["style"],
                f"{float(policy['aggregate']['successRate'] or 0.0):.2f}",
                f"{float(policy['aggregate']['exactMatchRate'] or 0.0):.2f}",
                f"{float(policy['aggregate']['meanMatchScore'] or 0.0):.3f}",
                f"{float(policy['runtime']['medianMs'] or 0.0):.3f}",
                f"{float(policy['readability']['score']):.1f}",
                f"{float(policy['extensibility']['score']):.1f}",
            ]
        )

    fixture_sections = []
    for fixture in report["fixtures"]:
        rows = []
        for policy in report["policies"]:
            fixture_report = next(item for item in policy["fixtures"] if item["fixtureId"] == fixture["fixtureId"])
            rows.append(
                [
                    policy["label"],
                    fixture_report["status"],
                    fixture_report.get("kind", "n/a"),
                    f"{float(fixture_report.get('matchScore') or 0.0):.3f}"
                    if fixture_report["status"] == "ok"
                    else "n/a",
                    "yes" if fixture_report.get("exactMatch") else "no",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Kind", "Match", "Exact"],
                "rows": rows,
            }
        )

    return {
        "title": "Live Localization Stream Import",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Policy",
            "Tier",
            "Style",
            "Success",
            "Exact",
            "Shape",
            "Runtime (ms)",
            "Readability",
            "Extensibility",
        ],
        "comparisonRows": comparison_rows,
        "fixtureSections": fixture_sections,
        "highlights": [
            f"Best policy fit: `{report['highlights']['bestFit']['label']}`",
            f"Fastest median runtime: `{report['highlights']['fastestMedianRuntime']['label']}`",
            f"Most readable implementation: `{report['highlights']['mostReadable']['label']}`",
            f"Broadest extension surface: `{report['highlights']['mostExtensible']['label']}`",
        ],
        "accepted": [
            "Stable web code uses `importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options, policy?)` as the only live-stream import contract.",
            "`alias_friendly` is the default production policy because it preserves the canonical websocket stream while still accepting wrapper aliases, top-level pose shortcuts, and message aliases from SDK and local tools.",
            "Alternative live-stream policies stay outside production until the same message fixtures show a better cross-client fit.",
        ],
        "deferred": [
            "`strict_canonical` stays experimental. It is readable, but it rejects cameraPose wrappers, top-level pose shortcuts, and clear aliases.",
            "`wrapped_pose` stays experimental. It works for pose wrappers, but it still drops top-level append shortcuts and clear aliases used by light clients.",
        ],
        "rules": [
            "Start live-stream import work with at least three policies, not one growing websocket handler.",
            "Compare policies on the same canonical, wrapped, shortcut, and clear-alias message fixtures before changing production defaults.",
            "Promote only the smallest import surface that keeps the normalized live estimate schema stable.",
        ],
        "stableInterfaceIntro": "The stable live localization stream import surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            importLiveLocalizationStreamMessage(
                previousEstimate,
                rawMessage,
                options = {},
                policy = 'alias_friendly',
            ) -> { kind, estimate }
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`importMessage(previousEstimate, rawMessage, options) -> { kind, estimate }`",
        ],
        "comparableInputs": [
            "Same live-message fixtures for every policy",
            "Same workload fixtures across canonical append, snapshot, wrapper aliases, top-level shortcuts, and clear aliases",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`apps/dreamwalker-web/src/`: stable live-stream import contract used by the panel",
            "`apps/dreamwalker-web/tools/`: discardable policy comparison harnesses and report generators",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the live-localization stream import lab and optionally refresh docs."""
    report = build_live_localization_stream_import_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            live_localization_stream_import_report=report,
        )
    summary = {
        "type": report["type"],
        "policyCount": len(report["policies"]),
        "fixtureCount": len(report["fixtures"]),
        "bestFit": report["highlights"]["bestFit"],
        "fastestMedianRuntime": report["highlights"]["fastestMedianRuntime"],
        "docs": docs,
    }
    print(json.dumps(summary, indent=2))
