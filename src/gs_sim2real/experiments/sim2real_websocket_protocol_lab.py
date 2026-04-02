"""Bridge experiment lab for sim2real websocket message protocol policies."""

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
    return _repo_root() / "apps" / "dreamwalker-web" / "tools" / "sim2real-websocket-protocol-lab.mjs"


def build_sim2real_websocket_protocol_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Run the web-side lab and parse its JSON report."""
    node_bin = shutil.which("node")
    if not node_bin:
        raise RuntimeError("node is required to run the sim2real websocket protocol experiment lab")

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


def build_sim2real_websocket_protocol_process_section(report: dict[str, Any]) -> dict[str, Any]:
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
                    fixture_report.get("summary", {}).get("type", "n/a"),
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
                "headers": ["Policy", "Status", "Type", "Match", "Exact"],
                "rows": rows,
            }
        )

    return {
        "title": "Sim2Real Websocket Protocol",
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
            "Stable web code uses `importSim2realWebsocketMessage(rawMessage, policy?)` as the only websocket response import contract.",
            "`alias_friendly` is the default production policy because it preserves canonical server responses while still accepting wrapped envelopes, message aliases, and field aliases from adapters and thin tools.",
            "Alternative websocket protocol policies stay outside production until the same ready/render/benchmark/error fixtures show a better cross-client fit.",
        ],
        "deferred": [
            "`strict_canonical` stays experimental. It is simple, but it rejects nested envelopes that already appear in wrapper tooling.",
            "`envelope_first` stays experimental. It handles wrappers, but it still rejects message aliases and field aliases used by SDK-style ready payloads and thin error emitters.",
        ],
        "rules": [
            "Start websocket protocol work with at least three import policies, not one growing socket message handler.",
            "Compare policies on the same ready, render-result, benchmark-report, and error fixtures before changing production defaults.",
            "Promote only the smallest import surface that keeps the normalized websocket message schema stable for the panel.",
        ],
        "stableInterfaceIntro": "The stable sim2real websocket import surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            importSim2realWebsocketMessage(
                rawMessage,
                policy = 'alias_friendly',
            ) -> normalized websocket message
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`importMessage(rawMessage) -> normalized websocket message`",
        ],
        "comparableInputs": [
            "Same websocket fixtures for every policy",
            "Same workload fixtures across canonical ready, wrapped render-result, aliased ready, wrapped benchmark report, and aliased errors",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`apps/dreamwalker-web/src/`: stable websocket message import contract used by the panel",
            "`apps/dreamwalker-web/tools/`: discardable protocol comparison harnesses and report generators",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the sim2real websocket protocol lab and optionally refresh docs."""
    report = build_sim2real_websocket_protocol_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            sim2real_websocket_protocol_report=report,
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
