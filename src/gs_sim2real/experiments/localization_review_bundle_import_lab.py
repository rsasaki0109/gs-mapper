"""Bridge experiment lab for localization review bundle import policies."""

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
    return _repo_root() / "apps" / "dreamwalker-web" / "tools" / "localization-review-bundle-import-lab.mjs"


def build_localization_review_bundle_import_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Run the web-side lab and parse its JSON report."""
    node_bin = shutil.which("node")
    if not node_bin:
        raise RuntimeError("node is required to run the localization review bundle import experiment lab")

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


def build_localization_review_bundle_import_process_section(report: dict[str, Any]) -> dict[str, Any]:
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
                    fixture_report.get("summary", {}).get("runCount", "n/a"),
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
                "headers": ["Policy", "Status", "Runs", "Match", "Exact"],
                "rows": rows,
            }
        )

    return {
        "title": "Localization Review Bundle Import",
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
            "Stable web code uses `importLocalizationReviewBundleDocument(rawDocument, policy?)` as the only review-bundle import contract.",
            "`alias_friendly` is the default production policy because it preserves canonical exports while still recovering linked captures and wrapper-friendly portable run shapes.",
            "Alternative review-bundle import policies stay outside production until the same canonical, linked-fallback, and alias-wrapper fixtures show a better cross-share fit.",
        ],
        "deferred": [
            "`strict_canonical` stays experimental. It is simple, but it rejects portable review bundles that rely on linked captures instead of embedded bundles.",
            "`linked_capture_fallback` stays experimental. It handles shared captures, but it still rejects wrapper aliases used by portable review-bundle adapters.",
        ],
        "rules": [
            "Start review-bundle import work with at least three policies, not one growing file-import handler.",
            "Compare policies on the same canonical, linked-fallback, and alias-wrapper fixtures before changing production defaults.",
            "Promote only the smallest import surface that keeps the restored capture shelf and run shelf schema stable.",
        ],
        "stableInterfaceIntro": "The stable localization review bundle import surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            importLocalizationReviewBundleDocument(
                rawDocument,
                policy = 'alias_friendly',
            ) -> normalized review bundle import payload
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`importDocument(rawDocument) -> normalized review bundle import payload`",
        ],
        "comparableInputs": [
            "Same review-bundle fixtures for every policy",
            "Same workload fixtures across canonical embedded bundles, linked-capture fallback, and alias-wrapped portable bundles",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`apps/dreamwalker-web/src/`: stable review-bundle import contract used by the panel",
            "`apps/dreamwalker-web/tools/`: discardable review-bundle comparison harnesses and report generators",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the localization review bundle import lab and optionally refresh docs."""
    report = build_localization_review_bundle_import_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            localization_review_bundle_import_report=report,
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
