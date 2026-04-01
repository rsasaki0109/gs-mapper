"""Experiment-first lab for route capture bundle import policies."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

from gs_sim2real.core.route_capture_bundle_import import (
    ResponsePoseFallbackRouteCaptureBundleImportPolicy,
    RouteAwareRouteCaptureBundleImportPolicy,
    RouteCaptureBundleImportPolicy,
    RouteCaptureBundleImportRequest,
    StrictCanonicalRouteCaptureBundleImportPolicy,
)


@dataclass(frozen=True)
class RouteCaptureBundleImportFixture:
    """Canonical route-capture bundle shared by every import policy."""

    fixture_id: str
    label: str
    intent: str
    request: RouteCaptureBundleImportRequest
    expected_fragment_label: str
    expected_capture_count: int
    expected_last_position: tuple[float, float, float]


EXPERIMENT_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES: tuple[RouteCaptureBundleImportPolicy, ...] = (
    StrictCanonicalRouteCaptureBundleImportPolicy(),
    ResponsePoseFallbackRouteCaptureBundleImportPolicy(),
    RouteAwareRouteCaptureBundleImportPolicy(),
)


def _render_response(position: list[float]) -> dict[str, Any]:
    return {
        "type": "render-result",
        "colorJpegBase64": "ZmFrZS1qcGVn",
        "pose": {
            "position": position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    }


def build_route_capture_bundle_import_fixtures() -> list[RouteCaptureBundleImportFixture]:
    """Build shared fixtures for bundle import comparisons."""
    canonical_bundle = {
        "type": "route-capture-bundle",
        "fragmentLabel": "Residency Canonical",
        "captures": [
            {
                "index": 0,
                "label": "gt:1",
                "pose": {"position": [0.0, 0.0, 0.0], "yawDegrees": 0.0},
                "response": _render_response([0.0, 0.0, 0.0]),
            },
            {
                "index": 1,
                "label": "gt:2",
                "pose": {"position": [1.0, 0.0, 0.0], "yawDegrees": 0.0},
                "response": _render_response([1.0, 0.0, 0.0]),
            },
        ],
    }
    response_pose_bundle = {
        "type": "route-capture-bundle",
        "fragmentLabel": "Residency Response Pose",
        "captures": [
            {
                "index": 0,
                "label": "gt:1",
                "response": _render_response([2.0, 0.0, 0.0]),
            }
        ],
    }
    route_pose_bundle = {
        "type": "route-capture-bundle",
        "fragmentLabel": "Residency Route Pose",
        "route": [
            {"index": 0, "position": [3.0, 0.0, 0.0], "yawDegrees": 15.0},
            {"index": 1, "position": [4.0, 0.0, 0.0], "yawDegrees": 15.0},
        ],
        "captures": [
            {
                "index": 0,
                "label": "gt:1",
                "response": _render_response([30.0, 0.0, 0.0]),
            },
            {
                "index": 1,
                "label": "gt:2",
                "response": _render_response([40.0, 0.0, 0.0]),
            },
        ],
    }
    return [
        RouteCaptureBundleImportFixture(
            fixture_id="canonical-bundle",
            label="Canonical Bundle",
            intent="Preserve current capture bundles with explicit capture poses.",
            request=RouteCaptureBundleImportRequest(canonical_bundle),
            expected_fragment_label="Residency Canonical",
            expected_capture_count=2,
            expected_last_position=(1.0, 0.0, 0.0),
        ),
        RouteCaptureBundleImportFixture(
            fixture_id="response-pose-fallback",
            label="Response Pose Fallback",
            intent="Recover bundle poses from render-result responses when capture.pose is missing.",
            request=RouteCaptureBundleImportRequest(response_pose_bundle),
            expected_fragment_label="Residency Response Pose",
            expected_capture_count=1,
            expected_last_position=(2.0, 0.0, 0.0),
        ),
        RouteCaptureBundleImportFixture(
            fixture_id="route-pose-fallback",
            label="Route Pose Fallback",
            intent="Recover capture poses from bundle.route when neither capture.pose nor response pose should define ground truth.",
            request=RouteCaptureBundleImportRequest(route_pose_bundle),
            expected_fragment_label="Residency Route Pose",
            expected_capture_count=2,
            expected_last_position=(4.0, 0.0, 0.0),
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_bundle_import_fixture(
    policy: RouteCaptureBundleImportPolicy,
    fixture: RouteCaptureBundleImportFixture,
) -> dict[str, Any]:
    """Run one bundle import policy on one canonical fixture."""
    started_at = perf_counter()
    try:
        parsed = policy.import_bundle(fixture.request)
    except Exception as exc:
        return {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "status": "error",
            "error": str(exc),
            "runtimeMs": float((perf_counter() - started_at) * 1000.0),
        }

    runtime_ms = float((perf_counter() - started_at) * 1000.0)
    captures = parsed.get("captures", []) if isinstance(parsed.get("captures"), list) else []
    last_capture = captures[-1] if captures else {}
    last_pose = last_capture.get("pose") if isinstance(last_capture, dict) else {}
    last_position = tuple(last_pose.get("position", [])) if isinstance(last_pose, dict) else ()
    fragment_label_match = parsed.get("fragmentLabel") == fixture.expected_fragment_label
    capture_count_match = len(captures) == fixture.expected_capture_count
    last_position_match = last_position == fixture.expected_last_position
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "fragmentLabel": parsed.get("fragmentLabel"),
        "captureCount": len(captures),
        "lastPosition": last_position,
        "fragmentLabelMatch": fragment_label_match,
        "captureCountMatch": capture_count_match,
        "lastPositionMatch": last_position_match,
        "schemaMatchScore": float(sum([fragment_label_match, capture_count_match, last_position_match]) / 3.0),
        "runtimeMs": runtime_ms,
    }


def benchmark_bundle_import_policy_runtime(
    policy: RouteCaptureBundleImportPolicy,
    fixtures: Sequence[RouteCaptureBundleImportFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure bundle import policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                policy.import_bundle(fixture.request)
            except Exception:
                continue
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))

    if not samples_ms:
        return {"repetitions": int(repetitions), "sampleCount": 0, "meanMs": None, "medianMs": None}

    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: RouteCaptureBundleImportPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.import_bundle))
    tree = ast.parse(source)
    branch_count = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in ast.walk(tree))
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", '"""', "'''", "#"))
    ]
    lines_of_code = len(lines)
    score = max(1.0, 10.0 - max(0, lines_of_code - 8) * 0.2 - max(0, branch_count - 2) * 0.8)
    return {
        "score": round(score, 1),
        "linesOfCode": lines_of_code,
        "branchCount": branch_count,
    }


def evaluate_extensibility(policy: RouteCaptureBundleImportPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "supportsCanonicalCapturePose": 2.0,
        "supportsResponsePoseFallback": 3.0,
        "supportsRoutePoseFallback": 3.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_bundle_import_policy(
    policy: RouteCaptureBundleImportPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one bundle import policy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    aggregate = {
        "successRate": float(len(successful) / max(1, len(fixture_reports))),
        "schemaMatchRate": _mean_or_none([report.get("schemaMatchScore") for report in successful]),
        "fragmentLabelMatchRate": _mean_or_none(
            [1.0 if report.get("fragmentLabelMatch") else 0.0 for report in successful]
        ),
        "failedFixtures": [report["fixtureId"] for report in fixture_reports if report["status"] != "ok"],
    }
    return {
        "name": policy.name,
        "label": policy.label,
        "style": policy.style,
        "tier": policy.tier,
        "capabilities": dict(policy.capabilities),
        "fixtures": list(fixture_reports),
        "aggregate": aggregate,
        "runtime": runtime_report,
        "readability": evaluate_readability(policy),
        "extensibility": evaluate_extensibility(policy),
    }


def build_route_capture_bundle_import_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare route capture bundle import policies on shared fixtures."""
    fixtures = build_route_capture_bundle_import_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "expectedFragmentLabel": fixture.expected_fragment_label,
            "expectedCaptureCount": fixture.expected_capture_count,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_ROUTE_CAPTURE_BUNDLE_IMPORT_POLICIES:
        fixture_reports = [evaluate_bundle_import_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_bundle_import_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_bundle_import_policy(policy, fixture_reports, runtime_report))

    best_fit = max(
        policy_reports,
        key=lambda report: (
            float(report["aggregate"]["successRate"] or 0.0),
            float(report["aggregate"]["schemaMatchRate"] or 0.0),
            float(report["aggregate"]["fragmentLabelMatchRate"] or 0.0),
        ),
    )
    fastest = min(
        (report for report in policy_reports if report["runtime"].get("medianMs") is not None),
        key=lambda report: float(report["runtime"]["medianMs"]),
    )
    most_readable = max(policy_reports, key=lambda report: float(report["readability"]["score"]))
    most_extensible = max(policy_reports, key=lambda report: float(report["extensibility"]["score"]))
    return {
        "protocol": "gs-sim2real-experiment-report/v1",
        "type": "route-capture-bundle-import-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "route-capture-bundle-import",
            "statement": (
                "Import ground-truth route capture bundles without freezing one bundle shape for canonical exports, "
                "response-pose recovery, and route-aware recovery flows."
            ),
            "stableInterface": "import_route_capture_bundle(RouteCaptureBundleImportRequest(...))",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": ["successRate", "schemaMatchRate", "fragmentLabelMatchRate"],
            "runtime": ["meanMs", "medianMs"],
            "readability": ["score", "linesOfCode", "branchCount"],
            "extensibility": ["score", "supportedCapabilities"],
            "heuristicNotice": "Readability/extensibility are generated heuristics, not objective truth.",
        },
        "policies": policy_reports,
        "highlights": {
            "bestFit": {
                "policy": best_fit["name"],
                "label": best_fit["label"],
                "schemaMatchRate": best_fit["aggregate"]["schemaMatchRate"],
            },
            "fastestMedianRuntime": {
                "policy": fastest["name"],
                "label": fastest["label"],
                "medianMs": fastest["runtime"]["medianMs"],
            },
            "mostReadable": {
                "policy": most_readable["name"],
                "label": most_readable["label"],
                "score": most_readable["readability"]["score"],
            },
            "mostExtensible": {
                "policy": most_extensible["name"],
                "label": most_extensible["label"],
                "score": most_extensible["extensibility"]["score"],
            },
        },
    }


def build_route_capture_bundle_import_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the bundle import report into a shared docs section."""
    comparison_rows = []
    for policy in report["policies"]:
        aggregate = policy["aggregate"]
        comparison_rows.append(
            [
                policy["label"],
                policy["tier"],
                policy["style"],
                f"{float(aggregate['successRate'] or 0.0):.2f}",
                f"{float(aggregate['schemaMatchRate'] or 0.0):.2f}",
                f"{float(aggregate['fragmentLabelMatchRate'] or 0.0):.2f}",
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
                    fixture_report.get("captureCount", "n/a"),
                    "yes" if fixture_report.get("fragmentLabelMatch") else "no",
                    f"{float(fixture_report.get('schemaMatchScore') or 0.0):.3f}"
                    if fixture_report["status"] == "ok"
                    else "n/a",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Captures", "Label Match", "Schema Match"],
                "rows": rows,
            }
        )

    return {
        "title": "Route Capture Bundle Import",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Policy",
            "Tier",
            "Style",
            "Success",
            "Schema",
            "Label",
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
            "Stable code uses `import_route_capture_bundle(RouteCaptureBundleImportRequest(...))` as the only ground-truth bundle import contract.",
            "`route_aware` is the default production policy because it preserves canonical bundles while still recovering poses from `route` and `response.pose` when captures are incomplete.",
            "Alternative bundle import policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-bundle fit.",
        ],
        "deferred": [
            "`strict_canonical` stays experimental. It is simple, but it rejects bundle variants that omit explicit capture poses.",
            "`response_pose_fallback` stays experimental. It recovers response poses, but it still ignores route-ground-truth bundles where `route` should define the capture pose.",
        ],
        "rules": [
            "Start ground-truth bundle work with at least three import policies, not one expanding normalizer.",
            "Compare policies on the same canonical, response-fallback, and route-fallback fixtures before changing production defaults.",
            "Promote only the smallest bundle import contract that preserves the benchmark-facing capture schema.",
        ],
        "stableInterfaceIntro": "The stable route-capture bundle import surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class RouteCaptureBundleImportRequest:
                input_like: Any

            def import_route_capture_bundle(
                request: RouteCaptureBundleImportRequest,
                *,
                policy: str = 'route_aware',
            ) -> dict[str, Any]: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`import_bundle(request) -> normalized route-capture-bundle dict`",
        ],
        "comparableInputs": [
            "Same `RouteCaptureBundleImportRequest` fixtures for every policy",
            "Same bundle fixtures across canonical capture poses, response-pose fallback, and route-pose fallback",
            "Same evaluation axes: schema match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable route capture bundle import contract for production benchmarking",
            "`src/gs_sim2real/experiments/`: discardable bundle import policies and comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the route capture bundle import lab and optionally refresh docs."""
    report = build_route_capture_bundle_import_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            route_capture_bundle_import_report=report,
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
