"""Experiment-first lab for sim2real query request import policies."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

from gs_sim2real.core.query_request_import import (
    AliasFriendlyQueryRequestImportPolicy,
    EnvelopeFirstQueryRequestImportPolicy,
    ImportedQueryRequest,
    QueryRequestImportPolicy,
    QueryRequestImportRequest,
    RenderQueryDefaults,
    StrictSchemaQueryRequestImportPolicy,
)


@dataclass(frozen=True)
class QueryRequestImportFixture:
    """Canonical query payload shared by every import policy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryRequestImportRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_REQUEST_IMPORT_POLICIES: tuple[QueryRequestImportPolicy, ...] = (
    StrictSchemaQueryRequestImportPolicy(),
    EnvelopeFirstQueryRequestImportPolicy(),
    AliasFriendlyQueryRequestImportPolicy(),
)


def build_query_request_import_fixtures() -> list[QueryRequestImportFixture]:
    """Build shared fixtures for query request import comparisons."""
    defaults = RenderQueryDefaults(
        width=640,
        height=480,
        fov_degrees=60.0,
        near_clip=0.05,
        far_clip=50.0,
        point_radius=1,
        timeout_ms=10_000,
    )
    estimate_document = {
        "type": "localization-estimate",
        "sourceType": "poses",
        "label": "Run A",
        "poses": [
            {
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
        ],
    }
    ground_truth_bundle = {
        "type": "route-capture-bundle",
        "captures": [],
    }
    return [
        QueryRequestImportFixture(
            fixture_id="canonical-render",
            label="Canonical Render Payload",
            intent="Preserve the current explicit render contract without reinterpretation.",
            request=QueryRequestImportRequest(
                payload={
                    "type": "render",
                    "pose": {
                        "position": [1.0, 2.0, 3.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                    "width": 320,
                    "height": 240,
                    "fovDegrees": 75.0,
                    "nearClip": 0.1,
                    "farClip": 40.0,
                    "pointRadius": 2,
                },
                defaults=defaults,
            ),
            expected_summary={
                "requestType": "render",
                "width": 320,
                "height": 240,
                "fovDegrees": 75.0,
                "pointRadius": 2,
                "position": (1.0, 2.0, 3.0),
            },
        ),
        QueryRequestImportFixture(
            fixture_id="enveloped-render-aliases",
            label="Enveloped Render Aliases",
            intent="Accept thin request wrappers from browser or SDK clients without changing the normalized output.",
            request=QueryRequestImportRequest(
                payload={
                    "requestType": "render",
                    "request": {
                        "cameraPose": {
                            "position": [0.0, 1.0, 2.0],
                            "quaternion": [0.0, 0.0, 0.0, 1.0],
                        },
                        "imageWidth": 160,
                        "imageHeight": 120,
                        "fov": 70.0,
                        "near": 0.2,
                        "far": 30.0,
                        "radius": 4,
                    },
                },
                defaults=defaults,
            ),
            expected_summary={
                "requestType": "render",
                "width": 160,
                "height": 120,
                "fovDegrees": 70.0,
                "pointRadius": 4,
                "position": (0.0, 1.0, 2.0),
            },
        ),
        QueryRequestImportFixture(
            fixture_id="pose-shortcut-render",
            label="Pose Shortcut Render",
            intent="Support low-friction local tooling that sends position/orientation directly at the top level.",
            request=QueryRequestImportRequest(
                payload={
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "width": 96,
                    "height": 64,
                    "fovDeg": 50.0,
                },
                defaults=defaults,
            ),
            expected_summary={
                "requestType": "render",
                "width": 96,
                "height": 64,
                "fovDegrees": 50.0,
                "pointRadius": 1,
                "position": (4.0, 5.0, 6.0),
            },
        ),
        QueryRequestImportFixture(
            fixture_id="canonical-benchmark",
            label="Canonical Image Benchmark",
            intent="Keep the existing localization image benchmark request schema stable.",
            request=QueryRequestImportRequest(
                payload={
                    "type": "localization-image-benchmark",
                    "groundTruthBundle": ground_truth_bundle,
                    "estimate": estimate_document,
                    "alignment": "timestamp",
                    "timeoutMs": 2500,
                    "maxFrames": 8,
                    "metrics": ["psnr", "lpips"],
                    "lpipsNet": "vgg",
                    "device": "auto",
                },
                defaults=defaults,
            ),
            expected_summary={
                "requestType": "localization-image-benchmark",
                "alignment": "timestamp",
                "timeoutMs": 2500,
                "maxFrames": 8,
                "metrics": ("psnr", "lpips"),
                "lpipsNet": "vgg",
                "device": "auto",
                "groundTruthType": "route-capture-bundle",
            },
        ),
        QueryRequestImportFixture(
            fixture_id="wrapped-benchmark-aliases",
            label="Wrapped Benchmark Aliases",
            intent="Accept benchmark wrappers and alias keys from heterogeneous clients without forcing one envelope shape.",
            request=QueryRequestImportRequest(
                payload={
                    "type": "image-benchmark",
                    "benchmark": {
                        "groundTruth": ground_truth_bundle,
                        "trajectory": estimate_document,
                        "alignmentMode": "index",
                        "responseTimeoutMs": 1800,
                        "frameLimit": 5,
                        "metricNames": ["LPIPS", "SSIM"],
                        "lpipsNetName": "alex",
                        "computeDevice": "cpu",
                    },
                },
                defaults=defaults,
            ),
            expected_summary={
                "requestType": "localization-image-benchmark",
                "alignment": "index",
                "timeoutMs": 1800,
                "maxFrames": 5,
                "metrics": ("lpips", "ssim"),
                "lpipsNet": "alex",
                "device": "cpu",
                "groundTruthType": "route-capture-bundle",
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def summarize_imported_query_request(imported: ImportedQueryRequest) -> dict[str, Any]:
    """Reduce the imported request to comparable fields for fixture scoring."""
    if imported.request_type == "render":
        if imported.render is None:
            raise ValueError("render import result is missing the render spec")
        render = imported.render
        return {
            "requestType": "render",
            "width": render.width,
            "height": render.height,
            "fovDegrees": render.fov_degrees,
            "pointRadius": render.point_radius,
            "position": render.position,
        }

    if imported.image_benchmark is None:
        raise ValueError("benchmark import result is missing the image benchmark spec")
    benchmark = imported.image_benchmark
    ground_truth_type = (
        benchmark.ground_truth_bundle.get("type") if isinstance(benchmark.ground_truth_bundle, dict) else None
    )
    return {
        "requestType": "localization-image-benchmark",
        "alignment": benchmark.alignment,
        "timeoutMs": benchmark.timeout_ms,
        "maxFrames": benchmark.max_frames,
        "metrics": benchmark.metrics,
        "lpipsNet": benchmark.lpips_net,
        "device": benchmark.device,
        "groundTruthType": ground_truth_type,
    }


def evaluate_query_request_import_fixture(
    policy: QueryRequestImportPolicy,
    fixture: QueryRequestImportFixture,
) -> dict[str, Any]:
    """Run one import policy on one canonical query payload."""
    started_at = perf_counter()
    try:
        imported = policy.import_request(fixture.request)
        summary = summarize_imported_query_request(imported)
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
    expected_keys = tuple(fixture.expected_summary)
    matched_keys = [summary.get(key) == fixture.expected_summary[key] for key in expected_keys]
    match_score = float(sum(matched_keys) / max(1, len(expected_keys)))
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "requestType": summary["requestType"],
        "matchScore": match_score,
        "exactMatch": match_score >= 0.999,
        "summary": summary,
        "runtimeMs": runtime_ms,
    }


def benchmark_query_request_import_policy_runtime(
    policy: QueryRequestImportPolicy,
    fixtures: Sequence[QueryRequestImportFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure query request import policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                policy.import_request(fixture.request)
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


def evaluate_readability(policy: QueryRequestImportPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.import_request))
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


def evaluate_extensibility(policy: QueryRequestImportPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "respectsCanonicalSchema": 2.0,
        "supportsEnvelopeWrappers": 2.5,
        "supportsAliasKeys": 3.0,
        "supportsPoseShortcuts": 2.5,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_query_request_import_policy(
    policy: QueryRequestImportPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one import policy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    aggregate = {
        "successRate": float(len(successful) / max(1, len(fixture_reports))),
        "exactMatchRate": _mean_or_none([1.0 if report.get("exactMatch") else 0.0 for report in successful]),
        "meanMatchScore": _mean_or_none([report.get("matchScore") for report in successful]),
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


def build_query_request_import_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare query request import policies on shared fixtures."""
    fixtures = build_query_request_import_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "expectedSummary": fixture.expected_summary,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_QUERY_REQUEST_IMPORT_POLICIES:
        fixture_reports = [evaluate_query_request_import_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_request_import_policy_runtime(
            policy,
            fixtures,
            repetitions=repetitions,
        )
        policy_reports.append(summarize_query_request_import_policy(policy, fixture_reports, runtime_report))

    best_fit = max(
        policy_reports,
        key=lambda report: (
            float(report["aggregate"]["successRate"] or 0.0),
            float(report["aggregate"]["meanMatchScore"] or 0.0),
            float(report["aggregate"]["exactMatchRate"] or 0.0),
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
        "type": "query-request-import-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "query-request-import",
            "statement": (
                "Import sim2real render and image benchmark query payloads without freezing one "
                "JSON envelope shape for browser panels, CLI tools, and lightweight clients."
            ),
            "stableInterface": "import_query_request(QueryRequestImportRequest(...))",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": ["successRate", "exactMatchRate", "meanMatchScore"],
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
                "meanMatchScore": best_fit["aggregate"]["meanMatchScore"],
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


def build_query_request_import_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the query request import report into a shared docs section."""
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
                    fixture_report.get("requestType", "n/a"),
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
        "title": "Query Request Import",
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
            "Stable code uses `import_query_request(QueryRequestImportRequest(...))` as the only query payload import contract.",
            "`alias_friendly` is the default production policy because it preserves the canonical schema while still accepting thin wrappers, alias keys, and pose shortcuts from browser and CLI clients.",
            "Alternative request import policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-client fit.",
        ],
        "deferred": [
            "`strict_schema` stays experimental. It is the fastest path, but it rejects lightweight clients that do not send the full canonical envelope.",
            "`envelope_first` stays experimental. It works for wrapped SDK payloads, but it still drops top-level pose shortcuts used by quick local tools.",
        ],
        "rules": [
            "Start query-request work with at least three import policies, not one monolithic parser.",
            "Compare policies on the same render and benchmark payload fixtures before changing production defaults.",
            "Promote only the smallest import contract that preserves the normalized render and benchmark request schema.",
        ],
        "stableInterfaceIntro": "The stable query request import surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class RenderQueryDefaults:
                width: int
                height: int
                fov_degrees: float
                near_clip: float
                far_clip: float
                point_radius: int
                timeout_ms: int = 10_000

            @dataclass(frozen=True)
            class QueryRequestImportRequest:
                payload: Any
                defaults: RenderQueryDefaults

            def import_query_request(
                request: QueryRequestImportRequest,
                *,
                policy: str = 'alias_friendly',
            ) -> ImportedQueryRequest: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`import_request(request) -> ImportedQueryRequest`",
        ],
        "comparableInputs": [
            "Same `QueryRequestImportRequest` fixtures for every policy",
            "Same workload fixtures across canonical render, wrapped render, pose shortcuts, canonical benchmark, and wrapped benchmark payloads",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable query request importer contract for production render-server handlers",
            "`src/gs_sim2real/experiments/`: discardable import policies and comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the query request import lab and optionally refresh docs."""
    report = build_query_request_import_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_request_import_report=report,
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
