"""Experiment-first lab for sim2real query response build policies."""

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

from gs_sim2real.core.query_response_build import (
    CORE_QUERY_RESPONSE_BUILD_POLICIES,
    QueryErrorResponseInput,
    QueryReadyDefaults,
    QueryReadyResponseInput,
    QueryRenderResultResponseInput,
    QueryResponseBuildPolicy,
    QUERY_RESPONSE_PROTOCOL_ID,
)


@dataclass(frozen=True)
class QueryResponseBuildFixture:
    """Canonical response-build workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    response_kind: str
    response_input: QueryRenderResultResponseInput | QueryReadyResponseInput | QueryErrorResponseInput
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_RESPONSE_BUILD_POLICIES: tuple[QueryResponseBuildPolicy, ...] = CORE_QUERY_RESPONSE_BUILD_POLICIES


def build_query_response_build_fixtures() -> list[QueryResponseBuildFixture]:
    """Build shared fixtures for response-build policy comparisons."""
    return [
        QueryResponseBuildFixture(
            fixture_id="render-result-canonical",
            label="Canonical Render Result",
            intent="Preserve the browser-facing render-result payload without losing explicit render settings.",
            response_kind="render-result",
            response_input=QueryRenderResultResponseInput(
                frame_id="dreamwalker_map",
                width=4,
                height=3,
                fov_degrees=70.0,
                near_clip=0.1,
                far_clip=25.0,
                point_radius=2,
                position=(1.0, 2.0, 3.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                camera_info={
                    "frameId": "dreamwalker_map",
                    "width": 4,
                    "height": 3,
                    "distortionModel": "plumb_bob",
                    "d": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "k": [4.0, 0.0, 2.0, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0],
                    "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "p": [4.0, 0.0, 2.0, 0.0, 0.0, 4.0, 1.5, 0.0, 0.0, 0.0, 1.0, 0.0],
                },
                color_jpeg_bytes=b"fake-jpeg-payload",
                depth_float32_bytes=bytes(range(24)),
            ),
            expected_summary={
                "type": "render-result",
                "protocol": QUERY_RESPONSE_PROTOCOL_ID,
                "keys": (
                    "cameraInfo",
                    "colorEncoding",
                    "colorJpegBase64",
                    "depthBase64",
                    "depthEncoding",
                    "farClip",
                    "fovDegrees",
                    "frameId",
                    "height",
                    "nearClip",
                    "pointRadius",
                    "pose",
                    "protocol",
                    "type",
                    "width",
                ),
                "frameId": "dreamwalker_map",
                "width": 4,
                "height": 3,
                "colorEncoding": "jpeg",
                "depthEncoding": "32FC1",
                "cameraInfoFrameId": "dreamwalker_map",
                "position": (1.0, 2.0, 3.0),
                "metaKeys": (),
            },
        ),
        QueryResponseBuildFixture(
            fixture_id="query-ready-canonical",
            label="Canonical Query Ready",
            intent="Keep the websocket handshake explicit enough for browser clients to discover defaults and request types.",
            response_kind="query-ready",
            response_input=QueryReadyResponseInput(
                transport="ws",
                endpoint="ws://127.0.0.1:8781/sim2real",
                frame_id="dreamwalker_map",
                renderer="gsplat",
                renderer_reason="auto-selected because gsplat, CUDA, and Gaussian PLY parameters are available",
                request_types=("render", "localization-image-benchmark"),
                defaults=QueryReadyDefaults(
                    width=1280,
                    height=720,
                    fov_degrees=60.0,
                    near_clip=0.05,
                    far_clip=50.0,
                    point_radius=1,
                ),
            ),
            expected_summary={
                "type": "query-ready",
                "protocol": QUERY_RESPONSE_PROTOCOL_ID,
                "keys": (
                    "defaults",
                    "endpoint",
                    "frameId",
                    "protocol",
                    "renderer",
                    "rendererReason",
                    "requestTypes",
                    "transport",
                    "type",
                ),
                "transport": "ws",
                "endpoint": "ws://127.0.0.1:8781/sim2real",
                "frameId": "dreamwalker_map",
                "renderer": "gsplat",
                "requestTypes": ("render", "localization-image-benchmark"),
                "defaultsKeys": ("farClip", "fovDegrees", "height", "nearClip", "pointRadius", "width"),
                "metaKeys": (),
            },
        ),
        QueryResponseBuildFixture(
            fixture_id="error-canonical",
            label="Canonical Error Payload",
            intent="Keep transport errors safe for clients while avoiding protocol drift.",
            response_kind="error",
            response_input=QueryErrorResponseInput(
                error="query timed out while waiting for the render thread",
                error_type="TimeoutError",
                error_code="query_timeout",
            ),
            expected_summary={
                "type": "error",
                "protocol": QUERY_RESPONSE_PROTOCOL_ID,
                "keys": ("error", "protocol", "type"),
                "error": "query timed out while waiting for the render thread",
                "metaKeys": (),
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def summarize_query_response_document(response: dict[str, Any]) -> dict[str, Any]:
    """Reduce a built response to comparable shape fields."""
    summary = {
        "type": response.get("type"),
        "protocol": response.get("protocol"),
        "keys": tuple(sorted(response.keys())),
        "metaKeys": tuple(sorted(response.get("meta", {}).keys())) if isinstance(response.get("meta"), dict) else (),
    }
    if response.get("type") == "render-result":
        pose = response.get("pose") if isinstance(response.get("pose"), dict) else {}
        camera_info = response.get("cameraInfo") if isinstance(response.get("cameraInfo"), dict) else {}
        summary.update(
            {
                "frameId": response.get("frameId"),
                "width": response.get("width"),
                "height": response.get("height"),
                "colorEncoding": response.get("colorEncoding"),
                "depthEncoding": response.get("depthEncoding"),
                "cameraInfoFrameId": camera_info.get("frameId"),
                "position": tuple(pose.get("position", ())),
            }
        )
    elif response.get("type") == "query-ready":
        defaults = response.get("defaults") if isinstance(response.get("defaults"), dict) else {}
        summary.update(
            {
                "transport": response.get("transport"),
                "endpoint": response.get("endpoint"),
                "frameId": response.get("frameId"),
                "renderer": response.get("renderer"),
                "requestTypes": tuple(response.get("requestTypes", ())),
                "defaultsKeys": tuple(sorted(defaults.keys())),
            }
        )
    elif response.get("type") == "error":
        summary["error"] = response.get("error")
    return summary


def _build_fixture_response(
    policy: QueryResponseBuildPolicy,
    fixture: QueryResponseBuildFixture,
) -> dict[str, Any]:
    if fixture.response_kind == "render-result":
        assert isinstance(fixture.response_input, QueryRenderResultResponseInput)
        return policy.build_render_result(fixture.response_input)
    if fixture.response_kind == "query-ready":
        assert isinstance(fixture.response_input, QueryReadyResponseInput)
        return policy.build_query_ready(fixture.response_input)
    assert isinstance(fixture.response_input, QueryErrorResponseInput)
    return policy.build_query_error(fixture.response_input)


def evaluate_query_response_build_fixture(
    policy: QueryResponseBuildPolicy,
    fixture: QueryResponseBuildFixture,
) -> dict[str, Any]:
    """Run one response-build policy on one canonical response source."""
    started_at = perf_counter()
    try:
        response = _build_fixture_response(policy, fixture)
        summary = summarize_query_response_document(response)
    except Exception as exc:
        return {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "responseKind": fixture.response_kind,
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
        "responseKind": fixture.response_kind,
        "status": "ok",
        "matchScore": match_score,
        "exactMatch": match_score >= 0.999,
        "summary": summary,
        "runtimeMs": runtime_ms,
    }


def benchmark_query_response_build_policy_runtime(
    policy: QueryResponseBuildPolicy,
    fixtures: Sequence[QueryResponseBuildFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure response-build runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                _build_fixture_response(policy, fixture)
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


def evaluate_readability(policy: QueryResponseBuildPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.__class__))
    tree = ast.parse(source)
    branch_count = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in ast.walk(tree))
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("class ", "def ", '"""', "'''", "#"))
    ]
    lines_of_code = len(lines)
    score = max(1.0, 10.0 - max(0, lines_of_code - 18) * 0.15 - max(0, branch_count - 3) * 0.8)
    return {
        "score": round(score, 1),
        "linesOfCode": lines_of_code,
        "branchCount": branch_count,
    }


def evaluate_extensibility(policy: QueryResponseBuildPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "preservesCanonicalProtocol": 2.0,
        "includesRenderSettings": 2.5,
        "includesRequestCatalog": 2.5,
        "includesDiagnosticMeta": 3.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_query_response_build_policy(
    policy: QueryResponseBuildPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one policy."""
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


def build_query_response_build_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare query-response build policies on shared fixtures."""
    fixtures = build_query_response_build_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "responseKind": fixture.response_kind,
            "expectedSummary": fixture.expected_summary,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_QUERY_RESPONSE_BUILD_POLICIES:
        fixture_reports = [evaluate_query_response_build_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_response_build_policy_runtime(
            policy,
            fixtures,
            repetitions=repetitions,
        )
        policy_reports.append(summarize_query_response_build_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-response-build-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "query-response-build",
            "statement": (
                "Build sim2real query responses in a way that stays comparable across websocket, "
                "browser, and CLI clients without hard-wiring one monolithic server helper."
            ),
            "stableInterface": (
                "build_render_result_response_document(...), "
                "build_query_ready_response_document(...), "
                "build_query_error_response_document(...)"
            ),
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


def build_query_response_build_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the response-build report into a shared docs section."""
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
                    fixture_report.get("responseKind", "n/a"),
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
        "title": "Query Response Build",
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
            "Stable server code uses `build_render_result_response_document(...)`, `build_query_ready_response_document(...)`, and `build_query_error_response_document(...)` as the only response-build surface.",
            "`browser_observable` is the production default because it preserves the current browser-facing query payload shape while keeping websocket/CLI clients explicit.",
            "Alternative response-build policies stay experimental until the same render-result, query-ready, and error fixtures show a better fit.",
        ],
        "deferred": [
            "`minimal_envelope` stays experimental. It is small and fast, but it drops render settings and request-catalog details the web client uses.",
            "`diagnostic_meta` stays experimental. It is a useful telemetry direction, but it expands the payload shape without a proven client need.",
        ],
        "rules": [
            "Compare at least three response-build policies before expanding server-side websocket payload helpers.",
            "Use the same render-result, query-ready, and error fixtures for every candidate policy.",
            "Keep production dependent only on the stable core builder functions, not on experiment-only policy details.",
        ],
        "stableInterfaceIntro": "The stable query response build surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            build_render_result_response_document(
                response_input,
                policy = 'browser_observable',
            ) -> dict[str, Any]

            build_query_ready_response_document(
                response_input,
                policy = 'browser_observable',
            ) -> dict[str, Any]

            build_query_error_response_document(
                response_input,
                policy = 'browser_observable',
            ) -> dict[str, Any]
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`build_render_result(response_input) -> dict[str, Any]`",
            "`build_query_ready(response_input) -> dict[str, Any]`",
            "`build_query_error(response_input) -> dict[str, Any]`",
        ],
        "comparableInputs": [
            "Same render-result fixture for every policy",
            "Same query-ready handshake fixture for every policy",
            "Same transport error fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable query-response build contract used by production servers",
            "`src/gs_sim2real/experiments/`: discardable response-build comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the query-response build lab and optionally refresh docs."""
    report = build_query_response_build_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_response_build_report=report,
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


__all__ = [
    "EXPERIMENT_QUERY_RESPONSE_BUILD_POLICIES",
    "build_query_response_build_experiment_report",
    "build_query_response_build_process_section",
    "build_query_response_build_fixtures",
    "run_cli",
]
