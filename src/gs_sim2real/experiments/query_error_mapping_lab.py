"""Experiment-first lab for sim2real query error-mapping policies."""

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

from gs_sim2real.core.query_error_mapping import (
    CORE_QUERY_ERROR_MAPPING_POLICIES,
    QueryErrorMappingPolicy,
    QueryErrorMappingRequest,
    resolve_query_error_mapping,
)


@dataclass(frozen=True)
class QueryErrorMappingFixture:
    """Canonical error-mapping workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryErrorMappingRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_ERROR_MAPPING_POLICIES: tuple[QueryErrorMappingPolicy, ...] = CORE_QUERY_ERROR_MAPPING_POLICIES


def build_query_error_mapping_fixtures() -> list[QueryErrorMappingFixture]:
    """Build shared fixtures for error-mapping policy comparisons."""
    return [
        QueryErrorMappingFixture(
            fixture_id="invalid-json",
            label="Invalid JSON",
            intent="Malformed websocket payloads should get a canonical parse error without transport-specific branching.",
            request=QueryErrorMappingRequest(
                event="invalid_json",
                detail="Expecting value: line 1 column 1 (char 0)",
                exception_type="JSONDecodeError",
                transport="ws",
            ),
            expected_summary={
                "error": "invalid JSON request: Expecting value: line 1 column 1 (char 0)",
                "errorType": "JSONDecodeError",
                "errorCode": "invalid_json_request",
            },
        ),
        QueryErrorMappingFixture(
            fixture_id="queue-rejected",
            label="Queue Rejected",
            intent="Rejected requests should report both the canonical failure and the policy reason.",
            request=QueryErrorMappingRequest(
                event="queue_rejected",
                reason="incoming request is lower priority than current queue contents",
                request_type="render",
                transport="ws",
            ),
            expected_summary={
                "error": "query queue rejected request: incoming request is lower priority than current queue contents",
                "errorType": "RuntimeError",
                "errorCode": "query_queue_rejected",
            },
        ),
        QueryErrorMappingFixture(
            fixture_id="queue-dropped",
            label="Queue Dropped",
            intent="Evicted queued work should explain that it was superseded instead of silently disappearing.",
            request=QueryErrorMappingRequest(
                event="queue_dropped",
                reason="evicted lower-priority queued work in favor of an interactive request",
                request_type="localization-image-benchmark",
                transport="ws",
            ),
            expected_summary={
                "error": "query dropped from queue: evicted lower-priority queued work in favor of an interactive request",
                "errorType": "RuntimeError",
                "errorCode": "query_queue_dropped",
            },
        ),
        QueryErrorMappingFixture(
            fixture_id="timeout",
            label="Timeout",
            intent="Queue wait timeouts should stay transport-safe and deterministic.",
            request=QueryErrorMappingRequest(
                event="query_timeout",
                transport="ws",
                request_type="render",
            ),
            expected_summary={
                "error": "query timed out while waiting for the render thread",
                "errorType": "TimeoutError",
                "errorCode": "query_timeout",
            },
        ),
        QueryErrorMappingFixture(
            fixture_id="shutdown",
            label="Server Shutdown",
            intent="Transport shutdown should surface a stable reconnect-safe error.",
            request=QueryErrorMappingRequest(
                event="server_shutdown",
                transport="ws",
            ),
            expected_summary={
                "error": "render query server is shutting down",
                "errorType": "RuntimeError",
                "errorCode": "query_server_shutdown",
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_error_mapping_fixture(
    policy: QueryErrorMappingPolicy,
    fixture: QueryErrorMappingFixture,
) -> dict[str, Any]:
    """Run one error-mapping policy on one canonical workload."""
    started_at = perf_counter()
    try:
        decision = resolve_query_error_mapping(fixture.request, policy=policy)
        summary = {
            "error": decision.error,
            "errorType": decision.error_type,
            "errorCode": decision.error_code,
        }
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
        "matchScore": match_score,
        "exactMatch": match_score >= 0.999,
        "summary": summary,
        "runtimeMs": runtime_ms,
    }


def benchmark_query_error_mapping_policy_runtime(
    policy: QueryErrorMappingPolicy,
    fixtures: Sequence[QueryErrorMappingFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure error-mapping policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            resolve_query_error_mapping(fixture.request, policy=policy)
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))
    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: QueryErrorMappingPolicy) -> dict[str, Any]:
    source = textwrap.dedent(inspect.getsource(policy.map_error))
    tree = ast.parse(source)
    branch_count = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in ast.walk(tree))
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", '"""', "'''", "#"))
    ]
    lines_of_code = len(lines)
    score = max(1.0, 10.0 - max(0, lines_of_code - 10) * 0.18 - max(0, branch_count - 3) * 0.8)
    return {"score": round(score, 1), "linesOfCode": lines_of_code, "branchCount": branch_count}


def evaluate_extensibility(policy: QueryErrorMappingPolicy) -> dict[str, Any]:
    weights = {
        "emitsStableCodes": 3.0,
        "preservesLiteralDetail": 2.5,
        "addsActionHints": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {"score": round(score, 1), "supportedCapabilities": supported}


def summarize_query_error_mapping_policy(
    policy: QueryErrorMappingPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
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


def build_query_error_mapping_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare error-mapping policies on shared fixtures."""
    fixtures = build_query_error_mapping_fixtures()
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
    for policy in EXPERIMENT_QUERY_ERROR_MAPPING_POLICIES:
        fixture_reports = [evaluate_query_error_mapping_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_error_mapping_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_error_mapping_policy(policy, fixture_reports, runtime_report))

    best_fit = max(
        policy_reports,
        key=lambda report: (
            float(report["aggregate"]["successRate"] or 0.0),
            float(report["aggregate"]["meanMatchScore"] or 0.0),
            float(report["aggregate"]["exactMatchRate"] or 0.0),
        ),
    )
    fastest = min(policy_reports, key=lambda report: float(report["runtime"]["medianMs"]))
    most_readable = max(policy_reports, key=lambda report: float(report["readability"]["score"]))
    most_extensible = max(policy_reports, key=lambda report: float(report["extensibility"]["score"]))
    return {
        "protocol": "gs-sim2real-experiment-report/v1",
        "type": "query-error-mapping-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "query-error-mapping",
            "statement": (
                "Map queue, timeout, parse, and shutdown failures intentionally instead of scattering ad hoc "
                "error strings across websocket transport code."
            ),
            "stableInterface": "resolve_query_error_mapping(QueryErrorMappingRequest(...))",
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


def build_query_error_mapping_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the error-mapping report into a shared docs section."""
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
                    f"{float(fixture_report.get('matchScore') or 0.0):.3f}"
                    if fixture_report["status"] == "ok"
                    else "n/a",
                    "yes" if fixture_report.get("exactMatch") else "no",
                    fixture_report.get("summary", {}).get("errorCode", "n/a"),
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Error Code"],
                "rows": rows,
            }
        )
    return {
        "title": "Query Error Mapping",
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
            "Stable transport code uses `resolve_query_error_mapping(QueryErrorMappingRequest(...))` as the only error-mapping surface.",
            "`structured_codes` is the production default because it keeps browser messages stable while preserving actionable queue-policy reasons.",
            "Alternative error-mapping strategies stay experimental until the same parse, queue, timeout, and shutdown fixtures show a better fit.",
        ],
        "deferred": [
            "`literal_passthrough` stays experimental. It preserves detail, but its messages drift with call sites and undermine comparability.",
            "`action_hint` stays experimental. It is friendly, but it changes canonical strings in ways that make transport regressions harder to diff.",
        ],
        "rules": [
            "Compare at least three error-mapping strategies before changing transport-visible failure messages.",
            "Use the same parse, queue, timeout, and shutdown fixtures for every policy.",
            "Keep websocket transport code dependent only on mapped error decisions, not on hand-written event strings.",
        ],
        "stableInterfaceIntro": "The stable query error-mapping surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            resolve_query_error_mapping(
                QueryErrorMappingRequest(...),
                policy = 'structured_codes',
            ) -> QueryErrorMappingDecision
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`map_error(request) -> QueryErrorMappingDecision`",
        ],
        "comparableInputs": [
            "Same invalid JSON fixture for every policy",
            "Same queue reject and queue drop fixtures for every policy",
            "Same timeout and shutdown fixtures for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable error-mapping contract used by interactive transports",
            "`src/gs_sim2real/experiments/`: discardable error-mapping comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the error-mapping lab and optionally refresh docs."""
    report = build_query_error_mapping_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_error_mapping_report=report,
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
    "EXPERIMENT_QUERY_ERROR_MAPPING_POLICIES",
    "build_query_error_mapping_experiment_report",
    "build_query_error_mapping_fixtures",
    "build_query_error_mapping_process_section",
    "run_cli",
]
