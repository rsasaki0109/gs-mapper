"""Experiment-first lab for sim2real query cancellation policies."""

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

from gs_sim2real.core.query_cancellation_policy import (
    CORE_QUERY_CANCELLATION_POLICIES,
    QueryCancellationPolicy,
    QueryCancellationRequest,
    resolve_query_cancellation_decision,
)
from gs_sim2real.core.query_queue_policy import QueuedQueryItem


@dataclass(frozen=True)
class QueryCancellationFixture:
    """Canonical cancellation workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryCancellationRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_CANCELLATION_POLICIES: tuple[QueryCancellationPolicy, ...] = CORE_QUERY_CANCELLATION_POLICIES


def _item(request_id: str, request_type: str, source_id: str, order: int) -> QueuedQueryItem:
    return QueuedQueryItem(
        request_id=request_id,
        request_type=request_type,
        transport="ws",
        submitted_order=order,
        source_id=source_id,
    )


def build_query_cancellation_policy_fixtures() -> list[QueryCancellationFixture]:
    """Build shared fixtures for cancellation policy comparisons."""
    return [
        QueryCancellationFixture(
            fixture_id="timeout-target-only",
            label="Timeout Target Only",
            intent="A timed-out queued render should be removed instead of lingering forever.",
            request=QueryCancellationRequest(
                pending_items=(_item("render-1", "render", "socket-a", 1),),
                event="timeout",
                target_request_id="render-1",
                source_id="socket-a",
            ),
            expected_summary={
                "canceledRequestIds": ("render-1",),
            },
        ),
        QueryCancellationFixture(
            fixture_id="disconnect-clears-source-backlog",
            label="Disconnect Clears Source Backlog",
            intent="A disconnected source should not leave its whole queued backlog behind.",
            request=QueryCancellationRequest(
                pending_items=(
                    _item("render-1", "render", "socket-a", 1),
                    _item("benchmark-1", "localization-image-benchmark", "socket-a", 2),
                    _item("render-2", "render", "socket-b", 3),
                ),
                event="connection_closed",
                target_request_id="render-1",
                source_id="socket-a",
            ),
            expected_summary={
                "canceledRequestIds": ("render-1", "benchmark-1"),
            },
        ),
        QueryCancellationFixture(
            fixture_id="shutdown-drains-everything",
            label="Shutdown Drains Everything",
            intent="Server shutdown must drain every queued request regardless of source.",
            request=QueryCancellationRequest(
                pending_items=(
                    _item("render-1", "render", "socket-a", 1),
                    _item("benchmark-1", "localization-image-benchmark", "socket-b", 2),
                ),
                event="shutdown",
            ),
            expected_summary={
                "canceledRequestIds": ("render-1", "benchmark-1"),
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_cancellation_fixture(
    policy: QueryCancellationPolicy,
    fixture: QueryCancellationFixture,
) -> dict[str, Any]:
    """Run one cancellation policy on one canonical workload."""
    started_at = perf_counter()
    try:
        decision = resolve_query_cancellation_decision(fixture.request, policy=policy)
        summary = {"canceledRequestIds": decision.canceled_request_ids}
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


def benchmark_query_cancellation_policy_runtime(
    policy: QueryCancellationPolicy,
    fixtures: Sequence[QueryCancellationFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure cancellation policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            resolve_query_cancellation_decision(fixture.request, policy=policy)
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))
    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: QueryCancellationPolicy) -> dict[str, Any]:
    source = textwrap.dedent(inspect.getsource(policy.cancel))
    tree = ast.parse(source)
    branch_count = sum(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.Match)) for node in ast.walk(tree))
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("def ", '"""', "'''", "#"))
    ]
    lines_of_code = len(lines)
    score = max(1.0, 10.0 - max(0, lines_of_code - 8) * 0.2 - max(0, branch_count - 2) * 0.8)
    return {"score": round(score, 1), "linesOfCode": lines_of_code, "branchCount": branch_count}


def evaluate_extensibility(policy: QueryCancellationPolicy) -> dict[str, Any]:
    weights = {
        "cancelsRequestedOnly": 2.5,
        "cancelsSourceBacklog": 3.0,
        "cancelsOnShutdown": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {"score": round(score, 1), "supportedCapabilities": supported}


def summarize_query_cancellation_policy(
    policy: QueryCancellationPolicy,
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


def build_query_cancellation_policy_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    fixtures = build_query_cancellation_policy_fixtures()
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
    for policy in EXPERIMENT_QUERY_CANCELLATION_POLICIES:
        fixture_reports = [evaluate_query_cancellation_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_cancellation_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_cancellation_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-cancellation-policy-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "query-cancellation-policy",
            "statement": (
                "Cancel orphaned queued sim2real work intentionally instead of scattering timeout and disconnect rules across transport code."
            ),
            "stableInterface": "resolve_query_cancellation_decision(QueryCancellationRequest(...))",
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


def build_query_cancellation_policy_process_section(report: dict[str, Any]) -> dict[str, Any]:
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
                    ",".join(fixture_report.get("summary", {}).get("canceledRequestIds", ())) or "none",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Canceled"],
                "rows": rows,
            }
        )
    return {
        "title": "Query Cancellation Policy",
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
            "Stable transport code uses `resolve_query_cancellation_decision(QueryCancellationRequest(...))` as the only cancellation surface.",
            "`cancel_source_backlog` is the production default because disconnected sources should not leave stale queued work behind.",
            "Alternative cancellation strategies stay experimental until the same timeout, disconnect, and shutdown fixtures show a better fit.",
        ],
        "deferred": [
            "`ignore_orphaned` stays experimental. It is simple, but it allows dead queues to accumulate.",
            "`cancel_requested_only` stays experimental. It improves single-request timeouts but still leaves same-source backlog behind.",
        ],
        "rules": [
            "Compare at least three cancellation strategies before changing timeout/disconnect cleanup in transport code.",
            "Use the same timeout, disconnect, and shutdown fixtures for every policy.",
            "Keep transport code dependent only on cancellation decisions, not on policy-specific branching.",
        ],
        "stableInterfaceIntro": "The stable query cancellation surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            resolve_query_cancellation_decision(
                QueryCancellationRequest(...),
                policy = 'cancel_source_backlog',
            ) -> QueryCancellationDecision
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`cancel(request) -> QueryCancellationDecision`",
        ],
        "comparableInputs": [
            "Same timeout fixture for every policy",
            "Same disconnect fixture for every policy",
            "Same shutdown fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable cancellation-policy contract used by interactive queue stores",
            "`src/gs_sim2real/experiments/`: discardable cancellation-policy comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    report = build_query_cancellation_policy_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_cancellation_policy_report=report,
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
    "EXPERIMENT_QUERY_CANCELLATION_POLICIES",
    "build_query_cancellation_policy_experiment_report",
    "build_query_cancellation_policy_fixtures",
    "build_query_cancellation_policy_process_section",
    "run_cli",
]
