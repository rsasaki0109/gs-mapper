"""Experiment-first lab for sim2real query timeout policies."""

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

from gs_sim2real.core.query_timeout_policy import (
    CORE_QUERY_TIMEOUT_POLICIES,
    QueryTimeoutPlan,
    QueryTimeoutPolicy,
    QueryTimeoutPolicyRequest,
)


@dataclass(frozen=True)
class QueryTimeoutFixture:
    """Canonical timeout-policy workload shared by every strategy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryTimeoutPolicyRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_TIMEOUT_POLICIES: tuple[QueryTimeoutPolicy, ...] = CORE_QUERY_TIMEOUT_POLICIES


def build_query_timeout_fixtures() -> list[QueryTimeoutFixture]:
    """Build shared fixtures for timeout-policy comparisons."""
    return [
        QueryTimeoutFixture(
            fixture_id="render-default-ws",
            label="Render Default WebSocket",
            intent="Keep normal render requests responsive while allowing one retry budget for transient browser transport stalls.",
            request=QueryTimeoutPolicyRequest(
                request_type="render",
                transport="ws",
                explicit_client_timeout_ms=10_000,
                allow_retry=True,
            ),
            expected_summary={
                "serverTimeoutSeconds": 30.0,
                "clientTimeoutMs": 10_000,
                "attemptTimeoutMs": 4_925,
                "maxAttempts": 2,
                "retryBackoffMs": 150,
            },
        ),
        QueryTimeoutFixture(
            fixture_id="benchmark-workload-floor",
            label="Benchmark Workload Floor",
            intent="Raise long image benchmark deadlines when the declared workload would outgrow a small explicit hint.",
            request=QueryTimeoutPolicyRequest(
                request_type="localization-image-benchmark",
                transport="ws",
                explicit_server_timeout_seconds=45.0,
                expected_work_units=12,
                allow_retry=False,
            ),
            expected_summary={
                "serverTimeoutSeconds": 96.0,
                "clientTimeoutMs": 101_000,
                "attemptTimeoutMs": 101_000,
                "maxAttempts": 1,
                "retryBackoffMs": 0,
            },
        ),
        QueryTimeoutFixture(
            fixture_id="bounded-explicit-render-hint",
            label="Bounded Explicit Render Hint",
            intent="Clamp large explicit server hints while preserving an intentionally short CLI-side render timeout.",
            request=QueryTimeoutPolicyRequest(
                request_type="render",
                transport="tcp",
                explicit_server_timeout_seconds=400.0,
                explicit_client_timeout_ms=2_500,
                allow_retry=True,
            ),
            expected_summary={
                "serverTimeoutSeconds": 300.0,
                "clientTimeoutMs": 2_500,
                "attemptTimeoutMs": 2_500,
                "maxAttempts": 1,
                "retryBackoffMs": 0,
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def summarize_query_timeout_plan(plan: QueryTimeoutPlan) -> dict[str, Any]:
    """Reduce a timeout plan to comparable fields."""
    return {
        "serverTimeoutSeconds": float(plan.server_timeout_seconds),
        "clientTimeoutMs": int(plan.client_timeout_ms),
        "attemptTimeoutMs": int(plan.attempt_timeout_ms),
        "maxAttempts": int(plan.max_attempts),
        "retryBackoffMs": int(plan.retry_backoff_ms),
    }


def evaluate_query_timeout_fixture(
    policy: QueryTimeoutPolicy,
    fixture: QueryTimeoutFixture,
) -> dict[str, Any]:
    """Run one timeout policy on one canonical request."""
    started_at = perf_counter()
    try:
        plan = policy.resolve_timeout_plan(fixture.request)
        summary = summarize_query_timeout_plan(plan)
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


def benchmark_query_timeout_policy_runtime(
    policy: QueryTimeoutPolicy,
    fixtures: Sequence[QueryTimeoutFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure timeout-policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                policy.resolve_timeout_plan(fixture.request)
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


def evaluate_readability(policy: QueryTimeoutPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.resolve_timeout_plan))
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


def evaluate_extensibility(policy: QueryTimeoutPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "respectsExplicitServerHint": 2.5,
        "scalesForWorkload": 3.0,
        "supportsRetryBudget": 2.5,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_query_timeout_policy(
    policy: QueryTimeoutPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one timeout policy."""
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


def build_query_timeout_policy_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare query timeout policies on shared fixtures."""
    fixtures = build_query_timeout_fixtures()
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
    for policy in EXPERIMENT_QUERY_TIMEOUT_POLICIES:
        fixture_reports = [evaluate_query_timeout_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_timeout_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_timeout_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-timeout-policy-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "query-timeout-policy",
            "statement": (
                "Resolve sim2real query timeouts and retry budgets without hard-coding one fixed deadline for "
                "browser renders, CLI renders, and image benchmark workloads."
            ),
            "stableInterface": "resolve_query_timeout_plan(QueryTimeoutPolicyRequest(...))",
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


def build_query_timeout_policy_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the timeout-policy report into a shared docs section."""
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
                    fixture_report.get("summary", {}).get("maxAttempts", "n/a"),
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Attempts"],
                "rows": rows,
            }
        )

    return {
        "title": "Query Timeout Policy",
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
            "Stable server/client code uses `resolve_query_timeout_plan(QueryTimeoutPolicyRequest(...))` as the only timeout-policy surface.",
            "`workload_aware_retry` is the default production policy because it preserves normal render latency, scales benchmark deadlines with work size, and keeps a small retry budget for websocket renders.",
            "Alternative timeout strategies stay outside production until the same render-default, benchmark-workload, and bounded-explicit fixtures show a better fit.",
        ],
        "deferred": [
            "`fixed_deadline` stays experimental. It is simple, but it underfits long benchmark workloads and leaves websocket render retries unavailable.",
            "`hint_bounded` stays experimental. It respects hints, but it still trusts undersized benchmark deadlines too readily.",
        ],
        "rules": [
            "Compare at least three timeout policies before changing query deadlines in transport code.",
            "Use the same render and benchmark fixtures for every candidate policy.",
            "Keep transport loops dependent only on the resolved timeout plan, not on policy-specific branching.",
        ],
        "stableInterfaceIntro": "The stable query timeout surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            resolve_query_timeout_plan(
                QueryTimeoutPolicyRequest(...),
                policy = 'workload_aware_retry',
            ) -> QueryTimeoutPlan
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`resolve_timeout_plan(request) -> QueryTimeoutPlan`",
        ],
        "comparableInputs": [
            "Same render-default fixture for every policy",
            "Same benchmark workload-floor fixture for every policy",
            "Same bounded explicit-hint fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable timeout-policy contract used by query transports and clients",
            "`src/gs_sim2real/experiments/`: discardable timeout-policy comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the query-timeout lab and optionally refresh docs."""
    report = build_query_timeout_policy_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_timeout_policy_report=report,
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
    "EXPERIMENT_QUERY_TIMEOUT_POLICIES",
    "build_query_timeout_fixtures",
    "build_query_timeout_policy_experiment_report",
    "build_query_timeout_policy_process_section",
    "run_cli",
]
