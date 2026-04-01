"""Experiment-first lab for sim2real query dedupe/coalescing policies."""

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

from gs_sim2real.core.query_coalescing_policy import (
    CORE_QUERY_COALESCING_POLICIES,
    QueryCoalescingPolicy,
    QueryCoalescingRequest,
    resolve_query_coalescing_decision,
)
from gs_sim2real.core.query_queue_policy import QueuedQueryItem


@dataclass(frozen=True)
class QueryCoalescingFixture:
    """Canonical coalescing workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryCoalescingRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_COALESCING_POLICIES: tuple[QueryCoalescingPolicy, ...] = CORE_QUERY_COALESCING_POLICIES


def _item(request_id: str, request_type: str, source_id: str, order: int, *, dedupe_key: str = "") -> QueuedQueryItem:
    return QueuedQueryItem(
        request_id=request_id,
        request_type=request_type,
        transport="ws",
        submitted_order=order,
        source_id=source_id,
        dedupe_key=dedupe_key,
    )


def build_query_coalescing_policy_fixtures() -> list[QueryCoalescingFixture]:
    """Build shared fixtures for coalescing policy comparisons."""
    return [
        QueryCoalescingFixture(
            fixture_id="benchmark-plus-render",
            label="Benchmark Plus Render",
            intent="A render should coexist with queued background benchmark work.",
            request=QueryCoalescingRequest(
                pending_items=(_item("benchmark-1", "localization-image-benchmark", "socket-a", 1),),
                incoming_item=_item("render-1", "render", "socket-a", 2, dedupe_key="pose-a"),
            ),
            expected_summary={
                "accepted": True,
                "pendingRequestIds": ("benchmark-1", "render-1"),
                "evictedRequestIds": (),
            },
        ),
        QueryCoalescingFixture(
            fixture_id="duplicate-render-same-source",
            label="Duplicate Render Same Source",
            intent="If the same source repeats a render, the latest request should replace the older pending one.",
            request=QueryCoalescingRequest(
                pending_items=(_item("render-1", "render", "socket-a", 1, dedupe_key="pose-a"),),
                incoming_item=_item("render-2", "render", "socket-a", 2, dedupe_key="pose-a"),
            ),
            expected_summary={
                "accepted": True,
                "pendingRequestIds": ("render-2",),
                "evictedRequestIds": ("render-1",),
            },
        ),
        QueryCoalescingFixture(
            fixture_id="latest-render-replaces-older-same-source",
            label="Latest Render Replaces Older Same Source",
            intent="A newer render from the same source should replace older pending renders even when the pose changed.",
            request=QueryCoalescingRequest(
                pending_items=(
                    _item("render-1", "render", "socket-a", 1, dedupe_key="pose-a"),
                    _item("render-b", "render", "socket-b", 2, dedupe_key="pose-b"),
                ),
                incoming_item=_item("render-2", "render", "socket-a", 3, dedupe_key="pose-c"),
            ),
            expected_summary={
                "accepted": True,
                "pendingRequestIds": ("render-b", "render-2"),
                "evictedRequestIds": ("render-1",),
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_coalescing_fixture(
    policy: QueryCoalescingPolicy,
    fixture: QueryCoalescingFixture,
) -> dict[str, Any]:
    """Run one coalescing policy on one canonical workload."""
    started_at = perf_counter()
    try:
        decision = resolve_query_coalescing_decision(fixture.request, policy=policy)
        summary = {
            "accepted": decision.accepted,
            "pendingRequestIds": decision.pending_request_ids,
            "evictedRequestIds": decision.evicted_request_ids,
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


def benchmark_query_coalescing_policy_runtime(
    policy: QueryCoalescingPolicy,
    fixtures: Sequence[QueryCoalescingFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            resolve_query_coalescing_decision(fixture.request, policy=policy)
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))
    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: QueryCoalescingPolicy) -> dict[str, Any]:
    source = textwrap.dedent(inspect.getsource(policy.coalesce))
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


def evaluate_extensibility(policy: QueryCoalescingPolicy) -> dict[str, Any]:
    weights = {
        "dedupesExactRender": 2.5,
        "replacesOlderRenderFromSource": 3.0,
        "preservesBackgroundBenchmark": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {"score": round(score, 1), "supportedCapabilities": supported}


def summarize_query_coalescing_policy(
    policy: QueryCoalescingPolicy,
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


def build_query_coalescing_policy_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    fixtures = build_query_coalescing_policy_fixtures()
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
    for policy in EXPERIMENT_QUERY_COALESCING_POLICIES:
        fixture_reports = [evaluate_query_coalescing_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_coalescing_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_coalescing_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-coalescing-policy-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "query-coalescing-policy",
            "statement": (
                "Coalesce duplicate interactive render requests intentionally instead of letting queues fill with obsolete previews."
            ),
            "stableInterface": "resolve_query_coalescing_decision(QueryCoalescingRequest(...))",
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


def build_query_coalescing_policy_process_section(report: dict[str, Any]) -> dict[str, Any]:
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
                    ",".join(fixture_report.get("summary", {}).get("evictedRequestIds", ())) or "none",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Evicted"],
                "rows": rows,
            }
        )
    return {
        "title": "Query Coalescing Policy",
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
            "Stable queue code uses `resolve_query_coalescing_decision(QueryCoalescingRequest(...))` as the only dedupe/coalescing surface.",
            "`latest_render_per_source` is the production default because browser previews care about the latest render, not every intermediate one.",
            "Alternative coalescing strategies stay experimental until the same coexistence, duplicate, and replace fixtures show a better fit.",
        ],
        "deferred": [
            "`keep_all` stays experimental. It is simple, but it keeps obsolete preview renders around.",
            "`exact_render_drop_new` stays experimental. It dedupes exact duplicates, but it still keeps older same-source renders when the pose changed.",
        ],
        "rules": [
            "Compare at least three coalescing strategies before changing queue admission for interactive render requests.",
            "Use the same coexistence, duplicate, and replace fixtures for every policy.",
            "Keep queue stores dependent only on coalescing decisions, not on policy-specific branching.",
        ],
        "stableInterfaceIntro": "The stable query coalescing surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            resolve_query_coalescing_decision(
                QueryCoalescingRequest(...),
                policy = 'latest_render_per_source',
            ) -> QueryCoalescingDecision
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`coalesce(request) -> QueryCoalescingDecision`",
        ],
        "comparableInputs": [
            "Same benchmark/render coexistence fixture for every policy",
            "Same duplicate render fixture for every policy",
            "Same latest-render replacement fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable coalescing contract used by interactive queue stores",
            "`src/gs_sim2real/experiments/`: discardable coalescing comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    report = build_query_coalescing_policy_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_coalescing_policy_report=report,
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
    "EXPERIMENT_QUERY_COALESCING_POLICIES",
    "build_query_coalescing_policy_experiment_report",
    "build_query_coalescing_policy_fixtures",
    "build_query_coalescing_policy_process_section",
    "run_cli",
]
