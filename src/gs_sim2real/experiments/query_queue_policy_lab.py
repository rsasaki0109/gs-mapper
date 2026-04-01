"""Experiment-first lab for sim2real query queue policies."""

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

from gs_sim2real.core.query_queue_policy import (
    CORE_QUERY_QUEUE_POLICIES,
    QueuedQueryItem,
    QueryQueuePolicy,
    QueryQueueState,
    admit_query_queue_item,
    dispatch_query_queue_item,
)


@dataclass(frozen=True)
class QueryQueueFixture:
    """Canonical queue workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    state: QueryQueueState
    incoming_item: QueuedQueryItem | None
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_QUEUE_POLICIES: tuple[QueryQueuePolicy, ...] = CORE_QUERY_QUEUE_POLICIES


def _item(request_id: str, request_type: str, order: int, *, work: int = 1) -> QueuedQueryItem:
    return QueuedQueryItem(
        request_id=request_id,
        request_type=request_type,
        transport="ws",
        submitted_order=order,
        expected_work_units=work,
    )


def build_query_queue_policy_fixtures() -> list[QueryQueueFixture]:
    """Build shared fixtures for queue policy comparisons."""
    return [
        QueryQueueFixture(
            fixture_id="single-render",
            label="Single Render",
            intent="A single render request should be admitted and dispatched immediately.",
            state=QueryQueueState(pending_items=(), max_pending=2),
            incoming_item=_item("render-1", "render", 1),
            expected_summary={
                "accepted": True,
                "pendingAfterAdmit": ("render-1",),
                "evicted": (),
                "dispatchRequestId": "render-1",
                "pendingAfterDispatch": (),
            },
        ),
        QueryQueueFixture(
            fixture_id="benchmark-then-render",
            label="Benchmark Then Render",
            intent="Interactive render work should leap ahead of queued background benchmark work.",
            state=QueryQueueState(
                pending_items=(_item("benchmark-1", "localization-image-benchmark", 1, work=12),),
                max_pending=3,
            ),
            incoming_item=_item("render-1", "render", 2),
            expected_summary={
                "accepted": True,
                "pendingAfterAdmit": ("render-1", "benchmark-1"),
                "evicted": (),
                "dispatchRequestId": "render-1",
                "pendingAfterDispatch": ("benchmark-1",),
            },
        ),
        QueryQueueFixture(
            fixture_id="evict-background-under-pressure",
            label="Evict Background Under Pressure",
            intent="When the queue is full, interactive render work should evict the worst background benchmark instead of being rejected.",
            state=QueryQueueState(
                pending_items=(
                    _item("benchmark-1", "localization-image-benchmark", 1, work=16),
                    _item("render-1", "render", 2),
                ),
                max_pending=2,
            ),
            incoming_item=_item("render-2", "render", 3),
            expected_summary={
                "accepted": True,
                "pendingAfterAdmit": ("render-1", "render-2"),
                "evicted": ("benchmark-1",),
                "dispatchRequestId": "render-1",
                "pendingAfterDispatch": ("render-2",),
            },
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_queue_fixture(
    policy: QueryQueuePolicy,
    fixture: QueryQueueFixture,
) -> dict[str, Any]:
    """Run one queue policy on one canonical queue workload."""
    started_at = perf_counter()
    try:
        if fixture.incoming_item is not None:
            admit = admit_query_queue_item(fixture.state, fixture.incoming_item, policy=policy)
            admitted_ids = admit.pending_request_ids
            dispatch_state_items = tuple(
                item
                for item in fixture.state.pending_items + (fixture.incoming_item,)
                if admit.accepted and item.request_id in set(admit.pending_request_ids)
            )
            dispatch_state = QueryQueueState(
                pending_items=tuple(sorted(dispatch_state_items, key=lambda item: admitted_ids.index(item.request_id))),
                max_pending=fixture.state.max_pending,
            )
        else:
            admit = None
            dispatch_state = fixture.state
            admitted_ids = tuple(item.request_id for item in fixture.state.pending_items)
        dispatch = dispatch_query_queue_item(dispatch_state, policy=policy)
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
    summary = {
        "accepted": True if admit is None else bool(admit.accepted),
        "pendingAfterAdmit": admitted_ids,
        "evicted": () if admit is None else admit.evicted_request_ids,
        "dispatchRequestId": dispatch.dispatch_request_id,
        "pendingAfterDispatch": dispatch.pending_request_ids,
    }
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


def benchmark_query_queue_policy_runtime(
    policy: QueryQueuePolicy,
    fixtures: Sequence[QueryQueueFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure queue policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            if fixture.incoming_item is not None:
                admit = admit_query_queue_item(fixture.state, fixture.incoming_item, policy=policy)
                dispatch_items = tuple(
                    item
                    for item in fixture.state.pending_items + (fixture.incoming_item,)
                    if admit.accepted and item.request_id in set(admit.pending_request_ids)
                )
                dispatch_state = QueryQueueState(dispatch_items, fixture.state.max_pending)
            else:
                dispatch_state = fixture.state
            dispatch_query_queue_item(dispatch_state, policy=policy)
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))
    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: QueryQueuePolicy) -> dict[str, Any]:
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


def evaluate_extensibility(policy: QueryQueuePolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "boundsQueue": 2.5,
        "prioritizesInteractiveRender": 3.0,
        "evictsBackgroundWork": 2.5,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_query_queue_policy(
    policy: QueryQueuePolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one queue policy."""
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


def build_query_queue_policy_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare queue policies on shared fixtures."""
    fixtures = build_query_queue_policy_fixtures()
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
    for policy in EXPERIMENT_QUERY_QUEUE_POLICIES:
        fixture_reports = [evaluate_query_queue_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_queue_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_queue_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-queue-policy-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "query-queue-policy",
            "statement": (
                "Manage interactive sim2real query backlogs without assuming that every pending request "
                "deserves strict FIFO treatment."
            ),
            "stableInterface": "admit_query_queue_item(QueryQueueState(...), item), dispatch_query_queue_item(QueryQueueState(...))",
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


def build_query_queue_policy_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the queue policy report into a shared docs section."""
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
                    fixture_report.get("summary", {}).get("dispatchRequestId", "n/a"),
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Dispatch"],
                "rows": rows,
            }
        )

    return {
        "title": "Query Queue Policy",
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
            "Stable server code uses `admit_query_queue_item(...)` and `dispatch_query_queue_item(...)` as the only queue policy surface.",
            "`interactive_first` is the production default because it keeps render interactions responsive and evicts lower-priority background work under pressure.",
            "Alternative queue policies stay experimental until the same single-render, benchmark-then-render, and pressure fixtures show a better fit.",
        ],
        "deferred": [
            "`fifo_unbounded` stays experimental. It is simple, but it allows heavy benchmark work to dominate interactive queues.",
            "`bounded_fifo` stays experimental. It caps backlog growth, but it still rejects interactive requests even when only background work is queued.",
        ],
        "rules": [
            "Compare at least three queue policies before changing interactive transport backlog behavior.",
            "Use the same single-render, mixed workload, and pressure fixtures for every candidate policy.",
            "Keep websocket server code dependent only on queue decisions, not on policy-specific branching.",
        ],
        "stableInterfaceIntro": "The stable query queue surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            admit_query_queue_item(
                QueryQueueState(...),
                item,
                policy = 'interactive_first',
            ) -> QueryQueueAdmitDecision

            dispatch_query_queue_item(
                QueryQueueState(...),
                policy = 'interactive_first',
            ) -> QueryQueueDispatchDecision
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`admit(state, item) -> QueryQueueAdmitDecision`",
            "`dispatch(state) -> QueryQueueDispatchDecision`",
        ],
        "comparableInputs": [
            "Same single render fixture for every policy",
            "Same mixed benchmark/render fixture for every policy",
            "Same pressure fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable queue policy contract used by websocket transport code",
            "`src/gs_sim2real/experiments/`: discardable queue policy comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the query queue lab and optionally refresh docs."""
    report = build_query_queue_policy_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_queue_policy_report=report,
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
    "EXPERIMENT_QUERY_QUEUE_POLICIES",
    "build_query_queue_policy_experiment_report",
    "build_query_queue_policy_fixtures",
    "build_query_queue_policy_process_section",
    "run_cli",
]
