"""Experiment-first lab for sim2real query source-identity policies."""

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

from gs_sim2real.core.query_source_identity import (
    CORE_QUERY_SOURCE_IDENTITY_POLICIES,
    QuerySourceIdentityPolicy,
    QuerySourceIdentityRequest,
    resolve_query_source_identity,
)


@dataclass(frozen=True)
class QuerySourceIdentityFixture:
    """Canonical source-identity workload shared by every policy."""

    fixture_id: str
    label: str
    intent: str
    request: QuerySourceIdentityRequest
    expected_summary: dict[str, Any]


EXPERIMENT_QUERY_SOURCE_IDENTITY_POLICIES: tuple[QuerySourceIdentityPolicy, ...] = CORE_QUERY_SOURCE_IDENTITY_POLICIES


def build_query_source_identity_fixtures() -> list[QuerySourceIdentityFixture]:
    """Build shared fixtures for source-identity policy comparisons."""
    return [
        QuerySourceIdentityFixture(
            fixture_id="ws-remote-address",
            label="WebSocket Remote Address",
            intent="Browser websocket sessions should use the observable remote socket address when it exists.",
            request=QuerySourceIdentityRequest(
                transport="ws",
                connection_serial=7,
                endpoint="ws://127.0.0.1:8781/sim2real",
                remote_host="127.0.0.1",
                remote_port=50123,
            ),
            expected_summary={"sourceId": "ws-127.0.0.1-50123"},
        ),
        QuerySourceIdentityFixture(
            fixture_id="client-hint-fallback",
            label="Client Hint Fallback",
            intent="When the transport cannot expose a remote address, a client hint should stay human-readable but collision-safe.",
            request=QuerySourceIdentityRequest(
                transport="ws",
                connection_serial=8,
                endpoint="ws://127.0.0.1:8781/sim2real",
                client_hint="Route Replay Panel",
            ),
            expected_summary={"sourceId": "ws-route-replay-panel-client-8"},
        ),
        QuerySourceIdentityFixture(
            fixture_id="endpoint-fallback",
            label="Endpoint Fallback",
            intent="When neither remote address nor client hint exists, the endpoint should still scope the source identity.",
            request=QuerySourceIdentityRequest(
                transport="zmq",
                connection_serial=3,
                endpoint="tcp://127.0.0.1:5588",
            ),
            expected_summary={"sourceId": "zmq-127.0.0.1-5588-client-3"},
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_source_identity_fixture(
    policy: QuerySourceIdentityPolicy,
    fixture: QuerySourceIdentityFixture,
) -> dict[str, Any]:
    """Run one source-identity policy on one canonical workload."""
    started_at = perf_counter()
    try:
        decision = resolve_query_source_identity(fixture.request, policy=policy)
        summary = {"sourceId": decision.source_id}
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


def benchmark_query_source_identity_policy_runtime(
    policy: QuerySourceIdentityPolicy,
    fixtures: Sequence[QuerySourceIdentityFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure source-identity policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            resolve_query_source_identity(fixture.request, policy=policy)
            samples_ms.append(float((perf_counter() - started_at) * 1000.0))
    return {
        "repetitions": int(repetitions),
        "sampleCount": len(samples_ms),
        "meanMs": float(mean(samples_ms)),
        "medianMs": float(median(samples_ms)),
    }


def evaluate_readability(policy: QuerySourceIdentityPolicy) -> dict[str, Any]:
    source = textwrap.dedent(inspect.getsource(policy.identify))
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


def evaluate_extensibility(policy: QuerySourceIdentityPolicy) -> dict[str, Any]:
    weights = {
        "usesRemoteAddress": 3.0,
        "usesEndpoint": 2.5,
        "usesClientHint": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {"score": round(score, 1), "supportedCapabilities": supported}


def summarize_query_source_identity_policy(
    policy: QuerySourceIdentityPolicy,
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


def build_query_source_identity_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare source-identity policies on shared fixtures."""
    fixtures = build_query_source_identity_fixtures()
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
    for policy in EXPERIMENT_QUERY_SOURCE_IDENTITY_POLICIES:
        fixture_reports = [evaluate_query_source_identity_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_source_identity_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_source_identity_policy(policy, fixture_reports, runtime_report))

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
        "type": "query-source-identity-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "query-source-identity",
            "statement": (
                "Assign interactive query source ids intentionally instead of hard-coding opaque transport counters "
                "inside websocket server code."
            ),
            "stableInterface": "resolve_query_source_identity(QuerySourceIdentityRequest(...))",
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


def build_query_source_identity_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the source-identity report into a shared docs section."""
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
                    fixture_report.get("summary", {}).get("sourceId", "n/a"),
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Match", "Exact", "Source Id"],
                "rows": rows,
            }
        )
    return {
        "title": "Query Source Identity",
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
            "Stable transport code uses `resolve_query_source_identity(QuerySourceIdentityRequest(...))` as the only source-identity surface.",
            "`remote_observable` is the production default because it preserves per-connection uniqueness while staying debuggable.",
            "Alternative source-id strategies stay experimental until the same remote-address, client-hint, and endpoint fallback fixtures show a better fit.",
        ],
        "deferred": [
            "`serial_only` stays experimental. It is cheap, but it hides which client or endpoint owns a backlog.",
            "`endpoint_scoped` stays experimental. It improves debuggability, but it still cannot distinguish same-endpoint clients without a remote address.",
        ],
        "rules": [
            "Compare at least three source-identity policies before changing how queue source ids are assigned.",
            "Use the same remote-address, client-hint, and endpoint-fallback fixtures for every policy.",
            "Keep websocket queue code dependent only on source identities, not on policy-specific transport branching.",
        ],
        "stableInterfaceIntro": "The stable query source-identity surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            resolve_query_source_identity(
                QuerySourceIdentityRequest(...),
                policy = 'remote_observable',
            ) -> QuerySourceIdentity
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`identify(request) -> QuerySourceIdentity`",
        ],
        "comparableInputs": [
            "Same websocket remote-address fixture for every policy",
            "Same client-hint fallback fixture for every policy",
            "Same endpoint fallback fixture for every policy",
            "Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable source-identity contract used by interactive transport code",
            "`src/gs_sim2real/experiments/`: discardable source-identity comparison harnesses and docs adapters",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the source-identity lab and optionally refresh docs."""
    report = build_query_source_identity_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_source_identity_report=report,
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
    "EXPERIMENT_QUERY_SOURCE_IDENTITY_POLICIES",
    "build_query_source_identity_experiment_report",
    "build_query_source_identity_fixtures",
    "build_query_source_identity_process_section",
    "run_cli",
]
