"""Experiment-first lab for sim2real query transport selection."""

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

from gs_sim2real.core.query_transport_selection import (
    CORE_QUERY_TRANSPORT_POLICIES,
    QueryTransportCapabilities,
    QueryTransportPolicy,
    QueryTransportPreferences,
    QueryTransportRequest,
    QueryTransportSelection,
    resolve_query_transport_endpoint,
)


@dataclass(frozen=True)
class QueryTransportFixture:
    """Canonical request shared by every query transport policy."""

    fixture_id: str
    label: str
    intent: str
    request: QueryTransportRequest
    preferred_transport: str
    transport_fitness: dict[str, float]


class ExplicitOnlyQueryTransportPolicy:
    """Keep auto mode minimal and mostly explicit-transport oriented."""

    name = "explicit_only"
    label = "Explicit Only"
    style = "manual-first"
    tier = "experiment"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": False,
        "supportsBrowserFirst": False,
    }

    def select(self, request: QueryTransportRequest) -> QueryTransportSelection:
        requested_transport = str(request.requested_transport or "auto").strip() or "auto"
        runtime = request.capabilities
        query_required = str(request.pose_source or "").strip() == "query"

        if requested_transport == "none":
            if query_required:
                raise RuntimeError("pose-source=query requires --query-transport zmq, ws, or auto")
            return QueryTransportSelection("none", "", "forced by --query-transport none")
        if requested_transport == "zmq":
            if not runtime.zmq_available:
                raise RuntimeError("query-transport=zmq requires the optional `pyzmq` package")
            return QueryTransportSelection(
                "zmq", resolve_query_transport_endpoint("zmq", request.endpoint), "forced by --query-transport zmq"
            )
        if requested_transport == "ws":
            if not runtime.ws_available:
                raise RuntimeError("query-transport=ws requires the optional `websockets` package")
            return QueryTransportSelection(
                "ws", resolve_query_transport_endpoint("ws", request.endpoint), "forced by --query-transport ws"
            )
        if requested_transport != "auto":
            raise RuntimeError(f"unsupported query transport: {requested_transport}")

        if query_required:
            if runtime.zmq_available:
                return QueryTransportSelection(
                    "zmq",
                    resolve_query_transport_endpoint("zmq", request.endpoint),
                    "auto-selected zmq because the explicit-only policy defaults to local tooling first",
                )
            if runtime.ws_available:
                return QueryTransportSelection(
                    "ws",
                    resolve_query_transport_endpoint("ws", request.endpoint),
                    "auto-selected ws because zmq is unavailable",
                )
            raise RuntimeError("pose-source=query requires the optional `pyzmq` or `websockets` packages")

        return QueryTransportSelection(
            "none",
            "",
            "auto-selected no query transport because the explicit-only policy disables optional transports by default",
        )


class BrowserFirstQueryTransportPolicy:
    """Bias auto mode toward browser-facing websocket clients."""

    name = "browser_first"
    label = "Browser First"
    style = "browser-priority"
    tier = "experiment"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": False,
        "supportsBrowserFirst": True,
    }

    def select(self, request: QueryTransportRequest) -> QueryTransportSelection:
        requested_transport = str(request.requested_transport or "auto").strip() or "auto"
        runtime = request.capabilities
        query_required = str(request.pose_source or "").strip() == "query"

        if requested_transport in {"none", "zmq", "ws"}:
            return CORE_QUERY_TRANSPORT_POLICIES["balanced"].select(request)
        if requested_transport != "auto":
            raise RuntimeError(f"unsupported query transport: {requested_transport}")

        if not query_required and not request.preferences.enable_query_transport:
            return QueryTransportSelection(
                "none",
                "",
                "auto-selected no query transport because the workload does not request interactive queries",
            )
        if runtime.ws_available:
            return QueryTransportSelection(
                "ws",
                resolve_query_transport_endpoint("ws", request.endpoint),
                "auto-selected ws because the policy always prefers browser-facing clients",
            )
        if runtime.zmq_available:
            return QueryTransportSelection(
                "zmq",
                resolve_query_transport_endpoint("zmq", request.endpoint),
                "auto-selected zmq because websockets are unavailable",
            )
        if query_required:
            raise RuntimeError("pose-source=query requires the optional `websockets` or `pyzmq` packages")
        return QueryTransportSelection(
            "none",
            "",
            "auto-selected no query transport because optional transport dependencies are unavailable",
        )


class BalancedQueryTransportExperimentPolicy:
    """Expose the stable query transport policy to the experiment harness."""

    name = "balanced"
    label = "Balanced Interactive Transport"
    style = "workload-aware"
    tier = "core"
    capabilities = dict(CORE_QUERY_TRANSPORT_POLICIES["balanced"].capabilities)

    def select(self, request: QueryTransportRequest) -> QueryTransportSelection:
        return CORE_QUERY_TRANSPORT_POLICIES["balanced"].select(request)


EXPERIMENT_QUERY_TRANSPORT_POLICIES: tuple[QueryTransportPolicy, ...] = (
    ExplicitOnlyQueryTransportPolicy(),
    BalancedQueryTransportExperimentPolicy(),
    BrowserFirstQueryTransportPolicy(),
)


def build_query_transport_fixtures() -> list[QueryTransportFixture]:
    """Build shared fixtures for query transport comparisons."""
    return [
        QueryTransportFixture(
            fixture_id="publish-only-static",
            label="Publish-Only Static Server",
            intent="Keep non-interactive deployments free from unused query sockets.",
            request=QueryTransportRequest(
                requested_transport="auto",
                pose_source="static",
                capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
                preferences=QueryTransportPreferences(enable_query_transport=False),
            ),
            preferred_transport="none",
            transport_fitness={"none": 1.0, "zmq": 0.3, "ws": 0.2},
        ),
        QueryTransportFixture(
            fixture_id="browser-query",
            label="Browser Query Mode",
            intent="Prefer websocket transport for browser-driven simulator requests.",
            request=QueryTransportRequest(
                requested_transport="auto",
                pose_source="query",
                capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
                preferences=QueryTransportPreferences(enable_query_transport=True, prefer_browser_clients=True),
            ),
            preferred_transport="ws",
            transport_fitness={"none": 0.0, "zmq": 0.55, "ws": 1.0},
        ),
        QueryTransportFixture(
            fixture_id="local-cli-query",
            label="Local CLI Query Mode",
            intent="Prefer zmq for local tooling when the workload is not browser-facing.",
            request=QueryTransportRequest(
                requested_transport="auto",
                pose_source="query",
                capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
                preferences=QueryTransportPreferences(enable_query_transport=True, prefer_local_cli=True),
            ),
            preferred_transport="zmq",
            transport_fitness={"none": 0.0, "zmq": 1.0, "ws": 0.55},
        ),
        QueryTransportFixture(
            fixture_id="ws-missing",
            label="WebSocket Dependency Missing",
            intent="Fallback cleanly to zmq when browser transport dependencies are absent.",
            request=QueryTransportRequest(
                requested_transport="auto",
                pose_source="query",
                capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=False),
                preferences=QueryTransportPreferences(enable_query_transport=True, prefer_browser_clients=True),
            ),
            preferred_transport="zmq",
            transport_fitness={"none": 0.0, "zmq": 1.0, "ws": 0.0},
        ),
        QueryTransportFixture(
            fixture_id="explicit-ws",
            label="Explicit WebSocket Override",
            intent="Respect hard operator choices even when other transports are also viable.",
            request=QueryTransportRequest(
                requested_transport="ws",
                pose_source="static",
                capabilities=QueryTransportCapabilities(zmq_available=True, ws_available=True),
                preferences=QueryTransportPreferences(enable_query_transport=True, prefer_local_cli=True),
            ),
            preferred_transport="ws",
            transport_fitness={"none": 0.0, "zmq": 0.2, "ws": 1.0},
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_query_transport_fixture(
    policy: QueryTransportPolicy,
    fixture: QueryTransportFixture,
) -> dict[str, Any]:
    """Run one query transport policy on one canonical fixture."""
    started_at = perf_counter()
    try:
        selection = policy.select(fixture.request)
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
    fitness = float(fixture.transport_fitness.get(selection.transport, 0.0))
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "selectedTransport": selection.transport,
        "endpoint": selection.endpoint,
        "preferredTransport": fixture.preferred_transport,
        "matchedPreferredTransport": selection.transport == fixture.preferred_transport,
        "fitness": fitness,
        "runtimeMs": runtime_ms,
    }


def benchmark_query_transport_policy_runtime(
    policy: QueryTransportPolicy,
    fixtures: Sequence[QueryTransportFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure query transport policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                policy.select(fixture.request)
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


def evaluate_readability(policy: QueryTransportPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.select))
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


def evaluate_extensibility(policy: QueryTransportPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "respectsExplicitRequests": 2.0,
        "usesRuntimeCapabilities": 3.0,
        "usesWorkloadPreferences": 3.0,
        "supportsBrowserFirst": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
    }


def summarize_query_transport_policy(
    policy: QueryTransportPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one query transport policy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    aggregate = {
        "successRate": float(len(successful) / max(1, len(fixture_reports))),
        "matchRate": _mean_or_none([1.0 if report.get("matchedPreferredTransport") else 0.0 for report in successful]),
        "meanFitness": _mean_or_none([report.get("fitness") for report in successful]),
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


def build_query_transport_selection_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare query transport policies on shared fixtures."""
    fixtures = build_query_transport_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "preferredTransport": fixture.preferred_transport,
            "requestedTransport": fixture.request.requested_transport,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_QUERY_TRANSPORT_POLICIES:
        fixture_reports = [evaluate_query_transport_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_query_transport_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_query_transport_policy(policy, fixture_reports, runtime_report))

    best_fit = max(
        policy_reports,
        key=lambda report: (
            float(report["aggregate"]["meanFitness"] or 0.0),
            float(report["aggregate"]["matchRate"] or 0.0),
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
        "type": "query-transport-selection-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "query-transport-selection",
            "statement": (
                "Select a sim2real query transport without freezing one universal choice for browser-facing "
                "interactive queries, local CLI tooling, and publish-only deployments."
            ),
            "stableInterface": "select_query_transport(QueryTransportRequest(...))",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": ["successRate", "matchRate", "meanFitness"],
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
                "meanFitness": best_fit["aggregate"]["meanFitness"],
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


def build_query_transport_selection_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the query transport report into a shared docs section."""
    comparison_rows = []
    for policy in report["policies"]:
        comparison_rows.append(
            [
                policy["label"],
                policy["tier"],
                policy["style"],
                f"{float(policy['aggregate']['successRate'] or 0.0):.2f}",
                f"{float(policy['aggregate']['matchRate'] or 0.0):.2f}",
                f"{float(policy['aggregate']['meanFitness'] or 0.0):.3f}",
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
                    fixture_report.get("selectedTransport", "n/a"),
                    fixture["preferredTransport"],
                    "yes" if fixture_report.get("matchedPreferredTransport") else "no",
                    f"{float(fixture_report.get('fitness') or 0.0):.3f}" if fixture_report["status"] == "ok" else "n/a",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Selected", "Preferred", "Match", "Fitness"],
                "rows": rows,
            }
        )

    return {
        "title": "Query Transport Selection",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Policy",
            "Tier",
            "Style",
            "Success",
            "Match",
            "Fitness",
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
            "Stable code uses `select_query_transport(QueryTransportRequest(...))` as the only transport-selection contract.",
            "`balanced` is the default production policy because it can keep publish-only deployments quiet, choose ws for browser-first query mode, and still honor local CLI preferences.",
            "Alternative transport policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-workload fit.",
        ],
        "deferred": [
            "`explicit_only` stays experimental. It is predictable, but it ignores browser-vs-CLI workload intent in auto mode.",
            "`browser_first` stays experimental. It works well for web clients, but it over-selects ws when local CLI tooling would prefer zmq.",
        ],
        "rules": [
            "Start query transport work with at least three policies, not one conditional chain.",
            "Compare policies on the same runtime capabilities and workload preferences before changing production defaults.",
            "Promote only the smallest transport contract that both browser-first and local tooling flows can share.",
        ],
        "stableInterfaceIntro": "The stable query transport surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class QueryTransportCapabilities:
                zmq_available: bool
                ws_available: bool

            @dataclass(frozen=True)
            class QueryTransportPreferences:
                enable_query_transport: bool = False
                prefer_browser_clients: bool = False
                prefer_local_cli: bool = False

            @dataclass(frozen=True)
            class QueryTransportRequest:
                requested_transport: str
                pose_source: str
                endpoint: str = ""
                capabilities: QueryTransportCapabilities = QueryTransportCapabilities(...)
                preferences: QueryTransportPreferences = QueryTransportPreferences()

            def select_query_transport(
                request: QueryTransportRequest,
                *,
                policy: str = 'balanced',
            ) -> QueryTransportSelection: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`select(request) -> QueryTransportSelection`",
        ],
        "comparableInputs": [
            "Same `QueryTransportRequest` fixtures for every policy",
            "Same workload fixtures (`publish-only-static`, `browser-query`, `local-cli-query`, `ws-missing`, `explicit-ws`)",
            "Same evaluation axes: fit, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable query transport contract for production render-server setup",
            "`src/gs_sim2real/experiments/`: discardable query transport policies and comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the query transport selection lab and optionally refresh docs."""
    report = build_query_transport_selection_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            query_transport_report=report,
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
