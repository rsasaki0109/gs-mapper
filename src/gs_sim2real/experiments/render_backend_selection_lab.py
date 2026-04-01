"""Experiment-first lab for sim2real render backend selection."""

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

from gs_sim2real.core.render_backend_selection import (
    CORE_RENDER_BACKEND_POLICIES,
    RenderBackendCapabilities,
    RenderBackendPolicy,
    RenderBackendPreferences,
    RenderBackendRequest,
    RenderBackendSelection,
)


@dataclass(frozen=True)
class RenderBackendFixture:
    """Canonical request shared by every backend-selection policy."""

    fixture_id: str
    label: str
    intent: str
    request: RenderBackendRequest
    preferred_backend: str
    backend_fitness: dict[str, float]


class ConservativeSimpleRenderBackendPolicy:
    """Minimize surprises by preferring the universal fallback."""

    name = "simple_safe"
    label = "Conservative Simple"
    style = "fallback-first"
    tier = "experiment"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": False,
        "supportsFastPreviewBias": True,
    }

    def select(self, request: RenderBackendRequest) -> RenderBackendSelection:
        requested_backend = str(request.requested_backend or "auto").strip() or "auto"
        runtime = request.capabilities

        if requested_backend == "simple":
            return RenderBackendSelection("simple", "forced by --renderer simple")
        if requested_backend == "gsplat":
            return CORE_RENDER_BACKEND_POLICIES["balanced"].select(request)
        if requested_backend != "auto":
            raise RuntimeError(f"unsupported renderer selection mode: {requested_backend}")
        if not runtime.has_gaussian_splat:
            return RenderBackendSelection(
                "simple",
                "fallback because the PLY does not contain Gaussian scale/rotation parameters",
            )
        if not runtime.gsplat_available:
            return RenderBackendSelection(
                "simple",
                "fallback because the optional `gsplat` package is not installed",
            )
        if not runtime.cuda_available:
            return RenderBackendSelection("simple", "fallback because CUDA is not available for gsplat")
        return RenderBackendSelection(
            "simple", "auto-selected simple because the conservative policy avoids optional GPU paths"
        )


class FidelityFirstRenderBackendPolicy:
    """Prefer the highest-fidelity backend whenever it is feasible."""

    name = "fidelity_first"
    label = "Fidelity First"
    style = "quality-priority"
    tier = "experiment"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": False,
        "supportsFastPreviewBias": False,
    }

    def select(self, request: RenderBackendRequest) -> RenderBackendSelection:
        requested_backend = str(request.requested_backend or "auto").strip() or "auto"
        runtime = request.capabilities

        if requested_backend in {"simple", "gsplat"}:
            return CORE_RENDER_BACKEND_POLICIES["balanced"].select(request)
        if requested_backend != "auto":
            raise RuntimeError(f"unsupported renderer selection mode: {requested_backend}")
        if runtime.has_gaussian_splat and runtime.gsplat_available and runtime.cuda_available:
            return RenderBackendSelection(
                "gsplat",
                "auto-selected gsplat because the policy always prefers the highest-fidelity feasible backend",
            )
        return CORE_RENDER_BACKEND_POLICIES["balanced"].select(request)


class BalancedRenderBackendExperimentPolicy:
    """Expose the stable backend-selection policy to the experiment harness."""

    name = "balanced"
    label = "Balanced Capability Gate"
    style = "capability-gated"
    tier = "core"
    capabilities = dict(CORE_RENDER_BACKEND_POLICIES["balanced"].capabilities)

    def select(self, request: RenderBackendRequest) -> RenderBackendSelection:
        return CORE_RENDER_BACKEND_POLICIES["balanced"].select(request)


EXPERIMENT_RENDER_BACKEND_POLICIES: tuple[RenderBackendPolicy, ...] = (
    ConservativeSimpleRenderBackendPolicy(),
    BalancedRenderBackendExperimentPolicy(),
    FidelityFirstRenderBackendPolicy(),
)


def build_render_backend_fixtures() -> list[RenderBackendFixture]:
    """Build shared fixtures for backend-selection comparisons."""
    return [
        RenderBackendFixture(
            fixture_id="plain-point-cloud",
            label="Plain Point Cloud",
            intent="Keep a path that works when the PLY lacks Gaussian parameters.",
            request=RenderBackendRequest(
                requested_backend="auto",
                capabilities=RenderBackendCapabilities(
                    has_gaussian_splat=False,
                    gsplat_available=True,
                    cuda_available=True,
                ),
                preferences=RenderBackendPreferences(
                    prefer_low_startup_latency=True,
                    prefer_visual_fidelity=False,
                ),
            ),
            preferred_backend="simple",
            backend_fitness={"simple": 1.0, "gsplat": 0.0},
        ),
        RenderBackendFixture(
            fixture_id="interactive-preview",
            label="Interactive Browser Preview",
            intent="Bias toward quick startup for teleop and browser reconnection loops.",
            request=RenderBackendRequest(
                requested_backend="auto",
                capabilities=RenderBackendCapabilities(
                    has_gaussian_splat=True,
                    gsplat_available=True,
                    cuda_available=True,
                ),
                preferences=RenderBackendPreferences(
                    prefer_low_startup_latency=True,
                    prefer_visual_fidelity=False,
                ),
            ),
            preferred_backend="simple",
            backend_fitness={"simple": 0.95, "gsplat": 0.55},
        ),
        RenderBackendFixture(
            fixture_id="offline-benchmark",
            label="Offline Benchmark Capture",
            intent="Prefer the highest-fidelity backend when precomputing benchmark evidence.",
            request=RenderBackendRequest(
                requested_backend="auto",
                capabilities=RenderBackendCapabilities(
                    has_gaussian_splat=True,
                    gsplat_available=True,
                    cuda_available=True,
                ),
                preferences=RenderBackendPreferences(
                    prefer_low_startup_latency=False,
                    prefer_visual_fidelity=True,
                ),
            ),
            preferred_backend="gsplat",
            backend_fitness={"simple": 0.6, "gsplat": 1.0},
        ),
        RenderBackendFixture(
            fixture_id="no-cuda-fallback",
            label="No CUDA Fallback",
            intent="Define graceful behavior when the optional high-fidelity path is unavailable at runtime.",
            request=RenderBackendRequest(
                requested_backend="auto",
                capabilities=RenderBackendCapabilities(
                    has_gaussian_splat=True,
                    gsplat_available=True,
                    cuda_available=False,
                ),
                preferences=RenderBackendPreferences(
                    prefer_low_startup_latency=False,
                    prefer_visual_fidelity=True,
                ),
            ),
            preferred_backend="simple",
            backend_fitness={"simple": 0.9, "gsplat": 0.0},
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_backend_fixture(
    policy: RenderBackendPolicy,
    fixture: RenderBackendFixture,
) -> dict[str, Any]:
    """Run one backend-selection policy on one fixture."""
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
    fitness = float(fixture.backend_fitness.get(selection.name, 0.0))
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "selectedBackend": selection.name,
        "reason": selection.reason,
        "preferredBackend": fixture.preferred_backend,
        "matchedPreferredBackend": selection.name == fixture.preferred_backend,
        "fitness": fitness,
        "runtimeMs": runtime_ms,
    }


def benchmark_backend_policy_runtime(
    policy: RenderBackendPolicy,
    fixtures: Sequence[RenderBackendFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure policy execution time on the shared fixtures."""
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


def evaluate_readability(policy: RenderBackendPolicy) -> dict[str, Any]:
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
    notes = []
    if lines_of_code <= 12:
        notes.append("short enough to audit during incident response")
    else:
        notes.append("multiple conditional branches need fixture coverage")
    if branch_count >= 4:
        notes.append("policy behavior is distributed across several runtime gates")
    else:
        notes.append("control flow stays shallow")
    return {
        "score": round(score, 1),
        "linesOfCode": lines_of_code,
        "branchCount": branch_count,
        "notes": notes,
    }


def evaluate_extensibility(policy: RenderBackendPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "respectsExplicitRequests": 2.0,
        "usesRuntimeCapabilities": 3.0,
        "usesWorkloadPreferences": 3.0,
        "supportsFastPreviewBias": 2.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    notes: list[str] = []
    if policy.capabilities.get("respectsExplicitRequests"):
        notes.append("still obeys hard operator choices")
    if policy.capabilities.get("usesRuntimeCapabilities"):
        notes.append("adapts to missing CUDA or missing gsplat")
    if policy.capabilities.get("usesWorkloadPreferences"):
        notes.append("can express preview vs benchmark intent")
    if policy.capabilities.get("supportsFastPreviewBias"):
        notes.append("can protect browser-first workflows from heavy startup")
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
        "notes": notes,
    }


def summarize_backend_policy(
    policy: RenderBackendPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one policy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    success_rate = float(len(successful) / max(1, len(fixture_reports)))
    aggregate = {
        "successRate": success_rate,
        "matchRate": _mean_or_none([1.0 if report.get("matchedPreferredBackend") else 0.0 for report in successful]),
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


def build_render_backend_selection_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare backend-selection policies on shared fixtures."""
    fixtures = build_render_backend_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "preferredBackend": fixture.preferred_backend,
            "requestedBackend": fixture.request.requested_backend,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_RENDER_BACKEND_POLICIES:
        fixture_reports = [evaluate_backend_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_backend_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_backend_policy(policy, fixture_reports, runtime_report))

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
        "type": "render-backend-selection-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "render-backend-selection",
            "statement": (
                "Select a render backend for sim2real workloads without freezing one universal policy for "
                "interactive preview, offline benchmarking, and degraded runtimes."
            ),
            "stableInterface": "select_render_backend(RenderBackendRequest(...))",
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


def build_render_backend_selection_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the backend-selection report into a shared docs section."""
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
                    fixture_report.get("selectedBackend", "n/a"),
                    fixture["preferredBackend"],
                    "yes" if fixture_report.get("matchedPreferredBackend") else "no",
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
        "title": "Render Backend Selection",
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
            "Stable code uses `select_render_backend(RenderBackendRequest(...))` as the only backend-selection contract.",
            "`balanced` is the default production policy because it keeps the existing capability gates and adds a preview-oriented latency escape hatch.",
            "Alternative policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a clear advantage.",
        ],
        "deferred": [
            "`simple_safe` stays experimental. It is easy to reason about, but it leaves benchmark quality on the table when gsplat is available.",
            "`fidelity_first` stays experimental. It improves offline quality, but it ignores interactive preview latency requirements.",
        ],
        "rules": [
            "Start backend-selection work with at least three policies, not one conditional chain.",
            "Compare policies on the same runtime capabilities and workload preferences before changing production defaults.",
            "Promote only the policy interface that multiple workloads can agree on.",
        ],
        "stableInterfaceIntro": "The stable render-backend selection surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class RenderBackendCapabilities:
                has_gaussian_splat: bool
                gsplat_available: bool
                cuda_available: bool

            @dataclass(frozen=True)
            class RenderBackendPreferences:
                prefer_low_startup_latency: bool = False
                prefer_visual_fidelity: bool = True

            @dataclass(frozen=True)
            class RenderBackendRequest:
                requested_backend: str
                capabilities: RenderBackendCapabilities
                preferences: RenderBackendPreferences = RenderBackendPreferences()

            def select_render_backend(
                request: RenderBackendRequest,
                *,
                policy: str = 'balanced',
            ) -> RenderBackendSelection: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`select(request) -> RenderBackendSelection`",
        ],
        "comparableInputs": [
            "Same `RenderBackendRequest` fixtures for every policy",
            "Same workload fixtures (`plain-point-cloud`, `interactive-preview`, `offline-benchmark`, `no-cuda-fallback`)",
            "Same evaluation axes: fit, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable backend-selection policy contract for production render code",
            "`src/gs_sim2real/experiments/`: discardable backend-selection policies and comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the render-backend selection lab and optionally refresh docs."""
    report = build_render_backend_selection_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(docs_dir=args.docs_dir, render_backend_report=report)
    summary = {
        "type": report["type"],
        "policyCount": len(report["policies"]),
        "fixtureCount": len(report["fixtures"]),
        "bestFit": report["highlights"]["bestFit"],
        "fastestMedianRuntime": report["highlights"]["fastestMedianRuntime"],
        "docs": docs,
    }
    print(json.dumps(summary, indent=2))
