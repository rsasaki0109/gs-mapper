"""Experiment-first lab for localization estimate import policies."""

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

from gs_sim2real.core.localization_estimate_import import (
    FallbackCascadeImportPolicy,
    LocalizationEstimateImportPolicy,
    LocalizationEstimateImportRequest,
    StrictContentGateImportPolicy,
    SuffixAwareRepairImportPolicy,
)


@dataclass(frozen=True)
class LocalizationEstimateImportFixture:
    """Canonical document shared by every import policy."""

    fixture_id: str
    label: str
    intent: str
    request: LocalizationEstimateImportRequest
    expected_source_type: str
    expected_pose_count: int
    expected_label: str


EXPERIMENT_LOCALIZATION_ESTIMATE_IMPORT_POLICIES: tuple[LocalizationEstimateImportPolicy, ...] = (
    StrictContentGateImportPolicy(),
    FallbackCascadeImportPolicy(),
    SuffixAwareRepairImportPolicy(),
)


def build_localization_estimate_import_fixtures() -> list[LocalizationEstimateImportFixture]:
    """Build shared fixtures for import policy comparisons."""
    canonical_json = json.dumps(
        {
            "type": "localization-estimate",
            "label": "ORB-SLAM3 Run A",
            "sourceType": "poses",
            "poses": [
                {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "timestampSeconds": 0.0,
                },
                {
                    "position": [1.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "timestampSeconds": 1.0,
                },
            ],
        }
    )
    commented_json = "\n".join(
        [
            "// exported from web review bundle",
            json.dumps(
                {
                    "type": "localization-estimate",
                    "label": "Commented Export",
                    "sourceType": "poses",
                    "poses": [
                        {
                            "position": [0.0, 0.0, 0.0],
                            "orientation": [0.0, 0.0, 0.0, 1.0],
                            "timestampSeconds": 0.0,
                        }
                    ],
                }
            ),
        ]
    )
    tum_text = "\n".join(
        [
            "# TUM trajectory",
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0",
            "1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0",
        ]
    )
    bracketed_text = "\n".join(
        [
            "[",
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0",
            "1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0",
            "]",
        ]
    )
    return [
        LocalizationEstimateImportFixture(
            fixture_id="canonical-json",
            label="Canonical JSON Estimate",
            intent="Preserve already-normalized localization estimates without reinterpreting them.",
            request=LocalizationEstimateImportRequest(canonical_json, file_name="orbslam_run_a.json"),
            expected_source_type="poses",
            expected_pose_count=2,
            expected_label="ORB-SLAM3 Run A",
        ),
        LocalizationEstimateImportFixture(
            fixture_id="tum-text",
            label="TUM Trajectory Text",
            intent="Keep a low-friction path for line-oriented trajectory logs.",
            request=LocalizationEstimateImportRequest(tum_text, file_name="orbslam3_camera_trajectory.txt"),
            expected_source_type="tum-trajectory-text",
            expected_pose_count=2,
            expected_label="orbslam3_camera_trajectory",
        ),
        LocalizationEstimateImportFixture(
            fixture_id="commented-json",
            label="Commented JSON Export",
            intent="Repair leading metadata comments emitted by experiment tooling without misclassifying the document as a trajectory.",
            request=LocalizationEstimateImportRequest(commented_json, file_name="commented_export.json"),
            expected_source_type="poses",
            expected_pose_count=1,
            expected_label="Commented Export",
        ),
        LocalizationEstimateImportFixture(
            fixture_id="bracketed-text-log",
            label="Bracketed Text Log",
            intent="Recover from text logs that are wrapped in lightweight brackets but are still line-oriented trajectories.",
            request=LocalizationEstimateImportRequest(bracketed_text, file_name="trajectory.log"),
            expected_source_type="tum-trajectory-text",
            expected_pose_count=2,
            expected_label="trajectory",
        ),
    ]


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_import_fixture(
    policy: LocalizationEstimateImportPolicy,
    fixture: LocalizationEstimateImportFixture,
) -> dict[str, Any]:
    """Run one import policy on one canonical document fixture."""
    started_at = perf_counter()
    try:
        parsed = policy.import_document(fixture.request)
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
    pose_count = len(parsed.get("poses", [])) if isinstance(parsed.get("poses"), list) else 0
    source_type_match = parsed.get("sourceType") == fixture.expected_source_type
    pose_count_match = pose_count == fixture.expected_pose_count
    label_match = parsed.get("label") == fixture.expected_label
    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": "ok",
        "sourceType": parsed.get("sourceType"),
        "poseCount": pose_count,
        "labelValue": parsed.get("label"),
        "sourceTypeMatch": source_type_match,
        "poseCountMatch": pose_count_match,
        "labelMatch": label_match,
        "schemaMatchScore": float(sum([source_type_match, pose_count_match, label_match]) / 3.0),
        "runtimeMs": runtime_ms,
    }


def benchmark_import_policy_runtime(
    policy: LocalizationEstimateImportPolicy,
    fixtures: Sequence[LocalizationEstimateImportFixture],
    *,
    repetitions: int,
) -> dict[str, float | int | None]:
    """Measure import policy runtime on shared fixtures."""
    samples_ms: list[float] = []
    for _ in range(max(1, int(repetitions))):
        for fixture in fixtures:
            started_at = perf_counter()
            try:
                policy.import_document(fixture.request)
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


def evaluate_readability(policy: LocalizationEstimateImportPolicy) -> dict[str, Any]:
    """Estimate readability from source shape. Heuristic, not normative."""
    source = textwrap.dedent(inspect.getsource(policy.import_document))
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
        notes.append("short enough to inspect while adding a new file format")
    else:
        notes.append("multiple branches require fixture coverage to stay trustworthy")
    if branch_count >= 4:
        notes.append("control flow mixes detection and repair logic")
    else:
        notes.append("control flow stays shallow")
    return {
        "score": round(score, 1),
        "linesOfCode": lines_of_code,
        "branchCount": branch_count,
        "notes": notes,
    }


def evaluate_extensibility(policy: LocalizationEstimateImportPolicy) -> dict[str, Any]:
    """Estimate extensibility from declared capability surface. Heuristic, not normative."""
    weights = {
        "usesFileNameHints": 3.0,
        "supportsCommentRepair": 3.0,
        "fallsBackAcrossFormats": 3.0,
    }
    supported = [key for key, enabled in policy.capabilities.items() if enabled]
    score = sum(weight for key, weight in weights.items() if policy.capabilities.get(key))
    notes = []
    if policy.capabilities.get("usesFileNameHints"):
        notes.append("can incorporate file-system context without changing callers")
    if policy.capabilities.get("supportsCommentRepair"):
        notes.append("can recover from exported metadata comments")
    if policy.capabilities.get("fallsBackAcrossFormats"):
        notes.append("can try multiple decoders before giving up")
    return {
        "score": round(score, 1),
        "supportedCapabilities": supported,
        "notes": notes,
    }


def summarize_import_policy(
    policy: LocalizationEstimateImportPolicy,
    fixture_reports: Sequence[dict[str, Any]],
    runtime_report: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate quality, runtime, readability, and extensibility for one import policy."""
    successful = [report for report in fixture_reports if report["status"] == "ok"]
    success_rate = float(len(successful) / max(1, len(fixture_reports)))
    aggregate = {
        "successRate": success_rate,
        "schemaMatchRate": _mean_or_none([report.get("schemaMatchScore") for report in successful]),
        "sourceTypeMatchRate": _mean_or_none([1.0 if report.get("sourceTypeMatch") else 0.0 for report in successful]),
        "labelMatchRate": _mean_or_none([1.0 if report.get("labelMatch") else 0.0 for report in successful]),
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


def build_localization_estimate_import_experiment_report(*, repetitions: int = 200) -> dict[str, Any]:
    """Compare localization estimate import policies on shared fixtures."""
    fixtures = build_localization_estimate_import_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "fileName": fixture.request.file_name,
            "expectedSourceType": fixture.expected_source_type,
            "expectedPoseCount": fixture.expected_pose_count,
        }
        for fixture in fixtures
    ]
    policy_reports = []
    for policy in EXPERIMENT_LOCALIZATION_ESTIMATE_IMPORT_POLICIES:
        fixture_reports = [evaluate_import_fixture(policy, fixture) for fixture in fixtures]
        runtime_report = benchmark_import_policy_runtime(policy, fixtures, repetitions=repetitions)
        policy_reports.append(summarize_import_policy(policy, fixture_reports, runtime_report))

    best_schema = max(
        policy_reports,
        key=lambda report: (
            float(report["aggregate"]["schemaMatchRate"] or 0.0),
            float(report["aggregate"]["successRate"] or 0.0),
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
        "type": "localization-estimate-import-experiment-report",
        "createdAt": datetime.now(UTC).isoformat(),
        "problem": {
            "name": "localization-estimate-import",
            "statement": (
                "Import localization estimate documents without freezing one parser path for normalized JSON, "
                "line-oriented trajectories, and lightly corrupted experiment exports."
            ),
            "stableInterface": "import_localization_estimate_document(LocalizationEstimateImportRequest(...))",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": ["successRate", "schemaMatchRate", "sourceTypeMatchRate", "labelMatchRate"],
            "runtime": ["meanMs", "medianMs"],
            "readability": ["score", "linesOfCode", "branchCount"],
            "extensibility": ["score", "supportedCapabilities"],
            "heuristicNotice": "Readability/extensibility are generated heuristics, not objective truth.",
        },
        "policies": policy_reports,
        "highlights": {
            "bestSchemaMatch": {
                "policy": best_schema["name"],
                "label": best_schema["label"],
                "schemaMatchRate": best_schema["aggregate"]["schemaMatchRate"],
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


def build_localization_estimate_import_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the importer report into a shared docs section."""
    comparison_rows = []
    for policy in report["policies"]:
        aggregate = policy["aggregate"]
        comparison_rows.append(
            [
                policy["label"],
                policy["tier"],
                policy["style"],
                f"{float(aggregate['successRate'] or 0.0):.2f}",
                f"{float(aggregate['schemaMatchRate'] or 0.0):.2f}",
                f"{float(aggregate['sourceTypeMatchRate'] or 0.0):.2f}",
                f"{float(aggregate['labelMatchRate'] or 0.0):.2f}",
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
                    fixture_report.get("sourceType", "n/a"),
                    str(fixture_report.get("poseCount", "n/a")),
                    "yes" if fixture_report.get("labelMatch") else "no",
                    f"{float(fixture_report.get('schemaMatchScore') or 0.0):.2f}"
                    if fixture_report["status"] == "ok"
                    else "n/a",
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Policy", "Status", "Source", "Poses", "Label Match", "Schema Score"],
                "rows": rows,
            }
        )

    return {
        "title": "Localization Estimate Import",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Policy",
            "Tier",
            "Style",
            "Success",
            "Schema",
            "Source",
            "Label",
            "Runtime (ms)",
            "Readability",
            "Extensibility",
        ],
        "comparisonRows": comparison_rows,
        "fixtureSections": fixture_sections,
        "highlights": [
            f"Best schema preservation: `{report['highlights']['bestSchemaMatch']['label']}`",
            f"Fastest median runtime: `{report['highlights']['fastestMedianRuntime']['label']}`",
            f"Most readable implementation: `{report['highlights']['mostReadable']['label']}`",
            f"Broadest extension surface: `{report['highlights']['mostExtensible']['label']}`",
        ],
        "accepted": [
            "Stable code uses `import_localization_estimate_document(LocalizationEstimateImportRequest(...))` as the importer contract.",
            "`suffix_aware` is the default production policy because it preserves normalized JSON, repairs commented exports, and still accepts line-oriented trajectories.",
            "Alternative importer policies stay in `src/gs_sim2real/experiments/` until they improve shared fixture quality without expanding the stable surface.",
        ],
        "deferred": [
            "`strict_content_gate` stays experimental. It is simple, but it misclassifies or rejects lightly malformed exports.",
            "`fallback_cascade` stays experimental. It recovers more often than strict parsing, but it still lacks file-hint and comment-repair behavior.",
        ],
        "rules": [
            "Start parser work with at least three importer policies, not one branching function.",
            "Compare policies on the same raw documents and the same schema-match metrics before changing production defaults.",
            "Promote only the smallest importer contract that multiple formats can share.",
        ],
        "stableInterfaceIntro": "The stable localization-estimate importer surface is intentionally small:",
        "stableInterfaceCode": textwrap.dedent(
            """
            @dataclass(frozen=True)
            class LocalizationEstimateImportRequest:
                raw_text: str
                file_name: str | None = None

            def import_localization_estimate_document(
                request: LocalizationEstimateImportRequest,
                *,
                policy: str = 'suffix_aware',
            ) -> dict[str, Any]: ...
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `capabilities`",
            "`import_document(request) -> normalized localization-estimate dict`",
        ],
        "comparableInputs": [
            "Same raw document fixtures for every policy",
            "Same fixtures (`canonical-json`, `tum-text`, `commented-json`, `bracketed-text-log`)",
            "Same evaluation axes: schema preservation, runtime, readability heuristic, extensibility heuristic",
        ],
        "boundary": [
            "`src/gs_sim2real/core/`: stable importer contract for production evaluation code",
            "`src/gs_sim2real/experiments/`: discardable importer policies and comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the localization estimate import lab and optionally refresh docs."""
    report = build_localization_estimate_import_experiment_report(repetitions=args.repetitions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(
            docs_dir=args.docs_dir,
            localization_import_report=report,
        )
    summary = {
        "type": report["type"],
        "policyCount": len(report["policies"]),
        "fixtureCount": len(report["fixtures"]),
        "bestSchemaMatch": report["highlights"]["bestSchemaMatch"],
        "fastestMedianRuntime": report["highlights"]["fastestMedianRuntime"],
        "docs": docs,
    }
    print(json.dumps(summary, indent=2))
