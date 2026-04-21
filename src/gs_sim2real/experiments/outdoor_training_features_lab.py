"""Experiment lab for outdoor training feature bundles."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


@dataclass(frozen=True)
class OutdoorTrainingFixture:
    """Canonical outdoor scenario shared by every feature profile."""

    fixture_id: str
    label: str
    intent: str
    has_lidar_depth: bool
    has_sky_masks: bool
    exposure_variance: float
    pose_noise: float
    sky_fraction: float
    interactive_budget: bool = False


@dataclass(frozen=True)
class OutdoorTrainingProfile:
    """A comparable training feature bundle."""

    name: str
    label: str
    tier: str
    style: str
    config_path: str | None
    use_depth: bool
    use_appearance: bool
    use_pose_refinement: bool
    use_sky_mask: bool
    relative_cost: float


EXPERIMENT_OUTDOOR_TRAINING_PROFILES: tuple[OutdoorTrainingProfile, ...] = (
    OutdoorTrainingProfile(
        name="rgb_only",
        label="RGB Only",
        tier="baseline",
        style="fast-preview",
        config_path=None,
        use_depth=False,
        use_appearance=False,
        use_pose_refinement=False,
        use_sky_mask=False,
        relative_cost=1.0,
    ),
    OutdoorTrainingProfile(
        name="depth_supervised",
        label="Depth Supervised",
        tier="core",
        style="geometry-anchored",
        config_path="configs/training_depth.yaml",
        use_depth=True,
        use_appearance=False,
        use_pose_refinement=False,
        use_sky_mask=False,
        relative_cost=1.2,
    ),
    OutdoorTrainingProfile(
        name="depth_appearance",
        label="Depth + Appearance",
        tier="core",
        style="exposure-robust",
        config_path="configs/training_appearance.yaml",
        use_depth=True,
        use_appearance=True,
        use_pose_refinement=False,
        use_sky_mask=False,
        relative_cost=1.35,
    ),
    OutdoorTrainingProfile(
        name="depth_appearance_pose",
        label="Depth + Appearance + Pose",
        tier="production-default",
        style="outdoor-supervised",
        config_path="configs/training_ba.yaml",
        use_depth=True,
        use_appearance=True,
        use_pose_refinement=True,
        use_sky_mask=False,
        relative_cost=1.6,
    ),
    OutdoorTrainingProfile(
        name="sky_masked_full",
        label="Sky-Masked Full",
        tier="experiment",
        style="sky-aware",
        config_path=None,
        use_depth=True,
        use_appearance=True,
        use_pose_refinement=True,
        use_sky_mask=True,
        relative_cost=1.75,
    ),
)


def build_outdoor_training_fixtures() -> list[OutdoorTrainingFixture]:
    """Build canonical outdoor scenarios for feature-bundle comparisons."""
    return [
        OutdoorTrainingFixture(
            fixture_id="autoware-fused-multibag",
            label="Autoware Fused Multi-Bag",
            intent="Compare the current production supervised stack under multi-session exposure and pose drift.",
            has_lidar_depth=True,
            has_sky_masks=False,
            exposure_variance=0.75,
            pose_noise=0.45,
            sky_fraction=0.35,
        ),
        OutdoorTrainingFixture(
            fixture_id="mcd-valid-gnss-single-session",
            label="MCD Valid-GNSS Single Session",
            intent="Keep the supervised MCD path focused on GNSS/LiDAR geometry without overfitting to exposure changes.",
            has_lidar_depth=True,
            has_sky_masks=False,
            exposure_variance=0.25,
            pose_noise=0.55,
            sky_fraction=0.25,
        ),
        OutdoorTrainingFixture(
            fixture_id="waymo-masked-front-camera",
            label="Waymo Masked Front Camera",
            intent="Exercise the only currently modelled case where sky/dynamic masks are available as preprocessing artifacts.",
            has_lidar_depth=True,
            has_sky_masks=True,
            exposure_variance=0.45,
            pose_noise=0.25,
            sky_fraction=0.45,
        ),
        OutdoorTrainingFixture(
            fixture_id="pose-free-slam-comparison",
            label="Pose-Free SLAM Comparison",
            intent="Avoid selecting supervised-only features when comparing external SLAM artifacts with no LiDAR depth maps.",
            has_lidar_depth=False,
            has_sky_masks=False,
            exposure_variance=0.35,
            pose_noise=0.65,
            sky_fraction=0.20,
            interactive_budget=True,
        ),
    ]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def evaluate_outdoor_training_profile(
    profile: OutdoorTrainingProfile,
    fixture: OutdoorTrainingFixture,
) -> dict[str, Any]:
    """Score one feature profile on one outdoor fixture."""
    missing_inputs: list[str] = []
    quality = 0.35

    if profile.use_depth:
        if fixture.has_lidar_depth:
            quality += 0.26 + 0.12 * fixture.pose_noise
        else:
            quality -= 0.22
            missing_inputs.append("lidar_depth")
    elif fixture.has_lidar_depth:
        quality -= 0.10 + 0.08 * fixture.pose_noise

    if profile.use_appearance:
        quality += 0.06 + 0.22 * fixture.exposure_variance
    elif fixture.exposure_variance > 0.4:
        quality -= 0.10 * fixture.exposure_variance

    if profile.use_pose_refinement:
        quality += 0.05 + 0.16 * fixture.pose_noise
    elif fixture.pose_noise > 0.4:
        quality -= 0.08 * fixture.pose_noise

    if profile.use_sky_mask:
        if fixture.has_sky_masks:
            quality += 0.05 + 0.18 * fixture.sky_fraction
        else:
            quality -= 0.20
            missing_inputs.append("sky_masks")
    elif fixture.sky_fraction > 0.35:
        quality -= 0.06 * fixture.sky_fraction

    if fixture.interactive_budget and profile.relative_cost > 1.35:
        quality -= 0.10 * (profile.relative_cost - 1.35)

    cost_penalty = 0.06 * max(0.0, profile.relative_cost - 1.0)
    input_readiness = 1.0 - min(1.0, len(missing_inputs) / 2.0)
    fit_score = _clamp01(quality - cost_penalty)
    status = "ok" if not missing_inputs else "missing-inputs"

    return {
        "fixtureId": fixture.fixture_id,
        "label": fixture.label,
        "intent": fixture.intent,
        "status": status,
        "qualityScore": _clamp01(quality),
        "fitScore": fit_score,
        "inputReadiness": input_readiness,
        "relativeCost": float(profile.relative_cost),
        "missingInputs": missing_inputs,
    }


def summarize_outdoor_training_profile(
    profile: OutdoorTrainingProfile,
    fixture_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate fixture scores for one feature profile."""
    success = [1.0 if item["status"] == "ok" else 0.0 for item in fixture_reports]
    fit_scores = [float(item["fitScore"]) for item in fixture_reports]
    quality_scores = [float(item["qualityScore"]) for item in fixture_reports]
    readiness_scores = [float(item["inputReadiness"]) for item in fixture_reports]
    return {
        "name": profile.name,
        "label": profile.label,
        "tier": profile.tier,
        "style": profile.style,
        "configPath": profile.config_path,
        "features": {
            "depth": profile.use_depth,
            "appearance": profile.use_appearance,
            "poseRefinement": profile.use_pose_refinement,
            "skyMask": profile.use_sky_mask,
        },
        "relativeCost": float(profile.relative_cost),
        "fixtures": list(fixture_reports),
        "aggregate": {
            "successRate": _mean_or_none(success),
            "meanFitScore": _mean_or_none(fit_scores),
            "meanQualityScore": _mean_or_none(quality_scores),
            "meanInputReadiness": _mean_or_none(readiness_scores),
        },
    }


def build_outdoor_training_features_experiment_report() -> dict[str, Any]:
    """Compare outdoor 3DGS training feature bundles on shared fixtures."""
    fixtures = build_outdoor_training_fixtures()
    fixture_summaries = [
        {
            "fixtureId": fixture.fixture_id,
            "label": fixture.label,
            "intent": fixture.intent,
            "hasLidarDepth": fixture.has_lidar_depth,
            "hasSkyMasks": fixture.has_sky_masks,
            "exposureVariance": fixture.exposure_variance,
            "poseNoise": fixture.pose_noise,
            "skyFraction": fixture.sky_fraction,
            "interactiveBudget": fixture.interactive_budget,
        }
        for fixture in fixtures
    ]
    profile_reports = [
        summarize_outdoor_training_profile(
            profile,
            [evaluate_outdoor_training_profile(profile, fixture) for fixture in fixtures],
        )
        for profile in EXPERIMENT_OUTDOOR_TRAINING_PROFILES
    ]

    best_overall = max(
        profile_reports,
        key=lambda report: (
            float(report["aggregate"]["meanFitScore"] or 0.0),
            float(report["aggregate"]["successRate"] or 0.0),
        ),
    )
    lowest_cost_viable = min(
        (report for report in profile_reports if float(report["aggregate"]["meanFitScore"] or 0.0) >= 0.60),
        key=lambda report: float(report["relativeCost"]),
    )
    waymo_fixture_id = "waymo-masked-front-camera"
    best_sky_fixture = max(
        profile_reports,
        key=lambda report: next(
            float(item["fitScore"]) for item in report["fixtures"] if item["fixtureId"] == waymo_fixture_id
        ),
    )

    return {
        "protocol": "gs-sim2real-experiment-report/v1",
        "type": "outdoor-training-features-experiment-report",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "problem": {
            "name": "outdoor-training-features",
            "statement": (
                "Compare depth supervision, per-image appearance correction, pose refinement, and sky masking as "
                "separable outdoor 3DGS training features before promoting a new default."
            ),
            "stableInterface": "training config keys: depth_loss_weight, appearance_embedding, joint_pose_refinement",
        },
        "fixtures": fixture_summaries,
        "metrics": {
            "quality": ["meanQualityScore", "meanFitScore"],
            "inputReadiness": ["successRate", "meanInputReadiness"],
            "cost": ["relativeCost"],
            "heuristicNotice": "Scores are fixture heuristics for planning; real PSNR/LPIPS runs still own final claims.",
        },
        "profiles": profile_reports,
        "highlights": {
            "bestOverall": {
                "profile": best_overall["name"],
                "label": best_overall["label"],
                "meanFitScore": best_overall["aggregate"]["meanFitScore"],
            },
            "lowestCostViable": {
                "profile": lowest_cost_viable["name"],
                "label": lowest_cost_viable["label"],
                "relativeCost": lowest_cost_viable["relativeCost"],
            },
            "bestSkyFixture": {
                "profile": best_sky_fixture["name"],
                "label": best_sky_fixture["label"],
                "fixtureId": waymo_fixture_id,
            },
        },
    }


def build_outdoor_training_features_process_section(report: dict[str, Any]) -> dict[str, Any]:
    """Convert the outdoor training feature report into a shared docs section."""
    comparison_rows = []
    for profile in report["profiles"]:
        features = profile["features"]
        comparison_rows.append(
            [
                profile["label"],
                profile["tier"],
                profile["style"],
                "yes" if features["depth"] else "no",
                "yes" if features["appearance"] else "no",
                "yes" if features["poseRefinement"] else "no",
                "yes" if features["skyMask"] else "no",
                f"{float(profile['aggregate']['meanFitScore'] or 0.0):.3f}",
                f"{float(profile['aggregate']['successRate'] or 0.0):.2f}",
                f"{float(profile['relativeCost']):.2f}",
            ]
        )

    fixture_sections = []
    for fixture in report["fixtures"]:
        rows = []
        for profile in report["profiles"]:
            fixture_report = next(item for item in profile["fixtures"] if item["fixtureId"] == fixture["fixtureId"])
            missing = ", ".join(fixture_report["missingInputs"]) or "none"
            rows.append(
                [
                    profile["label"],
                    fixture_report["status"],
                    f"{float(fixture_report['fitScore']):.3f}",
                    f"{float(fixture_report['qualityScore']):.3f}",
                    missing,
                ]
            )
        fixture_sections.append(
            {
                "title": fixture["label"],
                "intent": fixture["intent"],
                "headers": ["Profile", "Status", "Fit", "Quality", "Missing Inputs"],
                "rows": rows,
            }
        )

    return {
        "title": "Outdoor Training Features",
        "updatedAt": report["createdAt"],
        "problemStatement": report["problem"]["statement"],
        "comparisonHeaders": [
            "Profile",
            "Tier",
            "Style",
            "Depth",
            "Appearance",
            "Pose",
            "Sky",
            "Fit",
            "Ready",
            "Cost",
        ],
        "comparisonRows": comparison_rows,
        "fixtureSections": fixture_sections,
        "highlights": [
            f"Best overall fit: `{report['highlights']['bestOverall']['label']}`",
            f"Lowest-cost viable profile: `{report['highlights']['lowestCostViable']['label']}`",
            f"Best sky-mask fixture fit: `{report['highlights']['bestSkyFixture']['label']}`",
        ],
        "accepted": [
            "Depth supervision, appearance correction, and pose refinement stay separable config-level features.",
            "`configs/training_ba.yaml` is the current supervised outdoor default when LiDAR depth exists and training budget allows pose refinement.",
            "Sky masking remains experimental until sky-mask artifacts and trainer loss masking are both available in the target dataset.",
        ],
        "deferred": [
            "`rgb_only` remains a fast baseline, not an outdoor quality default.",
            "`sky_masked_full` remains experimental because most current MCD/Autoware fixtures lack sky-mask artifacts.",
        ],
        "rules": [
            "Compare feature bundles on the same outdoor fixtures before changing training defaults.",
            "Do not enable a feature by default when its required artifacts are missing from the preprocessing contract.",
            "Keep feature-bundle selection in configs/docs; keep trainer code limited to the explicit feature keys it already supports.",
        ],
        "stableInterfaceIntro": "The stable surface is the training config feature contract, not an auto-selector:",
        "stableInterfaceCode": textwrap.dedent(
            """
            depth_loss_weight: float
            appearance_embedding: bool
            appearance_lr: float
            appearance_reg_weight: float
            joint_pose_refinement: bool
            joint_pose_lr: float
            joint_pose_reg_weight: float
            """
        ).strip(),
        "experimentContract": [
            "`name`, `label`, `style`, `tier`, `features`, `relativeCost`",
            "`evaluate_outdoor_training_profile(profile, fixture) -> score report`",
        ],
        "comparableInputs": [
            "Same outdoor scenario fixtures for every feature profile",
            "Same feature availability signals (`has_lidar_depth`, `has_sky_masks`)",
            "Same evaluation axes: fit, input readiness, relative cost",
        ],
        "boundary": [
            "`configs/`: stable named training feature bundles",
            "`src/gs_sim2real/train/`: explicit feature-key implementation, no automatic profile selection",
            "`src/gs_sim2real/experiments/`: discardable feature-bundle comparison harnesses",
        ],
    }


def run_cli(args: argparse.Namespace) -> None:
    """Run the outdoor training feature lab and optionally refresh docs."""
    report = build_outdoor_training_features_experiment_report()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    docs = None
    if args.write_docs:
        from .report_docs import write_repo_experiment_process_docs

        docs = write_repo_experiment_process_docs(docs_dir=args.docs_dir, outdoor_training_features_report=report)
    summary = {
        "type": report["type"],
        "profileCount": len(report["profiles"]),
        "fixtureCount": len(report["fixtures"]),
        "bestOverall": report["highlights"]["bestOverall"],
        "lowestCostViable": report["highlights"]["lowestCostViable"],
        "bestSkyFixture": report["highlights"]["bestSkyFixture"],
        "docs": docs,
    }
    print(json.dumps(summary, indent=2))


__all__ = [
    "EXPERIMENT_OUTDOOR_TRAINING_PROFILES",
    "OutdoorTrainingFixture",
    "OutdoorTrainingProfile",
    "build_outdoor_training_features_experiment_report",
    "build_outdoor_training_features_process_section",
    "build_outdoor_training_fixtures",
    "evaluate_outdoor_training_profile",
]
