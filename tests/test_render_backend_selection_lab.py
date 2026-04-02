"""Tests for experiment-first render backend selection."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.render_backend_selection import (
    RenderBackendCapabilities,
    RenderBackendPreferences,
    RenderBackendRequest,
    select_render_backend,
)
from gs_sim2real.experiments.render_backend_selection_lab import (
    build_render_backend_selection_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_select_render_backend_balanced_prefers_simple_for_fast_preview() -> None:
    selection = select_render_backend(
        RenderBackendRequest(
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
        )
    )

    assert selection.name == "simple"
    assert "prefers low startup latency" in selection.reason


def test_render_backend_selection_lab_report_compares_three_policies() -> None:
    report = build_render_backend_selection_experiment_report(repetitions=4)

    assert report["type"] == "render-backend-selection-experiment-report"
    assert len(report["fixtures"]) >= 4
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"simple_safe", "balanced", "fidelity_first"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "balanced"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_render_backend_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "# Experiments" in experiments_text
    assert "## Localization Alignment" in experiments_text
    assert "## Render Backend Selection" in experiments_text
    assert "## Render Backend Selection" in decisions_text
    assert "select_render_backend" in interfaces_text
