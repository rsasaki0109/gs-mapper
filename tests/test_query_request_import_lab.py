"""Tests for experiment-first query request import."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.query_request_import import (
    QueryRequestImportRequest,
    RenderQueryDefaults,
    import_query_request,
)
from gs_sim2real.experiments.query_request_import_lab import (
    build_query_request_import_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_import_query_request_alias_friendly_supports_pose_shortcut_render_payloads() -> None:
    imported = import_query_request(
        QueryRequestImportRequest(
            payload={
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "width": 160,
                "height": 120,
                "fovDeg": 55.0,
            },
            defaults=RenderQueryDefaults(
                width=640,
                height=480,
                fov_degrees=60.0,
                near_clip=0.05,
                far_clip=50.0,
                point_radius=1,
            ),
        )
    )

    assert imported.request_type == "render"
    assert imported.render is not None
    assert imported.render.position == (1.0, 2.0, 3.0)
    assert imported.render.width == 160
    assert imported.render.fov_degrees == 55.0


def test_query_request_import_lab_report_compares_three_policies() -> None:
    report = build_query_request_import_experiment_report(repetitions=4)

    assert report["type"] == "query-request-import-experiment-report"
    assert len(report["fixtures"]) >= 5
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"strict_schema", "envelope_first", "alias_friendly"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "alias_friendly"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_query_request_import_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Query Request Import" in experiments_text
    assert "## Query Request Import" in decisions_text
    assert "import_query_request" in interfaces_text
