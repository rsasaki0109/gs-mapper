"""Tests for experiment-first route capture bundle import."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.route_capture_bundle_import import (
    RouteCaptureBundleImportRequest,
    import_route_capture_bundle,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs
from gs_sim2real.experiments.route_capture_bundle_import_lab import (
    build_route_capture_bundle_import_experiment_report,
)


def test_import_route_capture_bundle_route_aware_recovers_route_pose() -> None:
    bundle = {
        "type": "route-capture-bundle",
        "fragmentLabel": "Route Pose",
        "route": [
            {
                "position": [4.0, 1.0, 0.5],
                "yawDegrees": 22.5,
            }
        ],
        "captures": [
            {
                "label": "gt:1",
                "response": {
                    "type": "render-result",
                    "colorJpegBase64": "ZmFrZS1qcGVn",
                    "pose": {
                        "position": [99.0, 99.0, 99.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        ],
    }

    parsed = import_route_capture_bundle(RouteCaptureBundleImportRequest(bundle))

    assert parsed["type"] == "route-capture-bundle"
    assert parsed["captures"][0]["pose"]["position"] == [4.0, 1.0, 0.5]
    assert parsed["captures"][0]["pose"]["yawDegrees"] == 22.5


def test_route_capture_bundle_import_lab_report_compares_three_policies() -> None:
    report = build_route_capture_bundle_import_experiment_report(repetitions=4)

    assert report["type"] == "route-capture-bundle-import-experiment-report"
    assert len(report["fixtures"]) >= 3
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"strict_canonical", "response_pose_fallback", "route_aware"}.issubset(policy_names)
    assert report["highlights"]["bestFit"]["policy"] == "route_aware"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_route_capture_bundle_import_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Route Capture Bundle Import" in experiments_text
    assert "## Route Capture Bundle Import" in decisions_text
    assert "import_route_capture_bundle" in interfaces_text
