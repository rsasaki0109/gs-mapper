"""Tests for experiment-first localization estimate import."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.core.localization_estimate_import import (
    LocalizationEstimateImportRequest,
    import_localization_estimate_document,
)
from gs_sim2real.experiments.localization_estimate_import_lab import (
    build_localization_estimate_import_experiment_report,
)
from gs_sim2real.experiments.report_docs import write_repo_experiment_process_docs


def test_import_localization_estimate_document_suffix_aware_repairs_commented_json() -> None:
    request = LocalizationEstimateImportRequest(
        raw_text="\n".join(
            [
                "// exported from experiment shelf",
                '{"type":"localization-estimate","label":"Commented Export","sourceType":"poses","poses":[{"position":[0,0,0],"orientation":[0,0,0,1],"timestampSeconds":0}]}',
            ]
        ),
        file_name="commented_export.json",
    )

    parsed = import_localization_estimate_document(request)

    assert parsed["sourceType"] == "poses"
    assert parsed["label"] == "Commented Export"
    assert len(parsed["poses"]) == 1


def test_localization_estimate_import_lab_report_compares_three_policies() -> None:
    report = build_localization_estimate_import_experiment_report(repetitions=4)

    assert report["type"] == "localization-estimate-import-experiment-report"
    assert len(report["fixtures"]) >= 4
    assert len(report["policies"]) >= 3
    policy_names = {policy["name"] for policy in report["policies"]}
    assert {"strict_content_gate", "fallback_cascade", "suffix_aware"}.issubset(policy_names)
    assert report["highlights"]["bestSchemaMatch"]["policy"] == "suffix_aware"
    for policy in report["policies"]:
        assert len(policy["fixtures"]) == len(report["fixtures"])
        assert policy["readability"]["score"] >= 1.0
        assert policy["extensibility"]["score"] >= 0.0


def test_repo_experiment_docs_include_localization_import_section(tmp_path: Path) -> None:
    outputs = write_repo_experiment_process_docs(docs_dir=tmp_path)

    experiments_text = Path(outputs["experiments"]).read_text(encoding="utf-8")
    decisions_text = Path(outputs["decisions"]).read_text(encoding="utf-8")
    interfaces_text = Path(outputs["interfaces"]).read_text(encoding="utf-8")

    assert "## Localization Estimate Import" in experiments_text
    assert "## Localization Estimate Import" in decisions_text
    assert "import_localization_estimate_document" in interfaces_text
