"""Snapshot test for scripts/smoke_route_policy_scenario_ci.py.

Guards the end-to-end smoke chain (matrix -> shard runs -> merge ->
manifest -> workflow materialization -> validation -> activation ->
review -> promotion) so the script keeps working as the modules under
``gs_sim2real.sim`` evolve.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "smoke_route_policy_scenario_ci.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("smoke_route_policy_scenario_ci", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_file_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_run_smoke_produces_passing_chain(tmp_path: Path) -> None:
    module = _load_script_module()

    log_lines: list[str] = []
    artifacts = module.run_smoke(tmp_path, log=log_lines.append)

    # Every gate must have logged PASS; no FAIL markers anywhere.
    gate_markers = [line for line in log_lines if line.startswith("[PASS]") or line.startswith("[FAIL]")]
    assert gate_markers, f"expected gate markers in log, got: {log_lines}"
    assert all(line.startswith("[PASS]") for line in gate_markers), gate_markers

    # The logged gates must cover every stage of the chain.
    expected_gate_suffixes = (
        "scenario-set run",
        "shard merge",
        "workflow validation",
        "workflow activation",
        "review artifact",
        "workflow promotion",
        "workflow adoption",
    )
    for suffix in expected_gate_suffixes:
        assert any(suffix in line for line in gate_markers), f"no PASS line for gate '{suffix}' in {gate_markers}"

    # Every returned artifact path exists under tmp_path.
    for key, path in artifacts.items():
        assert path.exists(), f"missing artifact {key} at {path}"
        assert tmp_path in path.parents or path == tmp_path, f"artifact {key} escaped tmp_path: {path}"

    # Both active workflows are pinned under tmp_path/.github/workflows/,
    # never near the real repo workflows directory.
    active = artifacts["active_workflow"]
    adopted_active = artifacts["adopted_active_workflow"]
    assert active.parent == tmp_path / ".github" / "workflows"
    assert adopted_active.parent == tmp_path / ".github" / "workflows"
    assert active != adopted_active, "manual and adopted active workflows must be distinct"
    repo_workflows = REPO_ROOT / ".github" / "workflows"
    assert repo_workflows not in active.parents
    assert repo_workflows not in adopted_active.parents

    # Promotion report should record the expected trigger config.
    promotion_payload = json.loads(artifacts["promotion"].read_text(encoding="utf-8"))
    assert promotion_payload["promoted"] is True
    assert promotion_payload["triggerMode"] == "pull-request"
    assert promotion_payload["pullRequestBranches"] == ["main"]

    # Adoption report carries the promoted triggers forward.
    adoption_payload = json.loads(artifacts["adoption"].read_text(encoding="utf-8"))
    assert adoption_payload["adopted"] is True
    assert adoption_payload["triggerMode"] == "pull-request"
    assert adoption_payload["pullRequestBranches"] == ["main"]
    assert adoption_payload["manualActiveWorkflowPath"] != adoption_payload["adoptedActiveWorkflowPath"]

    # Adopted YAML gets the pull_request trigger block; manual one only has
    # workflow_dispatch so the review diff is meaningful.
    manual_text = active.read_text(encoding="utf-8")
    adopted_text = adopted_active.read_text(encoding="utf-8")
    assert "workflow_dispatch" in manual_text
    assert "pull_request:" not in manual_text
    assert "workflow_dispatch" in adopted_text
    assert "pull_request:" in adopted_text
    assert "- 'main'" in adopted_text


def test_main_returns_zero_on_clean_run(tmp_path: Path) -> None:
    module = _load_script_module()

    rc = module.main(["--root", str(tmp_path / "run")])

    assert rc == 0
