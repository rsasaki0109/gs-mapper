"""GitHub Actions workflow materialization for scenario CI manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from os import path as os_path
from pathlib import Path
import shlex
import re
from typing import Any

from .policy_scenario_ci_manifest import (
    RoutePolicyScenarioCIManifest,
    load_route_policy_scenario_ci_manifest_json,
)


ROUTE_POLICY_SCENARIO_CI_WORKFLOW_VERSION = "gs-mapper-route-policy-scenario-ci-workflow/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowConfig:
    """Stable settings used to materialize a GitHub Actions workflow."""

    workflow_id: str = "route-policy-scenario-shards"
    workflow_name: str = "Route Policy Scenario Shards"
    runs_on: str = "ubuntu-latest"
    python_version: str = "3.11"
    install_command: str = 'pip install -e ".[dev]"'
    artifact_root: str | None = None
    artifact_retention_days: int = 7
    workflow_dispatch: bool = True
    push_branches: tuple[str, ...] = ()
    pull_request_branches: tuple[str, ...] = ()
    fail_fast: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.workflow_name):
            raise ValueError("workflow_name must not be empty")
        if not str(self.runs_on):
            raise ValueError("runs_on must not be empty")
        if not str(self.python_version):
            raise ValueError("python_version must not be empty")
        if not str(self.install_command):
            raise ValueError("install_command must not be empty")
        _positive_int(self.artifact_retention_days, "artifact_retention_days")
        push_branches = tuple(str(branch) for branch in self.push_branches)
        pull_request_branches = tuple(str(branch) for branch in self.pull_request_branches)
        if not self.workflow_dispatch and not push_branches and not pull_request_branches:
            raise ValueError("workflow must have at least one trigger")
        object.__setattr__(self, "push_branches", push_branches)
        object.__setattr__(self, "pull_request_branches", pull_request_branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow-config",
            "workflowId": self.workflow_id,
            "workflowName": self.workflow_name,
            "runsOn": self.runs_on,
            "pythonVersion": self.python_version,
            "installCommand": self.install_command,
            "artifactRoot": self.artifact_root,
            "artifactRetentionDays": self.artifact_retention_days,
            "workflowDispatch": self.workflow_dispatch,
            "pushBranches": list(self.push_branches),
            "pullRequestBranches": list(self.pull_request_branches),
            "failFast": self.fail_fast,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowMaterialization:
    """Materialized workflow YAML and compact metadata."""

    workflow_id: str
    manifest_id: str
    workflow_name: str
    workflow_yaml: str
    config: RoutePolicyScenarioCIWorkflowConfig
    workflow_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_WORKFLOW_VERSION

    def __post_init__(self) -> None:
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.workflow_name):
            raise ValueError("workflow_name must not be empty")
        if not str(self.workflow_yaml):
            raise ValueError("workflow_yaml must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow",
            "version": self.version,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "workflowName": self.workflow_name,
            "workflowPath": self.workflow_path,
            "workflowYaml": self.workflow_yaml,
            "config": self.config.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


def materialize_route_policy_scenario_ci_workflow(
    manifest: RoutePolicyScenarioCIManifest,
    *,
    config: RoutePolicyScenarioCIWorkflowConfig | None = None,
) -> RoutePolicyScenarioCIWorkflowMaterialization:
    """Render a GitHub Actions workflow from a scenario CI manifest."""

    resolved_config = config or RoutePolicyScenarioCIWorkflowConfig()
    artifact_root = resolved_config.artifact_root or _common_artifact_root(manifest)
    workflow_config = RoutePolicyScenarioCIWorkflowConfig(
        workflow_id=resolved_config.workflow_id,
        workflow_name=resolved_config.workflow_name,
        runs_on=resolved_config.runs_on,
        python_version=resolved_config.python_version,
        install_command=resolved_config.install_command,
        artifact_root=artifact_root,
        artifact_retention_days=resolved_config.artifact_retention_days,
        workflow_dispatch=resolved_config.workflow_dispatch,
        push_branches=resolved_config.push_branches,
        pull_request_branches=resolved_config.pull_request_branches,
        fail_fast=resolved_config.fail_fast,
        metadata=resolved_config.metadata,
    )
    workflow_yaml = _render_workflow_yaml(manifest, workflow_config)
    return RoutePolicyScenarioCIWorkflowMaterialization(
        workflow_id=workflow_config.workflow_id,
        manifest_id=manifest.manifest_id,
        workflow_name=workflow_config.workflow_name,
        workflow_yaml=workflow_yaml,
        config=workflow_config,
        metadata={
            "manifestId": manifest.manifest_id,
            "shardJobCount": manifest.shard_job_count,
            "scenarioCount": manifest.scenario_count,
        },
    )


def write_route_policy_scenario_ci_workflow_yaml(
    path: str | Path,
    materialization: RoutePolicyScenarioCIWorkflowMaterialization,
) -> Path:
    """Write a materialized GitHub Actions workflow YAML file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(materialization.workflow_yaml, encoding="utf-8")
    return output_path


def write_route_policy_scenario_ci_workflow_json(
    path: str | Path,
    materialization: RoutePolicyScenarioCIWorkflowMaterialization,
) -> Path:
    """Write workflow materialization metadata as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = materialization.to_dict()
    payload["workflowPath"] = materialization.workflow_path
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_workflow_json(path: str | Path) -> RoutePolicyScenarioCIWorkflowMaterialization:
    """Load workflow materialization metadata from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_workflow_from_dict(_mapping(payload, "ciWorkflow"))


def route_policy_scenario_ci_workflow_config_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowConfig:
    """Rebuild workflow materialization config from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow-config")
    return RoutePolicyScenarioCIWorkflowConfig(
        workflow_id=str(payload["workflowId"]),
        workflow_name=str(payload["workflowName"]),
        runs_on=str(payload["runsOn"]),
        python_version=str(payload["pythonVersion"]),
        install_command=str(payload["installCommand"]),
        artifact_root=None if payload.get("artifactRoot") is None else str(payload["artifactRoot"]),
        artifact_retention_days=int(payload.get("artifactRetentionDays", 7)),
        workflow_dispatch=bool(payload.get("workflowDispatch", True)),
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
        fail_fast=bool(payload.get("failFast", False)),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_workflow_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowMaterialization:
    """Rebuild workflow materialization metadata from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_WORKFLOW_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_WORKFLOW_VERSION:
        raise ValueError(f"unsupported route policy scenario CI workflow version: {version}")
    return RoutePolicyScenarioCIWorkflowMaterialization(
        workflow_id=str(payload["workflowId"]),
        manifest_id=str(payload["manifestId"]),
        workflow_name=str(payload["workflowName"]),
        workflow_path=None if payload.get("workflowPath") is None else str(payload["workflowPath"]),
        workflow_yaml=str(payload["workflowYaml"]),
        config=route_policy_scenario_ci_workflow_config_from_dict(_mapping(payload["config"], "config")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_workflow_markdown(
    materialization: RoutePolicyScenarioCIWorkflowMaterialization,
) -> str:
    """Render a compact Markdown summary for a materialized workflow."""

    config = materialization.config
    lines = [
        f"# Route Policy Scenario CI Workflow: {materialization.workflow_id}",
        f"- Manifest: {materialization.manifest_id}",
        f"- Workflow: {materialization.workflow_name}",
        f"- Runs on: {config.runs_on}",
        f"- Python: {config.python_version}",
        f"- Artifact root: {config.artifact_root or 'n/a'}",
        "",
        "| Trigger | Branches |",
        "| --- | --- |",
    ]
    if config.workflow_dispatch:
        lines.append("| workflow_dispatch | n/a |")
    if config.push_branches:
        lines.append(f"| push | {', '.join(config.push_branches)} |")
    if config.pull_request_branches:
        lines.append(f"| pull_request | {', '.join(config.pull_request_branches)} |")
    lines.extend(
        [
            "",
            "## Jobs",
            "",
            "| Job | Purpose |",
            "| --- | --- |",
            "| route-policy-scenario-shards | matrix shard execution |",
            "| route-policy-scenario-merge | shard artifact merge and history gate |",
        ]
    )
    return "\n".join(lines) + "\n"


def run_cli(args: Any) -> None:
    """Run the route policy scenario-ci-workflow CLI."""

    manifest = load_route_policy_scenario_ci_manifest_json(getattr(args, "manifest"))
    config = RoutePolicyScenarioCIWorkflowConfig(
        workflow_id=str(getattr(args, "workflow_id")),
        workflow_name=str(getattr(args, "workflow_name")),
        runs_on=str(getattr(args, "runs_on")),
        python_version=str(getattr(args, "python_version")),
        install_command=str(getattr(args, "install_command")),
        artifact_root=getattr(args, "artifact_root", None),
        artifact_retention_days=int(getattr(args, "artifact_retention_days")),
        workflow_dispatch=not bool(getattr(args, "no_workflow_dispatch", False)),
        push_branches=tuple(getattr(args, "push_branch") or ()),
        pull_request_branches=tuple(getattr(args, "pull_request_branch") or ()),
        fail_fast=bool(getattr(args, "fail_fast", False)),
        metadata={"manifestPath": getattr(args, "manifest")},
    )
    materialization = materialize_route_policy_scenario_ci_workflow(manifest, config=config)
    workflow_path = write_route_policy_scenario_ci_workflow_yaml(getattr(args, "workflow_output"), materialization)
    materialization = RoutePolicyScenarioCIWorkflowMaterialization(
        workflow_id=materialization.workflow_id,
        manifest_id=materialization.manifest_id,
        workflow_name=materialization.workflow_name,
        workflow_yaml=materialization.workflow_yaml,
        config=materialization.config,
        workflow_path=workflow_path.as_posix(),
        metadata=materialization.metadata,
        version=materialization.version,
    )
    write_route_policy_scenario_ci_workflow_json(getattr(args, "index_output"), materialization)
    markdown = render_route_policy_scenario_ci_workflow_markdown(materialization)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario CI workflow saved to: {workflow_path.as_posix()}")
    print(f"Scenario CI workflow index saved to: {getattr(args, 'index_output')}")


def _render_workflow_yaml(
    manifest: RoutePolicyScenarioCIManifest,
    config: RoutePolicyScenarioCIWorkflowConfig,
) -> str:
    lines: list[str] = [
        "# Generated by gs-mapper route-policy-scenario-ci-workflow.",
        "# Edit the scenario CI manifest and regenerate this workflow instead of editing by hand.",
        f"name: {_yaml_scalar(config.workflow_name)}",
        "",
        "on:",
    ]
    if config.workflow_dispatch:
        lines.append("  workflow_dispatch: {}")
    if config.push_branches:
        lines.extend(["  push:", "    branches:"])
        lines.extend(f"      - {_yaml_scalar(branch)}" for branch in config.push_branches)
    if config.pull_request_branches:
        lines.extend(["  pull_request:", "    branches:"])
        lines.extend(f"      - {_yaml_scalar(branch)}" for branch in config.pull_request_branches)

    lines.extend(
        [
            "",
            "jobs:",
            "  route-policy-scenario-shards:",
            "    name: Route policy shard ${{ matrix.shardId }}",
            f"    runs-on: {_yaml_scalar(config.runs_on)}",
            "    strategy:",
            f"      fail-fast: {_yaml_bool(config.fail_fast)}",
            "      matrix:",
            "        include:",
        ]
    )
    for job in manifest.shard_jobs:
        lines.extend(
            [
                f"          - jobId: {_yaml_scalar(job.job_id)}",
                f"            shardId: {_yaml_scalar(job.shard_id)}",
                f"            scenarioSetPath: {_yaml_scalar(job.scenario_set_path)}",
                f"            scenarioCount: {job.scenario_count}",
                f"            reportDir: {_yaml_scalar(job.report_dir)}",
                f"            runOutput: {_yaml_scalar(job.run_output)}",
                f"            historyOutput: {_yaml_scalar(job.history_output)}",
                f"            cacheKey: {_yaml_scalar(job.cache_key)}",
                f"            command: {_yaml_scalar(_shell_command(job.command))}",
            ]
        )

    artifact_root = config.artifact_root or "."
    lines.extend(
        [
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: actions/setup-python@v5",
            "        with:",
            f"          python-version: {_yaml_scalar(config.python_version)}",
            "      - name: Install dependencies",
            "        run: |",
            f"          {config.install_command}",
            "      - name: Run shard",
            "        run: |",
            "          ${{ matrix.command }}",
            "      - name: Upload shard artifacts",
            "        uses: actions/upload-artifact@v4",
            "        with:",
            "          name: route-policy-scenario-${{ matrix.shardId }}",
            f"          retention-days: {config.artifact_retention_days}",
            "          if-no-files-found: error",
        ]
    )
    upload_paths = _workflow_upload_paths(manifest, artifact_root)
    if len(upload_paths) == 1:
        lines.append(f"          path: {_yaml_scalar(upload_paths[0])}")
    else:
        lines.extend(["          path: |"])
        lines.extend(f"            {path}" for path in upload_paths)

    lines.extend(
        [
            "",
            "  route-policy-scenario-merge:",
            "    name: Merge route policy shard reports",
            f"    runs-on: {_yaml_scalar(config.runs_on)}",
            "    needs: route-policy-scenario-shards",
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: actions/setup-python@v5",
            "        with:",
            f"          python-version: {_yaml_scalar(config.python_version)}",
            "      - name: Install dependencies",
            "        run: |",
            f"          {config.install_command}",
            "      - name: Download shard artifacts",
            "        uses: actions/download-artifact@v4",
            "        with:",
            "          pattern: route-policy-scenario-*",
            "          merge-multiple: true",
            f"          path: {_yaml_scalar(artifact_root)}",
            "      - name: Merge shard reports",
            "        run: |",
            f"          {_shell_command(manifest.merge_job.command)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _workflow_upload_paths(manifest: RoutePolicyScenarioCIManifest, artifact_root: str) -> tuple[str, ...]:
    if artifact_root != ".":
        return (artifact_root,)
    paths: list[str] = []
    for job in manifest.shard_jobs:
        paths.extend((job.report_dir, job.run_output, job.history_output))
        if job.markdown_output is not None:
            paths.append(job.markdown_output)
        if job.history_markdown_output is not None:
            paths.append(job.history_markdown_output)
    return tuple(dict.fromkeys(paths))


def _common_artifact_root(manifest: RoutePolicyScenarioCIManifest) -> str:
    candidates: list[str] = []
    for job in manifest.shard_jobs:
        candidates.append(job.report_dir)
        candidates.append(str(Path(job.run_output).parent))
        candidates.append(str(Path(job.history_output).parent))
        if job.markdown_output is not None:
            candidates.append(str(Path(job.markdown_output).parent))
        if job.history_markdown_output is not None:
            candidates.append(str(Path(job.history_markdown_output).parent))
    common = os_path.commonpath(candidates) if candidates else "."
    return Path(common or ".").as_posix()


def _shell_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _yaml_bool(value: bool) -> str:
    return "true" if value else "false"


def _yaml_scalar(value: str) -> str:
    text = str(value)
    if text == "":
        return "''"
    return "'" + text.replace("'", "''") + "'"


def _record_type(payload: Mapping[str, Any], expected: str) -> None:
    record_type = payload.get("recordType")
    if record_type != expected:
        raise ValueError(f"expected {expected!r}, got {record_type!r}")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a mapping")


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be a sequence")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _positive_int(value: int, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"
