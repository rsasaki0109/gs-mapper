"""Activation guardrails for generated scenario CI workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from .policy_scenario_ci_workflow import (
    RoutePolicyScenarioCIWorkflowMaterialization,
    RoutePolicyScenarioCIWorkflowValidationReport,
    load_route_policy_scenario_ci_workflow_json,
    load_route_policy_scenario_ci_workflow_validation_json,
)


ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ACTIVATION_VERSION = "gs-mapper-route-policy-scenario-ci-workflow-activation/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowActivationCheck:
    """One activation guardrail check."""

    check_id: str
    passed: bool
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.check_id):
            raise ValueError("check_id must not be empty")
        if not str(self.message):
            raise ValueError("message must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow-activation-check",
            "checkId": self.check_id,
            "passed": bool(self.passed),
            "message": self.message,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowActivationReport:
    """Report produced when activating a generated workflow YAML file."""

    activation_id: str
    workflow_id: str
    manifest_id: str
    validation_id: str
    source_workflow_path: str
    active_workflow_path: str
    activated: bool
    checks: tuple[RoutePolicyScenarioCIWorkflowActivationCheck, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ACTIVATION_VERSION

    def __post_init__(self) -> None:
        if not str(self.activation_id):
            raise ValueError("activation_id must not be empty")
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.validation_id):
            raise ValueError("validation_id must not be empty")
        if not str(self.source_workflow_path):
            raise ValueError("source_workflow_path must not be empty")
        if not str(self.active_workflow_path):
            raise ValueError("active_workflow_path must not be empty")
        if not self.checks:
            raise ValueError("activation report must contain at least one check")

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow-activation",
            "version": self.version,
            "activationId": self.activation_id,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "validationId": self.validation_id,
            "sourceWorkflowPath": self.source_workflow_path,
            "activeWorkflowPath": self.active_workflow_path,
            "activated": bool(self.activated),
            "passed": self.passed,
            "failedChecks": list(self.failed_checks),
            "checkCount": len(self.checks),
            "checks": [check.to_dict() for check in self.checks],
            "metadata": _json_mapping(self.metadata),
        }


def activate_route_policy_scenario_ci_workflow(
    materialization: RoutePolicyScenarioCIWorkflowMaterialization,
    validation_report: RoutePolicyScenarioCIWorkflowValidationReport,
    *,
    source_workflow_path: str | Path,
    active_workflow_path: str | Path,
    activation_id: str | None = None,
    overwrite: bool = False,
) -> RoutePolicyScenarioCIWorkflowActivationReport:
    """Write a generated workflow to an active GitHub Actions path after guardrail checks pass."""

    resolved_activation_id = activation_id or f"{materialization.workflow_id}-activation"
    source_path = Path(source_workflow_path)
    active_path = Path(active_workflow_path)
    checks: list[RoutePolicyScenarioCIWorkflowActivationCheck] = [
        _check_equal(
            "validation-workflow-id",
            validation_report.workflow_id,
            materialization.workflow_id,
            "validation report references the materialized workflow",
        ),
        _check_equal(
            "validation-manifest-id",
            validation_report.manifest_id,
            materialization.manifest_id,
            "validation report references the materialized manifest",
        ),
        _passed("validation-passed", "workflow validation report passed")
        if validation_report.passed
        else _failed(
            "validation-passed",
            "workflow validation report failed",
            failedChecks=list(validation_report.failed_checks),
        ),
        _passed("active-path-root", "active workflow path is under .github/workflows")
        if _is_github_workflow_path(active_path)
        else _failed("active-path-root", "active workflow path must be under .github/workflows"),
        _passed("active-path-suffix", "active workflow path has a YAML suffix")
        if active_path.suffix in {".yml", ".yaml"}
        else _failed("active-path-suffix", "active workflow path must end with .yml or .yaml"),
    ]
    if validation_report.workflow_path is not None:
        checks.append(
            _passed("validation-source-path", "source workflow path matches validation report")
            if _same_path(validation_report.workflow_path, source_path)
            else _failed(
                "validation-source-path",
                "source workflow path does not match validation report",
                actual=source_path.as_posix(),
                expected=validation_report.workflow_path,
            )
        )

    source_text = _read_source_workflow(source_path, checks)
    if source_text is not None:
        checks.append(
            _check_equal(
                "source-content",
                source_text,
                materialization.workflow_yaml,
                "source workflow content matches materialization index",
            )
        )
    active_text = _read_existing_active_workflow(active_path, checks)
    if active_text is not None:
        if active_text == materialization.workflow_yaml:
            checks.append(_passed("active-output-existing-current", "active workflow already matches materialization"))
        elif overwrite:
            checks.append(_passed("active-output-overwrite", "active workflow exists and overwrite is enabled"))
        else:
            checks.append(_failed("active-output-overwrite", "active workflow exists and overwrite is disabled"))

    activated = all(check.passed for check in checks)
    activation_state = "blocked"
    if activated:
        if active_text == materialization.workflow_yaml:
            activation_state = "already-current"
        else:
            active_path.parent.mkdir(parents=True, exist_ok=True)
            active_path.write_text(materialization.workflow_yaml, encoding="utf-8")
            activation_state = "activated"

    return RoutePolicyScenarioCIWorkflowActivationReport(
        activation_id=resolved_activation_id,
        workflow_id=materialization.workflow_id,
        manifest_id=materialization.manifest_id,
        validation_id=validation_report.validation_id,
        source_workflow_path=source_path.as_posix(),
        active_workflow_path=active_path.as_posix(),
        activated=activated,
        checks=tuple(checks),
        metadata={
            "activationState": activation_state,
            "contentByteLength": len(materialization.workflow_yaml.encode("utf-8")),
            "contentSha256": hashlib.sha256(materialization.workflow_yaml.encode("utf-8")).hexdigest(),
            "overwrite": overwrite,
        },
    )


def write_route_policy_scenario_ci_workflow_activation_json(
    path: str | Path,
    report: RoutePolicyScenarioCIWorkflowActivationReport,
) -> Path:
    """Write workflow activation as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_workflow_activation_json(
    path: str | Path,
) -> RoutePolicyScenarioCIWorkflowActivationReport:
    """Load a workflow activation report JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_workflow_activation_from_dict(_mapping(payload, "ciWorkflowActivation"))


def route_policy_scenario_ci_workflow_activation_check_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowActivationCheck:
    """Rebuild one workflow activation check from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow-activation-check")
    return RoutePolicyScenarioCIWorkflowActivationCheck(
        check_id=str(payload["checkId"]),
        passed=bool(payload.get("passed", False)),
        message=str(payload["message"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_workflow_activation_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowActivationReport:
    """Rebuild a workflow activation report from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow-activation")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ACTIVATION_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ACTIVATION_VERSION:
        raise ValueError(f"unsupported route policy scenario CI workflow activation version: {version}")
    checks = tuple(
        route_policy_scenario_ci_workflow_activation_check_from_dict(_mapping(item, "check"))
        for item in _sequence(payload.get("checks", ()), "checks")
    )
    expected_check_count = payload.get("checkCount")
    if expected_check_count is not None and int(expected_check_count) != len(checks):
        raise ValueError("checkCount does not match loaded checks")
    return RoutePolicyScenarioCIWorkflowActivationReport(
        activation_id=str(payload["activationId"]),
        workflow_id=str(payload["workflowId"]),
        manifest_id=str(payload["manifestId"]),
        validation_id=str(payload["validationId"]),
        source_workflow_path=str(payload["sourceWorkflowPath"]),
        active_workflow_path=str(payload["activeWorkflowPath"]),
        activated=bool(payload.get("activated", False)),
        checks=checks,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_workflow_activation_markdown(
    report: RoutePolicyScenarioCIWorkflowActivationReport,
) -> str:
    """Render a compact Markdown summary for workflow activation."""

    lines = [
        f"# Route Policy Scenario CI Workflow Activation: {report.activation_id}",
        f"- Status: {'ACTIVATED' if report.activated else 'BLOCKED'}",
        f"- Workflow: {report.workflow_id}",
        f"- Manifest: {report.manifest_id}",
        f"- Validation: {report.validation_id}",
        f"- Source: {report.source_workflow_path}",
        f"- Active path: {report.active_workflow_path}",
        f"- Checks: {len(report.checks)}",
        "",
        "| Check | Pass | Message |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        lines.append(f"| {check.check_id} | {'yes' if check.passed else 'no'} | {check.message} |")
    return "\n".join(lines) + "\n"


def run_activation_cli(args: Any) -> None:
    """Run the route policy scenario-ci-workflow-activate CLI."""

    materialization = load_route_policy_scenario_ci_workflow_json(getattr(args, "workflow_index"))
    validation_report = load_route_policy_scenario_ci_workflow_validation_json(getattr(args, "validation_report"))
    source_workflow_path = getattr(args, "workflow", None) or materialization.workflow_path
    if source_workflow_path is None:
        raise ValueError("--workflow is required when workflow index has no workflowPath")
    report = activate_route_policy_scenario_ci_workflow(
        materialization,
        validation_report,
        source_workflow_path=source_workflow_path,
        active_workflow_path=getattr(args, "active_workflow_output"),
        activation_id=getattr(args, "activation_id", None),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    write_route_policy_scenario_ci_workflow_activation_json(getattr(args, "output"), report)
    markdown = render_route_policy_scenario_ci_workflow_activation_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario CI workflow activation saved to: {getattr(args, 'output')}")
    if bool(getattr(args, "fail_on_activation", False)) and not report.activated:
        raise SystemExit(2)


def _read_source_workflow(
    path: Path,
    checks: list[RoutePolicyScenarioCIWorkflowActivationCheck],
) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        checks.append(_failed("source-readable", f"source workflow could not be read: {exc}"))
        return None
    checks.append(_passed("source-readable", "source workflow is readable"))
    return text


def _read_existing_active_workflow(
    path: Path,
    checks: list[RoutePolicyScenarioCIWorkflowActivationCheck],
) -> str | None:
    if not path.exists():
        checks.append(_passed("active-output-available", "active workflow path is available"))
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        checks.append(_failed("active-output-readable", f"active workflow could not be read: {exc}"))
        return None


def _is_github_workflow_path(path: Path) -> bool:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == ".github" and index + 1 < len(parts) and parts[index + 1] == "workflows":
            return True
    return False


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).as_posix() == Path(right).as_posix()


def _check_equal(
    check_id: str,
    actual: Any,
    expected: Any,
    message: str,
) -> RoutePolicyScenarioCIWorkflowActivationCheck:
    if actual == expected:
        return _passed(check_id, message, actual=actual, expected=expected)
    return _failed(check_id, f"{message}; expected {expected!r}, got {actual!r}", actual=actual, expected=expected)


def _passed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowActivationCheck:
    return RoutePolicyScenarioCIWorkflowActivationCheck(
        check_id=check_id,
        passed=True,
        message=message,
        metadata=metadata,
    )


def _failed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowActivationCheck:
    return RoutePolicyScenarioCIWorkflowActivationCheck(
        check_id=check_id,
        passed=False,
        message=message,
        metadata=metadata,
    )


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
