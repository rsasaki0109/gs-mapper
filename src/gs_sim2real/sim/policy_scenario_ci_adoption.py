"""Promotion-backed adoption of trigger-enabled scenario CI workflows.

After a :class:`RoutePolicyScenarioCIWorkflowPromotionReport` passes, the
manual-only workflow YAML that was activated earlier still has
``workflow_dispatch``-only triggers. Adoption re-materializes the same
scenario CI manifest with the promoted ``trigger_mode`` / branch set,
re-runs validation and activation against a **separate** active path, and
records the outcome so the manual-only file stays reviewable alongside
the trigger-enabled one.

The adoption deliberately never overwrites the manual workflow path that
was activated earlier; instead it writes a parallel file under
``.github/workflows/`` so the two YAMLs can be diffed in review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from .policy_scenario_ci_activation import (
    RoutePolicyScenarioCIWorkflowActivationReport,
    activate_route_policy_scenario_ci_workflow,
)
from .policy_scenario_ci_manifest import RoutePolicyScenarioCIManifest
from .policy_scenario_ci_promotion import RoutePolicyScenarioCIWorkflowPromotionReport
from .policy_scenario_ci_workflow import (
    RoutePolicyScenarioCIWorkflowConfig,
    RoutePolicyScenarioCIWorkflowMaterialization,
    RoutePolicyScenarioCIWorkflowValidationReport,
    materialize_route_policy_scenario_ci_workflow,
    validate_route_policy_scenario_ci_workflow,
    write_route_policy_scenario_ci_workflow_yaml,
)


ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ADOPTION_VERSION = "gs-mapper-route-policy-scenario-ci-workflow-adoption/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowAdoptionCheck:
    """One adoption guardrail check."""

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
            "recordType": "route-policy-scenario-ci-workflow-adoption-check",
            "checkId": self.check_id,
            "passed": bool(self.passed),
            "message": self.message,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowAdoptionReport:
    """Report describing the promoted-trigger adoption of a scenario CI workflow."""

    adoption_id: str
    promotion_id: str
    review_id: str
    workflow_id: str
    manifest_id: str
    trigger_mode: str
    manual_active_workflow_path: str
    adopted_source_workflow_path: str
    adopted_active_workflow_path: str
    adopted: bool
    checks: tuple[RoutePolicyScenarioCIWorkflowAdoptionCheck, ...]
    push_branches: tuple[str, ...] = ()
    pull_request_branches: tuple[str, ...] = ()
    adopted_config: RoutePolicyScenarioCIWorkflowConfig | None = None
    adopted_validation: RoutePolicyScenarioCIWorkflowValidationReport | None = None
    adopted_activation: RoutePolicyScenarioCIWorkflowActivationReport | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_WORKFLOW_ADOPTION_VERSION

    def __post_init__(self) -> None:
        if not str(self.adoption_id):
            raise ValueError("adoption_id must not be empty")
        if not str(self.promotion_id):
            raise ValueError("promotion_id must not be empty")
        if not str(self.review_id):
            raise ValueError("review_id must not be empty")
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.manual_active_workflow_path):
            raise ValueError("manual_active_workflow_path must not be empty")
        if not str(self.adopted_source_workflow_path):
            raise ValueError("adopted_source_workflow_path must not be empty")
        if not str(self.adopted_active_workflow_path):
            raise ValueError("adopted_active_workflow_path must not be empty")
        if not self.checks:
            raise ValueError("adoption report must contain at least one check")
        object.__setattr__(self, "push_branches", tuple(str(branch) for branch in self.push_branches))
        object.__setattr__(
            self,
            "pull_request_branches",
            tuple(str(branch) for branch in self.pull_request_branches),
        )

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow-adoption",
            "version": self.version,
            "adoptionId": self.adoption_id,
            "promotionId": self.promotion_id,
            "reviewId": self.review_id,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "triggerMode": self.trigger_mode,
            "manualActiveWorkflowPath": self.manual_active_workflow_path,
            "adoptedSourceWorkflowPath": self.adopted_source_workflow_path,
            "adoptedActiveWorkflowPath": self.adopted_active_workflow_path,
            "adopted": bool(self.adopted),
            "passed": self.passed,
            "failedChecks": list(self.failed_checks),
            "checkCount": len(self.checks),
            "pushBranches": list(self.push_branches),
            "pullRequestBranches": list(self.pull_request_branches),
            "adoptedConfig": None if self.adopted_config is None else self.adopted_config.to_dict(),
            "adoptedValidation": None if self.adopted_validation is None else self.adopted_validation.to_dict(),
            "adoptedActivation": None if self.adopted_activation is None else self.adopted_activation.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "metadata": _json_mapping(self.metadata),
        }


def adopt_route_policy_scenario_ci_workflow(
    promotion_report: RoutePolicyScenarioCIWorkflowPromotionReport,
    manifest: RoutePolicyScenarioCIManifest,
    source_materialization: RoutePolicyScenarioCIWorkflowMaterialization,
    *,
    adopted_source_workflow_path: str | Path,
    adopted_active_workflow_path: str | Path,
    adoption_id: str | None = None,
    overwrite: bool = False,
) -> RoutePolicyScenarioCIWorkflowAdoptionReport:
    """Re-materialize the scenario CI workflow with promoted triggers and activate it.

    The adopted workflow is written to ``adopted_source_workflow_path`` and
    then activated to ``adopted_active_workflow_path``. Neither path may
    collide with the manual-only active workflow path recorded on
    ``promotion_report`` — the whole point of adoption is to produce a
    second, trigger-enabled YAML that reviewers can diff against the
    manual-only one.
    """

    resolved_adoption_id = adoption_id or f"{promotion_report.workflow_id}-adoption"
    source_path = Path(adopted_source_workflow_path)
    active_path = Path(adopted_active_workflow_path)
    manual_active_path = Path(promotion_report.active_workflow_path)

    adopted_config = _build_adopted_config(
        promotion_report=promotion_report,
        source_config=source_materialization.config,
    )

    gate_checks: list[RoutePolicyScenarioCIWorkflowAdoptionCheck] = [
        _passed("promotion-promoted", "promotion report is PROMOTED")
        if promotion_report.promoted
        else _failed(
            "promotion-promoted",
            "promotion report is not PROMOTED; adoption is blocked",
            failedChecks=list(promotion_report.failed_checks),
        ),
        _check_equal(
            "manifest-id",
            manifest.manifest_id,
            promotion_report.manifest_id,
            "manifest id matches promotion report",
        ),
        _check_equal(
            "workflow-id",
            source_materialization.workflow_id,
            promotion_report.workflow_id,
            "source materialization workflow id matches promotion report",
        ),
        _passed(
            "adopted-path-distinct-from-manual",
            "adopted active path differs from the manual-only active path",
        )
        if _posix(active_path) != _posix(manual_active_path)
        else _failed(
            "adopted-path-distinct-from-manual",
            "adopted active path must differ from manual active path",
            manualActiveWorkflowPath=_posix(manual_active_path),
            adoptedActiveWorkflowPath=_posix(active_path),
        ),
        _passed(
            "adopted-source-path-distinct",
            "adopted source path differs from the manual-only active path",
        )
        if _posix(source_path) != _posix(manual_active_path)
        else _failed(
            "adopted-source-path-distinct",
            "adopted source path must not collide with the manual active path",
            manualActiveWorkflowPath=_posix(manual_active_path),
            adoptedSourceWorkflowPath=_posix(source_path),
        ),
    ]

    if any(not check.passed for check in gate_checks):
        return _blocked_report(
            resolved_adoption_id,
            promotion_report,
            manifest,
            source_materialization,
            adopted_config,
            source_path,
            active_path,
            manual_active_path,
            tuple(gate_checks),
        )

    adopted_materialization = materialize_route_policy_scenario_ci_workflow(manifest, config=adopted_config)
    written_source_path = write_route_policy_scenario_ci_workflow_yaml(source_path, adopted_materialization)
    adopted_materialization = RoutePolicyScenarioCIWorkflowMaterialization(
        workflow_id=adopted_materialization.workflow_id,
        manifest_id=adopted_materialization.manifest_id,
        workflow_name=adopted_materialization.workflow_name,
        workflow_yaml=adopted_materialization.workflow_yaml,
        config=adopted_materialization.config,
        workflow_path=written_source_path.as_posix(),
        metadata=adopted_materialization.metadata,
        version=adopted_materialization.version,
    )
    adopted_validation = validate_route_policy_scenario_ci_workflow(
        manifest,
        adopted_materialization,
        validation_id=f"{resolved_adoption_id}-validation",
        workflow_path=written_source_path,
    )
    adopted_activation = activate_route_policy_scenario_ci_workflow(
        adopted_materialization,
        adopted_validation,
        source_workflow_path=written_source_path,
        active_workflow_path=active_path,
        activation_id=f"{resolved_adoption_id}-activation",
        overwrite=overwrite,
    )

    trigger_checks = _trigger_yaml_checks(
        adopted_materialization.workflow_yaml,
        promotion_report,
    )
    stage_checks = [
        _passed("adopted-validation-passed", "re-validation of adopted workflow passed")
        if adopted_validation.passed
        else _failed(
            "adopted-validation-passed",
            "re-validation of adopted workflow failed",
            failedChecks=list(adopted_validation.failed_checks),
        ),
        _passed("adopted-activation-active", "re-activation of adopted workflow was active")
        if adopted_activation.activated
        else _failed(
            "adopted-activation-active",
            "re-activation of adopted workflow was blocked",
            failedChecks=list(adopted_activation.failed_checks),
        ),
    ]

    checks = tuple([*gate_checks, *trigger_checks, *stage_checks])
    adopted = all(check.passed for check in checks)
    return RoutePolicyScenarioCIWorkflowAdoptionReport(
        adoption_id=resolved_adoption_id,
        promotion_id=promotion_report.promotion_id,
        review_id=promotion_report.review_id,
        workflow_id=promotion_report.workflow_id,
        manifest_id=promotion_report.manifest_id,
        trigger_mode=promotion_report.trigger_mode,
        manual_active_workflow_path=_posix(manual_active_path),
        adopted_source_workflow_path=_posix(written_source_path),
        adopted_active_workflow_path=_posix(active_path),
        adopted=adopted,
        checks=checks,
        push_branches=promotion_report.push_branches,
        pull_request_branches=promotion_report.pull_request_branches,
        adopted_config=adopted_materialization.config,
        adopted_validation=adopted_validation,
        adopted_activation=adopted_activation,
        metadata={
            "adoptedContentByteLength": len(adopted_materialization.workflow_yaml.encode("utf-8")),
            "adoptedContentSha256": hashlib.sha256(adopted_materialization.workflow_yaml.encode("utf-8")).hexdigest(),
            "overwrite": bool(overwrite),
        },
    )


def write_route_policy_scenario_ci_workflow_adoption_json(
    path: str | Path,
    report: RoutePolicyScenarioCIWorkflowAdoptionReport,
) -> Path:
    """Write workflow trigger adoption as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def render_route_policy_scenario_ci_workflow_adoption_markdown(
    report: RoutePolicyScenarioCIWorkflowAdoptionReport,
) -> str:
    """Render a compact Markdown summary for a workflow trigger adoption report."""

    lines = [
        f"# Route Policy Scenario CI Workflow Adoption: {report.adoption_id}",
        f"- Status: {'ADOPTED' if report.adopted else 'BLOCKED'}",
        f"- Promotion: {report.promotion_id}",
        f"- Review: {report.review_id}",
        f"- Workflow: {report.workflow_id}",
        f"- Manifest: {report.manifest_id}",
        f"- Trigger mode: {report.trigger_mode}",
        f"- Manual active path: {report.manual_active_workflow_path}",
        f"- Adopted source path: {report.adopted_source_workflow_path}",
        f"- Adopted active path: {report.adopted_active_workflow_path}",
        f"- Push branches: {_display_branches(report.push_branches)}",
        f"- Pull request branches: {_display_branches(report.pull_request_branches)}",
        f"- Checks: {len(report.checks)}",
        "",
        "| Check | Pass | Message |",
        "| --- | --- | --- |",
    ]
    for check in report.checks:
        lines.append(f"| {check.check_id} | {'yes' if check.passed else 'no'} | {check.message} |")
    return "\n".join(lines) + "\n"


def _build_adopted_config(
    *,
    promotion_report: RoutePolicyScenarioCIWorkflowPromotionReport,
    source_config: RoutePolicyScenarioCIWorkflowConfig,
) -> RoutePolicyScenarioCIWorkflowConfig:
    trigger_mode = promotion_report.trigger_mode
    push_branches = promotion_report.push_branches if trigger_mode in {"push", "push-and-pull-request"} else ()
    pull_request_branches = (
        promotion_report.pull_request_branches if trigger_mode in {"pull-request", "push-and-pull-request"} else ()
    )
    return RoutePolicyScenarioCIWorkflowConfig(
        workflow_id=source_config.workflow_id,
        workflow_name=source_config.workflow_name,
        runs_on=source_config.runs_on,
        python_version=source_config.python_version,
        install_command=source_config.install_command,
        artifact_root=source_config.artifact_root,
        artifact_retention_days=source_config.artifact_retention_days,
        workflow_dispatch=True,
        push_branches=push_branches,
        pull_request_branches=pull_request_branches,
        fail_fast=source_config.fail_fast,
        metadata={
            **source_config.metadata,
            "adoptedTriggerMode": trigger_mode,
            "adoptedFromPromotion": promotion_report.promotion_id,
        },
    )


def _trigger_yaml_checks(
    yaml_text: str,
    promotion_report: RoutePolicyScenarioCIWorkflowPromotionReport,
) -> tuple[RoutePolicyScenarioCIWorkflowAdoptionCheck, ...]:
    trigger_mode = promotion_report.trigger_mode
    checks: list[RoutePolicyScenarioCIWorkflowAdoptionCheck] = [
        _passed("workflow-dispatch-retained", "adopted workflow retains workflow_dispatch trigger")
        if "workflow_dispatch" in yaml_text
        else _failed("workflow-dispatch-retained", "adopted workflow missing workflow_dispatch trigger"),
    ]
    if trigger_mode in {"push", "push-and-pull-request"}:
        checks.append(
            _passed("push-trigger-emitted", "adopted workflow emits push trigger block")
            if "push:" in yaml_text
            else _failed("push-trigger-emitted", "adopted workflow is missing push trigger block")
        )
        for branch in promotion_report.push_branches:
            checks.append(
                _passed(
                    f"push-branch:{_slug(branch)}",
                    f"adopted workflow push trigger mentions branch {branch}",
                )
                if branch in yaml_text
                else _failed(
                    f"push-branch:{_slug(branch)}",
                    f"adopted workflow push trigger is missing branch {branch}",
                )
            )
    if trigger_mode in {"pull-request", "push-and-pull-request"}:
        checks.append(
            _passed("pull-request-trigger-emitted", "adopted workflow emits pull_request trigger block")
            if "pull_request:" in yaml_text
            else _failed(
                "pull-request-trigger-emitted",
                "adopted workflow is missing pull_request trigger block",
            )
        )
        for branch in promotion_report.pull_request_branches:
            checks.append(
                _passed(
                    f"pull-request-branch:{_slug(branch)}",
                    f"adopted workflow pull_request trigger mentions branch {branch}",
                )
                if branch in yaml_text
                else _failed(
                    f"pull-request-branch:{_slug(branch)}",
                    f"adopted workflow pull_request trigger is missing branch {branch}",
                )
            )
    return tuple(checks)


def _blocked_report(
    adoption_id: str,
    promotion_report: RoutePolicyScenarioCIWorkflowPromotionReport,
    manifest: RoutePolicyScenarioCIManifest,
    source_materialization: RoutePolicyScenarioCIWorkflowMaterialization,
    adopted_config: RoutePolicyScenarioCIWorkflowConfig,
    source_path: Path,
    active_path: Path,
    manual_active_path: Path,
    checks: tuple[RoutePolicyScenarioCIWorkflowAdoptionCheck, ...],
) -> RoutePolicyScenarioCIWorkflowAdoptionReport:
    # Return a blocked report without materializing / writing any YAML so
    # the manual workflow and the filesystem remain untouched.
    del manifest, source_materialization
    return RoutePolicyScenarioCIWorkflowAdoptionReport(
        adoption_id=adoption_id,
        promotion_id=promotion_report.promotion_id,
        review_id=promotion_report.review_id,
        workflow_id=promotion_report.workflow_id,
        manifest_id=promotion_report.manifest_id,
        trigger_mode=promotion_report.trigger_mode,
        manual_active_workflow_path=_posix(manual_active_path),
        adopted_source_workflow_path=_posix(source_path),
        adopted_active_workflow_path=_posix(active_path),
        adopted=False,
        checks=checks,
        push_branches=promotion_report.push_branches,
        pull_request_branches=promotion_report.pull_request_branches,
        adopted_config=adopted_config,
        adopted_validation=None,
        adopted_activation=None,
        metadata={"adoptionState": "blocked-pre-materialization"},
    )


def _display_branches(branches: Sequence[str]) -> str:
    return ", ".join(branches) if branches else "n/a"


def _posix(path: str | Path) -> str:
    return Path(path).as_posix()


def _check_equal(
    check_id: str,
    actual: Any,
    expected: Any,
    message: str,
) -> RoutePolicyScenarioCIWorkflowAdoptionCheck:
    if actual == expected:
        return _passed(check_id, message, actual=actual, expected=expected)
    return _failed(check_id, f"{message}; expected {expected!r}, got {actual!r}", actual=actual, expected=expected)


def _passed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowAdoptionCheck:
    return RoutePolicyScenarioCIWorkflowAdoptionCheck(
        check_id=check_id,
        passed=True,
        message=message,
        metadata=metadata,
    )


def _failed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowAdoptionCheck:
    return RoutePolicyScenarioCIWorkflowAdoptionCheck(
        check_id=check_id,
        passed=False,
        message=message,
        metadata=metadata,
    )


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


def _slug(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return slug or "unnamed"
