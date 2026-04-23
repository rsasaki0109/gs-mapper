"""Promotion gate for generated scenario CI workflow triggers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .policy_scenario_ci_review import (
    RoutePolicyScenarioCIReviewArtifact,
    load_route_policy_scenario_ci_review_json,
)


ROUTE_POLICY_SCENARIO_CI_WORKFLOW_PROMOTION_VERSION = "gs-mapper-route-policy-scenario-ci-workflow-promotion/v1"

WORKFLOW_PROMOTION_TRIGGER_MODES = ("pull-request", "push", "push-and-pull-request")


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowPromotionCheck:
    """One workflow trigger promotion check."""

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
            "recordType": "route-policy-scenario-ci-workflow-promotion-check",
            "checkId": self.check_id,
            "passed": bool(self.passed),
            "message": self.message,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIWorkflowPromotionReport:
    """Report produced before widening a generated workflow's triggers."""

    promotion_id: str
    review_id: str
    workflow_id: str
    manifest_id: str
    active_workflow_path: str
    trigger_mode: str
    promoted: bool
    checks: tuple[RoutePolicyScenarioCIWorkflowPromotionCheck, ...]
    push_branches: tuple[str, ...] = ()
    pull_request_branches: tuple[str, ...] = ()
    review_url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_WORKFLOW_PROMOTION_VERSION

    def __post_init__(self) -> None:
        if not str(self.promotion_id):
            raise ValueError("promotion_id must not be empty")
        if not str(self.review_id):
            raise ValueError("review_id must not be empty")
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.active_workflow_path):
            raise ValueError("active_workflow_path must not be empty")
        if not self.checks:
            raise ValueError("promotion report must contain at least one check")
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

    @property
    def trigger_branch_count(self) -> int:
        return len(self.push_branches) + len(self.pull_request_branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-workflow-promotion",
            "version": self.version,
            "promotionId": self.promotion_id,
            "reviewId": self.review_id,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "activeWorkflowPath": self.active_workflow_path,
            "triggerMode": self.trigger_mode,
            "promoted": bool(self.promoted),
            "passed": self.passed,
            "failedChecks": list(self.failed_checks),
            "checkCount": len(self.checks),
            "triggerBranchCount": self.trigger_branch_count,
            "pushBranches": list(self.push_branches),
            "pullRequestBranches": list(self.pull_request_branches),
            "reviewUrl": self.review_url,
            "checks": [check.to_dict() for check in self.checks],
            "metadata": _json_mapping(self.metadata),
        }


def promote_route_policy_scenario_ci_workflow(
    review_artifact: RoutePolicyScenarioCIReviewArtifact,
    *,
    trigger_mode: str = "pull-request",
    push_branches: Sequence[str] = (),
    pull_request_branches: Sequence[str] = ("main",),
    review_url: str | None = None,
    promotion_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIWorkflowPromotionReport:
    """Gate promotion from manual-only scenario CI dispatch to repository triggers."""

    resolved_trigger_mode = str(trigger_mode)
    resolved_push_branches = tuple(str(branch) for branch in push_branches)
    resolved_pull_request_branches = tuple(str(branch) for branch in pull_request_branches)
    resolved_review_url = _resolve_review_url(review_artifact, review_url)
    resolved_promotion_id = promotion_id or f"{review_artifact.workflow_id}-promotion"
    checks: list[RoutePolicyScenarioCIWorkflowPromotionCheck] = [
        _passed("trigger-mode", "trigger mode is supported")
        if resolved_trigger_mode in WORKFLOW_PROMOTION_TRIGGER_MODES
        else _failed(
            "trigger-mode",
            "trigger mode must be pull-request, push, or push-and-pull-request",
            triggerMode=resolved_trigger_mode,
        ),
        _passed("review-passed", "scenario CI review passed")
        if review_artifact.passed
        else _failed(
            "review-passed",
            "scenario CI review did not pass",
            failedShards=list(review_artifact.failed_shards),
            historyFailedChecks=list(review_artifact.history_failed_checks),
        ),
        _passed("validation-passed", "workflow validation passed")
        if review_artifact.validation_passed
        else _failed("validation-passed", "workflow validation did not pass"),
        _passed("activation-active", "workflow activation was active")
        if review_artifact.activation_activated
        else _failed("activation-active", "workflow activation was blocked"),
        _passed("shard-merge-passed", "scenario shard merge passed")
        if review_artifact.shard_merge_passed
        else _failed("shard-merge-passed", "scenario shard merge did not pass"),
        _passed("history-passed", "merged benchmark history passed")
        if review_artifact.history_passed
        else _failed(
            "history-passed",
            "merged benchmark history did not pass",
            historyFailedChecks=list(review_artifact.history_failed_checks),
        ),
        _passed("active-path-root", "active workflow path is under .github/workflows")
        if _is_github_workflow_path(review_artifact.active_workflow_path)
        else _failed("active-path-root", "active workflow path must be under .github/workflows"),
        _passed("active-path-suffix", "active workflow path has a YAML suffix")
        if Path(review_artifact.active_workflow_path).suffix in {".yml", ".yaml"}
        else _failed("active-path-suffix", "active workflow path must end with .yml or .yaml"),
        _passed("review-url", "review URL is attached to the promotion")
        if _valid_review_url(resolved_review_url)
        else _failed("review-url", "review URL must be an absolute http(s) URL"),
        _passed("shard-coverage", "review includes executed shard coverage")
        if review_artifact.shard_count > 0 and review_artifact.scenario_count > 0 and review_artifact.report_count > 0
        else _failed(
            "shard-coverage",
            "review must include at least one shard, scenario, and report",
            shardCount=review_artifact.shard_count,
            scenarioCount=review_artifact.scenario_count,
            reportCount=review_artifact.report_count,
        ),
        _passed("failed-shards", "review has no failed shards")
        if not review_artifact.failed_shards
        else _failed(
            "failed-shards", "review contains failed shards", failedShards=list(review_artifact.failed_shards)
        ),
    ]
    checks.extend(
        _trigger_branch_checks(
            resolved_trigger_mode,
            push_branches=resolved_push_branches,
            pull_request_branches=resolved_pull_request_branches,
        )
    )
    promoted = all(check.passed for check in checks)
    return RoutePolicyScenarioCIWorkflowPromotionReport(
        promotion_id=resolved_promotion_id,
        review_id=review_artifact.review_id,
        workflow_id=review_artifact.workflow_id,
        manifest_id=review_artifact.manifest_id,
        active_workflow_path=review_artifact.active_workflow_path,
        trigger_mode=resolved_trigger_mode,
        promoted=promoted,
        push_branches=resolved_push_branches,
        pull_request_branches=resolved_pull_request_branches,
        review_url=resolved_review_url,
        checks=tuple(checks),
        metadata={
            "reviewPassed": review_artifact.passed,
            "reviewShardCount": review_artifact.shard_count,
            "reviewScenarioCount": review_artifact.scenario_count,
            "reviewReportCount": review_artifact.report_count,
            **_json_mapping(metadata or {}),
        },
    )


def write_route_policy_scenario_ci_workflow_promotion_json(
    path: str | Path,
    report: RoutePolicyScenarioCIWorkflowPromotionReport,
) -> Path:
    """Write workflow trigger promotion as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_workflow_promotion_json(
    path: str | Path,
) -> RoutePolicyScenarioCIWorkflowPromotionReport:
    """Load a workflow trigger promotion report JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_workflow_promotion_from_dict(_mapping(payload, "ciWorkflowPromotion"))


def route_policy_scenario_ci_workflow_promotion_check_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowPromotionCheck:
    """Rebuild one workflow trigger promotion check from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow-promotion-check")
    return RoutePolicyScenarioCIWorkflowPromotionCheck(
        check_id=str(payload["checkId"]),
        passed=bool(payload.get("passed", False)),
        message=str(payload["message"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_workflow_promotion_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIWorkflowPromotionReport:
    """Rebuild a workflow trigger promotion report from JSON."""

    _record_type(payload, "route-policy-scenario-ci-workflow-promotion")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_WORKFLOW_PROMOTION_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_WORKFLOW_PROMOTION_VERSION:
        raise ValueError(f"unsupported route policy scenario CI workflow promotion version: {version}")
    checks = tuple(
        route_policy_scenario_ci_workflow_promotion_check_from_dict(_mapping(item, "check"))
        for item in _sequence(payload.get("checks", ()), "checks")
    )
    expected_check_count = payload.get("checkCount")
    if expected_check_count is not None and int(expected_check_count) != len(checks):
        raise ValueError("checkCount does not match loaded checks")
    return RoutePolicyScenarioCIWorkflowPromotionReport(
        promotion_id=str(payload["promotionId"]),
        review_id=str(payload["reviewId"]),
        workflow_id=str(payload["workflowId"]),
        manifest_id=str(payload["manifestId"]),
        active_workflow_path=str(payload["activeWorkflowPath"]),
        trigger_mode=str(payload["triggerMode"]),
        promoted=bool(payload.get("promoted", False)),
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
        review_url=None if payload.get("reviewUrl") is None else str(payload["reviewUrl"]),
        checks=checks,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_workflow_promotion_markdown(
    report: RoutePolicyScenarioCIWorkflowPromotionReport,
) -> str:
    """Render a compact Markdown summary for workflow trigger promotion."""

    lines = [
        f"# Route Policy Scenario CI Workflow Promotion: {report.promotion_id}",
        f"- Status: {'PROMOTED' if report.promoted else 'BLOCKED'}",
        f"- Workflow: {report.workflow_id}",
        f"- Manifest: {report.manifest_id}",
        f"- Review: {report.review_id}",
        f"- Review URL: {report.review_url or 'n/a'}",
        f"- Active workflow: {report.active_workflow_path}",
        f"- Trigger mode: {report.trigger_mode}",
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


def run_promotion_cli(args: Any) -> None:
    """Run the route policy scenario-ci-workflow-promote CLI."""

    review_artifact = load_route_policy_scenario_ci_review_json(getattr(args, "review"))
    report = promote_route_policy_scenario_ci_workflow(
        review_artifact,
        trigger_mode=getattr(args, "trigger_mode"),
        push_branches=tuple(getattr(args, "push_branch", None) or ()),
        pull_request_branches=tuple(getattr(args, "pull_request_branch", None) or ()),
        review_url=getattr(args, "review_url", None),
        promotion_id=getattr(args, "promotion_id", None),
    )
    write_route_policy_scenario_ci_workflow_promotion_json(getattr(args, "output"), report)
    markdown = render_route_policy_scenario_ci_workflow_promotion_markdown(report)
    if getattr(args, "markdown_output", None):
        output_path = Path(getattr(args, "markdown_output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Scenario CI workflow promotion saved to: {getattr(args, 'output')}")
    if bool(getattr(args, "fail_on_promotion", False)) and not report.promoted:
        raise SystemExit(2)


def _trigger_branch_checks(
    trigger_mode: str,
    *,
    push_branches: tuple[str, ...],
    pull_request_branches: tuple[str, ...],
) -> tuple[RoutePolicyScenarioCIWorkflowPromotionCheck, ...]:
    checks: list[RoutePolicyScenarioCIWorkflowPromotionCheck] = []
    needs_push = trigger_mode in {"push", "push-and-pull-request"}
    needs_pull_request = trigger_mode in {"pull-request", "push-and-pull-request"}
    allows_push = needs_push
    allows_pull_request = needs_pull_request
    checks.append(
        _passed("push-branches-present", "push trigger branches are configured")
        if (push_branches if needs_push else True)
        else _failed("push-branches-present", "push trigger promotion requires at least one push branch")
    )
    checks.append(
        _passed("pull-request-branches-present", "pull_request trigger branches are configured")
        if (pull_request_branches if needs_pull_request else True)
        else _failed(
            "pull-request-branches-present",
            "pull_request trigger promotion requires at least one pull request branch",
        )
    )
    checks.append(
        _passed("push-branches-mode", "push branches match trigger mode")
        if allows_push or not push_branches
        else _failed("push-branches-mode", "push branches were provided for a non-push trigger mode")
    )
    checks.append(
        _passed("pull-request-branches-mode", "pull request branches match trigger mode")
        if allows_pull_request or not pull_request_branches
        else _failed(
            "pull-request-branches-mode",
            "pull request branches were provided for a non-pull-request trigger mode",
        )
    )
    checks.append(_branch_policy_check("push-branch-policy", push_branches))
    checks.append(_branch_policy_check("pull-request-branch-policy", pull_request_branches))
    return tuple(checks)


def _branch_policy_check(
    check_id: str,
    branches: tuple[str, ...],
) -> RoutePolicyScenarioCIWorkflowPromotionCheck:
    invalid = tuple(branch for branch in branches if not _valid_literal_branch_name(branch))
    duplicates = tuple(branch for branch in branches if branches.count(branch) > 1)
    if invalid:
        return _failed(
            check_id,
            "trigger branches must be literal branch names without whitespace or wildcards",
            invalidBranches=list(dict.fromkeys(invalid)),
        )
    if duplicates:
        return _failed(check_id, "trigger branches must be unique", duplicateBranches=list(dict.fromkeys(duplicates)))
    return _passed(check_id, "trigger branches satisfy literal branch policy")


def _resolve_review_url(
    review_artifact: RoutePolicyScenarioCIReviewArtifact,
    review_url: str | None,
) -> str | None:
    if review_url:
        return str(review_url)
    metadata_url = review_artifact.metadata.get("pagesBaseUrl")
    if isinstance(metadata_url, str) and metadata_url:
        return metadata_url
    return None


def _valid_review_url(value: str | None) -> bool:
    if value is None:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _valid_literal_branch_name(value: str) -> bool:
    branch = str(value)
    if not branch or branch.strip() != branch:
        return False
    if any(character.isspace() or ord(character) < 32 for character in branch):
        return False
    if any(character in branch for character in ("*", "[", "]", "\\", "^", "~", ":", "?")):
        return False
    if branch in {".", ".."} or branch.startswith("/") or branch.endswith("/") or "//" in branch:
        return False
    return True


def _is_github_workflow_path(path: str | Path) -> bool:
    parts = Path(path).parts
    for index, part in enumerate(parts[:-1]):
        if part == ".github" and index + 1 < len(parts) and parts[index + 1] == "workflows":
            return True
    return False


def _display_branches(branches: Sequence[str]) -> str:
    return ", ".join(branches) if branches else "n/a"


def _passed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowPromotionCheck:
    return RoutePolicyScenarioCIWorkflowPromotionCheck(
        check_id=check_id,
        passed=True,
        message=message,
        metadata=metadata,
    )


def _failed(
    check_id: str,
    message: str,
    **metadata: Any,
) -> RoutePolicyScenarioCIWorkflowPromotionCheck:
    return RoutePolicyScenarioCIWorkflowPromotionCheck(
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
