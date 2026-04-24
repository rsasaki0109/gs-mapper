"""Review artifacts for generated scenario CI workflow changes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import difflib
from html import escape
import json
from pathlib import Path
from typing import Any

from .policy_scenario_ci_activation import (
    RoutePolicyScenarioCIWorkflowActivationReport,
    load_route_policy_scenario_ci_workflow_activation_json,
)
from .policy_scenario_ci_workflow import (
    RoutePolicyScenarioCIWorkflowValidationReport,
    load_route_policy_scenario_ci_workflow_validation_json,
)
from .policy_scenario_sharding import (
    RoutePolicyScenarioShardMergeReport,
    RoutePolicyScenarioShardRunSummary,
    load_route_policy_scenario_shard_merge_json,
)


ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION = "gs-mapper-route-policy-scenario-ci-review/v1"


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewShard:
    """Review-friendly summary of one executed scenario shard."""

    shard_id: str
    passed: bool
    scenario_count: int
    report_count: int
    run_path: str | None = None
    history_path: str | None = None
    report_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.shard_id):
            raise ValueError("shard_id must not be empty")
        _positive_int(self.scenario_count, "scenario_count")
        _non_negative_int(self.report_count, "report_count")
        object.__setattr__(self, "report_paths", tuple(str(path) for path in self.report_paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-review-shard",
            "shardId": self.shard_id,
            "passed": bool(self.passed),
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "runPath": self.run_path,
            "historyPath": self.history_path,
            "reportPaths": list(self.report_paths),
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewAdoption:
    """Review-friendly view of a promotion-backed adoption outcome."""

    adoption_id: str
    adopted: bool
    trigger_mode: str
    adopted_active_workflow_path: str
    adopted_source_workflow_path: str
    push_branches: tuple[str, ...] = ()
    pull_request_branches: tuple[str, ...] = ()
    workflow_diff: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.adoption_id):
            raise ValueError("adoption_id must not be empty")
        if not str(self.trigger_mode):
            raise ValueError("trigger_mode must not be empty")
        if not str(self.adopted_active_workflow_path):
            raise ValueError("adopted_active_workflow_path must not be empty")
        if not str(self.adopted_source_workflow_path):
            raise ValueError("adopted_source_workflow_path must not be empty")
        object.__setattr__(self, "push_branches", tuple(str(branch) for branch in self.push_branches))
        object.__setattr__(
            self,
            "pull_request_branches",
            tuple(str(branch) for branch in self.pull_request_branches),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-review-adoption",
            "adoptionId": self.adoption_id,
            "adopted": bool(self.adopted),
            "triggerMode": self.trigger_mode,
            "adoptedActiveWorkflowPath": self.adopted_active_workflow_path,
            "adoptedSourceWorkflowPath": self.adopted_source_workflow_path,
            "pushBranches": list(self.push_branches),
            "pullRequestBranches": list(self.pull_request_branches),
            "workflowDiff": self.workflow_diff,
            "metadata": _json_mapping(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicyScenarioCIReviewArtifact:
    """Static review artifact for scenario CI workflow publication."""

    review_id: str
    merge_id: str
    workflow_id: str
    manifest_id: str
    validation_id: str
    activation_id: str
    validation_passed: bool
    activation_activated: bool
    shard_merge_passed: bool
    history_passed: bool
    active_workflow_path: str
    source_workflow_path: str
    shards: tuple[RoutePolicyScenarioCIReviewShard, ...]
    history_failed_checks: tuple[str, ...] = ()
    adoption: RoutePolicyScenarioCIReviewAdoption | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION

    def __post_init__(self) -> None:
        if not str(self.review_id):
            raise ValueError("review_id must not be empty")
        if not str(self.merge_id):
            raise ValueError("merge_id must not be empty")
        if not str(self.workflow_id):
            raise ValueError("workflow_id must not be empty")
        if not str(self.manifest_id):
            raise ValueError("manifest_id must not be empty")
        if not str(self.validation_id):
            raise ValueError("validation_id must not be empty")
        if not str(self.activation_id):
            raise ValueError("activation_id must not be empty")
        if not str(self.active_workflow_path):
            raise ValueError("active_workflow_path must not be empty")
        if not str(self.source_workflow_path):
            raise ValueError("source_workflow_path must not be empty")
        if not self.shards:
            raise ValueError("review artifact must contain at least one shard")
        object.__setattr__(self, "history_failed_checks", tuple(str(check) for check in self.history_failed_checks))

    @property
    def passed(self) -> bool:
        return self.validation_passed and self.activation_activated and self.shard_merge_passed and self.history_passed

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def scenario_count(self) -> int:
        return sum(shard.scenario_count for shard in self.shards)

    @property
    def report_count(self) -> int:
        return sum(shard.report_count for shard in self.shards)

    @property
    def failed_shards(self) -> tuple[str, ...]:
        return tuple(shard.shard_id for shard in self.shards if not shard.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recordType": "route-policy-scenario-ci-review",
            "version": self.version,
            "reviewId": self.review_id,
            "passed": self.passed,
            "mergeId": self.merge_id,
            "workflowId": self.workflow_id,
            "manifestId": self.manifest_id,
            "validationId": self.validation_id,
            "activationId": self.activation_id,
            "validationPassed": bool(self.validation_passed),
            "activationActivated": bool(self.activation_activated),
            "shardMergePassed": bool(self.shard_merge_passed),
            "historyPassed": bool(self.history_passed),
            "historyFailedChecks": list(self.history_failed_checks),
            "activeWorkflowPath": self.active_workflow_path,
            "sourceWorkflowPath": self.source_workflow_path,
            "shardCount": self.shard_count,
            "scenarioCount": self.scenario_count,
            "reportCount": self.report_count,
            "failedShards": list(self.failed_shards),
            "shards": [shard.to_dict() for shard in self.shards],
            "adoption": None if self.adoption is None else self.adoption.to_dict(),
            "metadata": _json_mapping(self.metadata),
        }


def build_route_policy_scenario_ci_review_artifact(
    merge_report: RoutePolicyScenarioShardMergeReport,
    validation_report: RoutePolicyScenarioCIWorkflowValidationReport,
    activation_report: RoutePolicyScenarioCIWorkflowActivationReport,
    *,
    review_id: str | None = None,
    pages_base_url: str | None = None,
    adoption: RoutePolicyScenarioCIReviewAdoption | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIReviewArtifact:
    """Build a compact review artifact for a scenario CI workflow change.

    When ``adoption`` is provided, the adopted trigger mode, branches, and
    unified diff between the manual and adopted YAMLs ride along on the
    artifact so static Pages consumers can inspect the promotion-backed
    adoption without checking out the branch. The ``passed`` gate is
    unaffected — adoption presentation is purely additive.
    """

    resolved_review_id = review_id or f"{activation_report.workflow_id}-review"
    if validation_report.workflow_id != activation_report.workflow_id:
        raise ValueError("validation and activation reports must reference the same workflow")
    if validation_report.manifest_id != activation_report.manifest_id:
        raise ValueError("validation and activation reports must reference the same manifest")
    return RoutePolicyScenarioCIReviewArtifact(
        review_id=resolved_review_id,
        merge_id=merge_report.merge_id,
        workflow_id=activation_report.workflow_id,
        manifest_id=activation_report.manifest_id,
        validation_id=validation_report.validation_id,
        activation_id=activation_report.activation_id,
        validation_passed=validation_report.passed,
        activation_activated=activation_report.activated,
        shard_merge_passed=merge_report.passed,
        history_passed=merge_report.history.passed,
        active_workflow_path=activation_report.active_workflow_path,
        source_workflow_path=activation_report.source_workflow_path,
        history_failed_checks=merge_report.history.failed_checks,
        shards=tuple(_review_shard_from_run(shard_run) for shard_run in merge_report.shard_runs),
        adoption=adoption,
        metadata={
            "pagesBaseUrl": pages_base_url,
            "historyPath": merge_report.history_path,
            "historyMarkdownPath": merge_report.history_markdown_path,
            "validationFailedChecks": list(validation_report.failed_checks),
            "activationFailedChecks": list(activation_report.failed_checks),
            **_json_mapping(metadata or {}),
        },
    )


def build_route_policy_scenario_ci_review_adoption(
    *,
    adoption_id: str,
    adopted: bool,
    trigger_mode: str,
    adopted_active_workflow_path: str | Path,
    adopted_source_workflow_path: str | Path,
    manual_workflow_text: str,
    adopted_workflow_text: str,
    push_branches: Sequence[str] = (),
    pull_request_branches: Sequence[str] = (),
    diff_context_lines: int = 3,
    metadata: Mapping[str, Any] | None = None,
) -> RoutePolicyScenarioCIReviewAdoption:
    """Build a review-friendly adoption summary including a unified YAML diff."""

    diff_lines = list(
        difflib.unified_diff(
            manual_workflow_text.splitlines(keepends=True),
            adopted_workflow_text.splitlines(keepends=True),
            fromfile="manual",
            tofile="adopted",
            n=max(0, int(diff_context_lines)),
        )
    )
    workflow_diff = "".join(diff_lines) if diff_lines else None
    return RoutePolicyScenarioCIReviewAdoption(
        adoption_id=adoption_id,
        adopted=adopted,
        trigger_mode=trigger_mode,
        adopted_active_workflow_path=Path(adopted_active_workflow_path).as_posix(),
        adopted_source_workflow_path=Path(adopted_source_workflow_path).as_posix(),
        push_branches=tuple(push_branches),
        pull_request_branches=tuple(pull_request_branches),
        workflow_diff=workflow_diff,
        metadata=_json_mapping(metadata or {}),
    )


def write_route_policy_scenario_ci_review_json(
    path: str | Path,
    artifact: RoutePolicyScenarioCIReviewArtifact,
) -> Path:
    """Write a scenario CI review artifact as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_route_policy_scenario_ci_review_json(path: str | Path) -> RoutePolicyScenarioCIReviewArtifact:
    """Load a scenario CI review JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return route_policy_scenario_ci_review_from_dict(_mapping(payload, "ciReview"))


def write_route_policy_scenario_ci_review_bundle(
    output_dir: str | Path,
    artifact: RoutePolicyScenarioCIReviewArtifact,
    *,
    json_name: str = "review.json",
    markdown_name: str = "review.md",
    html_name: str = "index.html",
) -> dict[str, str]:
    """Write JSON, Markdown, and HTML files suitable for static Pages hosting."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = write_route_policy_scenario_ci_review_json(directory / json_name, artifact)
    markdown_path = directory / markdown_name
    markdown_path.write_text(render_route_policy_scenario_ci_review_markdown(artifact), encoding="utf-8")
    html_path = directory / html_name
    html_path.write_text(render_route_policy_scenario_ci_review_html(artifact), encoding="utf-8")
    return {
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "html": html_path.as_posix(),
    }


def route_policy_scenario_ci_review_shard_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewShard:
    """Rebuild one CI review shard from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review-shard")
    return RoutePolicyScenarioCIReviewShard(
        shard_id=str(payload["shardId"]),
        passed=bool(payload.get("passed", False)),
        scenario_count=int(payload["scenarioCount"]),
        report_count=int(payload["reportCount"]),
        run_path=None if payload.get("runPath") is None else str(payload["runPath"]),
        history_path=None if payload.get("historyPath") is None else str(payload["historyPath"]),
        report_paths=tuple(str(item) for item in _sequence(payload.get("reportPaths", ()), "reportPaths")),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_review_adoption_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewAdoption:
    """Rebuild a CI review adoption block from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review-adoption")
    return RoutePolicyScenarioCIReviewAdoption(
        adoption_id=str(payload["adoptionId"]),
        adopted=bool(payload.get("adopted", False)),
        trigger_mode=str(payload["triggerMode"]),
        adopted_active_workflow_path=str(payload["adoptedActiveWorkflowPath"]),
        adopted_source_workflow_path=str(payload["adoptedSourceWorkflowPath"]),
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
        workflow_diff=None if payload.get("workflowDiff") is None else str(payload["workflowDiff"]),
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
    )


def route_policy_scenario_ci_review_from_dict(
    payload: Mapping[str, Any],
) -> RoutePolicyScenarioCIReviewArtifact:
    """Rebuild a scenario CI review artifact from JSON."""

    _record_type(payload, "route-policy-scenario-ci-review")
    version = str(payload.get("version", ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION))
    if version != ROUTE_POLICY_SCENARIO_CI_REVIEW_VERSION:
        raise ValueError(f"unsupported route policy scenario CI review version: {version}")
    shards = tuple(
        route_policy_scenario_ci_review_shard_from_dict(_mapping(item, "reviewShard"))
        for item in _sequence(payload.get("shards", ()), "shards")
    )
    expected_shard_count = payload.get("shardCount")
    if expected_shard_count is not None and int(expected_shard_count) != len(shards):
        raise ValueError("shardCount does not match loaded review shards")
    adoption_payload = payload.get("adoption")
    adoption = (
        None
        if adoption_payload is None
        else route_policy_scenario_ci_review_adoption_from_dict(_mapping(adoption_payload, "adoption"))
    )
    return RoutePolicyScenarioCIReviewArtifact(
        review_id=str(payload["reviewId"]),
        merge_id=str(payload["mergeId"]),
        workflow_id=str(payload["workflowId"]),
        manifest_id=str(payload["manifestId"]),
        validation_id=str(payload["validationId"]),
        activation_id=str(payload["activationId"]),
        validation_passed=bool(payload.get("validationPassed", False)),
        activation_activated=bool(payload.get("activationActivated", False)),
        shard_merge_passed=bool(payload.get("shardMergePassed", False)),
        history_passed=bool(payload.get("historyPassed", False)),
        active_workflow_path=str(payload["activeWorkflowPath"]),
        source_workflow_path=str(payload["sourceWorkflowPath"]),
        history_failed_checks=tuple(
            str(item) for item in _sequence(payload.get("historyFailedChecks", ()), "historyFailedChecks")
        ),
        shards=shards,
        adoption=adoption,
        metadata=_json_mapping(_mapping(payload.get("metadata", {}), "metadata")),
        version=version,
    )


def render_route_policy_scenario_ci_review_markdown(artifact: RoutePolicyScenarioCIReviewArtifact) -> str:
    """Render a compact Markdown review artifact."""

    lines = [
        f"# Route Policy Scenario CI Review: {artifact.review_id}",
        f"- Status: {'PASS' if artifact.passed else 'FAIL'}",
        f"- Workflow: {artifact.workflow_id}",
        f"- Manifest: {artifact.manifest_id}",
        f"- Merge: {artifact.merge_id} ({'PASS' if artifact.shard_merge_passed else 'FAIL'})",
        f"- Validation: {artifact.validation_id} ({'PASS' if artifact.validation_passed else 'FAIL'})",
        f"- Activation: {artifact.activation_id} ({'ACTIVE' if artifact.activation_activated else 'BLOCKED'})",
        f"- Active workflow: {artifact.active_workflow_path}",
        f"- Shards: {artifact.shard_count}",
        f"- Scenarios: {artifact.scenario_count}",
        f"- Reports: {artifact.report_count}",
        "",
        "| Shard | Pass | Scenarios | Reports | Run |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for shard in artifact.shards:
        lines.append(
            "| "
            f"{shard.shard_id} | "
            f"{'yes' if shard.passed else 'no'} | "
            f"{shard.scenario_count} | "
            f"{shard.report_count} | "
            f"{shard.run_path or 'n/a'} |"
        )
    if artifact.history_failed_checks:
        lines.extend(["", "## History Failed Checks", ""])
        lines.extend(f"- {check}" for check in artifact.history_failed_checks)
    if artifact.adoption is not None:
        adoption = artifact.adoption
        lines.extend(
            [
                "",
                "## Adopted Workflow",
                "",
                f"- Adoption: {adoption.adoption_id} ({'ADOPTED' if adoption.adopted else 'BLOCKED'})",
                f"- Trigger mode: {adoption.trigger_mode}",
                f"- Adopted active path: {adoption.adopted_active_workflow_path}",
                f"- Adopted source path: {adoption.adopted_source_workflow_path}",
                f"- Push branches: {_display_branches(adoption.push_branches)}",
                f"- Pull request branches: {_display_branches(adoption.pull_request_branches)}",
            ]
        )
        if adoption.workflow_diff:
            lines.extend(
                [
                    "",
                    "```diff",
                    adoption.workflow_diff.rstrip("\n"),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def render_route_policy_scenario_ci_review_html(artifact: RoutePolicyScenarioCIReviewArtifact) -> str:
    """Render a self-contained static HTML review page."""

    status_class = "pass" if artifact.passed else "fail"
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(shard.shard_id)}</td>"
        f'<td><span class="pill {"pass" if shard.passed else "fail"}">{"PASS" if shard.passed else "FAIL"}</span></td>'
        f"<td>{shard.scenario_count}</td>"
        f"<td>{shard.report_count}</td>"
        f"<td>{_optional_link(shard.run_path)}</td>"
        "</tr>"
        for shard in artifact.shards
    )
    history_failed = "".join(f"<li>{escape(check)}</li>" for check in artifact.history_failed_checks)
    failed_section = (
        f"<section><h2>History Failed Checks</h2><ul>{history_failed}</ul></section>"
        if artifact.history_failed_checks
        else ""
    )
    adoption_section = _render_adoption_section_html(artifact.adoption)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(artifact.review_id)} CI Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8f4; color: #20231f; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 20px; margin: 32px 0 12px; letter-spacing: 0; }}
    .subtitle {{ color: #5b6259; margin: 0 0 24px; }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .metric {{ background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; padding: 14px; }}
    .metric span {{ display: block; color: #5b6259; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 10px; font-size: 12px; font-weight: 700; }}
    .pass {{ background: #dcefd8; color: #1e5a2b; }}
    .fail {{ background: #f7d6d2; color: #8a1f16; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e9ede5; text-align: left; vertical-align: top; }}
    th {{ background: #eef2ea; font-size: 13px; color: #424940; }}
    tr:last-child td {{ border-bottom: 0; }}
    a {{ color: #285b9b; }}
    code {{ background: #eef2ea; padding: 2px 5px; border-radius: 4px; }}
    pre.diff {{ background: #ffffff; border: 1px solid #dfe4da; border-radius: 8px; padding: 12px; overflow-x: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45; }}
    pre.diff .add {{ color: #1e5a2b; }}
    pre.diff .del {{ color: #8a1f16; }}
    pre.diff .hunk {{ color: #5b6259; }}
  </style>
</head>
<body>
  <main>
    <h1>Route Policy Scenario CI Review</h1>
    <p class="subtitle"><span class="pill {status_class}">{"PASS" if artifact.passed else "FAIL"}</span> {escape(artifact.review_id)}</p>
    <section class="grid">
      <div class="metric"><span>Workflow</span><strong>{escape(artifact.workflow_id)}</strong></div>
      <div class="metric"><span>Manifest</span><strong>{escape(artifact.manifest_id)}</strong></div>
      <div class="metric"><span>Shards</span><strong>{artifact.shard_count}</strong></div>
      <div class="metric"><span>Scenarios</span><strong>{artifact.scenario_count}</strong></div>
      <div class="metric"><span>Reports</span><strong>{artifact.report_count}</strong></div>
    </section>
    <section>
      <h2>Workflow Gate</h2>
      <p>Validation <span class="pill {"pass" if artifact.validation_passed else "fail"}">{"PASS" if artifact.validation_passed else "FAIL"}</span>
      Activation <span class="pill {"pass" if artifact.activation_activated else "fail"}">{"ACTIVE" if artifact.activation_activated else "BLOCKED"}</span></p>
      <p>Active workflow: <code>{escape(artifact.active_workflow_path)}</code></p>
      <p>Source workflow: <code>{escape(artifact.source_workflow_path)}</code></p>
    </section>
    <section>
      <h2>Shard Runs</h2>
      <table>
        <thead><tr><th>Shard</th><th>Status</th><th>Scenarios</th><th>Reports</th><th>Run</th></tr></thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </section>
    {failed_section}
    {adoption_section}
  </main>
</body>
</html>
"""


def run_review_cli(args: Any) -> None:
    """Run the route policy scenario-ci-review CLI."""

    merge_report = load_route_policy_scenario_shard_merge_json(getattr(args, "shard_merge"))
    validation_report = load_route_policy_scenario_ci_workflow_validation_json(getattr(args, "validation_report"))
    activation_report = load_route_policy_scenario_ci_workflow_activation_json(getattr(args, "activation_report"))
    adoption = _load_review_adoption(
        adoption_report_path=getattr(args, "adoption_report", None),
        manual_active_workflow_path=activation_report.active_workflow_path,
        manual_workflow_override=getattr(args, "manual_workflow", None),
        adopted_workflow_override=getattr(args, "adopted_workflow", None),
    )
    artifact = build_route_policy_scenario_ci_review_artifact(
        merge_report,
        validation_report,
        activation_report,
        review_id=getattr(args, "review_id", None),
        pages_base_url=getattr(args, "pages_base_url", None),
        adoption=adoption,
    )
    bundle_dir = getattr(args, "bundle_dir", None)
    if bundle_dir:
        paths = write_route_policy_scenario_ci_review_bundle(bundle_dir, artifact)
        print(f"Scenario CI review bundle saved to: {paths['html']}")
    else:
        write_route_policy_scenario_ci_review_json(getattr(args, "output"), artifact)
        if getattr(args, "markdown_output", None):
            markdown_path = Path(getattr(args, "markdown_output"))
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(render_route_policy_scenario_ci_review_markdown(artifact), encoding="utf-8")
        if getattr(args, "html_output", None):
            html_path = Path(getattr(args, "html_output"))
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(render_route_policy_scenario_ci_review_html(artifact), encoding="utf-8")
        print(f"Scenario CI review saved to: {getattr(args, 'output')}")
    print(render_route_policy_scenario_ci_review_markdown(artifact), end="")
    if bool(getattr(args, "fail_on_review", False)) and not artifact.passed:
        raise SystemExit(2)


def _load_review_adoption(
    *,
    adoption_report_path: str | Path | None,
    manual_active_workflow_path: str,
    manual_workflow_override: str | Path | None = None,
    adopted_workflow_override: str | Path | None = None,
) -> RoutePolicyScenarioCIReviewAdoption | None:
    if adoption_report_path is None:
        return None
    payload = json.loads(Path(adoption_report_path).read_text(encoding="utf-8"))
    payload = _mapping(payload, "adoptionReport")
    manual_text_path = Path(manual_workflow_override) if manual_workflow_override else Path(manual_active_workflow_path)
    adopted_text_path = (
        Path(adopted_workflow_override) if adopted_workflow_override else Path(payload["adoptedActiveWorkflowPath"])
    )
    manual_text = manual_text_path.read_text(encoding="utf-8")
    adopted_text = adopted_text_path.read_text(encoding="utf-8")
    return build_route_policy_scenario_ci_review_adoption(
        adoption_id=str(payload["adoptionId"]),
        adopted=bool(payload.get("adopted", False)),
        trigger_mode=str(payload["triggerMode"]),
        adopted_active_workflow_path=str(payload["adoptedActiveWorkflowPath"]),
        adopted_source_workflow_path=str(payload["adoptedSourceWorkflowPath"]),
        manual_workflow_text=manual_text,
        adopted_workflow_text=adopted_text,
        push_branches=tuple(str(item) for item in _sequence(payload.get("pushBranches", ()), "pushBranches")),
        pull_request_branches=tuple(
            str(item) for item in _sequence(payload.get("pullRequestBranches", ()), "pullRequestBranches")
        ),
    )


def _render_adoption_section_html(adoption: RoutePolicyScenarioCIReviewAdoption | None) -> str:
    if adoption is None:
        return ""
    status_pill = "pass" if adoption.adopted else "fail"
    status_label = "ADOPTED" if adoption.adopted else "BLOCKED"
    push_branches = _display_branches(adoption.push_branches)
    pr_branches = _display_branches(adoption.pull_request_branches)
    diff_block = (
        f'<pre class="diff">{_render_diff_html(adoption.workflow_diff)}</pre>'
        if adoption.workflow_diff
        else "<p>No diff available between manual and adopted workflows.</p>"
    )
    return f"""<section>
      <h2>Adopted Workflow</h2>
      <p><span class="pill {status_pill}">{status_label}</span> {escape(adoption.adoption_id)}</p>
      <p>Trigger mode: <code>{escape(adoption.trigger_mode)}</code></p>
      <p>Adopted active path: <code>{escape(adoption.adopted_active_workflow_path)}</code></p>
      <p>Adopted source path: <code>{escape(adoption.adopted_source_workflow_path)}</code></p>
      <p>Push branches: {escape(push_branches)}</p>
      <p>Pull request branches: {escape(pr_branches)}</p>
      {diff_block}
    </section>"""


def _render_diff_html(diff_text: str) -> str:
    rendered_lines: list[str] = []
    for raw_line in diff_text.splitlines():
        escaped = escape(raw_line)
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            rendered_lines.append(f'<span class="hunk">{escaped}</span>')
        elif raw_line.startswith("@@"):
            rendered_lines.append(f'<span class="hunk">{escaped}</span>')
        elif raw_line.startswith("+"):
            rendered_lines.append(f'<span class="add">{escaped}</span>')
        elif raw_line.startswith("-"):
            rendered_lines.append(f'<span class="del">{escaped}</span>')
        else:
            rendered_lines.append(escaped)
    return "\n".join(rendered_lines)


def _display_branches(branches: Sequence[str]) -> str:
    return ", ".join(branches) if branches else "n/a"


def _review_shard_from_run(shard_run: RoutePolicyScenarioShardRunSummary) -> RoutePolicyScenarioCIReviewShard:
    return RoutePolicyScenarioCIReviewShard(
        shard_id=shard_run.shard_id,
        passed=shard_run.passed,
        scenario_count=shard_run.scenario_count,
        report_count=shard_run.report_count,
        run_path=shard_run.run_path,
        history_path=shard_run.history_path,
        report_paths=shard_run.report_paths,
        metadata=shard_run.metadata,
    )


def _optional_link(path: str | None) -> str:
    if path is None:
        return "n/a"
    escaped = escape(path)
    return f'<a href="{escaped}">{escaped}</a>'


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


def _non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")
