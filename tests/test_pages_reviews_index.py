"""Tests for scripts/build_pages_reviews_index.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_pages_reviews_index.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_pages_reviews_index", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_review_bundle(
    root: Path,
    *,
    review_id: str,
    passed: bool = True,
    shard_count: int = 1,
    scenario_count: int = 1,
    report_count: int = 1,
    adoption: dict | None = None,
    write_bundle_html: bool = True,
) -> Path:
    bundle = root / review_id
    bundle.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "recordType": "route-policy-scenario-ci-review",
        "reviewId": review_id,
        "passed": bool(passed),
        "shardCount": shard_count,
        "scenarioCount": scenario_count,
        "reportCount": report_count,
    }
    if adoption is not None:
        payload["adoption"] = adoption
    (bundle / "review.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    if write_bundle_html:
        (bundle / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    return bundle


def test_script_file_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111


def test_collect_review_entries_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    entries = module.collect_review_entries(tmp_path / "missing")

    assert entries == []


def test_collect_review_entries_skips_non_bundle_children(tmp_path: Path) -> None:
    module = _load_script_module()

    # Valid bundle.
    _write_review_bundle(tmp_path, review_id="alpha")
    # Sub-dir without review.json should be ignored.
    (tmp_path / "not-a-bundle").mkdir()
    (tmp_path / "not-a-bundle" / "readme.txt").write_text("ignore me", encoding="utf-8")
    # File at the top level should be ignored.
    (tmp_path / "loose-file.txt").write_text("ignore me", encoding="utf-8")
    # Sub-dir whose review.json is not a review record (wrong recordType).
    (tmp_path / "wrong-type").mkdir()
    (tmp_path / "wrong-type" / "review.json").write_text(json.dumps({"recordType": "something-else"}), encoding="utf-8")

    entries = module.collect_review_entries(tmp_path)

    assert [entry.review_id for entry in entries] == ["alpha"]


def test_write_reviews_index_covers_mixed_bundles(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(
        tmp_path,
        review_id="bravo",
        passed=True,
        shard_count=2,
        scenario_count=4,
        report_count=4,
        adoption={
            "recordType": "route-policy-scenario-ci-review-adoption",
            "adoptionId": "bravo-adoption",
            "adopted": True,
            "triggerMode": "pull-request",
            "adoptedActiveWorkflowPath": ".github/workflows/bravo-adopted.yml",
            "adoptedSourceWorkflowPath": "runs/bravo-adopted.generated.yml",
        },
    )
    _write_review_bundle(
        tmp_path,
        review_id="alpha",
        passed=False,
        shard_count=1,
        scenario_count=1,
        report_count=1,
    )

    html_output = tmp_path / "index.html"
    json_output = tmp_path / "index.json"
    markdown_output = tmp_path / "index.md"

    entries = module.write_reviews_index(
        tmp_path,
        html_output=html_output,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    # Stable alphabetical sort so the index is deterministic.
    assert [entry.review_id for entry in entries] == ["alpha", "bravo"]

    index_payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert index_payload["recordType"] == "gs-mapper-route-policy-scenario-ci-reviews-index/v1"
    assert index_payload["reviewCount"] == 2
    assert index_payload["passCount"] == 1
    assert index_payload["failCount"] == 1
    assert [entry["reviewId"] for entry in index_payload["entries"]] == ["alpha", "bravo"]
    assert index_payload["entries"][1]["adoptionTriggerMode"] == "pull-request"
    assert index_payload["entries"][1]["adoptionAdopted"] is True
    assert index_payload["entries"][0]["adoptionAdopted"] is None

    html_text = html_output.read_text(encoding="utf-8")
    assert "Route Policy Scenario CI Reviews" in html_text
    assert 'href="alpha/index.html"' in html_text
    assert 'href="bravo/index.html"' in html_text
    # Pass / fail pills for both.
    assert '<span class="pill pass">PASS</span>' in html_text
    assert '<span class="pill fail">FAIL</span>' in html_text
    # Adoption pill + trigger mode for bravo.
    assert '<span class="pill pass">ADOPTED</span>' in html_text
    assert "<code>pull-request</code>" in html_text
    # Alpha has no adoption → "none" pill.
    assert '<span class="pill info">none</span>' in html_text

    markdown_text = markdown_output.read_text(encoding="utf-8")
    assert "| [alpha](alpha/index.html) | FAIL |" in markdown_text
    assert "| [bravo](bravo/index.html) | PASS |" in markdown_text
    assert "ADOPTED (`pull-request`)" in markdown_text


def test_main_writes_default_outputs_inside_reviews_dir(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(tmp_path, review_id="charlie")
    rc = module.main(["--reviews-dir", str(tmp_path)])

    assert rc == 0
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "index.json").is_file()
    payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert payload["reviewCount"] == 1


def test_empty_reviews_dir_still_writes_placeholder_index(tmp_path: Path) -> None:
    module = _load_script_module()

    html_output = tmp_path / "index.html"
    json_output = tmp_path / "index.json"
    entries = module.write_reviews_index(
        tmp_path,
        html_output=html_output,
        json_output=json_output,
    )

    assert entries == []
    assert json.loads(json_output.read_text(encoding="utf-8"))["reviewCount"] == 0
    assert "No review bundles published yet" in html_output.read_text(encoding="utf-8")


def test_entry_falls_back_to_bundle_dir_when_html_missing(tmp_path: Path) -> None:
    module = _load_script_module()

    _write_review_bundle(tmp_path, review_id="delta", write_bundle_html=False)

    entries = module.collect_review_entries(tmp_path)

    # Without a per-bundle index.html, the href points at the directory so the
    # browser shows the Pages directory listing or a fallback.
    assert entries[0].bundle_html == "delta"
