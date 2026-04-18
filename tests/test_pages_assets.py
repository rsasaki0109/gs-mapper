"""Sanity checks for GitHub Pages static assets (docs/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assets_dir() -> Path:
    return REPO_ROOT / "docs" / "assets"


def test_scenes_index_json_parses(assets_dir: Path) -> None:
    p = assets_dir / "scenes.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("type") == "web-scene-index"
    assert isinstance(data.get("scenes"), list)


def test_scenes_index_lists_outdoor_demo_first(assets_dir: Path) -> None:
    data = json.loads((assets_dir / "scenes.json").read_text(encoding="utf-8"))
    scenes = data["scenes"]
    assert len(scenes) >= 1
    assert scenes[0]["id"] == "outdoor-demo"
    assert scenes[0]["manifest"] == "assets/outdoor-demo/scene.json"


def test_outdoor_demo_manifest_and_referenced_binary_exist(assets_dir: Path) -> None:
    manifest_path = assets_dir / "outdoor-demo" / "scene.json"
    assert manifest_path.is_file()
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert man.get("type") == "web-scene-manifest"
    href = man["asset"]["href"]
    bin_path = (manifest_path.parent / href).resolve()
    assert bin_path.is_file(), f"missing binary referenced by outdoor-demo: {href}"


def test_webgpu_viewer_bundle_present() -> None:
    """shrekshao WebGPU viewer bundle must be present alongside the wrapper page."""
    docs_dir = REPO_ROOT / "docs"
    wrapper = docs_dir / "splat_webgpu.html"
    inner = docs_dir / "splat-webgpu" / "index.html"
    js = docs_dir / "splat-webgpu" / "assets" / "index.js"
    css = docs_dir / "splat-webgpu" / "assets" / "index.css"
    license_file = docs_dir / "splat-webgpu" / "LICENSE"
    for p in (wrapper, inner, js, css, license_file):
        assert p.is_file(), f"missing WebGPU viewer asset: {p}"
    # Inner html should reference the relative bundle assets.
    inner_text = inner.read_text(encoding="utf-8")
    assert "./assets/index.js" in inner_text
    assert "./assets/index.css" in inner_text
    # Bundle should not be empty.
    assert js.stat().st_size > 10_000
