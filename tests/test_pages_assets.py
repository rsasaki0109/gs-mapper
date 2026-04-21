"""Sanity checks for GitHub Pages static assets (docs/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assets_dir() -> Path:
    return REPO_ROOT / "docs" / "assets"


def _scene_picker_urls() -> list[str]:
    data = json.loads((REPO_ROOT / "docs" / "scenes-list.json").read_text(encoding="utf-8"))
    return [scene["url"] for scene in data.get("scenes", [])]


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


def test_dust3r_demo_splat_present(assets_dir: Path) -> None:
    """DUSt3R-derived splat must be bundled and match the 32-byte format."""
    splat = assets_dir / "outdoor-demo" / "outdoor-demo-dust3r.splat"
    assert splat.is_file(), "missing DUSt3R demo splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    # splat.html should link to the DUSt3R variant so visitors can toggle it.
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "outdoor-demo-dust3r.splat" in html


def test_bag6_mast3r_splat_present(assets_dir: Path) -> None:
    """MAST3R-derived bag6 splat must be bundled and linked from splat.html."""
    splat = assets_dir / "outdoor-demo" / "bag6-mast3r.splat"
    assert splat.is_file(), "missing MAST3R demo splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "bag6-mast3r.splat" in html


def test_bag6_vggt_slam_splat_present(assets_dir: Path) -> None:
    """VGGT-SLAM-derived bag6 splat must be bundled and linked from splat.html."""
    splat = assets_dir / "outdoor-demo" / "bag6-vggt-slam-20-15k.splat"
    assert splat.is_file(), "missing VGGT-SLAM bag6 demo splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "bag6-vggt-slam-20-15k.splat" in html


def test_bag6_mast3r_slam_splat_present(assets_dir: Path) -> None:
    """MASt3R-SLAM-derived bag6 splat must be bundled and linked from splat.html."""
    splat = assets_dir / "outdoor-demo" / "bag6-mast3r-slam-20-15k.splat"
    assert splat.is_file(), "missing MASt3R-SLAM bag6 demo splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "bag6-mast3r-slam-20-15k.splat" in html


def test_mcd_tuhh_day04_mast3r_splat_present(assets_dir: Path) -> None:
    """MAST3R-derived MCD splat must be bundled and linked from splat.html."""
    splat = assets_dir / "outdoor-demo" / "mcd-tuhh-day04-mast3r.splat"
    assert splat.is_file(), "missing MAST3R MCD demo splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "mcd-tuhh-day04-mast3r.splat" in html


def test_mcd_tuhh_day04_dust3r_splat_present(assets_dir: Path) -> None:
    """MCD tuhh_day_04 DUSt3R splat must be bundled and linked from splat.html."""
    splat = assets_dir / "outdoor-demo" / "mcd-tuhh-day04.splat"
    assert splat.is_file(), "missing MCD tuhh_day_04 DUSt3R splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "mcd-tuhh-day04.splat" in html


def test_mcd_ntu_day02_supervised_splat_present(assets_dir: Path) -> None:
    """The valid-GNSS MCD supervised splat must be bundled and linked."""
    splat = assets_dir / "outdoor-demo" / "mcd-ntu-day02-supervised.splat"
    assert splat.is_file(), "missing MCD ntu_day_02 supervised splat"
    size = splat.stat().st_size
    assert size > 1_000_000, f"splat looks too small ({size} bytes)"
    assert size % 32 == 0, f"splat is not 32-byte aligned ({size} bytes)"
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "mcd-ntu-day02-supervised.splat" in html
    assert "MCD ntu_day_02" in html


def test_mcd_tuhh_day04_zero_gnss_diagnostic_not_in_production_picker() -> None:
    """The rejected all-zero GNSS artifact must not be exposed as a production scene."""
    urls = set(_scene_picker_urls())
    assert "assets/outdoor-demo/mcd-tuhh-day04-supervised.splat" not in urls
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "mcd-tuhh-day04-supervised.splat" not in html
    assert "zero-GNSS diagnostic" not in html


def test_splat_html_has_scene_picker_with_all_bundled_splats(assets_dir: Path) -> None:
    """The <select id="sceneSelect"> must list every bundled .splat."""
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert 'id="sceneSelect"' in html, "scene picker <select> is missing"
    assert 'data-testid="scene-picker"' in html
    for url in _scene_picker_urls():
        assert f'value="{url}"' in html, f"scene picker does not expose {url}; add an <option> under #sceneSelect"
    # Picker JS is shared — each viewer must reference the single bootstrap file.
    assert "scene-picker.js" in html, 'splat.html must include <script src="scene-picker.js">'


def test_splat_spark_has_scene_picker_and_spark_wiring(assets_dir: Path) -> None:
    """Spark viewer ships the same picker and must use the SparkRenderer wrapper."""
    html = (REPO_ROOT / "docs" / "splat_spark.html").read_text(encoding="utf-8")
    # Same picker contract as splat.html.
    assert 'id="sceneSelect"' in html, "Spark viewer is missing the scene picker"
    for url in _scene_picker_urls():
        assert f'value="{url}"' in html, f"Spark picker does not expose {url}; add an <option> under #sceneSelect"
    assert "scene-picker.js" in html, 'splat_spark.html must include <script src="scene-picker.js">'
    # Spark 2.0 needs SparkRenderer added to the scene and three >= r179, otherwise
    # the canvas renders blank. Enforce both at the source level.
    assert "SparkRenderer" in html, "splat_spark.html must instantiate SparkRenderer"
    assert "scene.add(spark)" in html, "SparkRenderer instance must be added to the scene"
    # WebXR / "Enter VR" wiring: VRButton import + renderer.xr.enabled + button append.
    assert "VRButton" in html, "splat_spark.html must import VRButton for Spark 2.0's WebXR support"
    assert "renderer.xr.enabled = true" in html, "WebXR requires renderer.xr.enabled = true"
    assert "VRButton.createButton(renderer)" in html, "VRButton must be mounted on the page"
    # three version pin (any patch of r179 / r180 / higher is fine).
    import re

    version_matches = re.findall(r"three@0\.(\d+)\.\d+", html)
    assert version_matches, "splat_spark.html should import a pinned three.js version"
    assert all(int(v) >= 179 for v in version_matches), f"Spark 2.0 requires three >= r179; found {version_matches}"


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


def test_splat_webgpu_has_scene_picker(assets_dir: Path) -> None:
    """WebGPU wrapper ships the same picker as splat.html / splat_spark.html."""
    html = (REPO_ROOT / "docs" / "splat_webgpu.html").read_text(encoding="utf-8")
    assert 'id="sceneSelect"' in html, "WebGPU viewer is missing the scene picker"
    assert 'data-testid="scene-picker"' in html
    for url in _scene_picker_urls():
        assert f'value="{url}"' in html, f"WebGPU picker does not expose {url}; add an <option> under #sceneSelect"
    assert "scene-picker.js" in html, 'splat_webgpu.html must include <script src="scene-picker.js">'


def test_shared_scene_picker_assets_present(assets_dir: Path) -> None:
    """The shared scene-picker.js + scenes-list.json exist and agree with the bundled splats."""
    docs_dir = REPO_ROOT / "docs"
    js = docs_dir / "scene-picker.js"
    index = docs_dir / "scenes-list.json"
    assert js.is_file(), "shared docs/scene-picker.js is missing"
    assert index.is_file(), "shared docs/scenes-list.json is missing"
    # JS must contain the URL-sync hook the viewer tests previously asserted inline.
    js_text = js.read_text(encoding="utf-8")
    assert "location.assign" in js_text, "scene-picker.js must swap location on change"
    assert "scenes-list.json" in js_text, "scene-picker.js must fetch the shared config"
    # scenes-list.json should point only at production picker splats, and every
    # listed splat must exist on disk.
    import json as _json

    data = _json.loads(index.read_text(encoding="utf-8"))
    assert data.get("version") == "gs-mapper-scene-picker/v1"
    indexed = [scene["url"] for scene in data.get("scenes", [])]
    assert indexed, "scenes-list.json has no production scenes"
    for url in indexed:
        assert (REPO_ROOT / "docs" / url).is_file(), f"scenes-list.json points at missing asset {url}"
