"""Sanity checks for GitHub Pages static assets (docs/)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assets_dir() -> Path:
    return REPO_ROOT / "docs" / "assets"


def _scene_picker_scenes() -> list[dict[str, str]]:
    data = json.loads((REPO_ROOT / "docs" / "scenes-list.json").read_text(encoding="utf-8"))
    return data.get("scenes", [])


def _scene_picker_urls() -> list[str]:
    return [scene["url"] for scene in _scene_picker_scenes()]


def _scene_picker_options() -> list[tuple[str, str]]:
    return [(scene["url"], scene["label"]) for scene in _scene_picker_scenes()]


def _readme_preview_specs() -> list[tuple[str, str]]:
    return [(Path(scene["preview"]).stem, scene["url"]) for scene in _scene_picker_scenes()]


def _html_scene_options(path: Path) -> list[tuple[str, str]]:
    html = path.read_text(encoding="utf-8")
    options: list[tuple[str, str]] = []
    for match in re.finditer(r'<option\s+value="([^"]+)"[^>]*>(.*?)</option>', html, flags=re.DOTALL):
        label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(2))).strip()
        options.append((match.group(1), label))
    return options


def _readme_production_scene_rows() -> list[tuple[str, str, str]]:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"\| Scene \| Preview \| Pipeline \|\n"
        r"\|[- |]+\|\n"
        r"(?P<rows>(?:\| .+\|\n)+)"
        r"\nThe Autoware supervised default uses",
        readme,
    )
    assert match is not None, "README production scene table is missing"
    rows: list[tuple[str, str, str]] = []
    for line in match.group("rows").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 3, f"unexpected README scene row: {line}"
        image_link = re.search(r"!\[\]\(([^)]+)\)\]\([^)]*\?url=([^)]+)\)", cells[1])
        assert image_link is not None, f"README scene row has no preview/deep link: {line}"
        rows.append((cells[0], image_link.group(1), image_link.group(2)))
    return rows


def _load_script_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    path = REPO_ROOT / "docs" / "splat.html"
    html = path.read_text(encoding="utf-8")
    assert 'id="sceneSelect"' in html, "scene picker <select> is missing"
    assert 'data-testid="scene-picker"' in html
    assert _html_scene_options(path) == _scene_picker_options()
    # Picker JS is shared — each viewer must reference the single bootstrap file.
    assert "scene-picker.js" in html, 'splat.html must include <script src="scene-picker.js">'


def test_splat_spark_has_scene_picker_and_spark_wiring(assets_dir: Path) -> None:
    """Spark viewer ships the same picker and must use the SparkRenderer wrapper."""
    path = REPO_ROOT / "docs" / "splat_spark.html"
    html = path.read_text(encoding="utf-8")
    # Same picker contract as splat.html.
    assert 'id="sceneSelect"' in html, "Spark viewer is missing the scene picker"
    assert _html_scene_options(path) == _scene_picker_options()
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
    path = REPO_ROOT / "docs" / "splat_webgpu.html"
    html = path.read_text(encoding="utf-8")
    assert 'id="sceneSelect"' in html, "WebGPU viewer is missing the scene picker"
    assert 'data-testid="scene-picker"' in html
    assert _html_scene_options(path) == _scene_picker_options()
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
    for scene in data["scenes"]:
        url = scene["url"]
        preview = scene.get("preview")
        assert (REPO_ROOT / "docs" / url).is_file(), f"scenes-list.json points at missing asset {url}"
        assert preview, f"scenes-list.json scene {url} is missing preview"
        assert (REPO_ROOT / "docs" / preview).is_file(), f"scenes-list.json points at missing preview {preview}"


def test_scene_count_matches_documented_production_bundle() -> None:
    """The public demo currently ships 8 production scenes."""
    assert len(_scene_picker_urls()) == 8


def test_readme_preview_script_covers_every_production_scene() -> None:
    """README thumbnail capture should not silently omit a picker scene."""
    module = _load_script_module(REPO_ROOT / "scripts" / "capture_readme_splat_previews.py")
    assert module.SCENES == _readme_preview_specs()


def test_hero_gif_script_uses_shared_scene_list() -> None:
    """Hero GIF recording should follow docs/scenes-list.json instead of a stale hard-coded subset."""
    module = _load_script_module(REPO_ROOT / "scripts" / "record_demo_gif.py")
    assert module._scene_urls() == _scene_picker_urls()


def test_readme_production_scene_table_matches_shared_scene_list() -> None:
    """README scene rows should stay in the same order as docs/scenes-list.json."""
    expected = [(scene["label"], f"docs/{scene['preview']}", scene["url"]) for scene in _scene_picker_scenes()]
    assert _readme_production_scene_rows() == expected


def test_readme_preview_images_cover_every_production_scene() -> None:
    """README preview PNGs should exist and be full-canvas captures."""
    for scene in _scene_picker_scenes():
        preview = REPO_ROOT / "docs" / scene["preview"]
        assert preview.stat().st_size > 50_000, f"preview looks too small: {preview}"
        with Image.open(preview) as image:
            assert image.size == (1280, 720), f"preview should be a full-canvas 1280x720 grab: {preview}"


def test_splat_html_supports_embed_mode() -> None:
    """splat.html must honor ?embed=1 so index.html can inline the viewer as a hero."""
    html = (REPO_ROOT / "docs" / "splat.html").read_text(encoding="utf-8")
    assert "body.embed" in html, "splat.html must define body.embed CSS for hero embed"
    assert "body.embed #info" in html, "embed mode should hide the info block"
    assert "body.embed .nohf" in html, "embed mode should also hide hf.space chrome"
    assert "params.get('embed')" in html, "splat.html must read ?embed=1 from the query string"
    assert "classList.add('embed')" in html, "embed mode must toggle the body class"
    assert "KeyP" in html, "embed mode must trigger the built-in carousel via the KeyP shortcut"


def test_readme_quickstart_split_lists_three_entry_points() -> None:
    """README Quickstart table must surface the photos / external SLAM / Physical AI trio."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # Section heading + deep-dive section pointers
    assert "## Quickstart — pick your entry point" in readme
    assert "[Bring Your Own Photos](#bring-your-own-photos-one-shot-pose-free)" in readme
    assert "[Import External SLAM Results](#import-external-slam-results)" in readme
    assert "[Physical AI benchmark path](#physical-ai-benchmark-path)" in readme
    # Minimum commands the Quickstart table advertises must stay runnable.
    assert "gs-mapper photos-to-splat --images ./my_photos --output outputs/my_splat" in readme
    assert "scripts/plan_external_slam_imports.py --format shell" in readme
    assert "scripts/generate_sim_catalog.py --output docs/sim-scenes.json" in readme
    # The old generic "## Quick Start" was renamed so it doesn't collide with the new entry map.
    assert "## Quick Start\n" not in readme, "rename the legacy Quick Start to CLI reference"
    assert "## CLI reference" in readme


def test_index_hero_embeds_live_outdoor_splat() -> None:
    """docs/index.html hero must show the live WebGL splat viewer, not just the fallback GIF."""
    html = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "hero-bg-splat" in html, "hero iframe needs the hero-bg-splat class"
    assert "hero-bg-fallback" in html, "hero must retain the GIF as a reduced-motion / fallback layer"
    assert "splat.html?url=" in html, "hero iframe must point at splat.html with a ?url= scene"
    assert "embed=1" in html, "hero iframe must request embed mode"
    assert "mcd-ntu-day02-supervised.splat" in html, (
        "hero should default to the supervised ntu_day_02 scene (largest outdoor production splat)"
    )
    assert "pointer-events: none" in html, "hero iframe must not capture clicks from the hero buttons"
    assert "prefers-reduced-motion" in html, "hero must fall back for users with reduced motion"
