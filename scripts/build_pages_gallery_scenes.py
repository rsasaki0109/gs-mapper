#!/usr/bin/env python3
"""Build photo-derived static scene bundles for the GitHub Pages demo."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gs_sim2real.viewer.web_export import points_to_scene_bundle


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS_DIR = REPO_ROOT / "docs" / "assets"
SCENE_INDEX_PATH = DOCS_ASSETS_DIR / "scenes.json"


@dataclass(frozen=True)
class GallerySceneSpec:
    scene_id: str
    label: str
    description: str
    image_path: Path
    output_dir: Path
    target_points: int = 18000


SCENE_SPECS = (
    GallerySceneSpec(
        scene_id="street-gallery",
        label="Street Gallery",
        description="Photo-derived point scene built from the street sample gallery image.",
        image_path=REPO_ROOT / "docs" / "gallery" / "street" / "street_view_01.jpg",
        output_dir=DOCS_ASSETS_DIR / "street-gallery",
    ),
    GallerySceneSpec(
        scene_id="campus-gallery",
        label="Campus Gallery",
        description="Photo-derived point scene built from the campus sample gallery image.",
        image_path=REPO_ROOT / "docs" / "gallery" / "campus" / "campus_view_01.jpg",
        output_dir=DOCS_ASSETS_DIR / "campus-gallery",
    ),
    GallerySceneSpec(
        scene_id="indoor-gallery",
        label="Indoor Gallery",
        description="Photo-derived point scene built from the indoor sample gallery image.",
        image_path=REPO_ROOT / "docs" / "gallery" / "indoor" / "indoor_view_01.jpg",
        output_dir=DOCS_ASSETS_DIR / "indoor-gallery",
    ),
)


def _build_photo_point_cloud(image_path: Path, *, target_points: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((320, 320))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape

    stride = max(1, int(math.sqrt((height * width) / max(target_points, 1))))
    rgb = rgb[::stride, ::stride]
    height, width, _ = rgb.shape

    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grad_y, grad_x = np.gradient(luma)
    edge_strength = np.clip(np.abs(grad_x) + np.abs(grad_y), 0.0, 1.0)

    horizontal = grid_x - 0.5
    vertical = 0.5 - grid_y
    perspective = 0.85 + 0.9 * (1.0 - grid_y)
    pseudo_depth = 1.2 + 0.75 * (1.0 - luma) + 0.65 * (1.0 - grid_y) + 0.25 * edge_strength
    ripple = 0.08 * np.sin(horizontal * np.pi * 5.0) * (0.5 + edge_strength)

    positions = np.stack(
        [
            horizontal * 6.0 * perspective,
            vertical * 4.0 * perspective,
            -(pseudo_depth + ripple),
        ],
        axis=-1,
    ).reshape(-1, 3)

    colors = rgb.reshape(-1, 3)

    rng = np.random.default_rng(7)
    jitter = rng.normal(scale=0.0125, size=positions.shape).astype(np.float32)
    jitter[:, 2] *= 0.35
    positions = positions.astype(np.float32) + jitter
    return positions, colors.astype(np.float32)


def _build_front_camera(positions: np.ndarray) -> dict[str, list[float]]:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = (minimum + maximum) * 0.5
    extents = np.maximum(maximum - minimum, 1e-3)
    front_distance = max(float(extents[0]), float(extents[1])) * 1.05 + abs(float(minimum[2])) * 0.35
    position = np.array([center[0], center[1] * 0.1, maximum[2] + front_distance], dtype=np.float32)
    return {
        "position": position.tolist(),
        "target": center.astype(np.float32).tolist(),
        "up": [0.0, 1.0, 0.0],
    }


def build_pages_gallery_scenes() -> None:
    scene_entries: list[dict[str, str]] = []
    for spec in SCENE_SPECS:
        positions, colors = _build_photo_point_cloud(spec.image_path, target_points=spec.target_points)
        points_to_scene_bundle(
            positions,
            colors,
            str(spec.output_dir),
            asset_format="binary",
            scene_id=spec.scene_id,
            label=spec.label,
            description=spec.description,
            camera=_build_front_camera(positions),
        )
        scene_entries.append(
            {
                "id": spec.scene_id,
                "label": spec.label,
                "manifest": f"assets/{spec.scene_id}/scene.json",
                "description": spec.description,
            }
        )

    scene_index = {
        "version": "gs-sim2real-scene-index/v1",
        "type": "web-scene-index",
        "scenes": scene_entries,
    }
    SCENE_INDEX_PATH.write_text(json.dumps(scene_index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_pages_gallery_scenes()
