"""Tests for renderer-backed Physical AI observations."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    Pose3D,
    SplatAssetObservationRenderer,
    SplatRenderConfig,
    build_simulation_catalog,
    load_splat_point_cloud,
    render_splat_point_cloud,
    resolve_scene_asset_path,
)


def test_load_splat_point_cloud_reads_asset_and_subsamples(tmp_path: Path) -> None:
    asset_path = tmp_path / "scene.splat"
    write_test_splat(
        asset_path,
        [
            ((0.0, 0.0, 3.0), (0, 0, 255, 255)),
            ((0.0, 0.0, 5.0), (255, 0, 0, 255)),
            ((1.0, 0.0, 5.0), (0, 255, 0, 128)),
        ],
    )

    cloud = load_splat_point_cloud(asset_path, max_gaussians=2)

    assert cloud.gaussian_count == 3
    assert cloud.loaded_count == 2
    assert cloud.positions.shape == (2, 3)
    assert cloud.colors.min() >= 0.0
    assert cloud.colors.max() <= 1.0
    assert cloud.opacities[0] == 1.0


def test_render_splat_point_cloud_projects_rgb_and_depth(tmp_path: Path) -> None:
    asset_path = tmp_path / "scene.splat"
    write_test_splat(
        asset_path,
        [
            ((0.0, 0.0, 3.0), (0, 0, 255, 255)),
            ((0.0, 0.0, 5.0), (255, 0, 0, 255)),
            ((1.0, 0.0, 5.0), (0, 255, 0, 255)),
        ],
    )
    cloud = load_splat_point_cloud(asset_path)
    config = SplatRenderConfig(width=64, height=48, far_clip=20.0, point_radius=1)
    pose = Pose3D(position=(0.0, 0.0, 0.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")

    rgb, depth = render_splat_point_cloud(cloud, pose, config)

    assert rgb.shape == (48, 64, 3)
    assert depth.shape == (48, 64)
    assert depth[24, 32] == np.float32(3.0)
    assert rgb[24, 32, 2] > rgb[24, 32, 0]
    assert np.count_nonzero(depth < 20.0) >= 2


def test_splat_asset_renderer_feeds_headless_environment_observation(tmp_path: Path) -> None:
    asset_path = tmp_path / "assets" / "scene-one" / "scene-one.splat"
    write_test_splat(
        asset_path,
        [
            ((0.0, 0.0, 3.0), (0, 0, 255, 255)),
            ((0.5, 0.0, 4.0), (0, 255, 0, 255)),
        ],
    )
    catalog = build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/scene-one/scene-one.splat",
                    "label": "Scene One",
                    "summary": "Tiny renderer integration fixture",
                }
            ]
        },
        docs_root=tmp_path,
        site_url="https://example.test/gs/",
    )
    env = HeadlessPhysicalAIEnvironment(
        catalog,
        observation_renderer=SplatAssetObservationRenderer(
            tmp_path,
            config=SplatRenderConfig(width=64, height=48, far_clip=20.0, point_radius=1),
        ),
    )
    env.reset("scene-one")

    observation = env.render_observation(
        ObservationRequest(
            pose=Pose3D(
                position=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=env.state.pose.frame_id,
            ),
            sensor_id="rgb-forward",
        )
    )

    outputs = observation.outputs
    jpeg = base64.b64decode(outputs["rgb"]["jpegBase64"])
    assert outputs["mode"] == "splat-raster-rgb"
    assert outputs["sceneId"] == "scene-one"
    assert outputs["assetUrl"] == "assets/scene-one/scene-one.splat"
    assert outputs["rgb"]["width"] == 64
    assert outputs["rgb"]["height"] == 48
    assert jpeg.startswith(b"\xff\xd8")
    assert outputs["cameraInfo"]["frameId"] == env.state.pose.frame_id
    assert outputs["depthStats"]["validPixelCount"] > 0


def test_splat_asset_renderer_returns_depth_proxy_payload(tmp_path: Path) -> None:
    asset_path = tmp_path / "assets" / "scene-one" / "scene-one.splat"
    write_test_splat(
        asset_path,
        [
            ((0.0, 0.0, 3.0), (0, 0, 255, 255)),
            ((0.0, 0.0, 5.0), (255, 0, 0, 255)),
            ((1.0, 0.0, 5.0), (0, 255, 0, 255)),
        ],
    )
    catalog = build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/scene-one/scene-one.splat",
                    "label": "Scene One",
                    "summary": "Tiny depth integration fixture",
                }
            ]
        },
        docs_root=tmp_path,
        site_url="https://example.test/gs/",
    )
    env = HeadlessPhysicalAIEnvironment(
        catalog,
        observation_renderer=SplatAssetObservationRenderer(
            tmp_path,
            config=SplatRenderConfig(width=64, height=48, far_clip=20.0, point_radius=1),
        ),
    )
    env.reset("scene-one")

    observation = env.render_observation(
        ObservationRequest(
            pose=Pose3D(
                position=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=env.state.pose.frame_id,
            ),
            sensor_id="depth-proxy",
            outputs=("depth", "validity-mask"),
        )
    )

    outputs = observation.outputs
    depth = np.frombuffer(base64.b64decode(outputs["depth"]["depthBase64"]), dtype="<f4").reshape(48, 64)
    mask = np.frombuffer(base64.b64decode(outputs["validityMask"]["maskBase64"]), dtype=np.uint8).reshape(48, 64)
    assert outputs["mode"] == "splat-raster-depth"
    assert outputs["depth"]["encoding"] == "float32-le"
    assert outputs["validityMask"]["encoding"] == "uint8-0-or-1"
    assert outputs["depthStats"]["validPixelCount"] == int(mask.sum())
    assert depth[24, 32] == np.float32(3.0)
    assert mask[24, 32] == 1


def test_splat_asset_renderer_returns_lidar_ray_proxy_payload(tmp_path: Path) -> None:
    asset_path = tmp_path / "assets" / "scene-one" / "scene-one.splat"
    write_test_splat(
        asset_path,
        [
            ((0.0, 0.0, 3.0), (0, 0, 255, 255)),
            ((0.0, 0.0, 5.0), (255, 0, 0, 255)),
            ((1.0, 0.0, 5.0), (0, 255, 0, 255)),
        ],
    )
    catalog = build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/scene-one/scene-one.splat",
                    "label": "Scene One",
                    "summary": "Tiny LiDAR integration fixture",
                }
            ]
        },
        docs_root=tmp_path,
        site_url="https://example.test/gs/",
    )
    env = HeadlessPhysicalAIEnvironment(
        catalog,
        observation_renderer=SplatAssetObservationRenderer(
            tmp_path,
            config=SplatRenderConfig(width=64, height=48, far_clip=20.0, point_radius=1, ray_stride_pixels=1),
        ),
    )
    env.reset("scene-one")

    observation = env.render_observation(
        ObservationRequest(
            pose=Pose3D(
                position=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id=env.state.pose.frame_id,
            ),
            sensor_id="lidar-ray-proxy",
            outputs=("ranges", "points"),
        )
    )

    outputs = observation.outputs
    ranges = np.frombuffer(base64.b64decode(outputs["ranges"]["rangesBase64"]), dtype="<f4")
    points = np.frombuffer(base64.b64decode(outputs["points"]["pointsBase64"]), dtype="<f4").reshape(-1, 3)
    assert outputs["mode"] == "splat-raster-lidar"
    assert outputs["ranges"]["encoding"] == "float32-le"
    assert outputs["points"]["encoding"] == "float32-le-xyz"
    assert outputs["points"]["coordinateFrame"] == env.state.pose.frame_id
    assert outputs["rayStats"]["rayCount"] == ranges.shape[0] == points.shape[0]
    assert np.any(np.isclose(ranges, 3.0))
    assert np.any(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 1], 0.0) & np.isclose(points[:, 2], 3.0))


def test_resolve_scene_asset_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes docs root"):
        resolve_scene_asset_path(tmp_path, "../outside.splat")


def write_test_splat(path: Path, rows: list[tuple[tuple[float, float, float], tuple[int, int, int, int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("position", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rotation", "u1", 4),
        ]
    )
    raw = np.empty(len(rows), dtype=dtype)
    for index, (position, rgba) in enumerate(rows):
        raw["position"][index] = position
        raw["scale"][index] = (0.01, 0.01, 0.01)
        raw["rgba"][index] = rgba
        raw["rotation"][index] = (128, 128, 128, 255)
    raw.tofile(path)
