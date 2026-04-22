"""Renderer adapters for Physical AI simulation observations."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image as PILImage

from .contract import SceneEnvironment
from .interfaces import Observation, ObservationRequest, Pose3D


@dataclass(frozen=True, slots=True)
class SplatRenderConfig:
    """Camera and raster settings for local .splat observation renders."""

    width: int = 320
    height: int = 240
    fov_degrees: float = 60.0
    near_clip: float = 0.05
    far_clip: float = 80.0
    point_radius: int = 1
    jpeg_quality: int = 85
    max_gaussians: int | None = 120_000
    ray_stride_pixels: int = 8
    max_ray_count: int | None = 4096


@dataclass(frozen=True, slots=True)
class SplatPointCloud:
    """Decoded subset of an antimatter15-style .splat asset."""

    path: Path
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    gaussian_count: int
    loaded_count: int


@dataclass(frozen=True, slots=True)
class SplatRenderFrame:
    """One local render from a decoded .splat asset."""

    cloud: SplatPointCloud
    rgb: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True, slots=True)
class SplatRaySamples:
    """LiDAR-like ray samples derived from a depth render."""

    ranges: np.ndarray
    camera_points: np.ndarray
    world_points: np.ndarray
    pixel_coordinates: np.ndarray


class ObservationRenderer(Protocol):
    """Renderer adapter used by concrete Physical AI environments."""

    def can_render(self, scene: SceneEnvironment, request: ObservationRequest) -> bool:
        """Return whether this adapter can render the request."""

    def render_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        """Render one observation for ``scene`` and ``request``."""


class SplatAssetObservationRenderer:
    """Render RGB, depth, and LiDAR-like observations from bundled `.splat` assets."""

    _DEPTH_OUTPUTS = frozenset({"depth", "validity-mask"})
    _LIDAR_OUTPUTS = frozenset({"ranges", "points"})

    def __init__(self, docs_root: str | Path, *, config: SplatRenderConfig | None = None):
        self.docs_root = Path(docs_root)
        self.config = config or SplatRenderConfig()
        self._cache: dict[Path, SplatPointCloud] = {}

    def can_render(self, scene: SceneEnvironment, request: ObservationRequest) -> bool:
        if not scene.asset_url.endswith(".splat"):
            return False
        if request.sensor_id == "rgb-forward":
            return tuple(request.outputs) == ("rgb",)
        if request.sensor_id == "depth-proxy":
            requested_outputs = set(request.outputs)
            return bool(requested_outputs) and requested_outputs.issubset(self._DEPTH_OUTPUTS)
        if request.sensor_id == "lidar-ray-proxy":
            requested_outputs = set(request.outputs)
            return bool(requested_outputs) and requested_outputs.issubset(self._LIDAR_OUTPUTS)
        return False

    def render_observation(self, scene: SceneEnvironment, request: ObservationRequest) -> Observation:
        if not self.can_render(scene, request):
            raise ValueError(f"splat asset renderer cannot render {request.sensor_id} outputs {request.outputs}")

        frame = self._render_frame(scene, request)
        if request.sensor_id == "depth-proxy":
            return self._render_depth_observation(scene, request, frame)
        if request.sensor_id == "lidar-ray-proxy":
            return self._render_lidar_observation(scene, request, frame)
        return self._render_rgb_observation(scene, request, frame)

    def _render_rgb_observation(
        self,
        scene: SceneEnvironment,
        request: ObservationRequest,
        frame: SplatRenderFrame,
    ) -> Observation:
        jpeg_bytes = encode_rgb_to_jpeg(frame.rgb, quality=self.config.jpeg_quality)
        outputs = self._base_outputs(scene, request, frame)
        outputs.update(
            {
                "mode": "splat-raster-rgb",
                "rgb": {
                    "encoding": "jpeg",
                    "width": int(self.config.width),
                    "height": int(self.config.height),
                    "jpegBase64": base64.b64encode(jpeg_bytes).decode("ascii"),
                    "byteLength": len(jpeg_bytes),
                },
                "depthStats": build_depth_stats(frame.depth, far_clip=self.config.far_clip),
            }
        )
        return Observation(sensor_id=request.sensor_id, pose=request.pose, outputs=outputs)

    def _render_depth_observation(
        self,
        scene: SceneEnvironment,
        request: ObservationRequest,
        frame: SplatRenderFrame,
    ) -> Observation:
        validity_mask = build_validity_mask(frame.depth, far_clip=self.config.far_clip)
        depth_bytes = encode_depth_float32(frame.depth)
        mask_bytes = validity_mask.astype(np.uint8).tobytes()
        outputs = self._base_outputs(scene, request, frame)
        outputs["mode"] = "splat-raster-depth"
        if "depth" in request.outputs:
            outputs["depth"] = {
                "encoding": "float32-le",
                "unit": "meter",
                "width": int(self.config.width),
                "height": int(self.config.height),
                "depthBase64": base64.b64encode(depth_bytes).decode("ascii"),
                "byteLength": len(depth_bytes),
                "nearClipMeters": float(self.config.near_clip),
                "farClipMeters": float(self.config.far_clip),
            }
        if "validity-mask" in request.outputs:
            outputs["validityMask"] = {
                "encoding": "uint8-0-or-1",
                "width": int(self.config.width),
                "height": int(self.config.height),
                "maskBase64": base64.b64encode(mask_bytes).decode("ascii"),
                "byteLength": len(mask_bytes),
            }
        outputs["depthStats"] = build_depth_stats(frame.depth, far_clip=self.config.far_clip)
        return Observation(sensor_id=request.sensor_id, pose=request.pose, outputs=outputs)

    def _render_lidar_observation(
        self,
        scene: SceneEnvironment,
        request: ObservationRequest,
        frame: SplatRenderFrame,
    ) -> Observation:
        samples = build_lidar_ray_samples(frame.depth, request.pose, self.config)
        outputs = self._base_outputs(scene, request, frame)
        outputs["mode"] = "splat-raster-lidar"
        outputs["rayStats"] = {
            "rayCount": int(samples.ranges.shape[0]),
            "stridePixels": max(int(self.config.ray_stride_pixels), 1),
            "maxRayCount": self.config.max_ray_count,
            "farClipMeters": float(self.config.far_clip),
        }
        if "ranges" in request.outputs:
            range_bytes = encode_float32_array(samples.ranges)
            outputs["ranges"] = {
                "encoding": "float32-le",
                "unit": "meter",
                "count": int(samples.ranges.shape[0]),
                "rangesBase64": base64.b64encode(range_bytes).decode("ascii"),
                "byteLength": len(range_bytes),
            }
        if "points" in request.outputs:
            point_bytes = encode_float32_array(samples.world_points)
            outputs["points"] = {
                "encoding": "float32-le-xyz",
                "coordinateFrame": request.pose.frame_id,
                "components": ["x", "y", "z"],
                "count": int(samples.world_points.shape[0]),
                "pointsBase64": base64.b64encode(point_bytes).decode("ascii"),
                "byteLength": len(point_bytes),
            }
        return Observation(sensor_id=request.sensor_id, pose=request.pose, outputs=outputs)

    def _render_frame(self, scene: SceneEnvironment, request: ObservationRequest) -> SplatRenderFrame:
        cloud = self._load_cloud(scene)
        rgb, depth = render_splat_point_cloud(cloud, request.pose, self.config)
        return SplatRenderFrame(cloud=cloud, rgb=rgb, depth=depth)

    def _base_outputs(
        self,
        scene: SceneEnvironment,
        request: ObservationRequest,
        frame: SplatRenderFrame,
    ) -> dict[str, object]:
        return {
            "sceneId": scene.scene_id,
            "assetUrl": scene.asset_url,
            "renderer": "splat-asset-simple",
            "gaussianCount": frame.cloud.gaussian_count,
            "loadedGaussianCount": frame.cloud.loaded_count,
            "requestedOutputs": list(request.outputs),
            "cameraInfo": build_camera_info(
                width=self.config.width,
                height=self.config.height,
                fov_degrees=self.config.fov_degrees,
                frame_id=request.pose.frame_id,
            ),
        }

    def _load_cloud(self, scene: SceneEnvironment) -> SplatPointCloud:
        path = resolve_scene_asset_path(self.docs_root, scene.asset_url)
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        cloud = load_splat_point_cloud(path, max_gaussians=self.config.max_gaussians)
        self._cache[path] = cloud
        return cloud


def resolve_scene_asset_path(docs_root: str | Path, asset_url: str) -> Path:
    """Resolve a scene asset URL under a local docs root."""

    root = Path(docs_root)
    path = (root / asset_url).resolve()
    root_resolved = root.resolve()
    if root_resolved not in path.parents and path != root_resolved:
        raise ValueError(f"scene asset escapes docs root: {asset_url}")
    return path


def load_splat_point_cloud(path: str | Path, *, max_gaussians: int | None = None) -> SplatPointCloud:
    """Load position/color/opacity fields from an antimatter15 `.splat` asset."""

    asset_path = Path(path)
    if asset_path.suffix != ".splat":
        raise ValueError(f"expected a .splat asset, got: {asset_path}")
    if not asset_path.is_file():
        raise FileNotFoundError(f"splat asset not found: {asset_path}")

    dtype = np.dtype(
        [
            ("position", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rotation", "u1", 4),
        ]
    )
    asset_size = asset_path.stat().st_size
    if asset_size <= 0 or asset_size % dtype.itemsize != 0:
        raise ValueError(f"splat asset must be non-empty and {dtype.itemsize}-byte aligned: {asset_path}")

    raw = np.fromfile(asset_path, dtype=dtype)
    gaussian_count = int(raw.shape[0])
    if max_gaussians is not None and max_gaussians > 0 and gaussian_count > max_gaussians:
        indices = np.linspace(0, gaussian_count - 1, num=int(max_gaussians), dtype=np.int64)
        raw = raw[indices]

    rgba = raw["rgba"].astype(np.float32) / 255.0
    return SplatPointCloud(
        path=asset_path,
        positions=raw["position"].astype(np.float32),
        colors=rgba[:, :3],
        opacities=rgba[:, 3],
        gaussian_count=gaussian_count,
        loaded_count=int(raw.shape[0]),
    )


def render_splat_point_cloud(
    cloud: SplatPointCloud,
    pose: Pose3D,
    config: SplatRenderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a simple RGB/depth frame by projecting .splat centers."""

    width, height = _validate_image_shape(config.width, config.height)
    near_clip = float(config.near_clip)
    far_clip = float(config.far_clip)
    if near_clip <= 0.0 or far_clip <= near_clip:
        raise ValueError("clip planes must satisfy 0 < near_clip < far_clip")

    fx, fy, cx, cy = compute_camera_intrinsics(width, height, config.fov_degrees)
    rotation_world_to_camera, translation = world_to_camera_transform(pose)
    camera_space = cloud.positions @ rotation_world_to_camera.T + translation
    depths = camera_space[:, 2]
    valid = (depths > near_clip) & (depths < far_clip)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), far_clip, dtype=np.float32)
    if not np.any(valid):
        return rgb, depth

    camera_space = camera_space[valid]
    depths = depths[valid]
    colors = cloud.colors[valid]
    opacities = cloud.opacities[valid]

    px = np.rint((camera_space[:, 0] / depths) * fx + cx).astype(np.int32)
    py = np.rint(cy - (camera_space[:, 1] / depths) * fy).astype(np.int32)
    px, py, depths, colors, opacities = expand_point_footprints(
        px,
        py,
        depths,
        colors,
        opacities,
        point_radius=config.point_radius,
    )

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(in_bounds):
        return rgb, depth

    px = px[in_bounds]
    py = py[in_bounds]
    depths = depths[in_bounds]
    colors = colors[in_bounds]
    opacities = opacities[in_bounds]

    pixel_indices = py * width + px
    sort_order = np.lexsort((depths, pixel_indices))
    pixel_indices = pixel_indices[sort_order]
    depths = depths[sort_order]
    colors = colors[sort_order]
    opacities = opacities[sort_order]

    _, first_indices = np.unique(pixel_indices, return_index=True)
    pixel_indices = pixel_indices[first_indices]
    depths = depths[first_indices]
    colors = colors[first_indices]
    opacities = opacities[first_indices]

    rgb.reshape(-1, 3)[pixel_indices] = np.clip(colors * opacities[:, None] * 255.0, 0.0, 255.0).astype(np.uint8)
    depth.reshape(-1)[pixel_indices] = depths.astype(np.float32)
    return rgb, depth


def compute_camera_intrinsics(width: int, height: int, fov_degrees: float) -> tuple[float, float, float, float]:
    """Compute pinhole intrinsics from vertical field of view."""

    width, height = _validate_image_shape(width, height)
    if not math.isfinite(fov_degrees) or fov_degrees <= 0.0 or fov_degrees >= 179.0:
        raise ValueError("fov_degrees must be within (0, 179)")
    fov_radians = math.radians(float(fov_degrees))
    fy = height / (2.0 * math.tan(fov_radians * 0.5))
    fx = fy * (width / height)
    return (float(fx), float(fy), width * 0.5, height * 0.5)


def build_camera_info(width: int, height: int, fov_degrees: float, frame_id: str) -> dict[str, object]:
    """Build a compact camera-info payload for observation metadata."""

    fx, fy, cx, cy = compute_camera_intrinsics(width, height, fov_degrees)
    return {
        "frameId": frame_id,
        "width": int(width),
        "height": int(height),
        "distortionModel": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
    }


def build_validity_mask(depth: np.ndarray, *, far_clip: float) -> np.ndarray:
    """Return a uint8 mask where finite values before far clip are valid."""

    depth_array = np.asarray(depth, dtype=np.float32)
    return (np.isfinite(depth_array) & (depth_array < float(far_clip))).astype(np.uint8)


def build_depth_stats(depth: np.ndarray, *, far_clip: float) -> dict[str, object]:
    """Summarize valid depth coverage for an observation."""

    depth_array = np.asarray(depth, dtype=np.float32)
    valid = build_validity_mask(depth_array, far_clip=far_clip).astype(bool)
    valid_depth = depth_array[valid]
    valid_pixel_count = int(valid_depth.size)
    return {
        "validPixelCount": valid_pixel_count,
        "validPixelRatio": valid_pixel_count / float(depth_array.size),
        "minMeters": float(valid_depth.min()) if valid_pixel_count else None,
        "maxMeters": float(valid_depth.max()) if valid_pixel_count else None,
        "farClipMeters": float(far_clip),
    }


def encode_depth_float32(depth: np.ndarray) -> bytes:
    """Encode a depth image as little-endian float32 bytes."""

    return encode_float32_array(depth)


def encode_float32_array(values: np.ndarray) -> bytes:
    """Encode an array as little-endian float32 bytes."""

    return np.asarray(values, dtype="<f4").tobytes()


def build_lidar_ray_samples(depth: np.ndarray, pose: Pose3D, config: SplatRenderConfig) -> SplatRaySamples:
    """Build LiDAR-like ranges and world points from a depth image."""

    depth_array = np.asarray(depth, dtype=np.float32)
    height, width = depth_array.shape
    stride = max(int(config.ray_stride_pixels), 1)
    valid = build_validity_mask(depth_array, far_clip=config.far_clip).astype(bool)
    sampled = np.zeros_like(valid, dtype=bool)
    sampled[::stride, ::stride] = True
    sample_mask = valid & sampled
    pixel_y, pixel_x = np.nonzero(sample_mask)

    if config.max_ray_count is not None and config.max_ray_count > 0 and pixel_x.size > config.max_ray_count:
        indices = np.linspace(0, pixel_x.size - 1, num=int(config.max_ray_count), dtype=np.int64)
        pixel_x = pixel_x[indices]
        pixel_y = pixel_y[indices]

    if pixel_x.size == 0:
        empty_float = np.empty((0,), dtype=np.float32)
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.int32)
        return SplatRaySamples(
            ranges=empty_float,
            camera_points=empty_points,
            world_points=empty_points,
            pixel_coordinates=empty_pixels,
        )

    fx, fy, cx, cy = compute_camera_intrinsics(width, height, config.fov_degrees)
    z = depth_array[pixel_y, pixel_x].astype(np.float32)
    x = ((pixel_x.astype(np.float32) - cx) / fx) * z
    y = ((cy - pixel_y.astype(np.float32)) / fy) * z
    camera_points = np.column_stack([x, y, z]).astype(np.float32)
    ranges = np.linalg.norm(camera_points, axis=1).astype(np.float32)
    rotation_camera_to_world = quaternion_to_rotation_matrix(pose.orientation_xyzw)
    position = np.asarray(pose.position, dtype=np.float32)
    world_points = (camera_points @ rotation_camera_to_world.T + position).astype(np.float32)
    pixel_coordinates = np.column_stack([pixel_x, pixel_y]).astype(np.int32)
    return SplatRaySamples(
        ranges=ranges,
        camera_points=camera_points,
        world_points=world_points,
        pixel_coordinates=pixel_coordinates,
    )


def world_to_camera_transform(pose: Pose3D) -> tuple[np.ndarray, np.ndarray]:
    """Return world-to-camera rotation and translation for a camera-to-world pose."""

    rotation_camera_to_world = quaternion_to_rotation_matrix(pose.orientation_xyzw)
    position = np.asarray(pose.position, dtype=np.float32)
    rotation_world_to_camera = rotation_camera_to_world.T
    translation = -rotation_world_to_camera @ position
    return rotation_world_to_camera, translation


def quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    """Convert an `[x, y, z, w]` quaternion to a 3x3 rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def expand_point_footprints(
    px: np.ndarray,
    py: np.ndarray,
    depths: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    *,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast projected points over a circular pixel footprint."""

    offsets = build_footprint_offsets(point_radius)
    if offsets.shape[0] <= 1:
        return px, py, depths, colors, opacities

    px = px[:, None] + offsets[None, :, 0]
    py = py[:, None] + offsets[None, :, 1]
    depths = np.broadcast_to(depths[:, None], px.shape).reshape(-1)
    colors = np.broadcast_to(colors[:, None, :], (colors.shape[0], offsets.shape[0], 3)).reshape(-1, 3)
    opacities = np.broadcast_to(opacities[:, None], px.shape).reshape(-1)
    return px.reshape(-1), py.reshape(-1), depths, colors, opacities


def build_footprint_offsets(point_radius: int) -> np.ndarray:
    """Build integer pixel offsets for a circular point footprint."""

    radius = max(int(point_radius), 0)
    if radius == 0:
        return np.array([[0, 0]], dtype=np.int32)

    radius_sq = radius * radius
    offsets = [
        (x_offset, y_offset)
        for y_offset in range(-radius, radius + 1)
        for x_offset in range(-radius, radius + 1)
        if x_offset * x_offset + y_offset * y_offset <= radius_sq
    ]
    return np.asarray(offsets, dtype=np.int32)


def encode_rgb_to_jpeg(rgb_image: np.ndarray, *, quality: int = 85) -> bytes:
    """Encode an RGB uint8 array into JPEG bytes."""

    image = PILImage.fromarray(np.asarray(rgb_image, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    return buffer.getvalue()


def _validate_image_shape(width: int, height: int) -> tuple[int, int]:
    resolved_width = int(width)
    resolved_height = int(height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("image width and height must be positive")
    return resolved_width, resolved_height
