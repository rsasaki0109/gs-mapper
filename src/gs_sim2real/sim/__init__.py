"""Physical AI simulation contracts for GS Mapper scenes."""

from .contract import (
    DEFAULT_SITE_URL,
    AxisAlignedBounds,
    CoordinateFrame,
    EvaluationTask,
    SceneEnvironment,
    SensorModel,
    SensorRig,
    SimulationCatalog,
    TrajectoryEpisode,
    Vec3,
    build_simulation_catalog,
    load_scene_picker_catalog,
    load_simulation_catalog_from_scene_picker,
    render_simulation_catalog_json,
)
from .costmap import TrajectoryCollisionSummary, summarize_collision_queries
from .footprint import RobotFootprint
from .headless import (
    HeadlessEnvironmentState,
    HeadlessPhysicalAIEnvironment,
)
from .interfaces import (
    AgentAction,
    CollisionQuery,
    Observation,
    ObservationRequest,
    PhysicalAIEnvironment,
    Pose3D,
    TrajectoryScore,
)
from .occupancy import (
    OccupancyQuery,
    VoxelOccupancyGrid,
    build_occupancy_grid_from_lidar_observation,
    decode_lidar_points_observation,
    point_to_voxel_cell,
)
from .planning import (
    OccupancyPlanningContext,
    PlanningViewpointKey,
)
from .rendering import (
    ObservationRenderer,
    SplatAssetObservationRenderer,
    SplatPointCloud,
    SplatRenderConfig,
    load_splat_point_cloud,
    render_splat_point_cloud,
    resolve_scene_asset_path,
)
from .route_planning import (
    RouteCandidate,
    RouteEvaluation,
    RoutePlan,
    select_best_route,
)

__all__ = [
    "DEFAULT_SITE_URL",
    "AgentAction",
    "AxisAlignedBounds",
    "CollisionQuery",
    "CoordinateFrame",
    "EvaluationTask",
    "HeadlessEnvironmentState",
    "HeadlessPhysicalAIEnvironment",
    "Observation",
    "ObservationRenderer",
    "ObservationRequest",
    "OccupancyQuery",
    "OccupancyPlanningContext",
    "PhysicalAIEnvironment",
    "Pose3D",
    "PlanningViewpointKey",
    "RobotFootprint",
    "RouteCandidate",
    "RouteEvaluation",
    "RoutePlan",
    "SceneEnvironment",
    "SensorModel",
    "SensorRig",
    "SplatAssetObservationRenderer",
    "SplatPointCloud",
    "SplatRenderConfig",
    "SimulationCatalog",
    "TrajectoryEpisode",
    "TrajectoryCollisionSummary",
    "TrajectoryScore",
    "Vec3",
    "VoxelOccupancyGrid",
    "build_occupancy_grid_from_lidar_observation",
    "build_simulation_catalog",
    "decode_lidar_points_observation",
    "load_scene_picker_catalog",
    "load_splat_point_cloud",
    "load_simulation_catalog_from_scene_picker",
    "render_splat_point_cloud",
    "render_simulation_catalog_json",
    "point_to_voxel_cell",
    "resolve_scene_asset_path",
    "select_best_route",
    "summarize_collision_queries",
]
