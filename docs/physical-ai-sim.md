# Physical AI Simulation Contract

GS Mapper is moving from a 3DGS demo repository toward a scene environment layer for Physical AI agents. The first contract is intentionally small: it standardizes what each reconstructed scene exposes before adding a full physics engine or renderer backend.

Generated catalog: [`docs/sim-scenes.json`](sim-scenes.json)

Generator:

```bash
python3 scripts/generate_sim_catalog.py --output docs/sim-scenes.json
```

## Contract

Each `SceneEnvironment` describes:

- `sceneId`, `label`, `summary`, and links to the `.splat`, preview image, and live viewer deep link
- coordinate frame: unit, up axis, gravity vector, handedness, and scale status
- world bounds with `declared`, `estimated`, or `placeholder` confidence
- sensor rig: RGB, depth-proxy, and LiDAR-ray-proxy contracts
- train/eval trajectory episodes
- evaluation tasks for localization, viewpoint planning, waypoint navigation, and mapping coverage

The Python API lives in `gs_sim2real.sim`:

```python
from pathlib import Path

from gs_sim2real.sim import load_simulation_catalog_from_scene_picker

catalog = load_simulation_catalog_from_scene_picker(Path("docs/scenes-list.json"))
scene = catalog.scene_by_id("bag6-mast3r")

print(scene.coordinate_frame.scale_status)
print(scene.viewer_url)
print([task.task_id for task in scene.evaluation_tasks])
```

## Current Readiness

`rgb-forward` is ready through the existing splat viewers and the local `.splat` observation renderer. `depth-proxy` is backed by the same local rasterizer and returns float32 depth plus a validity mask. `lidar-ray-proxy` samples that depth image into LiDAR-like ranges and world-frame points. Those ray points can be converted into a sparse voxel occupancy grid for geometry-aware collision checks.

Metric and estimated-metric scenes expose navigation and mapping tasks. Relative-scale scenes expose localization and viewpoint-planning tasks, but avoid waypoint-navigation scoring until a metric alignment is provided.

## Scene Sources

The simulation catalog is derived from the existing public scene picker at `docs/scenes-list.json`. This keeps the Physical AI environment list synchronized with the eight public bundled splats.

The supervised default `outdoor-demo` scene reads declared bounds from `docs/assets/outdoor-demo/scene.json`. Other bundled splats currently use README trajectory extents as estimated bounds. This is explicit in each `bounds.confidence` value and prevents downstream agents from treating estimated boxes as surveyed collision geometry.

## Environment Interface

Concrete environments should implement `PhysicalAIEnvironment`:

- `reset(scene_id, seed=None)`
- `step(action)`
- `render_observation(request)`
- `query_collision(pose)`
- `sample_goal(scene_id, seed=None)`
- `score_trajectory(scene_id, trajectory)`

Payload dataclasses are available for stable integration:

- `Pose3D`
- `AgentAction`
- `ObservationRequest`
- `Observation`
- `CollisionQuery`
- `TrajectoryScore`

## Headless Backend

`HeadlessPhysicalAIEnvironment` is the first executable backend. It does not simulate dynamics yet; it gives agents a deterministic bounds-based environment that can reset scenes, sample goals, apply simple actions, reject out-of-bounds poses, and return observations. Without an injected renderer, observations stay metadata-only.

```python
from pathlib import Path

from gs_sim2real.sim import (
    AgentAction,
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    load_simulation_catalog_from_scene_picker,
)

catalog = load_simulation_catalog_from_scene_picker(Path("docs/scenes-list.json"))
env = HeadlessPhysicalAIEnvironment(catalog)

reset = env.reset("outdoor-demo")
goal = env.sample_goal("outdoor-demo", seed=42)
transition = env.step(AgentAction("twist", {"linearX": 0.5}, duration_seconds=1.0))
collision = env.query_collision(env.state.pose)
observation = env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward"))
```

For local renderer-backed RGB, depth, and ray proxies, inject `SplatAssetObservationRenderer`. It reads the same bundled `.splat` assets as the public viewer and returns JPEG-backed `rgb-forward` observations, float32 `depth-proxy` observations with validity masks, or LiDAR-like `lidar-ray-proxy` ranges and points.

```python
from pathlib import Path

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    ObservationRequest,
    SplatAssetObservationRenderer,
    SplatRenderConfig,
    load_simulation_catalog_from_scene_picker,
)

docs_root = Path("docs")
catalog = load_simulation_catalog_from_scene_picker(docs_root / "scenes-list.json")
renderer = SplatAssetObservationRenderer(
    docs_root,
    config=SplatRenderConfig(width=320, height=240, far_clip=80.0, point_radius=1),
)
env = HeadlessPhysicalAIEnvironment(catalog, observation_renderer=renderer)

env.reset("outdoor-demo")
observation = env.render_observation(ObservationRequest(pose=env.state.pose, sensor_id="rgb-forward"))
jpeg_base64 = observation.outputs["rgb"]["jpegBase64"]

depth = env.render_observation(
    ObservationRequest(
        pose=env.state.pose,
        sensor_id="depth-proxy",
        outputs=("depth", "validity-mask"),
    )
)
depth_base64 = depth.outputs["depth"]["depthBase64"]
mask_base64 = depth.outputs["validityMask"]["maskBase64"]

lidar = env.render_observation(
    ObservationRequest(
        pose=env.state.pose,
        sensor_id="lidar-ray-proxy",
        outputs=("ranges", "points"),
    )
)
ranges_base64 = lidar.outputs["ranges"]["rangesBase64"]
points_base64 = lidar.outputs["points"]["pointsBase64"]
```

To turn those ray points into lightweight collision geometry, build an occupancy grid and inject it into the headless environment. The backend can query either the pose point or a conservative circular robot footprint.

```python
from gs_sim2real.sim import (
    OccupancyPlanningContext,
    RobotFootprint,
    RouteCandidate,
    build_occupancy_grid_from_lidar_observation,
    build_route_policy_sample,
    replan_after_blocked_rollout,
    rollout_route,
    rollout_route_with_replanning,
    select_best_route,
)

occupancy = build_occupancy_grid_from_lidar_observation(
    lidar,
    voxel_size_meters=0.5,
    inflation_radius_meters=0.5,
)
env.set_occupancy_grid(occupancy)
env.set_robot_footprint(RobotFootprint(radius_meters=0.45, height_meters=1.2))

collision = env.query_collision(env.state.pose)
```

For repeated route checks, use `OccupancyPlanningContext` to cache the LiDAR-to-occupancy result by scene, viewpoint, and voxel settings:

```python
planning = OccupancyPlanningContext(voxel_size_meters=0.5, inflation_radius_meters=0.5)
occupancy = planning.set_environment_occupancy(env, scene_id="outdoor-demo", pose=env.state.pose)
cache_info = planning.cache_info()
```

For route-level planning, pass candidate trajectories through the same scoring path. The selected route is ranked by pass/fail, collision rate, collision count, clearance, and path length.

```python
plan = select_best_route(
    env,
    scene_id="outdoor-demo",
    candidates=(
        RouteCandidate("nominal", (env.state.pose, goal)),
        RouteCandidate("detour", (env.state.pose, waypoint, goal)),
    ),
    planning_context=planning,
)
best_route = plan.selected.candidate
```

To execute the selected route through the environment step contract, roll it out as either absolute `teleport` steps or fixed-duration `twist` segments. The rollout records every transition plus applied/collision status for each action.

```python
rollout = rollout_route(
    env,
    plan.selected,
    action_type="teleport",
    stop_on_collision=True,
)
route_passed = rollout.passed
rollout_metrics = rollout.metrics()
```

For closed-loop recovery, feed a blocked rollout into replanning. Candidate continuations are automatically anchored at the last applied pose before being scored and optionally executed.

```python
replan = replan_after_blocked_rollout(
    env,
    scene_id="outdoor-demo",
    rollout=rollout,
    candidates=(
        RouteCandidate("recover-left", (left_waypoint, goal)),
        RouteCandidate("recover-right", (right_waypoint, goal)),
    ),
    planning_context=planning,
    execute=True,
)

closed_loop = rollout_route_with_replanning(
    env,
    scene_id="outdoor-demo",
    initial_route=plan.selected,
    replan_candidate_batches=((RouteCandidate("detour", (waypoint, goal)),),),
)
```

For policy integration, compact the planning or rollout record into numeric features plus a scalar reward. This keeps agent-facing observations stable while preserving full debug records separately.

```python
sample = build_route_policy_sample(closed_loop)
agent_observation = sample.observation.features
agent_reward = sample.reward.reward
agent_terminal = sample.reward.terminal
```

Supported actions:

- `twist`: `linearX`, `linearY`, `linearZ` or `vx`, `vy`, `vz`
- `teleport`: absolute `x`, `y`, `z` plus optional `qx`, `qy`, `qz`, `qw`

The backend always blocks poses outside `SceneEnvironment.bounds`. When a `VoxelOccupancyGrid` is set, in-bounds collision checks also reject poses that fall into occupied voxels. When a `RobotFootprint` is set, the occupancy query checks every voxel touched by the circular body radius and height instead of only the pose point. `score_trajectory()` uses the same collision path and reports `collision-rate`, `collision-count`, clearance metrics, and per-reason notes.

## Next Implementation Layer

The next useful layer is Gymnasium-style environment adapters: wrap reset/step around policy feedback samples so learned agents can consume GS Mapper scenes directly.
