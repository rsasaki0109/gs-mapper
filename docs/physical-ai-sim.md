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

`rgb-forward` is ready through the existing splat viewers. `depth-proxy` and `lidar-ray-proxy` are contract-only placeholders so agents, robotics bridges, and future render backends can agree on names and payload shapes before the ray-query implementation lands.

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

`HeadlessPhysicalAIEnvironment` is the first executable backend. It does not render pixels or simulate dynamics yet; it gives agents a deterministic bounds-based environment that can reset scenes, sample goals, apply simple actions, reject out-of-bounds poses, and return metadata-only observations.

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

Supported actions:

- `twist`: `linearX`, `linearY`, `linearZ` or `vx`, `vy`, `vz`
- `teleport`: absolute `x`, `y`, `z` plus optional `qx`, `qy`, `qz`, `qw`

The backend blocks poses outside `SceneEnvironment.bounds`. This is not a replacement for collision geometry; it is the minimal runtime needed to let agent code exercise the same `reset/step/render/query/score` loop before renderer-backed RGB, depth, and ray queries are added.

## Next Implementation Layer

The next useful layer is renderer-backed observation. `rgb-forward` should move from metadata-only responses to real offscreen splat renders, then `depth-proxy` and `lidar-ray-proxy` can be backed by renderer depth buffers or splat ray marching.
