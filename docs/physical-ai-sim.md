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
    RoutePolicyEnvConfig,
    RoutePolicyGoalSpec,
    RoutePolicyGoalSuite,
    RoutePolicyGymAdapter,
    RoutePolicyImitationFitConfig,
    RoutePolicyBenchmarkRegressionThresholds,
    RoutePolicyMatrixConfigSpec,
    RoutePolicyMatrixGoalSuiteSpec,
    RoutePolicyMatrixRegistrySpec,
    RoutePolicyMatrixSceneSpec,
    RoutePolicyQualityThresholds,
    RoutePolicyRegistry,
    RoutePolicyRegistryEntry,
    RoutePolicyScenarioMatrix,
    RoutePolicyScenarioSet,
    RoutePolicyScenarioSpec,
    RoutePolicyScenarioCIReviewArtifact,
    RoutePolicyScenarioCIWorkflowConfig,
    RoutePolicyScenarioCIWorkflowActivationReport,
    RoutePolicyScenarioCIWorkflowPromotionReport,
    RoutePolicyScenarioCIWorkflowValidationReport,
    activate_route_policy_scenario_ci_workflow,
    build_route_policy_scenario_ci_manifest,
    build_route_policy_scenario_ci_review_artifact,
    build_route_policy_scenario_shard_plan,
    build_route_policy_benchmark_history,
    build_occupancy_grid_from_lidar_observation,
    build_route_policy_replay_batch,
    build_route_policy_replay_schema,
    build_route_policy_sample,
    collect_route_policy_dataset,
    evaluate_route_policy_baselines,
    evaluate_route_policy_dataset_quality,
    evaluate_route_policy_imitation_model,
    expand_route_policy_scenario_matrix_to_directory,
    fit_route_policy_imitation_model,
    load_route_policy_benchmark_history_json,
    load_route_policy_goal_suite_json,
    load_route_policy_imitation_model_json,
    load_route_policy_registry_json,
    load_route_policy_scenario_ci_manifest_json,
    load_route_policy_scenario_ci_review_json,
    load_route_policy_scenario_ci_workflow_activation_json,
    load_route_policy_scenario_ci_workflow_json,
    load_route_policy_scenario_ci_workflow_promotion_json,
    load_route_policy_scenario_ci_workflow_validation_json,
    load_route_policy_scenario_matrix_json,
    load_route_policy_scenario_set_json,
    load_route_policy_scenario_set_run_json,
    load_route_policy_scenario_shard_plan_json,
    iter_route_policy_replay_batches,
    load_route_policy_transitions_jsonl,
    materialize_route_policy_scenario_ci_workflow,
    merge_route_policy_scenario_shard_run_jsons,
    promote_route_policy_scenario_ci_workflow,
    replan_after_blocked_rollout,
    render_route_policy_benchmark_history_markdown,
    render_route_policy_benchmark_markdown,
    render_route_policy_quality_markdown,
    render_route_policy_scenario_ci_manifest_markdown,
    render_route_policy_scenario_ci_review_html,
    render_route_policy_scenario_ci_review_markdown,
    render_route_policy_scenario_ci_workflow_activation_markdown,
    render_route_policy_scenario_ci_workflow_markdown,
    render_route_policy_scenario_ci_workflow_promotion_markdown,
    render_route_policy_scenario_ci_workflow_validation_markdown,
    render_route_policy_scenario_matrix_markdown,
    render_route_policy_scenario_set_markdown,
    render_route_policy_scenario_shard_merge_markdown,
    render_route_policy_scenario_shard_plan_markdown,
    run_route_policy_imitation_benchmark,
    run_route_policy_scenario_set,
    validate_route_policy_scenario_ci_workflow,
    rollout_route,
    rollout_route_with_replanning,
    select_best_route,
    write_route_policy_benchmark_history_json,
    write_route_policy_benchmark_report_json,
    write_route_policy_dataset_json,
    write_route_policy_goal_suite_json,
    write_route_policy_imitation_model_json,
    write_route_policy_registry_json,
    write_route_policy_scenario_ci_manifest_json,
    write_route_policy_scenario_ci_review_bundle,
    write_route_policy_scenario_ci_review_json,
    write_route_policy_scenario_ci_workflow_activation_json,
    write_route_policy_scenario_ci_workflow_json,
    write_route_policy_scenario_ci_workflow_promotion_json,
    write_route_policy_scenario_ci_workflow_validation_json,
    write_route_policy_scenario_ci_workflow_yaml,
    write_route_policy_scenario_matrix_expansion_json,
    write_route_policy_scenario_matrix_json,
    write_route_policy_scenario_set_json,
    write_route_policy_scenario_set_run_json,
    write_route_policy_scenario_shard_merge_json,
    write_route_policy_scenario_shard_plan_json,
    write_route_policy_scenario_shards_from_expansion,
    write_route_policy_transitions_jsonl,
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

For learned policy loops, wrap the same route feedback with the Gymnasium-style adapter. It avoids a hard `gymnasium` dependency, but follows the same `reset()` and `step()` return shape.

```python
policy_env = RoutePolicyGymAdapter(
    env,
    RoutePolicyEnvConfig(scene_id="outdoor-demo", max_steps=64),
)

observation, info = policy_env.reset(seed=7, goal=goal)
observation, reward, terminated, truncated, info = policy_env.step(
    {"routeId": "agent-waypoint", "target": goal.to_dict()}
)
```

To build replay data for imitation learning or offline RL, collect adapter episodes into a stable dataset envelope. JSON preserves full episodes; JSONL flattens transitions for streaming training jobs.

```python
def direct_goal_policy(observation, info):
    return {"routeId": f"direct-{info['stepIndex']}", "target": info["goal"]}


dataset = collect_route_policy_dataset(
    (policy_env,),
    direct_goal_policy,
    episode_count=16,
    dataset_id="outdoor-demo-policy-rollouts",
    seed_start=100,
)
write_route_policy_dataset_json("runs/outdoor-policy-rollouts.json", dataset)
write_route_policy_transitions_jsonl("runs/outdoor-policy-transitions.jsonl", dataset)
```

Gate those rollouts before they enter training. The quality report keeps pass/fail checks and metrics separate from the raw replay data, so CI can reject weak datasets without changing the collector.

```python
quality = evaluate_route_policy_dataset_quality(
    dataset,
    thresholds=RoutePolicyQualityThresholds(
        min_success_rate=0.95,
        max_collision_rate=0.01,
        max_truncation_rate=0.05,
        min_scene_count=1,
        min_episode_count=16,
    ),
)
print(render_route_policy_quality_markdown(quality))
assert quality.passed
```

For simple policy comparisons, collect each named baseline against the same adapter, seed range, and goals. The returned evaluation keeps each policy dataset and its QA report isolated.

```python
baselines = evaluate_route_policy_baselines(
    (policy_env,),
    {
        "direct": direct_goal_policy,
        "agent": learned_policy,
    },
    episode_count=16,
    evaluation_id="outdoor-demo-baselines",
    seed_start=100,
)
print(baselines.best_policy_name)
```

For offline training, load JSONL transitions back into a flat table and pin a numeric feature schema. Missing columns are filled with `0.0`, which lets older replay files keep working while the schema evolves.

```python
transition_table = load_route_policy_transitions_jsonl("runs/outdoor-policy-transitions.jsonl")
schema = build_route_policy_replay_schema(
    transition_table,
    action_keys=("payload.target.position.0", "payload.target.position.1", "payload.target.position.2"),
)

for batch in iter_route_policy_replay_batches(transition_table, batch_size=32, schema=schema, shuffle=True, seed=7):
    trainer.step(
        observations=batch.observation_matrix,
        actions=batch.action_matrix,
        rewards=batch.reward_vector,
        dones=batch.done_vector,
    )

full_batch = build_route_policy_replay_batch(transition_table, schema=schema)
```

For a dependency-free reference baseline, fit the built-in k-nearest-neighbor imitation model from the same batch. It decodes the predicted action vector back into an adapter-compatible route action and can be evaluated through the same baseline gate as hand-written policies.

```python
imitation_model = fit_route_policy_imitation_model(
    full_batch,
    config=RoutePolicyImitationFitConfig(neighbor_count=3),
    metadata={"source": "outdoor-demo-policy-rollouts"},
)

imitation_eval = evaluate_route_policy_imitation_model(
    (policy_env,),
    imitation_model,
    episode_count=16,
    evaluation_id="outdoor-demo-imitation",
    seed_start=100,
    thresholds=RoutePolicyQualityThresholds(min_success_rate=0.8),
)
print(imitation_eval.best_policy_name)
```

Persist fitted imitation models when the benchmark needs to run in a separate job from replay collection. The saved model contains the replay schema and training vectors, so it can be loaded without the original JSONL file.

```python
write_route_policy_imitation_model_json("runs/outdoor-imitation-model.json", imitation_model)
loaded_model = load_route_policy_imitation_model_json("runs/outdoor-imitation-model.json")

benchmark_report = run_route_policy_imitation_benchmark(
    (policy_env,),
    loaded_model,
    episode_count=16,
    benchmark_id="outdoor-demo-policy-benchmark",
    include_direct_baseline=True,
    seed_start=100,
)
write_route_policy_benchmark_report_json("runs/outdoor-policy-benchmark.json", benchmark_report)
print(render_route_policy_benchmark_markdown(benchmark_report))
```

The same flow is available from the CLI:

```bash
gs-mapper route-policy-benchmark \
  --transitions-jsonl runs/outdoor-policy-transitions.jsonl \
  --scene-catalog docs/scenes-list.json \
  --scene-id outdoor-demo \
  --action-keys payload.target.position.0 payload.target.position.1 payload.target.position.2 \
  --episode-count 16 \
  --include-direct-baseline \
  --model-output runs/outdoor-imitation-model.json \
  --output runs/outdoor-policy-benchmark.json \
  --markdown-output runs/outdoor-policy-benchmark.md
```

For repeatable comparisons, move fixed goals and saved policy paths into versioned JSON artifacts. Registry model paths are resolved relative to the registry file, so the bundle can move as one directory.

```python
goal_suite = RoutePolicyGoalSuite(
    suite_id="outdoor-demo-fixed-goals",
    scene_id="outdoor-demo",
    goals=(
        RoutePolicyGoalSpec("near", (0.25, 0.0, 0.0)),
        RoutePolicyGoalSpec("far", (0.5, 0.0, 0.0)),
    ),
)
write_route_policy_goal_suite_json("runs/outdoor-goals.json", goal_suite)

registry = RoutePolicyRegistry(
    registry_id="outdoor-demo-policies",
    policies=(
        RoutePolicyRegistryEntry(policy_name="direct", policy_type="direct-goal"),
        RoutePolicyRegistryEntry(
            policy_name="imitation-k3",
            policy_type="imitation-model",
            model_path="outdoor-imitation-model.json",
        ),
    ),
)
write_route_policy_registry_json("runs/outdoor-policies.json", registry)
```

```bash
gs-mapper route-policy-benchmark \
  --policy-registry runs/outdoor-policies.json \
  --goal-suite runs/outdoor-goals.json \
  --scene-catalog docs/scenes-list.json \
  --episode-count 16 \
  --output runs/outdoor-policy-registry-benchmark.json \
  --markdown-output runs/outdoor-policy-registry-benchmark.md
```

Once single-run reports are stable, collect them into a history artifact. The history keeps only compact per-policy metric snapshots, so CI jobs can compare commits or datasets without rebuilding rollout datasets.

```python
history = build_route_policy_benchmark_history(
    (
        "runs/commit-a/outdoor-policy-registry-benchmark.json",
        "runs/commit-b/outdoor-policy-registry-benchmark.json",
    ),
    baseline_report="runs/baseline/outdoor-policy-registry-benchmark.json",
    history_id="outdoor-demo-policy-history",
    thresholds=RoutePolicyBenchmarkRegressionThresholds(
        max_success_rate_drop=0.05,
        max_collision_rate_increase=0.01,
        max_truncation_rate_increase=0.02,
        max_mean_reward_drop=0.25,
    ),
)
write_route_policy_benchmark_history_json("runs/outdoor-policy-history.json", history)
print(render_route_policy_benchmark_history_markdown(history))

loaded_history = load_route_policy_benchmark_history_json("runs/outdoor-policy-history.json")
assert loaded_history.passed
```

The same regression gate is available from the CLI. Add `--fail-on-regression` in CI to return exit status 2 when a current report falls outside the blessed baseline envelope.

```bash
gs-mapper route-policy-benchmark-history \
  --report runs/commit-a/outdoor-policy-registry-benchmark.json \
  --report runs/commit-b/outdoor-policy-registry-benchmark.json \
  --baseline-report runs/baseline/outdoor-policy-registry-benchmark.json \
  --history-id outdoor-demo-policy-history \
  --max-success-rate-drop 0.05 \
  --max-collision-rate-increase 0.01 \
  --max-truncation-rate-increase 0.02 \
  --max-mean-reward-drop 0.25 \
  --output runs/outdoor-policy-history.json \
  --markdown-output runs/outdoor-policy-history.md \
  --fail-on-regression
```

To scale beyond one scene, define a scenario set. Each scenario pins a scene catalog, optional scene id, optional goal suite, and optional simulator/evaluation overrides. The runner executes the same policy registry for every scenario, writes one benchmark report per scenario, and sends those reports into the history gate. Relative scenario paths and the embedded `policyRegistryPath` resolve from the scenario-set JSON directory.

```python
scenario_set = RoutePolicyScenarioSet(
    scenario_set_id="outdoor-demo-scenarios",
    policy_registry_path="outdoor-policies.json",
    episode_count=16,
    seed_start=100,
    max_steps=64,
    scenarios=(
        RoutePolicyScenarioSpec(
            scenario_id="short-goals",
            scene_catalog="../docs/scenes-list.json",
            scene_id="outdoor-demo",
            goal_suite_path="outdoor-short-goals.json",
        ),
        RoutePolicyScenarioSpec(
            scenario_id="long-goals",
            scene_catalog="../docs/scenes-list.json",
            scene_id="outdoor-demo",
            goal_suite_path="outdoor-long-goals.json",
            max_steps=96,
        ),
    ),
)
write_route_policy_scenario_set_json("runs/scenarios/outdoor-scenarios.json", scenario_set)

loaded_scenarios = load_route_policy_scenario_set_json("runs/scenarios/outdoor-scenarios.json")
registry = load_route_policy_registry_json("runs/scenarios/outdoor-policies.json")
scenario_report = run_route_policy_scenario_set(
    loaded_scenarios,
    registry,
    report_dir="runs/scenarios/reports",
    scenario_set_base_path="runs/scenarios",
    registry_base_path="runs/scenarios",
    policy_registry_path="runs/scenarios/outdoor-policies.json",
    history_output="runs/scenarios/history.json",
)
write_route_policy_scenario_set_run_json("runs/scenarios/scenario-run.json", scenario_report)
print(render_route_policy_scenario_set_markdown(scenario_report))

loaded_run = load_route_policy_scenario_set_run_json("runs/scenarios/scenario-run.json")
assert loaded_run.passed
```

The CLI keeps the same artifact boundaries:

```bash
gs-mapper route-policy-scenario-set \
  --scenario-set runs/scenarios/outdoor-scenarios.json \
  --report-dir runs/scenarios/reports \
  --output runs/scenarios/scenario-run.json \
  --markdown-output runs/scenarios/scenario-run.md \
  --history-output runs/scenarios/history.json \
  --history-markdown-output runs/scenarios/history.md \
  --baseline-report runs/baseline/outdoor-policy-registry-benchmark.json \
  --max-success-rate-drop 0.05 \
  --max-collision-rate-increase 0.01 \
  --max-truncation-rate-increase 0.02 \
  --fail-on-regression
```

For broader coverage, create a scenario matrix and expand it into one scenario-set JSON per policy registry. The matrix is a Cartesian product over scene catalogs, goal suites, and simulator/evaluation configs, while keeping policy registries as separate scenario-set artifacts. Matrix paths are authored relative to the matrix JSON directory; generated scenario-set files are rewritten so their paths resolve from the generated file directory.

```python
matrix = RoutePolicyScenarioMatrix(
    matrix_id="outdoor-demo-matrix",
    registries=(
        RoutePolicyMatrixRegistrySpec("direct", "outdoor-policies-direct.json"),
        RoutePolicyMatrixRegistrySpec("imitation", "outdoor-policies-imitation.json"),
    ),
    scenes=(
        RoutePolicyMatrixSceneSpec(
            "outdoor-demo",
            "../docs/scenes-list.json",
            scene_id="outdoor-demo",
        ),
    ),
    goal_suites=(
        RoutePolicyMatrixGoalSuiteSpec("short", "outdoor-short-goals.json"),
        RoutePolicyMatrixGoalSuiteSpec("long", "outdoor-long-goals.json"),
    ),
    configs=(
        RoutePolicyMatrixConfigSpec("fast", episode_count=8, seed_start=100, max_steps=64),
        RoutePolicyMatrixConfigSpec("stress", episode_count=32, seed_start=1000, max_steps=96),
    ),
    episode_count=16,
    seed_start=100,
)
write_route_policy_scenario_matrix_json("runs/scenarios/outdoor-matrix.json", matrix)

loaded_matrix = load_route_policy_scenario_matrix_json("runs/scenarios/outdoor-matrix.json")
expansion = expand_route_policy_scenario_matrix_to_directory(
    loaded_matrix,
    "runs/scenarios/generated",
    matrix_base_path="runs/scenarios",
)
write_route_policy_scenario_matrix_expansion_json("runs/scenarios/matrix-expansion.json", expansion)
print(render_route_policy_scenario_matrix_markdown(expansion))
```

```bash
gs-mapper route-policy-scenario-matrix \
  --matrix runs/scenarios/outdoor-matrix.json \
  --output-dir runs/scenarios/generated \
  --index-output runs/scenarios/matrix-expansion.json \
  --markdown-output runs/scenarios/matrix-expansion.md
```

For CI-sized execution, split the generated scenario sets into shard scenario-set files. Each shard is still a normal `RoutePolicyScenarioSet`, so CI jobs can run shards with the existing `route-policy-scenario-set` command. The final merge step reads the shard run JSON files, collects every per-scenario benchmark report, and rebuilds one global history gate.

```python
shard_plan = write_route_policy_scenario_shards_from_expansion(
    expansion,
    "runs/scenarios/shards",
    max_scenarios_per_shard=4,
    shard_plan_id="outdoor-demo-shards",
)
write_route_policy_scenario_shard_plan_json("runs/scenarios/shard-plan.json", shard_plan)
print(render_route_policy_scenario_shard_plan_markdown(shard_plan))

# Run each shard independently with run_route_policy_scenario_set(...) or the CLI below.
# Then merge all shard run JSON files:
merge = merge_route_policy_scenario_shard_run_jsons(
    (
        "runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json",
        "runs/scenarios/shard-runs/outdoor-matrix-direct-shard-002.json",
    ),
    merge_id="outdoor-demo-shard-merge",
    history_output="runs/scenarios/shard-history.json",
    history_markdown_output="runs/scenarios/shard-history.md",
)
write_route_policy_scenario_shard_merge_json("runs/scenarios/shard-merge.json", merge)
print(render_route_policy_scenario_shard_merge_markdown(merge))
```

```bash
gs-mapper route-policy-scenario-shards \
  --expansion runs/scenarios/matrix-expansion.json \
  --max-scenarios-per-shard 4 \
  --shard-plan-id outdoor-demo-shards \
  --output-dir runs/scenarios/shards \
  --index-output runs/scenarios/shard-plan.json \
  --markdown-output runs/scenarios/shard-plan.md

gs-mapper route-policy-scenario-set \
  --scenario-set runs/scenarios/shards/outdoor-matrix-direct-shard-001.json \
  --report-dir runs/scenarios/shard-reports/001 \
  --output runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json \
  --history-output runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001-history.json

gs-mapper route-policy-scenario-shard-merge \
  --run runs/scenarios/shard-runs/outdoor-matrix-direct-shard-001.json \
  --run runs/scenarios/shard-runs/outdoor-matrix-direct-shard-002.json \
  --merge-id outdoor-demo-shard-merge \
  --history-output runs/scenarios/shard-history.json \
  --history-markdown-output runs/scenarios/shard-history.md \
  --output runs/scenarios/shard-merge.json \
  --markdown-output runs/scenarios/shard-merge.md \
  --fail-on-regression
```

To make CI matrix jobs stable, generate a manifest from the shard plan. The manifest includes a `matrix.include` list for shard jobs, expected report paths, cache keys, generated CLI commands, and the merge job dependencies.

```python
ci_manifest = build_route_policy_scenario_ci_manifest(
    shard_plan,
    manifest_id="outdoor-demo-ci",
    report_dir="runs/scenarios/ci/reports",
    run_output_dir="runs/scenarios/ci/runs",
    history_output_dir="runs/scenarios/ci/histories",
    merge_id="outdoor-demo-shard-merge",
    merge_output="runs/scenarios/ci/shard-merge.json",
    merge_history_output="runs/scenarios/ci/shard-history.json",
    cache_key_prefix="outdoor-demo-policy",
    fail_on_regression=True,
)
write_route_policy_scenario_ci_manifest_json("runs/scenarios/ci-manifest.json", ci_manifest)
print(render_route_policy_scenario_ci_manifest_markdown(ci_manifest))
```

```bash
gs-mapper route-policy-scenario-ci-manifest \
  --shard-plan runs/scenarios/shard-plan.json \
  --manifest-id outdoor-demo-ci \
  --report-dir runs/scenarios/ci/reports \
  --run-output-dir runs/scenarios/ci/runs \
  --history-output-dir runs/scenarios/ci/histories \
  --merge-id outdoor-demo-shard-merge \
  --merge-output runs/scenarios/ci/shard-merge.json \
  --merge-history-output runs/scenarios/ci/shard-history.json \
  --cache-key-prefix outdoor-demo-policy \
  --fail-on-regression \
  --output runs/scenarios/ci-manifest.json \
  --markdown-output runs/scenarios/ci-manifest.md
```

The manifest can also materialize a GitHub Actions workflow. By default the workflow is manual-only (`workflow_dispatch`) so generated YAML can be reviewed before enabling push or pull-request triggers.

```python
workflow = materialize_route_policy_scenario_ci_workflow(
    ci_manifest,
    config=RoutePolicyScenarioCIWorkflowConfig(
        workflow_id="outdoor-demo-policy-shards",
        workflow_name="Outdoor Demo Policy Shards",
        artifact_root="runs/scenarios/ci",
        push_branches=("main",),
        pull_request_branches=("main",),
    ),
)
write_route_policy_scenario_ci_workflow_yaml(
    ".github/workflows/outdoor-demo-policy-shards.generated.yml",
    workflow,
)
write_route_policy_scenario_ci_workflow_json("runs/scenarios/ci-workflow.json", workflow)
print(render_route_policy_scenario_ci_workflow_markdown(workflow))
```

```bash
gs-mapper route-policy-scenario-ci-workflow \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-id outdoor-demo-policy-shards \
  --workflow-name "Outdoor Demo Policy Shards" \
  --artifact-root runs/scenarios/ci \
  --push-branch main \
  --pull-request-branch main \
  --workflow-output .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --index-output runs/scenarios/ci-workflow.json \
  --markdown-output runs/scenarios/ci-workflow.md
```

Validate the generated YAML before enabling the workflow. The validator parses the YAML, checks materialization metadata, verifies shard matrix entries and commands against the manifest, and writes a JSON report that can be used as a review gate.

```python
validation = validate_route_policy_scenario_ci_workflow(
    ci_manifest,
    workflow,
    workflow_path=".github/workflows/outdoor-demo-policy-shards.generated.yml",
)
write_route_policy_scenario_ci_workflow_validation_json(
    "runs/scenarios/ci-workflow-validation.json",
    validation,
)
print(render_route_policy_scenario_ci_workflow_validation_markdown(validation))
```

```bash
gs-mapper route-policy-scenario-ci-workflow-validate \
  --manifest runs/scenarios/ci-manifest.json \
  --workflow-index runs/scenarios/ci-workflow.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --output runs/scenarios/ci-workflow-validation.json \
  --markdown-output runs/scenarios/ci-workflow-validation.md \
  --fail-on-validation
```

After validation passes, activate the workflow into a real GitHub Actions path. Activation refuses to write unless the validation report passed, the validated source path matches the source file being activated, the source YAML matches the materialization index, and the destination is under `.github/workflows/`.

```python
activation = activate_route_policy_scenario_ci_workflow(
    workflow,
    validation,
    source_workflow_path=".github/workflows/outdoor-demo-policy-shards.generated.yml",
    active_workflow_path=".github/workflows/outdoor-demo-policy-shards.yml",
)
write_route_policy_scenario_ci_workflow_activation_json(
    "runs/scenarios/ci-workflow-activation.json",
    activation,
)
print(render_route_policy_scenario_ci_workflow_activation_markdown(activation))
```

```bash
gs-mapper route-policy-scenario-ci-workflow-activate \
  --workflow-index runs/scenarios/ci-workflow.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --workflow .github/workflows/outdoor-demo-policy-shards.generated.yml \
  --active-workflow-output .github/workflows/outdoor-demo-policy-shards.yml \
  --output runs/scenarios/ci-workflow-activation.json \
  --markdown-output runs/scenarios/ci-workflow-activation.md \
  --fail-on-activation
```

Publish a review bundle for GitHub Pages once shard runs have been merged. The bundle contains `review.json`, `review.md`, and an `index.html` page that summarizes shard status, workflow validation, workflow activation, and the active workflow path.

```python
review = build_route_policy_scenario_ci_review_artifact(
    shard_merge,
    validation,
    activation,
    review_id="outdoor-demo-policy-review",
    pages_base_url="https://example.github.io/gs-mapper/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_review_bundle(
    "docs/reviews/outdoor-demo-policy",
    review,
)
print(render_route_policy_scenario_ci_review_markdown(review))
```

```bash
gs-mapper route-policy-scenario-ci-review \
  --shard-merge runs/scenarios/ci/shard-merge.json \
  --validation-report runs/scenarios/ci-workflow-validation.json \
  --activation-report runs/scenarios/ci-workflow-activation.json \
  --review-id outdoor-demo-policy-review \
  --pages-base-url https://example.github.io/gs-mapper/reviews/outdoor-demo-policy/ \
  --bundle-dir docs/reviews/outdoor-demo-policy \
  --fail-on-review
```

Before widening from manual dispatch to repository triggers, write a promotion report. The gate requires the review to pass, validation and activation to be green, shard merge and history checks to pass, the active workflow path to stay under `.github/workflows/`, a published review URL, and literal branch names for the requested trigger mode.

```python
promotion = promote_route_policy_scenario_ci_workflow(
    review,
    trigger_mode="pull-request",
    pull_request_branches=("main",),
    review_url="https://example.github.io/gs-mapper/reviews/outdoor-demo-policy/",
)
write_route_policy_scenario_ci_workflow_promotion_json(
    "runs/scenarios/ci-workflow-promotion.json",
    promotion,
)
print(render_route_policy_scenario_ci_workflow_promotion_markdown(promotion))
```

```bash
gs-mapper route-policy-scenario-ci-workflow-promote \
  --review runs/scenarios/ci-review.json \
  --review-url https://example.github.io/gs-mapper/reviews/outdoor-demo-policy/ \
  --trigger-mode pull-request \
  --pull-request-branch main \
  --output runs/scenarios/ci-workflow-promotion.json \
  --markdown-output runs/scenarios/ci-workflow-promotion.md \
  --fail-on-promotion
```

After the promotion report passes, the adoption step re-materializes the scenario CI manifest with the promoted trigger mode / branches, re-runs validation and activation against a **separate** active workflow path, and records a per-gate adoption report. The manual-only active workflow file is never overwritten — adoption always writes a parallel YAML so the two files can be diffed side by side before flipping the repo over to the trigger-enabled workflow.

```python
from gs_sim2real.sim import (
    adopt_route_policy_scenario_ci_workflow,
    write_route_policy_scenario_ci_workflow_adoption_json,
)

adoption = adopt_route_policy_scenario_ci_workflow(
    promotion,
    manifest,
    materialization,  # the manual-only materialization that produced the activated YAML
    adopted_source_workflow_path="runs/scenarios/ci-workflow-adopted.generated.yml",
    adopted_active_workflow_path=".github/workflows/outdoor-demo-policy-shards-adopted.yml",
)
write_route_policy_scenario_ci_workflow_adoption_json(
    "runs/scenarios/ci-workflow-adoption.json",
    adoption,
)
```

Adoption gates:

- `promotion-promoted`: only adopt from a PROMOTED report.
- `manifest-id` / `workflow-id`: the adoption must target the same manifest and workflow the promotion report was built from.
- `adopted-path-distinct-from-manual`: the adopted active workflow path must differ from the promoted `active_workflow_path`; collisions fail pre-materialization so the manual YAML is never touched.
- `adopted-source-path-distinct`: the staged YAML source path must also be separate from the manual active path.
- `workflow-dispatch-retained`: adopted YAML keeps `workflow_dispatch` so on-demand reruns still work.
- `push-trigger-emitted` / `pull-request-trigger-emitted`: required trigger blocks appear when the promoted `trigger_mode` demands them.
- Per-branch gates (`push-branch:<name>` / `pull-request-branch:<name>`): each promoted branch literally appears in the adopted YAML.
- `adopted-validation-passed` / `adopted-activation-active`: the re-run validation and activation reports must themselves pass.

A minimal end-to-end recipe that walks matrix expansion all the way through adoption lives at `scripts/smoke_route_policy_scenario_ci.py`.

Supported actions:

- `twist`: `linearX`, `linearY`, `linearZ` or `vx`, `vy`, `vz`
- `teleport`: absolute `x`, `y`, `z` plus optional `qx`, `qy`, `qz`, `qw`

The backend always blocks poses outside `SceneEnvironment.bounds`. When a `VoxelOccupancyGrid` is set, in-bounds collision checks also reject poses that fall into occupied voxels. When a `RobotFootprint` is set, the occupancy query checks every voxel touched by the circular body radius and height instead of only the pose point. `score_trajectory()` uses the same collision path and reports `collision-rate`, `collision-count`, clearance metrics, and per-reason notes.

## Next Implementation Layer

The scenario CI chain from matrix expansion through promotion-backed adoption is now covered by `scripts/smoke_route_policy_scenario_ci.py`. The next useful layer is surfacing the adoption gate via a dedicated CLI command (today the library API is driven from Python or the smoke script) and wiring the adopted workflow path into the Pages review bundle so reviewers can see both the manual and trigger-enabled YAMLs without checking out the branch.
