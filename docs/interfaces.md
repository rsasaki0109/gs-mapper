# Interfaces

Only the surfaces that survived comparison should be treated as stable.

## Localization Alignment

### Stable Core

The stable localization-alignment surface is intentionally small:

```python
@dataclass(frozen=True)
class PoseSample:
    index: int
    label: str
    position: tuple[float, float, float]
    yaw_degrees: float
    timestamp_seconds: float | None
    response: dict[str, Any] | None = None
    relative_timestamp_seconds: float | None = None

@dataclass(frozen=True)
class AlignmentPair:
    pair_index: int
    ground_truth: PoseSample
    estimate: PoseSample
    time_delta_seconds: float | None
    interpolation_kind: str

def align_pose_samples(
    ground_truth_poses: Sequence[PoseSample],
    estimate_poses: Sequence[PoseSample],
    *,
    alignment: str = 'auto',
) -> tuple[str, list[AlignmentPair]]: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `align(ground_truth_poses, estimate_poses) -> list[AlignmentPair]`

### Comparable Inputs

- Same `PoseSample` arrays for every strategy
- Same canonical fixtures (`ordered-index`, `reordered-timestamp`, `sparse-timestamp`)
- Same evaluation axes: quality, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable, minimal, dependency surface for production code
- `src/gs_sim2real/experiments/`: discardable strategies and comparison harnesses

## Render Backend Selection

### Stable Core

The stable render-backend selection surface is intentionally small:

```python
@dataclass(frozen=True)
class RenderBackendCapabilities:
    has_gaussian_splat: bool
    gsplat_available: bool
    cuda_available: bool

@dataclass(frozen=True)
class RenderBackendPreferences:
    prefer_low_startup_latency: bool = False
    prefer_visual_fidelity: bool = True

@dataclass(frozen=True)
class RenderBackendRequest:
    requested_backend: str
    capabilities: RenderBackendCapabilities
    preferences: RenderBackendPreferences = RenderBackendPreferences()

def select_render_backend(
    request: RenderBackendRequest,
    *,
    policy: str = 'balanced',
) -> RenderBackendSelection: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `select(request) -> RenderBackendSelection`

### Comparable Inputs

- Same `RenderBackendRequest` fixtures for every policy
- Same workload fixtures (`plain-point-cloud`, `interactive-preview`, `offline-benchmark`, `no-cuda-fallback`)
- Same evaluation axes: fit, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable backend-selection policy contract for production render code
- `src/gs_sim2real/experiments/`: discardable backend-selection policies and comparison harnesses

## Localization Estimate Import

### Stable Core

The stable localization-estimate importer surface is intentionally small:

```python
@dataclass(frozen=True)
class LocalizationEstimateImportRequest:
    raw_text: str
    file_name: str | None = None

def import_localization_estimate_document(
    request: LocalizationEstimateImportRequest,
    *,
    policy: str = 'suffix_aware',
) -> dict[str, Any]: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `import_document(request) -> normalized localization-estimate dict`

### Comparable Inputs

- Same raw document fixtures for every policy
- Same fixtures (`canonical-json`, `tum-text`, `commented-json`, `bracketed-text-log`)
- Same evaluation axes: schema preservation, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable importer contract for production evaluation code
- `src/gs_sim2real/experiments/`: discardable importer policies and comparison harnesses

## Localization Review Bundle Import

### Stable Core

The stable localization review bundle import surface is intentionally small:

```python
importLocalizationReviewBundleDocument(
    rawDocument,
    policy = 'alias_friendly',
) -> normalized review bundle import payload
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `importDocument(rawDocument) -> normalized review bundle import payload`

### Comparable Inputs

- Same review-bundle fixtures for every policy
- Same workload fixtures across canonical embedded bundles, linked-capture fallback, and alias-wrapped portable bundles
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `apps/dreamwalker-web/src/`: stable review-bundle import contract used by the panel
- `apps/dreamwalker-web/tools/`: discardable review-bundle comparison harnesses and report generators

## Query Cancellation Policy

### Stable Core

The stable query cancellation surface is intentionally small:

```python
resolve_query_cancellation_decision(
    QueryCancellationRequest(...),
    policy = 'cancel_source_backlog',
) -> QueryCancellationDecision
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `cancel(request) -> QueryCancellationDecision`

### Comparable Inputs

- Same timeout fixture for every policy
- Same disconnect fixture for every policy
- Same shutdown fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable cancellation-policy contract used by interactive queue stores
- `src/gs_sim2real/experiments/`: discardable cancellation-policy comparison harnesses and docs adapters

## Query Coalescing Policy

### Stable Core

The stable query coalescing surface is intentionally small:

```python
resolve_query_coalescing_decision(
    QueryCoalescingRequest(...),
    policy = 'latest_render_per_source',
) -> QueryCoalescingDecision
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `coalesce(request) -> QueryCoalescingDecision`

### Comparable Inputs

- Same benchmark/render coexistence fixture for every policy
- Same duplicate render fixture for every policy
- Same latest-render replacement fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable coalescing contract used by interactive queue stores
- `src/gs_sim2real/experiments/`: discardable coalescing comparison harnesses and docs adapters

## Query Error Mapping

### Stable Core

The stable query error-mapping surface is intentionally small:

```python
resolve_query_error_mapping(
    QueryErrorMappingRequest(...),
    policy = 'structured_codes',
) -> QueryErrorMappingDecision
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `map_error(request) -> QueryErrorMappingDecision`

### Comparable Inputs

- Same invalid JSON fixture for every policy
- Same queue reject and queue drop fixtures for every policy
- Same timeout and shutdown fixtures for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable error-mapping contract used by interactive transports
- `src/gs_sim2real/experiments/`: discardable error-mapping comparison harnesses and docs adapters

## Query Transport Selection

### Stable Core

The stable query transport surface is intentionally small:

```python
@dataclass(frozen=True)
class QueryTransportCapabilities:
    zmq_available: bool
    ws_available: bool

@dataclass(frozen=True)
class QueryTransportPreferences:
    enable_query_transport: bool = False
    prefer_browser_clients: bool = False
    prefer_local_cli: bool = False

@dataclass(frozen=True)
class QueryTransportRequest:
    requested_transport: str
    pose_source: str
    endpoint: str = ""
    capabilities: QueryTransportCapabilities = QueryTransportCapabilities(...)
    preferences: QueryTransportPreferences = QueryTransportPreferences()

def select_query_transport(
    request: QueryTransportRequest,
    *,
    policy: str = 'balanced',
) -> QueryTransportSelection: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `select(request) -> QueryTransportSelection`

### Comparable Inputs

- Same `QueryTransportRequest` fixtures for every policy
- Same workload fixtures (`publish-only-static`, `browser-query`, `local-cli-query`, `ws-missing`, `explicit-ws`)
- Same evaluation axes: fit, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable query transport contract for production render-server setup
- `src/gs_sim2real/experiments/`: discardable query transport policies and comparison harnesses

## Query Request Import

### Stable Core

The stable query request import surface is intentionally small:

```python
@dataclass(frozen=True)
class RenderQueryDefaults:
    width: int
    height: int
    fov_degrees: float
    near_clip: float
    far_clip: float
    point_radius: int
    timeout_ms: int = 10_000

@dataclass(frozen=True)
class QueryRequestImportRequest:
    payload: Any
    defaults: RenderQueryDefaults

def import_query_request(
    request: QueryRequestImportRequest,
    *,
    policy: str = 'alias_friendly',
) -> ImportedQueryRequest: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `import_request(request) -> ImportedQueryRequest`

### Comparable Inputs

- Same `QueryRequestImportRequest` fixtures for every policy
- Same workload fixtures across canonical render, wrapped render, pose shortcuts, canonical benchmark, and wrapped benchmark payloads
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable query request importer contract for production render-server handlers
- `src/gs_sim2real/experiments/`: discardable import policies and comparison harnesses

## Query Queue Policy

### Stable Core

The stable query queue surface is intentionally small:

```python
admit_query_queue_item(
    QueryQueueState(...),
    item,
    policy = 'interactive_first',
) -> QueryQueueAdmitDecision

dispatch_query_queue_item(
    QueryQueueState(...),
    policy = 'interactive_first',
) -> QueryQueueDispatchDecision
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `admit(state, item) -> QueryQueueAdmitDecision`
- `dispatch(state) -> QueryQueueDispatchDecision`

### Comparable Inputs

- Same single render fixture for every policy
- Same mixed benchmark/render fixture for every policy
- Same pressure fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable queue policy contract used by websocket transport code
- `src/gs_sim2real/experiments/`: discardable queue policy comparison harnesses and docs adapters

## Query Source Identity

### Stable Core

The stable query source-identity surface is intentionally small:

```python
resolve_query_source_identity(
    QuerySourceIdentityRequest(...),
    policy = 'remote_observable',
) -> QuerySourceIdentity
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `identify(request) -> QuerySourceIdentity`

### Comparable Inputs

- Same websocket remote-address fixture for every policy
- Same client-hint fallback fixture for every policy
- Same endpoint fallback fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable source-identity contract used by interactive transport code
- `src/gs_sim2real/experiments/`: discardable source-identity comparison harnesses and docs adapters

## Query Timeout Policy

### Stable Core

The stable query timeout surface is intentionally small:

```python
resolve_query_timeout_plan(
    QueryTimeoutPolicyRequest(...),
    policy = 'workload_aware_retry',
) -> QueryTimeoutPlan
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `resolve_timeout_plan(request) -> QueryTimeoutPlan`

### Comparable Inputs

- Same render-default fixture for every policy
- Same benchmark workload-floor fixture for every policy
- Same bounded explicit-hint fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable timeout-policy contract used by query transports and clients
- `src/gs_sim2real/experiments/`: discardable timeout-policy comparison harnesses and docs adapters

## Query Response Build

### Stable Core

The stable query response build surface is intentionally small:

```python
build_render_result_response_document(
    response_input,
    policy = 'browser_observable',
) -> dict[str, Any]

build_query_ready_response_document(
    response_input,
    policy = 'browser_observable',
) -> dict[str, Any]

build_query_error_response_document(
    response_input,
    policy = 'browser_observable',
) -> dict[str, Any]
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `build_render_result(response_input) -> dict[str, Any]`
- `build_query_ready(response_input) -> dict[str, Any]`
- `build_query_error(response_input) -> dict[str, Any]`

### Comparable Inputs

- Same render-result fixture for every policy
- Same query-ready handshake fixture for every policy
- Same transport error fixture for every policy
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable query-response build contract used by production servers
- `src/gs_sim2real/experiments/`: discardable response-build comparison harnesses and docs adapters

## Live Localization Stream Import

### Stable Core

The stable live localization stream import surface is intentionally small:

```python
importLiveLocalizationStreamMessage(
    previousEstimate,
    rawMessage,
    options = {},
    policy = 'alias_friendly',
) -> { kind, estimate }
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `importMessage(previousEstimate, rawMessage, options) -> { kind, estimate }`

### Comparable Inputs

- Same live-message fixtures for every policy
- Same workload fixtures across canonical append, snapshot, wrapper aliases, top-level shortcuts, and clear aliases
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `apps/dreamwalker-web/src/`: stable live-stream import contract used by the panel
- `apps/dreamwalker-web/tools/`: discardable policy comparison harnesses and report generators

## Route Capture Bundle Import

### Stable Core

The stable route-capture bundle import surface is intentionally small:

```python
@dataclass(frozen=True)
class RouteCaptureBundleImportRequest:
    input_like: Any

def import_route_capture_bundle(
    request: RouteCaptureBundleImportRequest,
    *,
    policy: str = 'route_aware',
) -> dict[str, Any]: ...
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `import_bundle(request) -> normalized route-capture-bundle dict`

### Comparable Inputs

- Same `RouteCaptureBundleImportRequest` fixtures for every policy
- Same bundle fixtures across canonical capture poses, response-pose fallback, and route-pose fallback
- Same evaluation axes: schema match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `src/gs_sim2real/core/`: stable route capture bundle import contract for production benchmarking
- `src/gs_sim2real/experiments/`: discardable bundle import policies and comparison harnesses

## Sim2Real Websocket Protocol

### Stable Core

The stable sim2real websocket import surface is intentionally small:

```python
importSim2realWebsocketMessage(
    rawMessage,
    policy = 'alias_friendly',
) -> normalized websocket message
```

### Experiment Contract

- `name`, `label`, `style`, `tier`, `capabilities`
- `importMessage(rawMessage) -> normalized websocket message`

### Comparable Inputs

- Same websocket fixtures for every policy
- Same workload fixtures across canonical ready, wrapped render-result, aliased ready, wrapped benchmark report, and aliased errors
- Same evaluation axes: shape match, runtime, readability heuristic, extensibility heuristic

### Boundary

- `apps/dreamwalker-web/src/`: stable websocket message import contract used by the panel
- `apps/dreamwalker-web/tools/`: discardable protocol comparison harnesses and report generators

