# Decisions

Updated: 2026-04-02T09:43:15.585Z

## Localization Alignment

### Accepted

- Stable code uses `gs_sim2real.core.localization_alignment.align_pose_samples()` as the only contract that production callers should depend on.
- `auto` keeps two stable behaviors instead of one universal algorithm: `index` for order-only logs and `timestamp` for timestamped logs.
- New alignment ideas must land in `src/gs_sim2real/experiments/` first and only graduate after they outperform current core behavior on shared fixtures.

### Deferred

- `timestamp_nearest` stays experimental. It is fast and simple, but it loses the middle frame on sparse trajectories where interpolation clearly wins.
- We are not freezing a larger abstract interface yet. The current core keeps only `PoseSample`, `AlignmentPair`, and `align_pose_samples(...)`.

### Operating Rules

1. Start with at least three concrete strategies for any new alignment problem.
2. Keep inputs and metrics identical across strategies before discussing architecture.
3. Promote only the minimum surface that survived comparison; delete or quarantine the rest.

## Render Backend Selection

### Accepted

- Stable code uses `select_render_backend(RenderBackendRequest(...))` as the only backend-selection contract.
- `balanced` is the default production policy because it keeps the existing capability gates and adds a preview-oriented latency escape hatch.
- Alternative policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a clear advantage.

### Deferred

- `simple_safe` stays experimental. It is easy to reason about, but it leaves benchmark quality on the table when gsplat is available.
- `fidelity_first` stays experimental. It improves offline quality, but it ignores interactive preview latency requirements.

### Operating Rules

1. Start backend-selection work with at least three policies, not one conditional chain.
2. Compare policies on the same runtime capabilities and workload preferences before changing production defaults.
3. Promote only the policy interface that multiple workloads can agree on.

## Localization Estimate Import

### Accepted

- Stable code uses `import_localization_estimate_document(LocalizationEstimateImportRequest(...))` as the importer contract.
- `suffix_aware` is the default production policy because it preserves normalized JSON, repairs commented exports, and still accepts line-oriented trajectories.
- Alternative importer policies stay in `src/gs_sim2real/experiments/` until they improve shared fixture quality without expanding the stable surface.

### Deferred

- `strict_content_gate` stays experimental. It is simple, but it misclassifies or rejects lightly malformed exports.
- `fallback_cascade` stays experimental. It recovers more often than strict parsing, but it still lacks file-hint and comment-repair behavior.

### Operating Rules

1. Start parser work with at least three importer policies, not one branching function.
2. Compare policies on the same raw documents and the same schema-match metrics before changing production defaults.
3. Promote only the smallest importer contract that multiple formats can share.

## Localization Review Bundle Import

### Accepted

- Stable web code uses `importLocalizationReviewBundleDocument(rawDocument, policy?)` as the only review-bundle import contract.
- `alias_friendly` is the default production policy because it preserves canonical exports while still recovering linked captures and wrapper-friendly portable run shapes.
- Alternative review-bundle import policies stay outside production until the same canonical, linked-fallback, and alias-wrapper fixtures show a better cross-share fit.

### Deferred

- `strict_canonical` stays experimental. It is simple, but it rejects portable review bundles that rely on linked captures instead of embedded bundles.
- `linked_capture_fallback` stays experimental. It handles shared captures, but it still rejects wrapper aliases used by portable review-bundle adapters.

### Operating Rules

1. Start review-bundle import work with at least three policies, not one growing file-import handler.
2. Compare policies on the same canonical, linked-fallback, and alias-wrapper fixtures before changing production defaults.
3. Promote only the smallest import surface that keeps the restored capture shelf and run shelf schema stable.

## Query Cancellation Policy

### Accepted

- Stable transport code uses `resolve_query_cancellation_decision(QueryCancellationRequest(...))` as the only cancellation surface.
- `cancel_source_backlog` is the production default because disconnected sources should not leave stale queued work behind.
- Alternative cancellation strategies stay experimental until the same timeout, disconnect, and shutdown fixtures show a better fit.

### Deferred

- `ignore_orphaned` stays experimental. It is simple, but it allows dead queues to accumulate.
- `cancel_requested_only` stays experimental. It improves single-request timeouts but still leaves same-source backlog behind.

### Operating Rules

1. Compare at least three cancellation strategies before changing timeout/disconnect cleanup in transport code.
2. Use the same timeout, disconnect, and shutdown fixtures for every policy.
3. Keep transport code dependent only on cancellation decisions, not on policy-specific branching.

## Query Coalescing Policy

### Accepted

- Stable queue code uses `resolve_query_coalescing_decision(QueryCoalescingRequest(...))` as the only dedupe/coalescing surface.
- `latest_render_per_source` is the production default because browser previews care about the latest render, not every intermediate one.
- Alternative coalescing strategies stay experimental until the same coexistence, duplicate, and replace fixtures show a better fit.

### Deferred

- `keep_all` stays experimental. It is simple, but it keeps obsolete preview renders around.
- `exact_render_drop_new` stays experimental. It dedupes exact duplicates, but it still keeps older same-source renders when the pose changed.

### Operating Rules

1. Compare at least three coalescing strategies before changing queue admission for interactive render requests.
2. Use the same coexistence, duplicate, and replace fixtures for every policy.
3. Keep queue stores dependent only on coalescing decisions, not on policy-specific branching.

## Query Error Mapping

### Accepted

- Stable transport code uses `resolve_query_error_mapping(QueryErrorMappingRequest(...))` as the only error-mapping surface.
- `structured_codes` is the production default because it keeps browser messages stable while preserving actionable queue-policy reasons.
- Alternative error-mapping strategies stay experimental until the same parse, queue, timeout, and shutdown fixtures show a better fit.

### Deferred

- `literal_passthrough` stays experimental. It preserves detail, but its messages drift with call sites and undermine comparability.
- `action_hint` stays experimental. It is friendly, but it changes canonical strings in ways that make transport regressions harder to diff.

### Operating Rules

1. Compare at least three error-mapping strategies before changing transport-visible failure messages.
2. Use the same parse, queue, timeout, and shutdown fixtures for every policy.
3. Keep websocket transport code dependent only on mapped error decisions, not on hand-written event strings.

## Query Transport Selection

### Accepted

- Stable code uses `select_query_transport(QueryTransportRequest(...))` as the only transport-selection contract.
- `balanced` is the default production policy because it can keep publish-only deployments quiet, choose ws for browser-first query mode, and still honor local CLI preferences.
- Alternative transport policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-workload fit.

### Deferred

- `explicit_only` stays experimental. It is predictable, but it ignores browser-vs-CLI workload intent in auto mode.
- `browser_first` stays experimental. It works well for web clients, but it over-selects ws when local CLI tooling would prefer zmq.

### Operating Rules

1. Start query transport work with at least three policies, not one conditional chain.
2. Compare policies on the same runtime capabilities and workload preferences before changing production defaults.
3. Promote only the smallest transport contract that both browser-first and local tooling flows can share.

## Query Request Import

### Accepted

- Stable code uses `import_query_request(QueryRequestImportRequest(...))` as the only query payload import contract.
- `alias_friendly` is the default production policy because it preserves the canonical schema while still accepting thin wrappers, alias keys, and pose shortcuts from browser and CLI clients.
- Alternative request import policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-client fit.

### Deferred

- `strict_schema` stays experimental. It is the fastest path, but it rejects lightweight clients that do not send the full canonical envelope.
- `envelope_first` stays experimental. It works for wrapped SDK payloads, but it still drops top-level pose shortcuts used by quick local tools.

### Operating Rules

1. Start query-request work with at least three import policies, not one monolithic parser.
2. Compare policies on the same render and benchmark payload fixtures before changing production defaults.
3. Promote only the smallest import contract that preserves the normalized render and benchmark request schema.

## Query Queue Policy

### Accepted

- Stable server code uses `admit_query_queue_item(...)` and `dispatch_query_queue_item(...)` as the only queue policy surface.
- `interactive_first` is the production default because it keeps render interactions responsive and evicts lower-priority background work under pressure.
- Alternative queue policies stay experimental until the same single-render, benchmark-then-render, and pressure fixtures show a better fit.

### Deferred

- `fifo_unbounded` stays experimental. It is simple, but it allows heavy benchmark work to dominate interactive queues.
- `bounded_fifo` stays experimental. It caps backlog growth, but it still rejects interactive requests even when only background work is queued.

### Operating Rules

1. Compare at least three queue policies before changing interactive transport backlog behavior.
2. Use the same single-render, mixed workload, and pressure fixtures for every candidate policy.
3. Keep websocket server code dependent only on queue decisions, not on policy-specific branching.

## Query Source Identity

### Accepted

- Stable transport code uses `resolve_query_source_identity(QuerySourceIdentityRequest(...))` as the only source-identity surface.
- `remote_observable` is the production default because it preserves per-connection uniqueness while staying debuggable.
- Alternative source-id strategies stay experimental until the same remote-address, client-hint, and endpoint fallback fixtures show a better fit.

### Deferred

- `serial_only` stays experimental. It is cheap, but it hides which client or endpoint owns a backlog.
- `endpoint_scoped` stays experimental. It improves debuggability, but it still cannot distinguish same-endpoint clients without a remote address.

### Operating Rules

1. Compare at least three source-identity policies before changing how queue source ids are assigned.
2. Use the same remote-address, client-hint, and endpoint-fallback fixtures for every policy.
3. Keep websocket queue code dependent only on source identities, not on policy-specific transport branching.

## Query Timeout Policy

### Accepted

- Stable server/client code uses `resolve_query_timeout_plan(QueryTimeoutPolicyRequest(...))` as the only timeout-policy surface.
- `workload_aware_retry` is the default production policy because it preserves normal render latency, scales benchmark deadlines with work size, and keeps a small retry budget for websocket renders.
- Alternative timeout strategies stay outside production until the same render-default, benchmark-workload, and bounded-explicit fixtures show a better fit.

### Deferred

- `fixed_deadline` stays experimental. It is simple, but it underfits long benchmark workloads and leaves websocket render retries unavailable.
- `hint_bounded` stays experimental. It respects hints, but it still trusts undersized benchmark deadlines too readily.

### Operating Rules

1. Compare at least three timeout policies before changing query deadlines in transport code.
2. Use the same render and benchmark fixtures for every candidate policy.
3. Keep transport loops dependent only on the resolved timeout plan, not on policy-specific branching.

## Query Response Build

### Accepted

- Stable server code uses `build_render_result_response_document(...)`, `build_query_ready_response_document(...)`, and `build_query_error_response_document(...)` as the only response-build surface.
- `browser_observable` is the production default because it preserves the current browser-facing query payload shape while keeping websocket/CLI clients explicit.
- Alternative response-build policies stay experimental until the same render-result, query-ready, and error fixtures show a better fit.

### Deferred

- `minimal_envelope` stays experimental. It is small and fast, but it drops render settings and request-catalog details the web client uses.
- `diagnostic_meta` stays experimental. It is a useful telemetry direction, but it expands the payload shape without a proven client need.

### Operating Rules

1. Compare at least three response-build policies before expanding server-side websocket payload helpers.
2. Use the same render-result, query-ready, and error fixtures for every candidate policy.
3. Keep production dependent only on the stable core builder functions, not on experiment-only policy details.

## Live Localization Stream Import

### Accepted

- Stable web code uses `importLiveLocalizationStreamMessage(previousEstimate, rawMessage, options, policy?)` as the only live-stream import contract.
- `alias_friendly` is the default production policy because it preserves the canonical websocket stream while still accepting wrapper aliases, top-level pose shortcuts, and message aliases from SDK and local tools.
- Alternative live-stream policies stay outside production until the same message fixtures show a better cross-client fit.

### Deferred

- `strict_canonical` stays experimental. It is readable, but it rejects cameraPose wrappers, top-level pose shortcuts, and clear aliases.
- `wrapped_pose` stays experimental. It works for pose wrappers, but it still drops top-level append shortcuts and clear aliases used by light clients.

### Operating Rules

1. Start live-stream import work with at least three policies, not one growing websocket handler.
2. Compare policies on the same canonical, wrapped, shortcut, and clear-alias message fixtures before changing production defaults.
3. Promote only the smallest import surface that keeps the normalized live estimate schema stable.

## Route Capture Bundle Import

### Accepted

- Stable code uses `import_route_capture_bundle(RouteCaptureBundleImportRequest(...))` as the only ground-truth bundle import contract.
- `route_aware` is the default production policy because it preserves canonical bundles while still recovering poses from `route` and `response.pose` when captures are incomplete.
- Alternative bundle import policies stay in `src/gs_sim2real/experiments/` until shared fixtures show a better cross-bundle fit.

### Deferred

- `strict_canonical` stays experimental. It is simple, but it rejects bundle variants that omit explicit capture poses.
- `response_pose_fallback` stays experimental. It recovers response poses, but it still ignores route-ground-truth bundles where `route` should define the capture pose.

### Operating Rules

1. Start ground-truth bundle work with at least three import policies, not one expanding normalizer.
2. Compare policies on the same canonical, response-fallback, and route-fallback fixtures before changing production defaults.
3. Promote only the smallest bundle import contract that preserves the benchmark-facing capture schema.

## Sim2Real Websocket Protocol

### Accepted

- Stable web code uses `importSim2realWebsocketMessage(rawMessage, policy?)` as the only websocket response import contract.
- `alias_friendly` is the default production policy because it preserves canonical server responses while still accepting wrapped envelopes, message aliases, and field aliases from adapters and thin tools.
- Alternative websocket protocol policies stay outside production until the same ready/render/benchmark/error fixtures show a better cross-client fit.

### Deferred

- `strict_canonical` stays experimental. It is simple, but it rejects nested envelopes that already appear in wrapper tooling.
- `envelope_first` stays experimental. It handles wrappers, but it still rejects message aliases and field aliases used by SDK-style ready payloads and thin error emitters.

### Operating Rules

1. Start websocket protocol work with at least three import policies, not one growing socket message handler.
2. Compare policies on the same ready, render-result, benchmark-report, and error fixtures before changing production defaults.
3. Promote only the smallest import surface that keeps the normalized websocket message schema stable for the panel.

