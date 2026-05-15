"""End-to-end integration tests for live policy trace emission (Sprint 3 / PR C6).

PR C → C5 wired ``RoutePolicyTraceEmitter`` into four rollout paths:

- Gym adapter (PR C2/C3) — episode-level emission on ``step`` / ``reset``
- baseline benchmark runner (PR C3) — per-policy emitter factory
- direct ``rollout_route`` (PR C4) — per-segment ``record_step``
- closed-loop replanning (PR C5) — per-rollout episode pass-through

Each path has its own unit-level coverage; this module verifies the four
paths *compose* — that a single emitter / single JSONL sink can absorb
events from any combination of them without losing the episode-state
contract, that the on-disk JSONL is well-formed across all paths, and
that the downstream ``CorrelationEventWindow`` conversion stays valid
end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.robotics import (
    CORRELATION_EVENT_WINDOWS_VERSION,
    load_correlation_event_windows_json,
)
from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    JsonlPolicyTraceEventStream,
    Pose3D,
    RouteCandidate,
    RoutePolicyEnvConfig,
    RoutePolicyGymAdapter,
    RoutePolicyTraceEmitter,
    RouteRewardWeights,
    build_simulation_catalog,
    convert_policy_trace_events_to_event_windows,
    load_policy_trace_jsonl,
    rollout_route,
    rollout_route_with_replanning,
    write_correlation_event_windows_json,
)


def test_single_emitter_aggregates_gym_and_direct_rollout_route(tmp_path: Path) -> None:
    """One emitter / one JSONL sink should accept events from both paths."""

    sink = tmp_path / "trace.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    with JsonlPolicyTraceEventStream(sink) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)

        # Path 1: Gym adapter rollout — emitter opens "unit-scene-episode-0".
        adapter = RoutePolicyGymAdapter(
            env,
            RoutePolicyEnvConfig(
                scene_id="unit-scene",
                reward_weights=RouteRewardWeights(
                    distance_penalty_per_meter=0.0,
                    step_penalty=0.0,
                ),
            ),
            trace_emitter=emitter,
        )
        adapter.reset(goal=_unit_pose((0.25, 0.0, 0.0)))
        adapter.step((0.25, 0.0, 0.0))

        # Path 2: direct rollout_route on the same emitter, distinct episode index.
        env.reset("unit-scene")
        start = env.state.pose
        goal = _pose_like(start, position=(0.25, 0.0, 0.0))
        rollout_route(
            env,
            RouteCandidate("direct-route", (start, goal)),
            action_type="teleport",
            trace_emitter=emitter,
            trace_scene_id="direct-scene",
            trace_episode_index=1,
        )

    events = load_policy_trace_jsonl(sink)
    assert len(events) == 2, "one terminal event from each path"
    assert events[0].episode_id == "unit-scene-episode-0"
    assert events[1].episode_id == "direct-scene-episode-1"
    assert events[0].event_name == "goal_reached"
    assert events[1].event_name == "goal_reached"
    # Episode ids stay unique across paths so downstream consumers can
    # group events by episode without collision.
    assert {event.episode_id for event in events} == {
        "unit-scene-episode-0",
        "direct-scene-episode-1",
    }


def test_closed_loop_jsonl_is_chronological_with_unique_episode_ids(
    tmp_path: Path,
) -> None:
    """Closed-loop replanning must persist (collision, goal_reached) in order."""

    sink = tmp_path / "trace.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    with JsonlPolicyTraceEventStream(sink) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        closed_loop = rollout_route_with_replanning(
            env,
            scene_id="unit-scene",
            initial_route=RouteCandidate("initial-blocked", (start, outside)),
            replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
            action_type="teleport",
            trace_emitter=emitter,
            trace_scene_id="closed-loop-scene",
        )

    assert closed_loop.passed is True
    events = load_policy_trace_jsonl(sink)
    assert [event.event_name for event in events] == ["collision", "goal_reached"]
    assert [event.episode_id for event in events] == [
        "closed-loop-scene-episode-0",
        "closed-loop-scene-episode-1",
    ]
    # Both events share the trace_scene_id tag so downstream filtering by
    # scene picks up the whole closed-loop episode chain.
    for event in events:
        assert "scene:closed-loop-scene" in event.tags


def test_live_emitted_jsonl_converts_to_correlation_event_windows(
    tmp_path: Path,
) -> None:
    """Live JSONL must round-trip into the event-aligned correlation pipeline."""

    sink = tmp_path / "trace.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))

    with JsonlPolicyTraceEventStream(sink) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        rollout_route_with_replanning(
            env,
            scene_id="unit-scene",
            initial_route=RouteCandidate("initial-blocked", (start, outside)),
            replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
            action_type="teleport",
            trace_emitter=emitter,
            trace_scene_id="closed-loop-scene",
        )

    events = load_policy_trace_jsonl(sink)
    windows = convert_policy_trace_events_to_event_windows(
        events,
        half_width_seconds=0.25,
    )
    assert len(windows) == len(events)
    for window in windows:
        assert window.source == "policy_trace"
        assert "policy-trace" in window.tags
        assert window.end_time - window.start_time == 0.5

    windows_path = tmp_path / "windows.json"
    write_correlation_event_windows_json(windows_path, windows)
    payload = windows_path.read_text(encoding="utf-8")
    assert CORRELATION_EVENT_WINDOWS_VERSION in payload
    # The on-disk windows JSON loads back into matching CorrelationEventWindow
    # records so the gate that consumes this file sees consistent data.
    reloaded = load_correlation_event_windows_json(windows_path)
    assert reloaded == windows


def test_append_mode_jsonl_combines_traces_from_multiple_emitter_runs(
    tmp_path: Path,
) -> None:
    """Re-opening the JSONL sink in append mode preserves prior emitter output."""

    sink = tmp_path / "trace.jsonl"
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())

    # Run 1: closed-loop replanning, episodes 0 and 1.
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    goal = _pose_like(start, position=(0.5, 0.0, 0.0))
    with JsonlPolicyTraceEventStream(sink) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        rollout_route_with_replanning(
            env,
            scene_id="unit-scene",
            initial_route=RouteCandidate("initial-blocked", (start, outside)),
            replan_candidate_batches=((RouteCandidate("recovery", (goal,)),),),
            action_type="teleport",
            trace_emitter=emitter,
            trace_scene_id="run-1",
        )

    # Run 2: direct rollout_route, episode 5 (caller picks a higher offset
    # so the two runs together describe a single ordered timeline).
    env.reset("unit-scene")
    start_2 = env.state.pose
    goal_2 = _pose_like(start_2, position=(0.25, 0.0, 0.0))
    with JsonlPolicyTraceEventStream(sink) as stream:
        emitter = RoutePolicyTraceEmitter(stream=stream)
        rollout_route(
            env,
            RouteCandidate("run-2-route", (start_2, goal_2)),
            action_type="teleport",
            trace_emitter=emitter,
            trace_scene_id="run-2",
            trace_episode_index=5,
        )

    events = load_policy_trace_jsonl(sink)
    assert [event.episode_id for event in events] == [
        "run-1-episode-0",
        "run-1-episode-1",
        "run-2-episode-5",
    ]
    assert [event.event_name for event in events] == [
        "collision",
        "goal_reached",
        "goal_reached",
    ]


def _build_unit_catalog():
    return build_simulation_catalog(
        {
            "scenes": [
                {
                    "url": "assets/unit-scene/unit-scene.splat",
                    "label": "Unit Scene",
                    "summary": "Generic unit scene",
                }
            ]
        },
        docs_root=Path("."),
        site_url="https://example.test/gs/",
    )


def _unit_pose(position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="generic_world",
    )


def _pose_like(template: Pose3D, *, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=template.orientation_xyzw,
        frame_id=template.frame_id,
    )
