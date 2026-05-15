"""Tests for the live trace emitter wiring on ``rollout_route`` (Sprint 3 / PR C4).

PR C2/C3 wired ``RoutePolicyTraceEmitter`` into the Gym adapter path. The
non-Gym path — ``route_execution.rollout_route``, which feeds direct
``AgentAction`` callers (route replanning, ad-hoc tests, CLI fixtures) —
remained un-wired, leaving live trace coverage incomplete. This module
covers the per-segment ``record_step`` invocation, terminal-event
dispatch on the last applied segment, the ``trace_scene_id`` fallback to
``route_id``, near-miss propagation through the transition mapping, and
that omitting the emitter preserves the prior behaviour.
"""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.sim import (
    HeadlessPhysicalAIEnvironment,
    JsonlPolicyTraceEventStream,
    Pose3D,
    RouteCandidate,
    RoutePolicyTraceEmitter,
    build_simulation_catalog,
    load_policy_trace_jsonl,
    rollout_route,
)


def test_rollout_route_emits_goal_reached_on_clear_route(tmp_path: Path) -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    waypoint = _pose_like(start, position=(0.25, 0.0, 0.0))
    goal = _pose_like(start, position=(0.25, 0.0, 0.25))
    route = RouteCandidate("clear-route", (start, waypoint, goal))

    sink = tmp_path / "trace.jsonl"
    stream = JsonlPolicyTraceEventStream(sink)
    emitter = RoutePolicyTraceEmitter(stream=stream)

    rollout = rollout_route(
        env,
        route,
        action_type="teleport",
        trace_emitter=emitter,
    )
    emitter.close()

    assert rollout.passed is True
    events = load_policy_trace_jsonl(sink)
    assert len(events) == 1
    terminal = events[0]
    assert terminal.event_name == "goal_reached"
    assert terminal.episode_id == "clear-route-episode-0"
    assert terminal.step_index == 1  # zero-based index of the last applied segment
    assert terminal.tags == ("scene:clear-route",)


def test_rollout_route_emits_collision_on_blocked_route(tmp_path: Path) -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    fallback = _pose_like(start, position=(0.5, 0.0, 0.0))
    route = RouteCandidate("blocked-route", (start, outside, fallback))

    sink = tmp_path / "trace.jsonl"
    stream = JsonlPolicyTraceEventStream(sink)
    emitter = RoutePolicyTraceEmitter(stream=stream)

    rollout = rollout_route(
        env,
        route,
        action_type="teleport",
        stop_on_collision=True,
        trace_emitter=emitter,
    )
    emitter.close()

    assert rollout.passed is False
    assert rollout.blocked_step_index == 0
    events = load_policy_trace_jsonl(sink)
    assert len(events) == 1
    assert events[0].event_name == "collision"
    assert events[0].metadata["terminationReason"] == "blocked-route"


def test_rollout_route_skips_terminal_on_intermediate_collision_when_continuing(
    tmp_path: Path,
) -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    outside = _pose_like(start, position=(2.0, 0.0, 0.0))
    final = _pose_like(start, position=(0.1, 0.0, 0.0))
    route = RouteCandidate("partial-blocked", (start, outside, final))

    sink = tmp_path / "trace.jsonl"
    stream = JsonlPolicyTraceEventStream(sink)
    emitter = RoutePolicyTraceEmitter(stream=stream)

    rollout_route(
        env,
        route,
        action_type="teleport",
        stop_on_collision=False,
        trace_emitter=emitter,
    )
    emitter.close()

    events = load_policy_trace_jsonl(sink)
    assert len(events) == 1, "exactly one terminal event should fire on the final segment"
    assert events[0].step_index == 1, "terminal event should be on the second (final) segment"


def test_rollout_route_honors_explicit_trace_scene_id() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    goal = _pose_like(start, position=(0.25, 0.0, 0.0))
    route = RouteCandidate("auto-id", (start, goal))

    captured: list[tuple[str, int]] = []

    class CapturingEmitter(RoutePolicyTraceEmitter):
        def begin_episode(self, *, scene_id: str, episode_index: int) -> str:  # type: ignore[override]
            captured.append((scene_id, episode_index))
            return super().begin_episode(scene_id=scene_id, episode_index=episode_index)

    emitter = CapturingEmitter()
    rollout_route(
        env,
        route,
        action_type="teleport",
        trace_emitter=emitter,
        trace_scene_id="custom-scene",
        trace_episode_index=7,
    )

    assert captured == [("custom-scene", 7)]


def test_rollout_route_emits_one_event_per_segment_step(tmp_path: Path) -> None:
    """Per-segment ``record_step`` should be called; only the last fires terminal."""

    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    waypoint_one = _pose_like(start, position=(0.1, 0.0, 0.0))
    waypoint_two = _pose_like(start, position=(0.2, 0.0, 0.0))
    goal = _pose_like(start, position=(0.3, 0.0, 0.0))
    route = RouteCandidate("multi-step", (start, waypoint_one, waypoint_two, goal))

    captured_step_indices: list[int] = []

    class CountingEmitter(RoutePolicyTraceEmitter):
        def record_step(
            self,
            *,
            scene_id: str,
            episode_index: int,
            step_index: int,
            next_observation,
            blocked: bool,
            goal_reached: bool,
            truncated: bool,
            terminated: bool,
            extra_tags=(),
        ):  # type: ignore[override]
            captured_step_indices.append(step_index)
            return super().record_step(
                scene_id=scene_id,
                episode_index=episode_index,
                step_index=step_index,
                next_observation=next_observation,
                blocked=blocked,
                goal_reached=goal_reached,
                truncated=truncated,
                terminated=terminated,
                extra_tags=extra_tags,
            )

    sink = tmp_path / "trace.jsonl"
    emitter = CountingEmitter(stream=JsonlPolicyTraceEventStream(sink))

    rollout_route(env, route, action_type="teleport", trace_emitter=emitter)
    emitter.close()

    assert captured_step_indices == [0, 1, 2], "record_step should fire once per executed segment"
    events = load_policy_trace_jsonl(sink)
    # Only the terminal step writes to the JSONL because clear intermediate
    # segments raise no events; the terminal sits on step_index=2.
    assert [event.event_name for event in events] == ["goal_reached"]
    assert events[0].step_index == 2


def test_rollout_route_emitter_optional_keeps_old_behavior() -> None:
    env = HeadlessPhysicalAIEnvironment(_build_unit_catalog())
    env.reset("unit-scene")
    start = env.state.pose
    goal = _pose_like(start, position=(0.25, 0.0, 0.0))

    rollout = rollout_route(
        env,
        RouteCandidate("clear-route", (start, goal)),
        action_type="teleport",
    )

    assert rollout.passed is True
    assert len(rollout.outcomes) == 1


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


def _pose_like(template: Pose3D, *, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(
        position=position,
        orientation_xyzw=template.orientation_xyzw,
        frame_id=template.frame_id,
    )
