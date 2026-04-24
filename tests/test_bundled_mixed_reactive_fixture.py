"""Sanity checks for the bundled mixed-reactive DynamicObstacleTimeline fixture.

The committed JSON under ``docs/fixtures/dynamic-obstacles/mixed-reactive.json``
is advertised by the partial-information benchmark recipe in
``docs/physical-ai-sim.md``. This test pins the record type, obstacle ids,
reactive classification, and speed so the recipe's "copy this JSON" promise
stays honest when the DynamicObstacle API or serialization shifts.
"""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.sim import load_route_policy_dynamic_obstacle_timeline_json


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "docs" / "fixtures" / "dynamic-obstacles" / "mixed-reactive.json"


def test_bundled_mixed_reactive_fixture_loads_as_typed_timeline() -> None:
    assert FIXTURE.is_file(), f"missing bundled fixture: {FIXTURE}"
    timeline = load_route_policy_dynamic_obstacle_timeline_json(FIXTURE)

    assert timeline.timeline_id == "mixed-reactive-demo"
    assert timeline.obstacle_count == 3
    assert [obstacle.obstacle_id for obstacle in timeline.obstacles] == ["hunter", "runner", "bollard"]


def test_bundled_mixed_reactive_fixture_covers_all_three_reactive_modes() -> None:
    timeline = load_route_policy_dynamic_obstacle_timeline_json(FIXTURE)
    modes = {
        obstacle.obstacle_id: (
            "chase" if obstacle.chase_target_agent else "flee" if obstacle.flee_from_agent else "waypoint"
        )
        for obstacle in timeline.obstacles
    }
    speeds = {obstacle.obstacle_id: obstacle.chase_speed_m_per_step for obstacle in timeline.obstacles}

    assert modes == {"hunter": "chase", "runner": "flee", "bollard": "waypoint"}
    # Chase and flee share the same magnitude; the static bollard is bound to 0.0.
    assert speeds["hunter"] == 0.5
    assert speeds["runner"] == 0.5
    assert speeds["bollard"] == 0.0


def test_bundled_mixed_reactive_fixture_metadata_points_at_the_recipe() -> None:
    timeline = load_route_policy_dynamic_obstacle_timeline_json(FIXTURE)
    assert timeline.metadata.get("source", "").startswith("docs/physical-ai-sim.md")
