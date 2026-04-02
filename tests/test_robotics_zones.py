"""Tests for semantic zone parsing and costmap rasterization."""

from __future__ import annotations

from gs_sim2real.robotics.zones import (
    build_semantic_zone_map,
    build_zone_hit_payload,
    build_zone_summary_payload,
    find_zone_hits,
    rasterize_costmap,
)


def sample_zone_payload() -> dict:
    return {
        "frameId": "dreamwalker_map",
        "resolution": 1.0,
        "defaultCost": 5,
        "bounds": {
            "minX": 0.0,
            "maxX": 4.0,
            "minZ": 0.0,
            "maxZ": 4.0,
        },
        "zones": [
            {
                "id": "stage",
                "label": "Stage",
                "shape": "rect",
                "center": [1.0, 0.0, 1.0],
                "size": [2.0, 2.0],
                "cost": 30,
                "tags": ["safe"],
            },
            {
                "id": "pit",
                "label": "Pit",
                "shape": "circle",
                "center": [3.0, 0.0, 3.0],
                "radius": 0.8,
                "cost": 90,
                "tags": ["hazard"],
            },
        ],
    }


def test_build_semantic_zone_map_parses_shapes() -> None:
    zone_map = build_semantic_zone_map(sample_zone_payload())

    assert zone_map.frame_id == "dreamwalker_map"
    assert zone_map.resolution == 1.0
    assert len(zone_map.zones) == 2
    assert zone_map.zones[0].shape == "rect"
    assert zone_map.zones[1].shape == "circle"


def test_find_zone_hits_returns_matching_zones() -> None:
    zone_map = build_semantic_zone_map(sample_zone_payload())

    stage_hits = find_zone_hits(zone_map, 1.0, 1.0)
    pit_hits = find_zone_hits(zone_map, 3.0, 3.0)
    none_hits = find_zone_hits(zone_map, 0.1, 3.9)

    assert [zone.zone_id for zone in stage_hits] == ["stage"]
    assert [zone.zone_id for zone in pit_hits] == ["pit"]
    assert none_hits == ()


def test_rasterize_costmap_uses_max_zone_cost() -> None:
    zone_map = build_semantic_zone_map(sample_zone_payload())
    costmap = rasterize_costmap(zone_map)

    assert costmap.width == 4
    assert costmap.height == 4
    assert len(costmap.data) == 16
    assert max(costmap.data) == 90
    assert min(costmap.data) == 5


def test_zone_summary_and_hit_payloads_are_json_ready() -> None:
    zone_map = build_semantic_zone_map(sample_zone_payload())
    zone_hits = find_zone_hits(zone_map, 1.0, 1.0)

    summary = build_zone_summary_payload(zone_map)
    hit_payload = build_zone_hit_payload(zone_hits, 1.0, 1.0)

    assert summary["zoneCount"] == 2
    assert summary["zones"][0]["id"] == "stage"
    assert hit_payload["zones"][0]["label"] == "Stage"
