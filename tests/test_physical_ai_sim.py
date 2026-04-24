"""Tests for Physical AI simulation scene contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gs_sim2real.sim import (
    AgentAction,
    AxisAlignedBounds,
    CollisionQuery,
    ObservationRequest,
    Pose3D,
    TrajectoryScore,
    Vec3,
    load_simulation_catalog_from_scene_picker,
    render_simulation_catalog_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_simulation_catalog_wraps_all_public_scenes() -> None:
    catalog = load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json")

    assert catalog.version == "gs-mapper-physical-ai-sim/v1"
    assert len(catalog.scenes) == 8
    assert len(set(catalog.scene_ids())) == 8

    outdoor = catalog.scene_by_id("outdoor-demo")
    assert outdoor.coordinate_frame.scale_status == "metric"
    assert outdoor.bounds.confidence == "declared"
    assert outdoor.has_task("waypoint-navigation")
    assert outdoor.viewer_url.endswith("splat.html?url=assets/outdoor-demo/outdoor-demo.splat")
    assert "rgb-forward" in outdoor.sensor_rig.sensor_ids()
    depth_sensor = next(sensor for sensor in outdoor.sensor_rig.sensors if sensor.sensor_id == "depth-proxy")
    assert depth_sensor.status == "ready-via-splat-raster"
    lidar_sensor = next(sensor for sensor in outdoor.sensor_rig.sensors if sensor.sensor_id == "lidar-ray-proxy")
    assert lidar_sensor.status == "ready-via-depth-rays"
    imu_sensor = next(sensor for sensor in outdoor.sensor_rig.sensors if sensor.sensor_id == "imu-proxy")
    assert imu_sensor.status == "ready-via-kinematic-finite-diff"
    assert tuple(imu_sensor.outputs) == ("angular-velocity", "linear-acceleration")

    dust3r = catalog.scene_by_id("outdoor-demo-dust3r")
    assert dust3r.coordinate_frame.scale_status == "relative"
    assert dust3r.has_task("localization")
    assert not dust3r.has_task("waypoint-navigation")


def test_simulation_catalog_json_is_stable_and_contains_agent_contract() -> None:
    catalog = load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json")
    payload = json.loads(render_simulation_catalog_json(catalog))

    assert payload["sceneCount"] == 8
    first = payload["scenes"][0]
    assert first["sceneId"] == "outdoor-demo"
    assert payload["sourceCatalog"] == "docs/scenes-list.json"
    assert first["bounds"]["source"] == "assets/outdoor-demo/scene.json"
    assert first["sensorRig"]["sensors"][0]["sensorId"] == "rgb-forward"
    assert {task["taskId"] for task in first["evaluationTasks"]} >= {
        "localization",
        "viewpoint-planning",
        "waypoint-navigation",
    }
    assert first["taskSplit"][0]["split"] == "train"
    assert first["taskSplit"][1]["split"] == "eval"
    assert all("-" not in scene["coordinateFrame"]["frameId"] for scene in payload["scenes"])


def test_generate_sim_catalog_script_matches_checked_in_docs(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    output = tmp_path / "sim-scenes.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_sim_catalog.py",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    expected = output.read_text(encoding="utf-8")
    actual = (REPO_ROOT / "docs" / "sim-scenes.json").read_text(encoding="utf-8")
    assert actual == expected


def test_bounds_and_interface_payloads_are_json_friendly() -> None:
    bounds = AxisAlignedBounds(
        minimum=Vec3(-1, -2, -3),
        maximum=Vec3(4, 5, 6),
        source="unit-test",
        confidence="declared",
    )
    assert bounds.extent.to_list() == [5, 7, 9]
    assert bounds.contains(Vec3(0, 0, 0))
    assert not bounds.contains(Vec3(8, 0, 0))

    pose = Pose3D(position=(1.0, 2.0, 3.0), orientation_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map")
    request = ObservationRequest(pose=pose, sensor_id="rgb-forward")
    collision = CollisionQuery(pose=pose, collides=False, reason="inside-bounds", clearance_meters=1.5)
    score = TrajectoryScore(metrics={"goal-success": 1.0}, passed=True)
    action = AgentAction(action_type="twist", values={"linearX": 0.2, "angularZ": 0.1})

    json.dumps(request.to_dict())
    json.dumps(collision.to_dict())
    json.dumps(score.to_dict())
    json.dumps(action.to_dict())


def test_unknown_scene_lookup_raises_key_error() -> None:
    catalog = load_simulation_catalog_from_scene_picker(REPO_ROOT / "docs" / "scenes-list.json")

    with pytest.raises(KeyError, match="unknown simulation scene"):
        catalog.scene_by_id("missing")
