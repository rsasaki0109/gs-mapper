"""Tests for scripts/show_dynamic_obstacle_timeline.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from gs_sim2real.sim import (
    DynamicObstacle,
    DynamicObstacleTimeline,
    DynamicObstacleWaypoint,
    write_route_policy_dynamic_obstacle_timeline_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "show_dynamic_obstacle_timeline.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("show_dynamic_obstacle_timeline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mixed_timeline(path: Path) -> Path:
    return write_route_policy_dynamic_obstacle_timeline_json(
        path,
        DynamicObstacleTimeline(
            timeline_id="mixed-reactive-demo",
            obstacles=(
                DynamicObstacle(
                    obstacle_id="bollard",
                    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(0.0, 1.0, 0.0)),),
                    radius_meters=0.1,
                ),
                DynamicObstacle(
                    obstacle_id="hunter",
                    waypoints=(DynamicObstacleWaypoint(step_index=0, position=(3.0, 0.0, 0.0)),),
                    radius_meters=0.25,
                    chase_target_agent=True,
                    chase_speed_m_per_step=0.5,
                ),
            ),
        ),
    )


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "show_dynamic_obstacle_timeline.py should be executable"


def test_script_passes_python_syntax_check() -> None:
    result = subprocess.run(
        ["python3", "-m", "py_compile", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_markdown_prints_header_and_reactive_classification(tmp_path: Path, capsys) -> None:
    timeline_path = _write_mixed_timeline(tmp_path / "timeline.json")
    module = _load_script_module()

    rc = module.main([str(timeline_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Route Policy Dynamic Obstacle Timeline: mixed-reactive-demo" in captured.out
    assert "Reactive mode" in captured.out
    assert "| bollard | 0.1 | 1 | 0 | 0 | waypoint | 0.0 |" in captured.out
    assert "| hunter | 0.25 | 1 | 0 | 0 | chase | 0.5 |" in captured.out


def test_cli_json_round_trips_the_timeline(tmp_path: Path, capsys) -> None:
    timeline_path = _write_mixed_timeline(tmp_path / "timeline.json")
    module = _load_script_module()

    rc = module.main([str(timeline_path), "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["timelineId"] == "mixed-reactive-demo"
    assert payload["obstacleCount"] == 2
    chaser_payload = next(obs for obs in payload["obstacles"] if obs["obstacleId"] == "hunter")
    assert chaser_payload["chaseTargetAgent"] is True
    assert chaser_payload["chaseSpeedMPerStep"] == 0.5


def test_cli_writes_output_file_when_requested(tmp_path: Path) -> None:
    timeline_path = _write_mixed_timeline(tmp_path / "timeline.json")
    output_path = tmp_path / "summary.md"
    module = _load_script_module()

    rc = module.main([str(timeline_path), "--output", str(output_path)])

    assert rc == 0
    text = output_path.read_text(encoding="utf-8")
    assert "Route Policy Dynamic Obstacle Timeline" in text
    assert "hunter" in text
