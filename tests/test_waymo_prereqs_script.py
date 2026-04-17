"""Smoke tests for scripts/check_waymo_e2e_prereqs.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_waymo_e2e_prereqs.sh"


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & 0o111, "check_waymo_e2e_prereqs.sh should be executable"


def test_script_passes_shell_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_script_reports_missing_tfrecord_dir(tmp_path: Path) -> None:
    env_dir = tmp_path / "empty"
    env_dir.mkdir()
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"WAYMO_DATA_DIR": str(env_dir), "PATH": "/usr/bin:/bin"},
    )
    assert "[MISS] no *.tfrecord" in result.stdout
    assert result.returncode != 0


@pytest.mark.parametrize("marker", ["Python runtime", "Waymo SDK", "Input data", "Summary"])
def test_script_prints_each_section(marker: str) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert marker in result.stdout
