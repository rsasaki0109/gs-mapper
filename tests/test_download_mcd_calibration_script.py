"""Smoke tests for scripts/download_mcd_calibration.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "download_mcd_calibration.sh"


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "download_mcd_calibration.sh should be executable"


def test_script_passes_shell_syntax_check() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_script_rejects_missing_rig_arg() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()


def test_script_rejects_unknown_rig() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "quadruped"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()


def test_script_pins_the_two_known_google_drive_file_ids() -> None:
    # These IDs were scraped from https://mcdviral.github.io/Download.html on
    # 2026-04-20. If MCDVIRAL rotates them the script's runtime sanity check
    # will catch it (HTML instead of "body:"), but we still want the static
    # pin here so a reviewer can spot-check against the upstream page.
    text = SCRIPT.read_text(encoding="utf-8")
    assert "1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym" in text, "handheld calibration file ID missing"
    assert "1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL" in text, "ATV calibration file ID missing"
