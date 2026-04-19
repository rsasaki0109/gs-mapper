"""Smoke tests for scripts/run_mast3r.py.

Doesn't actually run MAST3R inference (that needs a GPU + 2.7 GB
checkpoint). Just verifies the wrapper is importable, --help parses,
and required args are enforced — mirroring the shape of
``tests/test_robotics_smoke_script.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_mast3r.py"


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "run_mast3r.py should be executable"


def test_script_passes_ast_parse() -> None:
    """The wrapper must parse as valid Python (cheaper than importing)."""
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_script_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "MAST3R" in result.stdout
    # Document the main flags that mirror run_dust3r.py.
    for flag in ("--image-dir", "--output", "--checkpoint", "--mast3r-root", "--scene-graph", "--subsample"):
        assert flag in result.stdout, f"missing flag {flag} in --help"


def test_script_rejects_missing_args() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--image-dir" in result.stderr or "--output" in result.stderr
