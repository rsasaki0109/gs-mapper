"""Tests for the thin scripts/check_mcd_gnss.py CLI wrapper."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from gs_sim2real.datasets.mcd_gnss_preflight import GnssPreflightSummary

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_mcd_gnss.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("check_mcd_gnss", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "check_mcd_gnss.py should be executable"


def test_script_passes_python_syntax_check() -> None:
    result = subprocess.run(
        ["python3", "-m", "py_compile", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_returns_nonzero_for_missing_session(tmp_path: Path, capsys) -> None:
    module = _load_script_module()

    rc = module.main([str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert rc == 1
    assert "no rosbag files" in captured.out


def test_cli_json_uses_preflight_module(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_script_module()

    def fake_scan(*args, **kwargs):
        return GnssPreflightSummary(data_dir=str(tmp_path), bag_count=1, topic="/vn200/GPS", valid_samples=3)

    monkeypatch.setattr(module, "scan_mcd_gnss", fake_scan)

    rc = module.main([str(tmp_path), "--gnss-topic", "/vn200/GPS", "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert '"topic": "/vn200/GPS"' in captured.out
    assert '"valid_samples": 3' in captured.out
