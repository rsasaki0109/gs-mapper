"""Tests for scripts/check_mcd_gnss.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_scan_rejects_zero_placeholder_navsat(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    bag_path = tmp_path / "session.bag"
    bag_path.write_bytes(b"bag")

    class FakeReader:
        def __init__(self, paths, **kwargs):
            self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
            self.topics = {"/vn200/GPS": SimpleNamespace(connections=[self.connection])}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, connections):
            for idx in range(3):
                yield self.connection, int((10 + idx) * 1e9), b""

        def deserialize(self, rawdata, msgtype):
            return SimpleNamespace(
                latitude=0.0,
                longitude=0.0,
                altitude=0.0,
                status=SimpleNamespace(status=0),
            )

    monkeypatch.setattr(module.MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

    summary = module.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS")

    assert not summary.ok
    assert summary.total_samples == 3
    assert summary.valid_samples == 0
    assert summary.zero_placeholder_samples == 3
    assert "all finite GNSS fixes are zero placeholders" in summary.failures


def test_scan_accepts_moving_valid_navsat_with_image_overlap(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    bag_path = tmp_path / "session.bag"
    bag_path.write_bytes(b"bag")
    ts_csv = tmp_path / "image_timestamps.csv"
    ts_csv.write_text(
        "filename,timestamp_ns\nframe_000000.jpg,10000000000\nframe_000001.jpg,11000000000\n",
        encoding="utf-8",
    )

    rows = [
        (10.0, 35.0, 139.0, 5.0),
        (11.0, 35.00002, 139.00003, 5.0),
        (12.0, 35.00004, 139.00006, 5.0),
    ]

    class FakeReader:
        def __init__(self, paths, **kwargs):
            self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
            self.topics = {"/vn200/GPS": SimpleNamespace(connections=[self.connection])}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, connections):
            for idx, row in enumerate(rows):
                yield self.connection, int(row[0] * 1e9), str(idx).encode("ascii")

        def deserialize(self, rawdata, msgtype):
            _, lat, lon, alt = rows[int(rawdata.decode("ascii"))]
            return SimpleNamespace(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                status=SimpleNamespace(status=0),
            )

    monkeypatch.setattr(module.MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

    summary = module.scan_mcd_gnss(
        tmp_path,
        gnss_topic="/vn200/GPS",
        image_timestamps=ts_csv,
        min_translation_m=1.0,
    )

    assert summary.ok
    assert summary.valid_samples == 3
    assert summary.translation_extent_m > 1.0
    assert summary.horizontal_extent_m > 1.0
    assert summary.image_timestamps is not None
    assert summary.image_timestamps.overlap_count == 2


def test_scan_rejects_altitude_spike_without_flattening(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    bag_path = tmp_path / "session.bag"
    bag_path.write_bytes(b"bag")

    rows = [
        (10.0, 35.0, 139.0, 10000.0),
        (11.0, 35.00002, 139.00003, 10.0),
        (12.0, 35.00004, 139.00006, 10.0),
    ]

    class FakeReader:
        def __init__(self, paths, **kwargs):
            self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
            self.topics = {"/vn200/GPS": SimpleNamespace(connections=[self.connection])}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, connections):
            for idx, row in enumerate(rows):
                yield self.connection, int(row[0] * 1e9), str(idx).encode("ascii")

        def deserialize(self, rawdata, msgtype):
            _, lat, lon, alt = rows[int(rawdata.decode("ascii"))]
            return SimpleNamespace(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                status=SimpleNamespace(status=0),
            )

    monkeypatch.setattr(module.MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

    summary = module.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS", max_vertical_extent_m=250.0)

    assert not summary.ok
    assert summary.horizontal_extent_m > 1.0
    assert summary.vertical_extent_m > 250.0
    assert any("vertical extent" in failure for failure in summary.failures)


def test_scan_can_flatten_altitude_spike(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    bag_path = tmp_path / "session.bag"
    bag_path.write_bytes(b"bag")

    rows = [
        (10.0, 35.0, 139.0, 10000.0),
        (11.0, 35.00002, 139.00003, 10.0),
        (12.0, 35.00004, 139.00006, 10.0),
    ]

    class FakeReader:
        def __init__(self, paths, **kwargs):
            self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
            self.topics = {"/vn200/GPS": SimpleNamespace(connections=[self.connection])}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def messages(self, connections):
            for idx, row in enumerate(rows):
                yield self.connection, int(row[0] * 1e9), str(idx).encode("ascii")

        def deserialize(self, rawdata, msgtype):
            _, lat, lon, alt = rows[int(rawdata.decode("ascii"))]
            return SimpleNamespace(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                status=SimpleNamespace(status=0),
            )

    monkeypatch.setattr(module.MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

    summary = module.scan_mcd_gnss(
        tmp_path,
        gnss_topic="/vn200/GPS",
        max_vertical_extent_m=250.0,
        flatten_altitude=True,
    )

    assert summary.ok
    assert summary.altitude_span_m > 9000.0
    assert summary.vertical_extent_m < 1e-3
    assert any("flattened altitude" in warning for warning in summary.warnings)


def test_cli_returns_nonzero_for_missing_session(tmp_path: Path, capsys) -> None:
    module = _load_script_module()

    rc = module.main([str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert rc == 1
    assert "no rosbag files" in captured.out
