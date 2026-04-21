"""Unit tests for importable MCD GNSS preflight logic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gs_sim2real.datasets import mcd_gnss_preflight as preflight


def _patch_navsat_reader(monkeypatch, rows: list[tuple[float, float, float, float]]) -> None:
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

    monkeypatch.setattr(preflight.MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))


def _touch_bag(tmp_path: Path) -> None:
    (tmp_path / "session.bag").write_bytes(b"bag")


def test_scan_rejects_zero_placeholder_navsat(monkeypatch, tmp_path: Path) -> None:
    _touch_bag(tmp_path)
    _patch_navsat_reader(monkeypatch, [(10.0, 0.0, 0.0, 0.0), (11.0, 0.0, 0.0, 0.0), (12.0, 0.0, 0.0, 0.0)])

    summary = preflight.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS")

    assert not summary.ok
    assert summary.total_samples == 3
    assert summary.valid_samples == 0
    assert summary.zero_placeholder_samples == 3
    assert "all finite GNSS fixes are zero placeholders" in summary.failures


def test_scan_accepts_moving_valid_navsat_with_image_overlap(monkeypatch, tmp_path: Path) -> None:
    _touch_bag(tmp_path)
    ts_csv = tmp_path / "image_timestamps.csv"
    ts_csv.write_text(
        "filename,timestamp_ns\nframe_000000.jpg,10000000000\nframe_000001.jpg,11000000000\n",
        encoding="utf-8",
    )
    _patch_navsat_reader(
        monkeypatch,
        [
            (10.0, 35.0, 139.0, 5.0),
            (11.0, 35.00002, 139.00003, 5.0),
            (12.0, 35.00004, 139.00006, 5.0),
        ],
    )

    summary = preflight.scan_mcd_gnss(
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
    _touch_bag(tmp_path)
    _patch_navsat_reader(
        monkeypatch,
        [
            (10.0, 35.0, 139.0, 10000.0),
            (11.0, 35.00002, 139.00003, 10.0),
            (12.0, 35.00004, 139.00006, 10.0),
        ],
    )

    summary = preflight.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS", max_vertical_extent_m=250.0)

    assert not summary.ok
    assert summary.horizontal_extent_m > 1.0
    assert summary.vertical_extent_m > 250.0
    assert any("vertical extent" in failure for failure in summary.failures)


def test_scan_can_flatten_altitude_spike(monkeypatch, tmp_path: Path) -> None:
    _touch_bag(tmp_path)
    _patch_navsat_reader(
        monkeypatch,
        [
            (10.0, 35.0, 139.0, 10000.0),
            (11.0, 35.00002, 139.00003, 10.0),
            (12.0, 35.00004, 139.00006, 10.0),
        ],
    )

    summary = preflight.scan_mcd_gnss(
        tmp_path,
        gnss_topic="/vn200/GPS",
        max_vertical_extent_m=250.0,
        flatten_altitude=True,
    )

    assert summary.ok
    assert summary.altitude_span_m > 9000.0
    assert summary.vertical_extent_m < 1e-3
    assert any("flattened altitude" in warning for warning in summary.warnings)


def test_scan_rejects_regressing_valid_timestamps(monkeypatch, tmp_path: Path) -> None:
    _touch_bag(tmp_path)
    _patch_navsat_reader(
        monkeypatch,
        [
            (10.0, 35.0, 139.0, 5.0),
            (12.0, 35.00002, 139.00003, 5.0),
            (11.0, 35.00004, 139.00006, 5.0),
        ],
    )

    summary = preflight.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS", min_translation_m=1.0)

    assert not summary.ok
    assert summary.nonmonotonic_valid_timestamps == 1
    assert summary.duplicate_valid_timestamps == 0
    assert any("timestamps regressed" in failure for failure in summary.failures)


def test_scan_warns_on_duplicate_valid_timestamps(monkeypatch, tmp_path: Path) -> None:
    _touch_bag(tmp_path)
    _patch_navsat_reader(
        monkeypatch,
        [
            (10.0, 35.0, 139.0, 5.0),
            (10.0, 35.00002, 139.00003, 5.0),
            (12.0, 35.00004, 139.00006, 5.0),
        ],
    )

    summary = preflight.scan_mcd_gnss(tmp_path, gnss_topic="/vn200/GPS", min_translation_m=1.0)

    assert summary.ok
    assert summary.nonmonotonic_valid_timestamps == 0
    assert summary.duplicate_valid_timestamps == 1
    assert any("timestamps repeated" in warning for warning in summary.warnings)
