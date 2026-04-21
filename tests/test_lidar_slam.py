"""Tests for LiDAR SLAM / GNSS trajectory import helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from gs_sim2real.preprocess.lidar_slam import LiDARSLAMProcessor


def _with_nmea_checksum(sentence: str) -> str:
    payload = sentence[1:] if sentence.startswith("$") else sentence
    payload = payload.split("*", 1)[0]
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    return f"${payload}*{checksum:02X}"


class TestNMEAHelpers:
    """Tests for NMEA parsing and ENU conversion."""

    def test_parse_nmea_time(self):
        """Should parse HHMMSS.sss into seconds from midnight."""
        assert LiDARSLAMProcessor._parse_nmea_time("123519") == 45319.0
        assert LiDARSLAMProcessor._parse_nmea_time("123519.5") == 45319.5
        assert LiDARSLAMProcessor._parse_nmea_time("") is None

    def test_parse_nmea_coordinate(self):
        """Should convert NMEA degree-minute coordinates into signed degrees."""
        lat = LiDARSLAMProcessor._parse_nmea_coordinate("4807.038", "N")
        lon = LiDARSLAMProcessor._parse_nmea_coordinate("01131.000", "E")
        south = LiDARSLAMProcessor._parse_nmea_coordinate("3407.000", "S")

        assert np.isclose(lat, 48.1173, atol=1e-9)
        assert np.isclose(lon, 11.5166666667, atol=1e-9)
        assert south < 0.0

    def test_wgs84_to_enu_reference_origin_is_zero(self):
        """Reference point should map to the local ENU origin."""
        east, north, up = LiDARSLAMProcessor._wgs84_to_enu(
            lat=35.0,
            lon=139.0,
            alt=42.0,
            ref_lat=35.0,
            ref_lon=139.0,
            ref_alt=42.0,
        )

        assert abs(east) < 1e-6
        assert abs(north) < 1e-6
        assert abs(up) < 1e-6


class TestNMEALoading:
    """Tests for converting NMEA logs into trajectory poses."""

    def test_load_nmea_trajectory_with_rmc_date_and_course(self, tmp_path):
        """Should create ENU poses and UTC timestamps from GGA/RMC sentences."""
        path = tmp_path / "track.nmea"
        path.write_text(
            "\n".join(
                [
                    _with_nmea_checksum("$GPRMC,123519,A,4807.038,N,01131.000,E,010.0,090.0,230394,,,A"),
                    _with_nmea_checksum("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"),
                    _with_nmea_checksum("$GPRMC,123520,A,4807.038,N,01131.001,E,010.0,090.0,230394,,,A"),
                    _with_nmea_checksum("$GPGGA,123520,4807.038,N,01131.001,E,1,08,0.9,545.4,M,46.9,M,,"),
                ]
            )
        )

        timestamps, poses = LiDARSLAMProcessor()._load_nmea_trajectory(path)

        assert len(timestamps) == 2
        assert len(poses) == 2

        expected_ts0 = datetime(1994, 3, 23, 12, 35, 19, tzinfo=timezone.utc).timestamp()
        assert timestamps[0] == expected_ts0
        assert timestamps[1] == expected_ts0 + 1.0

        assert np.allclose(poses[0][:3, 3], np.zeros(3), atol=1e-4)
        assert poses[1][0, 3] > 1.0
        assert abs(poses[1][1, 3]) < 0.2
        assert abs(poses[1][2, 3]) < 0.2
        assert np.allclose(poses[0][:3, :3], np.eye(3), atol=1e-6)

    def test_load_nmea_trajectory_without_date_falls_back_to_relative_seconds(self, tmp_path):
        """Should keep relative timestamps when the NMEA stream has no date info."""
        path = tmp_path / "track_no_date.nmea"
        path.write_text(
            "\n".join(
                [
                    _with_nmea_checksum("$GPGGA,000001,3500.000,N,13900.000,E,1,08,0.9,10.0,M,0.0,M,,"),
                    _with_nmea_checksum("$GPGGA,000002,3500.000,N,13900.001,E,1,08,0.9,10.0,M,0.0,M,,"),
                ]
            )
        )

        timestamps, poses = LiDARSLAMProcessor()._load_nmea_trajectory(path)

        assert timestamps == [0.0, 1.0]
        assert np.allclose(poses[0][:3, 3], np.zeros(3), atol=1e-4)
        assert poses[1][0, 3] > 1.0
        assert np.allclose(poses[0][:3, :3], np.eye(3), atol=1e-6)

    def test_load_nmea_crlf_and_blank_lines(self, tmp_path: Path) -> None:
        """Should tolerate Windows CRLF and stray blank lines."""
        path = tmp_path / "crlf.nmea"
        path.write_text(
            "\r\n\r\n"
            + "\r\n".join(
                [
                    _with_nmea_checksum("$GPGGA,000001,3500.000,N,13900.000,E,1,08,0.9,10.0,M,0.0,M,,"),
                    _with_nmea_checksum("$GPGGA,000002,3500.000,N,13900.001,E,1,08,0.9,10.0,M,0.0,M,,"),
                ]
            )
            + "\r\n"
        )

        timestamps, poses = LiDARSLAMProcessor()._load_nmea_trajectory(path)

        assert len(timestamps) == 2
        assert len(poses) == 2

    def test_load_nmea_multitalker_gngga(self, tmp_path: Path) -> None:
        """Should accept GNGGA / GNRMC talker IDs (multi-GNSS receivers)."""
        path = tmp_path / "gnss.nmea"
        path.write_text(
            "\n".join(
                [
                    _with_nmea_checksum("$GNRMC,123519,A,4807.038,N,01131.000,E,010.0,090.0,230394,,,A"),
                    _with_nmea_checksum("$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"),
                    _with_nmea_checksum("$GNRMC,123520,A,4807.038,N,01131.001,E,010.0,090.0,230394,,,A"),
                    _with_nmea_checksum("$GNGGA,123520,4807.038,N,01131.001,E,1,08,0.9,545.4,M,46.9,M,,"),
                ]
            )
        )

        timestamps, poses = LiDARSLAMProcessor()._load_nmea_trajectory(path)

        assert len(timestamps) == 2
        assert len(poses) == 2

    def test_load_nmea_utf8_bom(self, tmp_path: Path) -> None:
        """UTF-8 BOM should not break the first sentence."""
        path = tmp_path / "bom.nmea"
        path.write_bytes(
            (
                "\ufeff"
                + _with_nmea_checksum("$GPGGA,000001,3500.000,N,13900.000,E,1,08,0.9,10.0,M,0.0,M,,")
                + "\n"
                + _with_nmea_checksum("$GPGGA,000002,3500.000,N,13900.001,E,1,08,0.9,10.0,M,0.0,M,,")
                + "\n"
            ).encode("utf-8")
        )

        timestamps, poses = LiDARSLAMProcessor()._load_nmea_trajectory(path)

        assert len(timestamps) == 2

    def test_load_nmea_raises_when_no_valid_fixes(self, tmp_path: Path) -> None:
        """Empty file or only invalid fixes should raise."""
        empty = tmp_path / "empty.nmea"
        empty.write_text("")
        with pytest.raises(ValueError, match="No valid NMEA"):
            LiDARSLAMProcessor()._load_nmea_trajectory(empty)

        bad = tmp_path / "bad.nmea"
        bad.write_text(
            _with_nmea_checksum("$GPGGA,000001,3500.000,N,13900.000,E,0,08,0.9,10.0,M,0.0,M,,") + "\n"  # fix quality 0
        )
        with pytest.raises(ValueError, match="No valid NMEA"):
            LiDARSLAMProcessor()._load_nmea_trajectory(bad)


class TestCOLMAPExport:
    """Tests for COLMAP text export from trajectories."""

    def test_import_trajectory_applies_pinhole_calibration_json(self, tmp_path: Path) -> None:
        """Should write PINHOLE parameters from JSON instead of a heuristic focal length."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            cv2.imwrite(str(img_dir / f"frame_{i:06d}.jpg"), np.zeros((480, 640, 3), dtype=np.uint8))

        tum = tmp_path / "track.tum"
        tum.write_text("0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n")

        calib = tmp_path / "pinhole.json"
        calib.write_text(json.dumps({"width": 640, "height": 480, "fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0}))

        out = tmp_path / "colmap"
        sparse = LiDARSLAMProcessor().import_trajectory(
            tum,
            img_dir,
            out,
            trajectory_format="tum",
            pinhole_calib_path=calib,
        )

        assert Path(sparse).name == "0"
        cameras = (out / "sparse" / "0" / "cameras.txt").read_text()
        assert "PINHOLE 640 480 500.0 510.0 320.0 240.0" in cameras.replace("\n", " ")

    def test_align_images_uses_image_timestamps_csv(self, tmp_path: Path) -> None:
        """Should pair images to trajectory samples using image_timestamps.csv."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            cv2.imwrite(str(img_dir / f"f{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

        (img_dir / "image_timestamps.csv").write_text("filename,timestamp_ns\nf0.jpg,1000000000\nf1.jpg,2000000000\n")

        tum = tmp_path / "t.tum"
        tum.write_text("1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n")

        out = tmp_path / "out"
        LiDARSLAMProcessor().import_trajectory(
            tum,
            img_dir,
            out,
            trajectory_format="tum",
        )

        images_txt = (out / "sparse" / "0" / "images.txt").read_text()
        assert "f0.jpg" in images_txt
        assert "f1.jpg" in images_txt

    def test_import_multicam_vehicle_trajectory_writes_two_cameras(self, tmp_path: Path) -> None:
        """Multi-camera vehicle TUM + per-folder images -> two PINHOLE cameras in COLMAP."""
        root = tmp_path / "images"
        (root / "cam0").mkdir(parents=True)
        (root / "cam1").mkdir(parents=True)
        cv2.imwrite(str(root / "cam0" / "frame_000000.jpg"), np.zeros((60, 80, 3), dtype=np.uint8))
        cv2.imwrite(str(root / "cam1" / "frame_000000.jpg"), np.zeros((60, 80, 3), dtype=np.uint8))
        (root / "image_timestamps.csv").write_text(
            "filename,timestamp_ns\ncam0/frame_000000.jpg,1000000000\ncam1/frame_000000.jpg,1000000000\n"
        )

        tum = tmp_path / "veh.tum"
        tum.write_text("1.0 0 0 0 0 0 0 1\n2.0 10 0 0 0 0 0 1\n")

        ph = (100.0, 100.0, 40.0, 30.0, 80, 60)
        cameras = [
            {"subdir": "cam0", "camera_id": 1, "camera_frame": "", "T_base_cam": None, "pinhole": ph},
            {"subdir": "cam1", "camera_id": 2, "camera_frame": "", "T_base_cam": None, "pinhole": ph},
        ]

        out = tmp_path / "colmap"
        sparse = LiDARSLAMProcessor().import_multicam_vehicle_trajectory(
            tum,
            root,
            out,
            cameras=cameras,
        )

        assert Path(sparse).name == "0"
        cams = (out / "sparse" / "0" / "cameras.txt").read_text()
        assert "1 PINHOLE" in cams
        assert "2 PINHOLE" in cams
        imgs = (out / "sparse" / "0" / "images.txt").read_text()
        assert "cam0/frame_000000.jpg" in imgs
        assert "cam1/frame_000000.jpg" in imgs
