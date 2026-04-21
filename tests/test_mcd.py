"""Tests for the MCD rosbag loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gs_sim2real.datasets.mcd import MCDLoader, _apply_antenna_offset_base_link


class TestAntennaOffsetBaseLink:
    """Heading-based base_link antenna correction in ENU."""

    def test_east_motion_forward_offset(self):
        """Straight east motion: base x aligns with +E; subtract (1,0,0)_base from positions."""
        east = np.array([0.0, 1.0, 2.0, 3.0])
        north = np.zeros(4)
        up = np.zeros(4)
        oe, on, ou = _apply_antenna_offset_base_link(east, north, up, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(oe, east - 1.0)
        np.testing.assert_allclose(on, north)
        np.testing.assert_allclose(ou, up)

    def test_mutual_exclusive_navsat_offsets(self, tmp_path):
        """extract_navsat_trajectory rejects both ENU and base offsets."""
        loader = MCDLoader(tmp_path)
        with pytest.raises(ValueError, match="both"):
            loader.extract_navsat_trajectory(
                tmp_path / "out",
                antenna_offset_enu=(1.0, 0.0, 0.0),
                antenna_offset_base=(0.0, 0.0, 1.0),
            )

    def test_extract_navsat_trajectory_rejects_zero_placeholder_fixes(self, monkeypatch, tmp_path):
        """All-zero NavSatFix placeholders should not produce a static trajectory."""
        bag_path = tmp_path / "zero_gps.bag"
        bag_path.write_bytes(b"bag")

        class FakeReader:
            def __init__(self, paths, **kwargs):
                assert paths == [bag_path]
                self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
                self.topics = {
                    "/vn200/GPS": SimpleNamespace(connections=[self.connection]),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def messages(self, connections):
                assert connections == [self.connection]
                for idx in range(3):
                    yield self.connection, int((idx + 1) * 1e9), b""

            def deserialize(self, rawdata, msgtype):
                return SimpleNamespace(
                    latitude=0.0,
                    longitude=0.0,
                    altitude=0.0,
                    status=SimpleNamespace(status=0),
                )

        monkeypatch.setattr(MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

        with pytest.raises(ValueError, match="Need at least 2 NavSatFix samples"):
            MCDLoader(tmp_path).extract_navsat_trajectory(tmp_path / "out", gnss_topic="/vn200/GPS")

    def test_extract_navsat_trajectory_can_flatten_altitude_spike(self, monkeypatch, tmp_path):
        """Optional altitude flattening should prevent VectorNav warm-up spikes from entering TUM z."""
        bag_path = tmp_path / "gps_spike.bag"
        bag_path.write_bytes(b"bag")
        rows = [
            (1.0, 35.0, 139.0, 10000.0),
            (2.0, 35.00001, 139.00001, 5.0),
            (3.0, 35.00002, 139.00002, 5.0),
        ]

        class FakeReader:
            def __init__(self, paths, **kwargs):
                assert paths == [bag_path]
                self.connection = SimpleNamespace(topic="/vn200/GPS", msgtype="sensor_msgs/msg/NavSatFix")
                self.topics = {
                    "/vn200/GPS": SimpleNamespace(connections=[self.connection]),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def messages(self, connections):
                assert connections == [self.connection]
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

        monkeypatch.setattr(MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

        tum_path = Path(
            MCDLoader(tmp_path).extract_navsat_trajectory(
                tmp_path / "out",
                gnss_topic="/vn200/GPS",
                flatten_altitude=True,
            )
        )

        z_values = [float(line.split()[3]) for line in tum_path.read_text(encoding="utf-8").splitlines()]
        assert max(abs(z) for z in z_values) < 1e-3
        assert '"altitude_mode": "flattened_median"' in (tum_path.parent / "origin_wgs84.json").read_text(
            encoding="utf-8"
        )


class TestFindBagPaths:
    """Tests for rosbag path discovery."""

    def test_finds_rosbag1_file(self, tmp_path):
        """Should return a direct rosbag1 file path."""
        bag_path = tmp_path / "sample.bag"
        bag_path.write_bytes(b"bag")

        assert MCDLoader._find_bag_paths(bag_path) == [bag_path]

    def test_finds_rosbag2_directory(self, tmp_path):
        """Should return a rosbag2 directory when metadata.yaml is present."""
        bag_dir = tmp_path / "session"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:")

        assert MCDLoader._find_bag_paths(bag_dir) == [bag_dir]


class TestLoadPreExtracted:
    """Tests for pre-extracted image fallback."""

    def test_uses_dataset_images_directory(self, tmp_path):
        """Should return the dataset's images directory when present."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "frame_0000.jpg").write_bytes(b"\xff\xd8dummy")

        loader = MCDLoader(tmp_path)
        assert loader._load_pre_extracted(tmp_path / "output", max_frames=10) == str(images_dir)

    def test_raises_when_no_images_exist(self, tmp_path):
        """Should raise when neither bags nor pre-extracted images exist."""
        loader = MCDLoader(tmp_path)
        with pytest.raises(FileNotFoundError, match="No images found"):
            loader._load_pre_extracted(tmp_path / "output", max_frames=10)


class TestSelectConnection:
    """Tests for selecting bag topic connections."""

    def test_prefers_requested_topic(self):
        """Should select the requested topic when it exists and matches the type."""
        requested = SimpleNamespace(topic="/camera", msgtype="sensor_msgs/msg/Image")
        fallback = SimpleNamespace(topic="/other", msgtype="sensor_msgs/msg/Image")
        topics = {
            "/camera": SimpleNamespace(connections=[requested]),
            "/other": SimpleNamespace(connections=[fallback]),
        }

        selected = MCDLoader._select_connection(
            topics,
            requested_topic="/camera",
            preferred_topics=("/other",),
            allowed_msgtypes=MCDLoader.IMAGE_MSGTYPES,
        )

        assert selected is requested

    def test_falls_back_to_preferred_topics(self):
        """Should select the first preferred topic with a matching message type."""
        selected_conn = SimpleNamespace(topic="/d455t/color/image_raw", msgtype="sensor_msgs/msg/Image")
        topics = {
            "/d455t/color/image_raw": SimpleNamespace(connections=[selected_conn]),
            "/vn200/imu": SimpleNamespace(
                connections=[SimpleNamespace(topic="/vn200/imu", msgtype="sensor_msgs/msg/Imu")]
            ),
        }

        selected = MCDLoader._select_connection(
            topics,
            requested_topic=None,
            preferred_topics=MCDLoader.DEFAULT_IMAGE_TOPICS,
            allowed_msgtypes=MCDLoader.IMAGE_MSGTYPES,
        )

        assert selected is selected_conn

    def test_normalizes_requested_topic_lists(self):
        """Should accept comma-separated or list-based requested topics."""
        assert MCDLoader._normalize_requested_topics("/cam0,/cam1") == ["/cam0", "/cam1"]
        assert MCDLoader._normalize_requested_topics(["/cam0", " /cam1 "]) == ["/cam0", "/cam1"]
        assert MCDLoader._normalize_requested_topics(None) is None


class TestTopicListing:
    """Tests for MCD topic inspection helpers."""

    def test_infers_topic_roles(self):
        """Should map message types to image/lidar/imu roles."""
        assert MCDLoader._infer_topic_role("/d455t/color/image_raw", "sensor_msgs/msg/Image") == "image"
        assert MCDLoader._infer_topic_role("/os_cloud_node/points", "sensor_msgs/msg/PointCloud2") == "lidar"
        assert MCDLoader._infer_topic_role("/vn200/imu", "sensor_msgs/msg/Imu") == "imu"
        assert MCDLoader._infer_topic_role("/gnss/fix", "sensor_msgs/msg/NavSatFix") == "gnss"
        assert MCDLoader._infer_topic_role("/cam/camera_info", "sensor_msgs/msg/CameraInfo") == "calibration"
        assert MCDLoader._infer_topic_role("/misc/topic", "std_msgs/msg/String") == "other"


class TestCameraCalibrationHelpers:
    """Tests for CameraInfo topic naming and PINHOLE extraction."""

    def test_infer_camera_info_topic(self):
        """Should map raw_image topics to sibling camera_info."""
        assert (
            MCDLoader._infer_camera_info_topic("/lucid_vision/camera_0/raw_image")
            == "/lucid_vision/camera_0/camera_info"
        )

    def test_camera_info_to_pinhole_dict(self):
        """Should read intrinsics from CameraInfo K matrix."""
        msg = SimpleNamespace(
            width=1920,
            height=1080,
            k=(900.0, 0.0, 960.5, 0.0, 901.0, 540.5, 0.0, 0.0, 1.0),
            header=SimpleNamespace(frame_id="/camera_optical"),
        )
        d = MCDLoader._camera_info_to_pinhole_dict(msg)
        assert d == {
            "width": 1920,
            "height": 1080,
            "fx": 900.0,
            "fy": 901.0,
            "cx": 960.5,
            "cy": 540.5,
            "frame_id": "/camera_optical",
        }

    def test_list_topics_reports_role_and_default_status(self, monkeypatch, tmp_path):
        """Should summarize bag topics from AnyReader."""
        bag_dir = tmp_path / "session"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:")

        class FakeReader:
            def __init__(self, paths):
                assert paths == [bag_dir]
                self.topics = {
                    "/misc/topic": SimpleNamespace(msgtype="std_msgs/msg/String", msgcount=2),
                    "/os_cloud_node/points": SimpleNamespace(msgtype="sensor_msgs/msg/PointCloud2", msgcount=10),
                    "/d455t/color/image_raw": SimpleNamespace(msgtype="sensor_msgs/msg/Image", msgcount=5),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))

        topics = MCDLoader(bag_dir).list_topics()

        assert topics == [
            {
                "topic": "/d455t/color/image_raw",
                "msgtype": "sensor_msgs/msg/Image",
                "msgcount": 5,
                "role": "image",
                "is_preferred_default": True,
            },
            {
                "topic": "/misc/topic",
                "msgtype": "std_msgs/msg/String",
                "msgcount": 2,
                "role": "other",
                "is_preferred_default": False,
            },
            {
                "topic": "/os_cloud_node/points",
                "msgtype": "sensor_msgs/msg/PointCloud2",
                "msgcount": 10,
                "role": "lidar",
                "is_preferred_default": True,
            },
        ]


class TestReaderCreation:
    """Tests for constructing AnyReader instances with typestore fallback."""

    def test_create_reader_passes_default_typestore_for_rosbag2(self, tmp_path):
        """Should provide a default ROS2 typestore for rosbag2 directories."""
        pytest.importorskip("rosbags.typesys")
        bag_dir = tmp_path / "session"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:")

        class FakeReader:
            def __init__(self, paths, *, default_typestore=None):
                assert paths == [bag_dir]
                assert default_typestore is not None
                self.default_typestore = default_typestore

        reader = MCDLoader._create_reader(FakeReader, [bag_dir])

        assert reader.default_typestore is not None

    def test_create_reader_falls_back_when_reader_rejects_typestore(self, tmp_path):
        """Should retry without kwargs for tests or older reader shims."""
        bag_dir = tmp_path / "session"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:")

        class FakeReader:
            def __init__(self, paths):
                assert paths == [bag_dir]
                self.paths = paths

        reader = MCDLoader._create_reader(FakeReader, [bag_dir])

        assert reader.paths == [bag_dir]


class TestFrameExtraction:
    """Tests for image extraction from rosbag readers."""

    def test_extract_frames_supports_multiple_topics(self, monkeypatch, tmp_path):
        """Should write each requested topic into its own image subdirectory."""
        bag_dir = tmp_path / "session"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:")

        conn0 = SimpleNamespace(topic="/cam0", msgtype="sensor_msgs/msg/Image")
        conn1 = SimpleNamespace(topic="/cam1", msgtype="sensor_msgs/msg/Image")

        class FakeReader:
            def __init__(self, paths):
                assert paths == [bag_dir]
                self.topics = {
                    "/cam0": SimpleNamespace(connections=[conn0]),
                    "/cam1": SimpleNamespace(connections=[conn1]),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def messages(self, connections):
                assert connections == [conn0, conn1]
                yield conn0, 0, b"cam0-0"
                yield conn1, 1, b"cam1-0"
                yield conn0, 2, b"cam0-1"
                yield conn1, 3, b"cam1-1"

            def deserialize(self, rawdata, msgtype):
                del msgtype
                return SimpleNamespace(payload=rawdata)

        monkeypatch.setattr(MCDLoader, "_get_anyreader", staticmethod(lambda: FakeReader))
        monkeypatch.setattr(
            MCDLoader,
            "_decode_image_message",
            classmethod(lambda cls, msg, msgtype: (np.zeros((2, 2, 3), dtype=np.uint8), ".jpg")),
        )

        output = Path(
            MCDLoader(bag_dir).extract_frames(
                output_dir=tmp_path / "out",
                image_topic="/cam0,/cam1",
                max_frames=2,
                every_n=1,
            )
        )

        assert output == tmp_path / "out" / "images"
        assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*.jpg")) == [
            "cam0/frame_000000.jpg",
            "cam0/frame_000001.jpg",
            "cam1/frame_000000.jpg",
            "cam1/frame_000001.jpg",
        ]


class TestImageDecoding:
    """Tests for converting ROS image messages into OpenCV arrays."""

    def test_decodes_rgb8_image_and_converts_to_bgr(self):
        """Should convert raw rgb8 data into BGR order for OpenCV output."""
        rgb = np.array([[[255, 10, 20], [0, 40, 50]]], dtype=np.uint8)
        msg = SimpleNamespace(
            height=1,
            width=2,
            encoding="rgb8",
            step=6,
            data=rgb.tobytes(),
        )

        image, extension = MCDLoader._decode_image_message(msg, "sensor_msgs/msg/Image")

        assert extension == ".jpg"
        assert image.shape == (1, 2, 3)
        assert image[0, 0].tolist() == [20, 10, 255]

    def test_decodes_compressed_image(self):
        """Should decode compressed image bytes through OpenCV."""
        bgr = np.full((2, 3, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        assert ok is True
        msg = SimpleNamespace(data=encoded.tobytes())

        image, extension = MCDLoader._decode_image_message(msg, "sensor_msgs/msg/CompressedImage")

        assert extension == ".jpg"
        assert image.shape == (2, 3, 3)


class TestPointCloudConversion:
    """Tests for PointCloud2 decoding."""

    def test_converts_pointcloud2_to_numpy(self):
        """Should decode x/y/z/intensity fields into an ``Nx4`` float array."""
        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "intensity"],
                "formats": ["<f4", "<f4", "<f4", "<f4"],
                "offsets": [0, 4, 8, 12],
                "itemsize": 16,
            }
        )
        points = np.array(
            [
                (1.0, 2.0, 3.0, 0.5),
                (4.0, 5.0, 6.0, 0.75),
            ],
            dtype=dtype,
        )
        msg = SimpleNamespace(
            is_bigendian=False,
            point_step=16,
            width=2,
            height=1,
            data=points.tobytes(),
            fields=[
                SimpleNamespace(name="x", offset=0, datatype=7, count=1),
                SimpleNamespace(name="y", offset=4, datatype=7, count=1),
                SimpleNamespace(name="z", offset=8, datatype=7, count=1),
                SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
            ],
        )

        array = MCDLoader._pointcloud2_to_numpy(msg)

        assert array.shape == (2, 4)
        assert np.allclose(array[0], [1.0, 2.0, 3.0, 0.5])
        assert np.allclose(array[1], [4.0, 5.0, 6.0, 0.75])


def _write_tum(path: Path, rows: list[tuple[float, tuple[float, float, float], tuple[float, float, float, float]]]):
    with open(path, "w") as f:
        for ts, (tx, ty, tz), (qx, qy, qz, qw) in rows:
            f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


def _write_timestamps_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "timestamp_ns"])
        w.writerows(rows)


class TestMergeLidarFramesToWorld:
    """Tests for merging per-frame LiDAR NPYs using a TUM trajectory."""

    def test_identity_with_timestamps_csv_uses_inferred_yaw(self, tmp_path):
        """With identity quaternions, yaw is inferred from successive ENU positions (east motion => yaw=0)."""
        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()
        np.save(lidar_dir / "frame_000000.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        np.save(lidar_dir / "frame_000001.npy", np.array([[2.0, 0.0, 0.0]], dtype=np.float32))
        _write_timestamps_csv(
            lidar_dir / "timestamps.csv",
            [("frame_000000.npy", 0), ("frame_000001.npy", 1_000_000_000)],
        )

        tum = tmp_path / "traj.tum"
        _write_tum(
            tum,
            [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (1.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ],
        )

        out = tmp_path / "world.npy"
        loader = MCDLoader(tmp_path)
        loader.merge_lidar_frames_to_world(
            lidar_dir=lidar_dir,
            trajectory_path=tum,
            output_path=out,
        )

        merged = np.load(out)
        assert merged.dtype == np.float32
        assert merged.shape[0] == 2
        # East motion => yaw=0 => identity rotation; base frame origin translated.
        np.testing.assert_allclose(merged[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(merged[1], [12.0, 0.0, 0.0], atol=1e-6)

    def test_inferred_yaw_north_motion(self, tmp_path):
        """North motion along +y ENU => yaw=pi/2: x_base points to +North."""
        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()
        np.save(lidar_dir / "frame_000000.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        np.save(lidar_dir / "frame_000001.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        _write_timestamps_csv(
            lidar_dir / "timestamps.csv",
            [("frame_000000.npy", 0), ("frame_000001.npy", 1_000_000_000)],
        )

        tum = tmp_path / "traj.tum"
        _write_tum(
            tum,
            [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (1.0, (0.0, 10.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ],
        )

        out = tmp_path / "world.npy"
        MCDLoader(tmp_path).merge_lidar_frames_to_world(
            lidar_dir=lidar_dir,
            trajectory_path=tum,
            output_path=out,
        )
        merged = np.load(out)
        np.testing.assert_allclose(merged[0], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(merged[1], [0.0, 11.0, 0.0], atol=1e-6)

    def test_sequential_fallback_warns_when_timestamps_csv_missing(self, tmp_path):
        """Without timestamps.csv, logger.warning fires and points are still returned."""
        import logging

        from gs_sim2real.datasets import mcd as mcd_module

        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()
        np.save(lidar_dir / "frame_000000.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        np.save(lidar_dir / "frame_000001.npy", np.array([[2.0, 0.0, 0.0]], dtype=np.float32))

        tum = tmp_path / "traj.tum"
        _write_tum(
            tum,
            [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (1.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ],
        )

        out = tmp_path / "world.npy"
        captured: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda rec: captured.append(rec.getMessage())
        handler.setLevel(logging.WARNING)
        mcd_module.logger.addHandler(handler)
        try:
            MCDLoader(tmp_path).merge_lidar_frames_to_world(
                lidar_dir=lidar_dir,
                trajectory_path=tum,
                output_path=out,
            )
        finally:
            mcd_module.logger.removeHandler(handler)
        merged = np.load(out)
        assert merged.shape[0] == 2
        assert any("timestamps.csv" in msg for msg in captured)

    def test_t_base_lidar_applies_rotation(self, tmp_path):
        """T_base_lidar with a 90-degree roll should rotate LiDAR points before world transform."""
        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()
        np.save(lidar_dir / "frame_000000.npy", np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        _write_timestamps_csv(lidar_dir / "timestamps.csv", [("frame_000000.npy", 0)])

        tum = tmp_path / "traj.tum"
        _write_tum(
            tum,
            [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (1.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ],
        )

        T_base_lidar = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        out = tmp_path / "world.npy"
        MCDLoader(tmp_path).merge_lidar_frames_to_world(
            lidar_dir=lidar_dir,
            trajectory_path=tum,
            output_path=out,
            T_base_lidar=T_base_lidar,
        )
        merged = np.load(out)
        # T_base_lidar rotates (0,1,0) -> (0,0,1); vehicle yaw=0 (east motion) keeps that as-is in world.
        np.testing.assert_allclose(merged[0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_raises_when_no_lidar_frames(self, tmp_path):
        """Should raise FileNotFoundError when no frame_*.npy files exist."""
        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()

        tum = tmp_path / "traj.tum"
        _write_tum(tum, [(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))])

        with pytest.raises(FileNotFoundError):
            MCDLoader(tmp_path).merge_lidar_frames_to_world(
                lidar_dir=lidar_dir,
                trajectory_path=tum,
                output_path=tmp_path / "world.npy",
            )

    def test_max_points_subsamples(self, tmp_path):
        """max_points caps the merged point count with deterministic RNG."""
        lidar_dir = tmp_path / "lidar"
        lidar_dir.mkdir()
        pts = np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32)
        np.save(lidar_dir / "frame_000000.npy", pts)
        _write_timestamps_csv(lidar_dir / "timestamps.csv", [("frame_000000.npy", 0)])

        tum = tmp_path / "traj.tum"
        _write_tum(
            tum,
            [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                (1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ],
        )

        out = tmp_path / "world.npy"
        MCDLoader(tmp_path).merge_lidar_frames_to_world(
            lidar_dir=lidar_dir,
            trajectory_path=tum,
            output_path=out,
            max_points=100,
        )
        merged = np.load(out)
        assert merged.shape[0] == 100
