"""Tests for ROS TF helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gs_sim2real.datasets.ros_tf import (
    HybridTfLookup,
    StaticTfMap,
    TimestampedTfEdges,
    geometry_transform_to_matrix,
    load_static_calibration_yaml,
    merge_static_tf_maps,
    normalize_frame_id,
)


class TestNormalizeFrameId:
    def test_strips_leading_slash(self) -> None:
        assert normalize_frame_id("/base_link") == "base_link"


class TestGeometryTransform:
    def test_identity_transform(self) -> None:
        t = SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        T = geometry_transform_to_matrix(t)
        assert np.allclose(T, np.eye(4))


class TestStaticTfMap:
    def test_chains_transforms(self) -> None:
        """map <- base <- cam  =>  lookup(map, cam) = T_map_base @ T_base_cam."""
        T_map_base = np.eye(4)
        T_map_base[0, 3] = 10.0

        T_base_cam = np.eye(4)
        T_base_cam[2, 3] = 2.0

        m = StaticTfMap()
        m.add("map", "base", T_map_base)
        m.add("base", "cam", T_base_cam)

        T = m.lookup("map", "cam")
        assert T is not None
        expected = T_map_base @ T_base_cam
        assert np.allclose(T, expected)

    def test_lookup_missing_returns_none(self) -> None:
        m = StaticTfMap()
        m.add("a", "b", np.eye(4))
        assert m.lookup("x", "b") is None

    def test_get_parent_and_transform(self) -> None:
        T = np.eye(4)
        T[0, 3] = 3.0
        m = StaticTfMap()
        m.add("map", "cam", T)
        p, T2 = m.get_parent_and_transform("cam")
        assert p == "map"
        assert np.allclose(T2, T)


class TestTimestampedTfEdges:
    def test_nearest_picks_closest_stamp(self) -> None:
        e = TimestampedTfEdges()
        T0 = np.eye(4)
        T1 = np.eye(4)
        T1[1, 3] = 1.0
        e.add(1_000_000_000, "base", "cam", T0)
        e.add(2_000_000_000, "base", "cam", T1)
        e.finalize()
        got = e.nearest("base", "cam", 1_100_000_000)
        assert got is not None and np.allclose(got, T0)


class TestHybridTfLookup:
    def test_prefers_dynamic_sample_when_present(self) -> None:
        static = StaticTfMap()
        Ts = np.eye(4)
        Ts[2, 3] = 0.5
        static.add("base", "cam", Ts)

        dyn = TimestampedTfEdges()
        Td = np.eye(4)
        Td[2, 3] = 9.0
        dyn.add(1_000_000_000, "base", "cam", Td)
        dyn.finalize()

        h = HybridTfLookup(static, dyn)
        T = h.lookup("base", "cam", 1_000_000_000)
        assert T is not None and np.allclose(T, Td)


# Trimmed fixture modelled on the real MCDVIRAL handheld YAML (pulled from
# https://mcdviral.github.io/Download.html via scripts/download_mcd_calibration.sh
# — see tests/test_download_mcd_calibration_script.py for the upstream pin).
_MCD_HANDHELD_YAML_FIXTURE = """\
body:
    d455b_color:
        T:
        - [-0.0005063667779342879, 0.004390860953902336, 0.999990231918679, 0.0682063644957161]
        - [0.9999682435112625, 0.007955484179591332, 0.0004714238775566064, 0.004985284223667225]
        - [-0.007953336513078169, 0.9999587144535296, -0.0043947499084668095, 0.09386005406130697]
        - [0.0, 0.0, 0.0, 1.0]
        camera_model: pinhole
        distortion_coeffs: [-0.0438, 0.0358, -0.0018, 0.0005]
        distortion_model: radtan
        intrinsics: [387.14, 387.01, 324.73, 238.67]
        resolution: [640, 480]
        rostopic: /d455b/color/image_raw
        timeshift_cam_imu: -0.0014
    mid70:
        T:
        - [1.0, 0.0, 0.0, 0.2]
        - [0.0, 1.0, 0.0, -0.1]
        - [0.0, 0.0, 1.0, 0.0]
        - [0.0, 0.0, 0.0, 1.0]
        rostopic: /livox/lidar
    os_sensor:
        T:
        - [0.9, -0.4, 0.0, 0.15]
        - [0.4, 0.9, 0.0, -0.05]
        - [0.0, 0.0, 1.0, 0.3]
        - [0.0, 0.0, 0.0, 1.0]
        rostopic: /os1_cloud_node/points
"""


class TestLoadStaticCalibrationYaml:
    def test_parses_mcdviral_handheld_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "handheld.yaml"
        p.write_text(_MCD_HANDHELD_YAML_FIXTURE, encoding="utf-8")
        m = load_static_calibration_yaml(p)
        # One edge per sensor — three in this fixture.
        assert len(m) == 3
        for sensor in ("d455b_color", "mid70", "os_sensor"):
            edge = m.get_parent_and_transform(sensor)
            assert edge is not None, f"missing {sensor}"
            parent, T = edge
            # Default parent is "body" so the YAML can be plugged in as-is for
            # MCD sessions where we then set --mcd-base-frame=body.
            assert parent == "body"
            assert T.shape == (4, 4)
            assert np.allclose(T[3, :], [0, 0, 0, 1])

    def test_base_frame_override_relabels_parent(self, tmp_path: Path) -> None:
        p = tmp_path / "handheld.yaml"
        p.write_text(_MCD_HANDHELD_YAML_FIXTURE, encoding="utf-8")
        m = load_static_calibration_yaml(p, base_frame="base_link")
        # Every loaded edge should now chain under base_link so lookups that
        # expect MCD's default base_frame still work without renaming anything.
        for sensor in ("d455b_color", "mid70", "os_sensor"):
            parent, _ = m.get_parent_and_transform(sensor)
            assert parent == "base_link"

    def test_lookup_round_trips_through_yaml_T(self, tmp_path: Path) -> None:
        p = tmp_path / "handheld.yaml"
        p.write_text(_MCD_HANDHELD_YAML_FIXTURE, encoding="utf-8")
        m = load_static_calibration_yaml(p)
        # The mid70 T has a pure translation; lookup("body","mid70") should
        # recover it exactly (no chain involved).
        T = m.lookup("body", "mid70")
        assert T is not None
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, -0.1],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert np.allclose(T, expected)

    def test_missing_body_section_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("not_body: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="body:"):
            load_static_calibration_yaml(p)

    def test_skips_malformed_T_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "mixed.yaml"
        p.write_text(
            "body:\n"
            "    good:\n"
            "        T:\n"
            "        - [1.0, 0.0, 0.0, 0.0]\n"
            "        - [0.0, 1.0, 0.0, 0.0]\n"
            "        - [0.0, 0.0, 1.0, 0.0]\n"
            "        - [0.0, 0.0, 0.0, 1.0]\n"
            "    wrong_shape:\n"
            "        T:\n"
            "        - [1.0, 2.0, 3.0]\n"
            "    no_T_field:\n"
            "        rostopic: /foo\n",
            encoding="utf-8",
        )
        m = load_static_calibration_yaml(p)
        # Only `good` should make it in; the other two are skipped with a warning.
        assert len(m) == 1
        assert m.get_parent_and_transform("good") is not None
        assert m.get_parent_and_transform("wrong_shape") is None
        assert m.get_parent_and_transform("no_T_field") is None


class TestMergeStaticTfMaps:
    def test_later_maps_win_on_child_collision(self) -> None:
        """merge order mirrors the preprocess wiring — bag-derived edges override YAML."""
        a = StaticTfMap()
        Ta = np.eye(4)
        Ta[0, 3] = 1.0
        a.add("body", "cam", Ta)

        b = StaticTfMap()
        Tb = np.eye(4)
        Tb[0, 3] = 2.0
        b.add("body", "cam", Tb)

        merged = merge_static_tf_maps(a, b)
        # b was passed second → it wins. In the CLI wiring we pass the
        # calibration-YAML map first and the bag-derived map second, so bag
        # values override the fallback YAML when both exist.
        _, T = merged.get_parent_and_transform("cam")
        assert np.allclose(T, Tb)

    def test_preserves_edges_from_both_maps(self) -> None:
        a = StaticTfMap()
        a.add("body", "cam0", np.eye(4))
        b = StaticTfMap()
        b.add("body", "cam1", np.eye(4))
        merged = merge_static_tf_maps(a, b)
        assert len(merged) == 2

    def test_skips_none_inputs(self) -> None:
        a = StaticTfMap()
        a.add("body", "cam", np.eye(4))
        merged = merge_static_tf_maps(a, None)  # type: ignore[arg-type]
        assert len(merged) == 1
