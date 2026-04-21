"""Tests for ROS TF helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


class TestMergeStaticTfMaps:
    def test_latter_map_overrides_child(self) -> None:
        a = StaticTfMap()
        a.add("base", "cam", np.eye(4))
        b = StaticTfMap()
        Tb = np.eye(4)
        Tb[0, 3] = 5.0
        b.add("base", "cam", Tb)
        m = merge_static_tf_maps(a, b)
        T = m.lookup("base", "cam")
        assert T is not None and np.allclose(T[0, 3], 5.0)


class TestLoadStaticCalibrationYaml:
    def test_loads_body_edges(self, tmp_path) -> None:
        p = tmp_path / "calib.yaml"
        p.write_text(
            "body:\n  mycam:\n    T:\n    - [1, 0, 0, 1]\n    - [0, 1, 0, 0]\n    - [0, 0, 1, 0]\n    - [0, 0, 0, 1]\n",
            encoding="utf-8",
        )
        m = load_static_calibration_yaml(p, base_frame="base_link")
        T = m.lookup("base_link", "mycam")
        assert T is not None and np.allclose(T[0, 3], 1.0)

    def test_missing_body_raises(self, tmp_path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("foo: {}\n", encoding="utf-8")
        try:
            load_static_calibration_yaml(p)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


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
