"""Tests for point-cloud loading utilities."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from gs_sim2real.preprocess.depth_from_lidar import load_pointcloud, project_lidar_to_image


def test_load_pointcloud_reads_binary_ply_with_uchar_rgb(tmp_path: Path) -> None:
    ply_path = tmp_path / "rgb_binary.ply"
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = struct.pack(
        "<fffBBBfffBBB",
        1.25,
        -2.5,
        3.0,
        10,
        20,
        30,
        4.5,
        5.5,
        6.5,
        40,
        50,
        60,
    )
    ply_path.write_bytes(header + body)

    points = load_pointcloud(ply_path)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [1.25, -2.5, 3.0, 10.0, 20.0, 30.0],
                [4.5, 5.5, 6.5, 40.0, 50.0, 60.0],
            ],
            dtype=np.float64,
        ),
    )


def test_load_pointcloud_reads_ascii_ply_with_rgb(tmp_path: Path) -> None:
    ply_path = tmp_path / "rgb_ascii.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "1.0 2.0 3.0 11 22 33",
                "4.0 5.0 6.0 44 55 66",
                "",
            ]
        ),
        encoding="ascii",
    )

    points = load_pointcloud(ply_path)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [1.0, 2.0, 3.0, 11.0, 22.0, 33.0],
                [4.0, 5.0, 6.0, 44.0, 55.0, 66.0],
            ],
            dtype=np.float64,
        ),
    )


def test_project_lidar_to_image_ignores_extra_point_channels() -> None:
    points = np.array(
        [
            [0.0, 0.0, 2.0, 10.0, 20.0, 30.0],
            [2.0, 0.0, 2.0, 40.0, 50.0, 60.0],
        ],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    depth = project_lidar_to_image(points, np.eye(4), intrinsic, height=4, width=4)

    assert depth[1, 1] == 2.0
    assert depth[1, 2] == 2.0
