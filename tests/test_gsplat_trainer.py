"""Tests for gsplat trainer data loading helpers."""

from __future__ import annotations

from pathlib import Path

from gs_sim2real.train.gsplat_trainer import GsplatTrainer


def test_load_images_txt_preserves_entries_with_blank_points_lines(tmp_path: Path) -> None:
    images_txt = tmp_path / "images.txt"
    images_txt.write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 frame_000000.jpg",
                "",
                "2 1 0 0 0 1 0 0 1 frame_000001.jpg",
                "",
                "3 1 0 0 0 2 0 0 1 nested/frame_000002.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )

    images = GsplatTrainer._load_images_txt(object(), images_txt)

    assert list(images) == [1, 2, 3]
    assert images[2]["tvec"] == [1.0, 0.0, 0.0]
    assert images[3]["name"] == "nested/frame_000002.jpg"
