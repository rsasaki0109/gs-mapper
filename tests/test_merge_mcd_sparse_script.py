"""Smoke tests for scripts/merge_mcd_sparse.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "merge_mcd_sparse.py"


def _make_fake_bag(root: Path, tag: str, img_id_base: int, pt_id_base: int) -> None:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    (images / "cam_a").mkdir(parents=True, exist_ok=True)

    (sparse / "cameras.txt").write_text("# Camera list\n1 PINHOLE 640 480 400.0 400.0 320.0 240.0\n", encoding="utf-8")
    # image lines: COLMAP expects blank line between entries (no track row written here).
    img_lines = [
        f"{img_id_base + 0} 1 0 0 0 {float(img_id_base)} 0 0 1 cam_a/frame_000000.jpg",
        "",
        f"{img_id_base + 1} 1 0 0 0 {float(img_id_base + 1)} 0 0 1 cam_a/frame_000001.jpg",
        "",
    ]
    (sparse / "images.txt").write_text("# Image list\n" + "\n".join(img_lines) + "\n", encoding="utf-8")
    pt_lines = [
        f"{pt_id_base + 0} 1.0 2.0 3.0 128 128 128 0.0",
        f"{pt_id_base + 1} 4.0 5.0 6.0 200 100 50 0.0",
    ]
    (sparse / "points3D.txt").write_text("# 3D point list\n" + "\n".join(pt_lines) + "\n", encoding="utf-8")
    # create the actual image stubs so the copy step has something to grab
    (images / "cam_a" / "frame_000000.jpg").write_bytes(b"stub")
    (images / "cam_a" / "frame_000001.jpg").write_bytes(b"stub")


def test_merge_concatenates_two_bags_with_renumbering(tmp_path: Path) -> None:
    bag_a = tmp_path / "bag_a"
    bag_b = tmp_path / "bag_b"
    out = tmp_path / "merged"
    _make_fake_bag(bag_a, "a", img_id_base=1, pt_id_base=1)
    _make_fake_bag(bag_b, "b", img_id_base=1, pt_id_base=1)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--inputs",
            str(bag_a),
            str(bag_b),
            "--tags",
            "a",
            "b",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    images_txt = (out / "sparse" / "0" / "images.txt").read_text(encoding="utf-8")
    assert "a/cam_a/frame_000000.jpg" in images_txt
    assert "b/cam_a/frame_000000.jpg" in images_txt
    # image IDs should not collide
    ids = [
        line.split()[0]
        for line in images_txt.splitlines()
        if line and not line.startswith("#") and len(line.split()) >= 10
    ]
    assert len(ids) == len(set(ids))

    points_txt = (out / "sparse" / "0" / "points3D.txt").read_text(encoding="utf-8")
    pids = [line.split()[0] for line in points_txt.splitlines() if line and not line.startswith("#")]
    assert len(pids) == len(set(pids)) == 4  # 2 per bag

    # Images copied with bag tag prefix.
    assert (out / "images" / "a" / "cam_a" / "frame_000000.jpg").is_file()
    assert (out / "images" / "b" / "cam_a" / "frame_000000.jpg").is_file()
