#!/usr/bin/env python3
"""Merge two or more MCD pose-seeded sparse outputs into a single COLMAP model.

All inputs must have been preprocessed with a common ENU origin
(``--mcd-reference-origin`` or ``--mcd-reference-bag``), so their
``images.txt`` / ``points3D.txt`` coordinates already agree. This script:

- concatenates ``sparse/0/points3D.txt`` (renumbering point IDs),
- concatenates ``sparse/0/images.txt`` (renumbering image IDs and prefixing
  ``bag_tag/`` to the image ``NAME`` column to avoid collisions),
- unions ``sparse/0/cameras.txt`` (renumbering camera IDs),
- copies the actual image files into the output's ``images/<bag_tag>/`` tree,
- (optionally) copies per-image ``depth/<bag_tag>/`` npy trees.

Usage:
    python3 scripts/merge_mcd_sparse.py \
        --inputs outputs/bag2_full outputs/bag4_full \
        --tags bag2 bag4 \
        --output outputs/multibag
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _load_cameras(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _load_images(path: Path) -> list[tuple[str, str]]:
    """Return list of (pose_line, name)."""
    out = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for ln in raw:
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 10:
            continue
        name = parts[-1]
        out.append((ln, name))
    return out


def _load_points(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Preprocessed bag output dirs")
    parser.add_argument("--tags", nargs="+", required=True, help="Bag tag per input (same length)")
    parser.add_argument("--output", required=True, help="Merged COLMAP output directory")
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Also copy <input>/depth/* into <output>/depth/<tag>/*",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.tags):
        parser.error(f"--inputs ({len(args.inputs)}) and --tags ({len(args.tags)}) length mismatch")

    out_dir = Path(args.output)
    sparse_out = out_dir / "sparse" / "0"
    images_out = out_dir / "images"
    sparse_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    cameras_lines: list[str] = []
    images_lines: list[str] = []
    points_lines: list[str] = []

    camera_offset = 0
    image_offset = 0
    point_offset = 0

    for in_dir, tag in zip(args.inputs, args.tags):
        in_path = Path(in_dir)
        in_sparse = in_path / "sparse" / "0"
        in_images = in_path / "images"
        in_cameras = in_sparse / "cameras.txt"
        in_images_txt = in_sparse / "images.txt"
        in_points_txt = in_sparse / "points3D.txt"
        for p in (in_cameras, in_images_txt, in_points_txt):
            if not p.is_file():
                raise FileNotFoundError(f"Missing {p}")

        # --- cameras.txt: renumber camera IDs (they're the first column).
        cam_remap: dict[int, int] = {}
        for line in _load_cameras(in_cameras):
            parts = line.split(maxsplit=1)
            old_id = int(parts[0])
            new_id = old_id + camera_offset
            cam_remap[old_id] = new_id
            cameras_lines.append(f"{new_id} {parts[1]}")
        camera_offset += max(cam_remap) if cam_remap else 0

        # --- images.txt: renumber image ID + camera ID, prefix name with tag/.
        bag_image_count = 0
        for ln, name in _load_images(in_images_txt):
            parts = ln.split()
            old_img_id = int(parts[0])
            old_cam_id = int(parts[-2])
            new_img_id = old_img_id + image_offset
            new_cam_id = cam_remap.get(old_cam_id, old_cam_id)
            prefixed = f"{tag}/{name}"
            new_line = " ".join([str(new_img_id)] + parts[1:-2] + [str(new_cam_id), prefixed])
            images_lines.append(new_line)
            images_lines.append("")  # COLMAP expects blank line (no tracks)
            bag_image_count += 1
        # COLMAP image IDs must stay unique across bags.
        image_offset += bag_image_count

        # --- points3D.txt: renumber point ID.
        bag_point_count = 0
        for line in _load_points(in_points_txt):
            parts = line.split(maxsplit=1)
            old_pid = int(parts[0])
            new_pid = old_pid + point_offset
            points_lines.append(f"{new_pid} {parts[1]}")
            bag_point_count += 1
        point_offset += bag_point_count

        # --- Copy image files into <output>/images/<tag>/<original-subdir>/*
        dst_root = images_out / tag
        for src in in_images.rglob("*"):
            if src.is_file():
                rel = src.relative_to(in_images)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

        # --- Optionally copy depth maps too.
        if args.include_depth:
            in_depth = in_path / "depth"
            if in_depth.is_dir():
                dst_depth = out_dir / "depth" / tag
                for src in in_depth.rglob("*"):
                    if src.is_file():
                        rel = src.relative_to(in_depth)
                        dst = dst_depth / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.copy2(src, dst)

        print(f"  [{tag}] cameras={len(cam_remap)}, images={bag_image_count}, points={bag_point_count}")

    (sparse_out / "cameras.txt").write_text("# Camera list\n" + "\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_out / "images.txt").write_text("# Image list\n" + "\n".join(images_lines) + "\n", encoding="utf-8")
    (sparse_out / "points3D.txt").write_text("# 3D point list\n" + "\n".join(points_lines) + "\n", encoding="utf-8")
    print(f"\nMerged COLMAP sparse written to {sparse_out}")
    print(f"Merged images under {images_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
