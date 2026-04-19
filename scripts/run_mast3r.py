#!/usr/bin/env python3
"""End-to-end MAST3R pose-free preprocessing.

Mirror of ``scripts/run_dust3r.py`` but for the MAST3R backend
(metric-aware descendant of DUSt3R). Runs MAST3R inference + sparse
global alignment on a directory of images and exports the result as a
COLMAP-text sparse model, ready for gsplat training. All real logic
lives in ``gs_sim2real.preprocess.pose_free``; this script is a thin CLI
wrapper around it.

Requires a local clone of ``naver/mast3r`` (with its ``dust3r`` + ``croco``
submodules). Point ``MAST3R_PATH`` at the clone root (or let the script
pick up ``/tmp/mast3r`` by default):

    export MAST3R_PATH=/tmp/mast3r
    python scripts/run_mast3r.py \\
        --image-dir outputs/bag6_colored/images/lucid_vision__camera_0__raw_image \\
        --output    outputs/bag6_mast3r \\
        --num-frames 20 \\
        --checkpoint $MAST3R_PATH/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAST3R and export a COLMAP-text sparse model.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--mast3r-root",
        type=Path,
        default=Path(os.environ.get("MAST3R_PATH", "/tmp/mast3r")),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir for MAST3R pairwise forward tensors (default: <output>/mast3r_cache)",
    )
    parser.add_argument("--num-frames", type=int, default=20, help="0 = keep all")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument(
        "--scene-graph",
        default="complete",
        help="MAST3R pair graph: 'complete' (all pairs, best quality; fits ~20 frames "
        "in 16 GB), 'swin-N' (sliding window of N), or 'oneref-K' (anchor to view K).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=8,
        help="Sparse GA pointcloud subsample stride (lower = denser pts3d, heavier).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

    checkpoint = args.checkpoint or (
        args.mast3r_root / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )

    processor = PoseFreeProcessor(
        method="mast3r",
        checkpoint=checkpoint,
        mast3r_root=args.mast3r_root,
        mast3r_cache=args.cache_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device,
        scene_graph=args.scene_graph,
        mast3r_subsample=args.subsample,
        max_points=args.max_points,
    )
    processor.estimate_poses(args.image_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
