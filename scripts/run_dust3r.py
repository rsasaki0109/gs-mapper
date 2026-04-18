#!/usr/bin/env python3
"""End-to-end DUSt3R pose-free preprocessing.

Runs DUSt3R inference + global alignment on a directory of images and exports
the result as a COLMAP-text sparse model, ready for gsplat training. All real
logic lives in ``gs_sim2real.preprocess.pose_free``; this script is a thin CLI
wrapper around it.

Requires a local clone of ``naver/dust3r`` and its ``croco`` submodule. Point
``DUST3R_PATH`` at the clone root (or let the script pick up ``/tmp/dust3r``
by default):

    export DUST3R_PATH=/tmp/dust3r
    python scripts/run_dust3r.py \\
        --image-dir outputs/mcd_ntu17/images \\
        --output    outputs/mcd_ntu17_dust3r \\
        --num-frames 20 \\
        --checkpoint $DUST3R_PATH/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DUSt3R and export a COLMAP-text sparse model.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dust3r-root", type=Path, default=Path(os.environ.get("DUST3R_PATH", "/tmp/dust3r")))
    parser.add_argument("--num-frames", type=int, default=30, help="0 = keep all")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align-iters", type=int, default=300)
    parser.add_argument("--align-lr", type=float, default=0.01)
    parser.add_argument("--align-schedule", default="cosine")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument(
        "--scene-graph",
        default="complete",
        help="DUSt3R pair graph: 'complete' (all pairs, best quality; fits ~20 frames "
        "in 16 GB), 'swin-N' (sliding window of N), or 'oneref-K' (anchor to view K).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from gs_sim2real.preprocess.pose_free import PoseFreeProcessor

    checkpoint = args.checkpoint or (args.dust3r_root / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    processor = PoseFreeProcessor(
        method="dust3r",
        checkpoint=checkpoint,
        dust3r_root=args.dust3r_root,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device,
        align_iters=args.align_iters,
        align_lr=args.align_lr,
        align_schedule=args.align_schedule,
        scene_graph=args.scene_graph,
        max_points=args.max_points,
    )
    processor.estimate_poses(args.image_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
