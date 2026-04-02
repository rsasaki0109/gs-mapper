#!/usr/bin/env python3
"""Run a demo of the full gs-sim2real pipeline.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --images /path/to/images
    python scripts/run_demo.py --iterations 1000 --no-viewer
    python scripts/run_demo.py --sample-images --num-images 20

This script runs the entire pipeline:
1. Download sample images (or use user-provided images)
2. Run COLMAP preprocessing
3. Train a 3DGS model (short training for demo)
4. Launch the web viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    """Run the full demo pipeline."""
    parser = argparse.ArgumentParser(description="Run gs-sim2real demo pipeline")
    parser.add_argument("--images", default=None, help="Path to input images directory")
    parser.add_argument("--output", default=None, help="Output directory (default: outputs/demo)")
    parser.add_argument(
        "--method",
        default="gsplat",
        choices=["gsplat", "nerfstudio"],
        help="Training method (default: gsplat)",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations (default: 1000 for demo)")
    parser.add_argument("--sample-images", action="store_true", help="Download sample images for demo")
    parser.add_argument("--num-images", type=int, default=10, help="Number of sample images (default: 10)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip COLMAP preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--no-viewer", action="store_true", help="Don't launch viewer at the end")
    parser.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else project_root / "outputs" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    colmap_dir = output_dir / "colmap"
    train_dir = output_dir / "train"

    print("=" * 60)
    print("  gs-sim2real Demo Pipeline")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Training method: {args.method}")
    print(f"  Iterations: {args.iterations}")
    print()

    # Step 1: Get images
    if args.images:
        images_dir = Path(args.images)
        if not images_dir.exists():
            print(f"Error: Image directory not found: {images_dir}")
            sys.exit(1)
        print(f"Using provided images from: {images_dir}")
    elif args.sample_images:
        print("Step 1: Downloading sample images...")
        from gs_sim2real.common.download import download_sample_images

        images_dir = download_sample_images(
            output_dir / "sample_data",
            num_images=args.num_images,
        )
    else:
        print("No images specified. Use --images <dir> or --sample-images.")
        print("\nExample:")
        print(f"  python {sys.argv[0]} --sample-images")
        print(f"  python {sys.argv[0]} --images /path/to/your/images")
        sys.exit(1)

    # Count images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
    print(f"  Found {len(image_files)} images\n")

    if len(image_files) < 3:
        print("Warning: At least 3 images are recommended for COLMAP reconstruction.")

    # Step 2: COLMAP preprocessing
    if not args.skip_preprocess:
        print("=" * 60)
        print("  Step 2: COLMAP Preprocessing")
        print("=" * 60)
        try:
            from gs_sim2real.preprocess.colmap import run_colmap

            sparse_dir = run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
            )
            print(f"  Sparse model at: {sparse_dir}\n")
        except FileNotFoundError as e:
            print(f"\n  COLMAP not available: {e}")
            print("  Skipping preprocessing. Training requires COLMAP output.")
            if not args.skip_train:
                print("  Use --skip-preprocess with pre-existing COLMAP data.")
                sys.exit(1)
    else:
        print("Skipping preprocessing (--skip-preprocess)\n")

    # Step 3: Training
    ply_path = None
    if not args.skip_train:
        print("=" * 60)
        print("  Step 3: Training")
        print("=" * 60)

        if args.method == "gsplat":
            try:
                from gs_sim2real.train.gsplat_trainer import train_gsplat

                ply_path = train_gsplat(
                    data_dir=colmap_dir,
                    output_dir=train_dir,
                    num_iterations=args.iterations,
                )
                print(f"  Model saved to: {ply_path}\n")
            except ImportError as e:
                print(f"  Error: {e}")
                sys.exit(1)
            except FileNotFoundError as e:
                print(f"  Error: {e}")
                print("  Make sure COLMAP preprocessing completed successfully.")
                sys.exit(1)
        else:
            try:
                from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

                train_nerfstudio(
                    data_dir=colmap_dir,
                    output_dir=train_dir,
                )
            except ImportError as e:
                print(f"  Error: {e}")
                sys.exit(1)
    else:
        print("Skipping training (--skip-train)\n")
        # Try to find existing PLY
        ply_candidates = list(train_dir.glob("*.ply"))
        if ply_candidates:
            ply_path = ply_candidates[0]
            print(f"  Using existing model: {ply_path}")

    # Step 4: Viewer
    if not args.no_viewer and ply_path is not None:
        print("=" * 60)
        print("  Step 4: Launching Viewer")
        print("=" * 60)
        from gs_sim2real.viewer.web_viewer import launch_viewer

        launch_viewer(ply_path, port=args.port)
    elif ply_path is None:
        print("No model to view.")

    print("\nDemo pipeline complete!")


if __name__ == "__main__":
    main()
