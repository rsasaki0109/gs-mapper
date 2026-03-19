"""Command-line interface for nerf-gs-playground.

Provides subcommands for the full 3DGS pipeline:
- download: Download datasets from supported sources
- preprocess: Run COLMAP or frame extraction on raw data
- train: Train a 3DGS model using gsplat or nerfstudio
- view: Launch the web viewer for a trained model
- run: Run the full pipeline end-to-end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="gs-playground",
        description="Multi-dataset 3D Gaussian Splatting reconstruction playground",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download
    dl = subparsers.add_parser("download", help="Download a dataset")
    dl.add_argument("--dataset", default=None, help="Dataset name (e.g. ggrt, covla, mcd)")
    dl.add_argument("--output", default=None, help="Output directory (default: data/)")
    dl.add_argument("--max-samples", type=int, default=None, help="Max samples to download")
    dl.add_argument("--sample-images", action="store_true", help="Download sample images for quick testing")

    # preprocess
    pp = subparsers.add_parser("preprocess", help="Preprocess images with COLMAP or frame extraction")
    pp.add_argument("--images", required=True, help="Input image directory or video file")
    pp.add_argument("--output", default="outputs/colmap", help="Output directory")
    pp.add_argument(
        "--method",
        choices=["colmap", "frames", "pose-free", "dust3r", "simple"],
        default="colmap",
        help="Preprocessing method (default: colmap). "
        "'pose-free' and 'dust3r' use DUSt3R for pose estimation; "
        "'simple' uses circular camera initialization.",
    )
    pp.add_argument("--fps", type=float, default=2.0, help="FPS for frame extraction (default: 2)")
    pp.add_argument("--max-frames", type=int, default=100, help="Max frames to extract (default: 100)")
    pp.add_argument(
        "--matching",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matching strategy (default: exhaustive)",
    )
    pp.add_argument("--no-gpu", action="store_true", help="Disable GPU for COLMAP")

    # train
    tr = subparsers.add_parser("train", help="Train a 3DGS model")
    tr.add_argument("--data", required=True, help="Preprocessed data directory")
    tr.add_argument("--output", default="outputs/train", help="Training output directory")
    tr.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    tr.add_argument("--iterations", type=int, default=30000, help="Number of training iterations")
    tr.add_argument("--config", default=None, help="Path to training config YAML override")

    # view
    vw = subparsers.add_parser("view", help="Launch the web viewer")
    vw.add_argument("--model", required=True, help="Path to the .ply file or COLMAP sparse dir")
    vw.add_argument("--host", default="0.0.0.0", help="Viewer host (default: 0.0.0.0)")
    vw.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")
    vw.add_argument("--colmap", action="store_true", help="View COLMAP sparse model instead of PLY")

    # export
    ex = subparsers.add_parser("export", help="Export PLY to web-friendly format")
    ex.add_argument("--model", required=True, help="Path to the .ply file")
    ex.add_argument(
        "--format",
        choices=["json", "binary"],
        default="json",
        help="Output format (default: json)",
    )
    ex.add_argument("--output", required=True, help="Output file path")
    ex.add_argument("--max-points", type=int, default=100000, help="Max points to export (default: 100000)")

    # benchmark
    bm = subparsers.add_parser("benchmark", help="Benchmark training backends")
    bm.add_argument("--data", required=True, help="Data directory for training")
    bm.add_argument("--iterations", type=int, default=1000, help="Number of training iterations (default: 1000)")
    bm.add_argument("--output", default="outputs/benchmark", help="Benchmark output directory")
    bm.add_argument("--dataset-name", default="default", help="Dataset name label (default: default)")
    bm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio", "both"],
        default="both",
        help="Backend to benchmark (default: both)",
    )

    # run (full pipeline)
    rn = subparsers.add_parser("run", help="Run the full pipeline end-to-end")
    rn.add_argument("--images", required=True, help="Input image directory")
    rn.add_argument("--output", default="outputs", help="Root output directory")
    rn.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    rn.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    rn.add_argument(
        "--preprocess-method",
        choices=["colmap", "pose-free", "dust3r", "simple"],
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    rn.add_argument("--skip-preprocess", action="store_true", help="Skip COLMAP preprocessing")
    rn.add_argument("--no-viewer", action="store_true", help="Skip launching the viewer")
    rn.add_argument("--port", type=int, default=8080, help="Viewer port (default: 8080)")

    return parser


def cmd_download(args: argparse.Namespace) -> None:
    """Handle the download subcommand."""
    if args.sample_images:
        from nerf_gs_playground.common.download import download_sample_images

        output_dir = Path(args.output) if args.output else Path("data/sample")
        download_sample_images(output_dir)
        return

    if args.dataset is None:
        print("Error: --dataset is required (unless using --sample-images).")
        sys.exit(1)

    from nerf_gs_playground.common.download import download_dataset

    output_dir = Path(args.output) if args.output else None
    download_dataset(
        name=args.dataset,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Handle the preprocess subcommand."""
    images_path = Path(args.images)
    output_dir = Path(args.output)

    if args.method == "frames":
        from nerf_gs_playground.preprocess.extract_frames import (
            extract_frames,
            extract_frames_from_dir,
        )

        if images_path.is_file():
            extract_frames(
                video_path=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        elif images_path.is_dir():
            extract_frames_from_dir(
                input_dir=images_path,
                output_dir=output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
            )
        else:
            print(f"Error: '{images_path}' is not a file or directory.")
            sys.exit(1)
    elif args.method in ("pose-free", "dust3r", "simple"):
        from nerf_gs_playground.preprocess.pose_free import run_pose_free

        # Map CLI method names to PoseFreeProcessor methods
        method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
        run_pose_free(
            image_dir=images_path,
            output_dir=output_dir,
            method=method_map[args.method],
        )
    else:
        from nerf_gs_playground.preprocess.colmap import run_colmap

        run_colmap(
            image_dir=images_path,
            output_dir=output_dir,
            matching=args.matching,
            use_gpu=not args.no_gpu,
        )


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the train subcommand."""
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Load config override if provided
    config = None
    if args.config:
        from nerf_gs_playground.common.config import load_config

        config = load_config(args.config)

    if args.method == "gsplat":
        from nerf_gs_playground.train.gsplat_trainer import train_gsplat

        ply_path = train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
            num_iterations=args.iterations,
        )
        print(f"\nTrained model saved to: {ply_path}")
    else:
        from nerf_gs_playground.train.nerfstudio_trainer import train_nerfstudio

        output = train_nerfstudio(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
        )
        print(f"\nNerfstudio output at: {output}")


def cmd_view(args: argparse.Namespace) -> None:
    """Handle the view subcommand."""
    from nerf_gs_playground.viewer.web_viewer import GaussianViewer

    viewer = GaussianViewer(host=args.host, port=args.port)

    if args.colmap:
        viewer.view_colmap(args.model)
    else:
        viewer.view_ply(args.model)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export subcommand."""
    if args.format == "json":
        from nerf_gs_playground.viewer.web_export import ply_to_json

        result = ply_to_json(args.model, args.output, max_points=args.max_points)
    else:
        from nerf_gs_playground.viewer.web_export import ply_to_binary

        result = ply_to_binary(args.model, args.output, max_points=args.max_points)

    print(f"Exported to: {result}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the benchmark subcommand."""
    from nerf_gs_playground.benchmark import Benchmark

    bench = Benchmark(data_dir=args.data, output_dir=args.output)

    if args.method in ("gsplat", "both"):
        print("Running gsplat benchmark...")
        bench.run_gsplat(num_iterations=args.iterations, dataset_name=args.dataset_name)

    if args.method in ("nerfstudio", "both"):
        print("Running nerfstudio benchmark...")
        bench.run_nerfstudio(num_iterations=args.iterations, dataset_name=args.dataset_name)

    print("\n" + bench.compare())
    bench.save_results()
    print(f"\nResults saved to: {Path(args.output) / 'benchmark_results.json'}")


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the run subcommand (full pipeline)."""
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    colmap_dir = output_dir / "colmap"
    train_dir = output_dir / "train"

    # Step 1: Preprocess
    if not args.skip_preprocess:
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method in ("pose-free", "dust3r", "simple"):
            from nerf_gs_playground.preprocess.pose_free import run_pose_free

            method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                method=method_map[preprocess_method],
            )
        else:
            from nerf_gs_playground.preprocess.colmap import run_colmap

            run_colmap(
                image_dir=images_dir,
                output_dir=colmap_dir,
            )
    else:
        print("Skipping preprocessing (--skip-preprocess)")

    # Step 2: Train
    print("\n" + "=" * 60)
    print("Step 2: Training")
    print("=" * 60)

    ply_path = None
    if args.method == "gsplat":
        from nerf_gs_playground.train.gsplat_trainer import train_gsplat

        ply_path = train_gsplat(
            data_dir=colmap_dir,
            output_dir=train_dir,
            num_iterations=args.iterations,
        )
    else:
        from nerf_gs_playground.train.nerfstudio_trainer import train_nerfstudio

        train_nerfstudio(
            data_dir=colmap_dir,
            output_dir=train_dir,
        )

    # Step 3: View
    if not args.no_viewer and ply_path is not None:
        print("\n" + "=" * 60)
        print("Step 3: Viewer")
        print("=" * 60)
        from nerf_gs_playground.viewer.web_viewer import launch_viewer

        launch_viewer(ply_path, port=args.port)

    print("\nPipeline complete!")


def main(argv: list[str] | None = None) -> None:
    """Entry point for the gs-playground CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "view": cmd_view,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
        "run": cmd_run,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
