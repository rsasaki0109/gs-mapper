"""Command-line interface for nerf-gs-sim2real.

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
        prog="gs-sim2real",
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
        choices=["colmap", "frames", "pose-free", "dust3r", "simple", "waymo"],
        default="colmap",
        help="Preprocessing method (default: colmap). "
        "'pose-free' and 'dust3r' use DUSt3R for pose estimation; "
        "'simple' uses circular camera initialization; "
        "'waymo' extracts frames from Waymo tfrecord files.",
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
    pp.add_argument(
        "--camera",
        default="FRONT",
        choices=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"],
        help="Waymo camera to extract (default: FRONT)",
    )
    pp.add_argument("--every-n", type=int, default=1, help="Extract every N-th frame for Waymo (default: 1)")

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
        choices=["json", "binary", "scene-bundle"],
        default="json",
        help="Output format (default: json)",
    )
    ex.add_argument("--output", required=True, help="Output file path")
    ex.add_argument("--max-points", type=int, default=100000, help="Max points to export (default: 100000)")
    ex.add_argument(
        "--bundle-asset-format",
        choices=["json", "binary"],
        default="binary",
        help="Asset format used inside --format scene-bundle (default: binary)",
    )
    ex.add_argument("--scene-id", default=None, help="Optional scene id for --format scene-bundle")
    ex.add_argument("--label", default=None, help="Optional scene label for --format scene-bundle")
    ex.add_argument("--description", default="", help="Optional scene description for --format scene-bundle")

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

    # demo (end-to-end: images -> splat -> DreamWalker teleop)
    dm = subparsers.add_parser("demo", help="End-to-end demo: images -> 3DGS -> DreamWalker robot teleop")
    dm.add_argument("--images", default=None, help="Input image directory or video file")
    dm.add_argument("--ply", default=None, help="Skip training, stage an existing PLY file directly")
    dm.add_argument("--output", default="outputs", help="Root output directory (default: outputs)")
    dm.add_argument(
        "--method",
        choices=["gsplat", "nerfstudio"],
        default="gsplat",
        help="Training method (default: gsplat)",
    )
    dm.add_argument("--iterations", type=int, default=1000, help="Training iterations (default: 1000)")
    dm.add_argument(
        "--preprocess-method",
        choices=["colmap", "pose-free", "dust3r", "simple"],
        default="colmap",
        help="Preprocessing method (default: colmap)",
    )
    dm.add_argument("--fragment", default="residency", help="DreamWalker fragment name (default: residency)")
    dm.add_argument("--no-launch", action="store_true", help="Skip launching the Vite dev server")

    # robotics ROS2 node
    rb = subparsers.add_parser("robotics-node", help="Launch the DreamWalker ROS2 bridge node scaffold")
    rb.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    rb.add_argument("--node-name", default="dreamwalker_bridge_node", help="ROS2 node name")
    rb.add_argument("--frame-id", default="dreamwalker_map", help="Expected map frame id")
    rb.add_argument("--log-period", type=float, default=2.0, help="Summary log period in seconds")
    rb.add_argument("--zones-file", default=None, help="Optional semantic zone JSON file")
    rb.add_argument("--costmap-period", type=float, default=10.0, help="Costmap republish period in seconds")
    rb.add_argument("--request-state-on-start", action="store_true", help="Publish request_state on startup")
    rb.add_argument(
        "--enable-image-relay",
        action="store_true",
        help="Subscribe to camera relay topics and log received frames",
    )
    rb.add_argument(
        "--demo-teleop",
        choices=["forward", "backward", "turn-left", "turn-right"],
        default=None,
        help="Publish one teleop command on startup",
    )
    rb.add_argument(
        "--demo-camera",
        choices=["front", "chase", "top"],
        default=None,
        help="Publish one camera command on startup",
    )
    rb.add_argument(
        "--demo-waypoint",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Publish one waypoint command on startup",
    )
    rb.add_argument(
        "--demo-pose2d",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA_RAD"),
        default=None,
        help="Publish one Pose2D command on startup",
    )

    # headless PLY render server
    sr = subparsers.add_parser(
        "sim2real-server",
        help="Headless PLY renderer that publishes RGB + depth to DreamWalker ROS2 topics",
    )
    sr.add_argument("--ply", required=True, help="Path to the trained PLY point cloud")
    sr.add_argument("--namespace", default="/dreamwalker", help="ROS topic namespace")
    sr.add_argument("--node-name", default="dreamwalker_sim2real_server", help="ROS2 node name")
    sr.add_argument("--frame-id", default="dreamwalker_map", help="Camera frame id for published messages")
    sr.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sr.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sr.add_argument("--fps", type=float, default=5.0, help="Publish rate in Hz")
    sr.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sr.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sr.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sr.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sr.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for camera output")
    sr.add_argument(
        "--renderer",
        choices=["auto", "simple", "gsplat"],
        default="auto",
        help="Rasterization backend. auto uses gsplat only when CUDA and Gaussian PLY parameters are available",
    )
    sr.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of points to load from the PLY for rendering",
    )
    sr.add_argument(
        "--pose-source",
        choices=["static", "robot_pose_stamped", "robot_pose2d", "query"],
        default="robot_pose_stamped",
        help="Source of camera poses used for rendering",
    )
    sr.add_argument(
        "--static-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Static camera position used when --pose-source static",
    )
    sr.add_argument(
        "--static-orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=(0.0, 0.0, 0.0, 1.0),
        help="Static camera orientation quaternion used when --pose-source static",
    )
    sr.add_argument(
        "--pose2d-z",
        type=float,
        default=0.0,
        help="Z position to use when pose source is robot_pose2d",
    )
    sr.add_argument(
        "--query-transport",
        choices=["auto", "none", "zmq", "ws"],
        default="none",
        help="Optional request-response transport for ad-hoc render queries",
    )
    sr.add_argument(
        "--query-endpoint",
        default="tcp://127.0.0.1:5588",
        help=(
            "Bind endpoint for the query transport when enabled. "
            "Defaults: tcp://127.0.0.1:5588 for zmq, ws://127.0.0.1:8781/sim2real for ws"
        ),
    )
    sr.add_argument(
        "--query-poll-period",
        type=float,
        default=0.01,
        help="Polling period in seconds for the query transport",
    )
    sr.add_argument("--run-once", action="store_true", help="Publish one frame and exit")

    # sim2real query client
    sq = subparsers.add_parser(
        "sim2real-query",
        help="Send one pose-based render query to a sim2real headless render server",
    )
    sq.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sq.add_argument("--timeout-ms", type=int, default=10000, help="Request timeout in milliseconds")
    sq.add_argument(
        "--position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="World-space camera position",
    )
    orientation = sq.add_mutually_exclusive_group()
    orientation.add_argument(
        "--orientation",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help="World-space camera orientation quaternion",
    )
    orientation.add_argument(
        "--yaw-degrees",
        type=float,
        default=None,
        help="Planar yaw in degrees, converted to a quaternion around +Z",
    )
    sq.add_argument("--width", type=int, default=640, help="Render width in pixels")
    sq.add_argument("--height", type=int, default=480, help="Render height in pixels")
    sq.add_argument("--fov-degrees", type=float, default=60.0, help="Vertical field of view in degrees")
    sq.add_argument("--near-clip", type=float, default=0.05, help="Near clip plane in meters")
    sq.add_argument("--far-clip", type=float, default=50.0, help="Far clip plane in meters")
    sq.add_argument("--point-radius", type=int, default=1, help="Projected point footprint radius in pixels")
    sq.add_argument("--jpeg-out", default=None, help="Optional path for the returned JPEG frame")
    sq.add_argument("--depth-out", default=None, help="Optional path for the returned depth .npy file")
    sq.add_argument("--camera-info-out", default=None, help="Optional path for the returned cameraInfo JSON")
    sq.add_argument("--response-out", default=None, help="Optional path for the full raw response JSON")

    # sim2real localization image benchmark
    sb = subparsers.add_parser(
        "sim2real-benchmark-images",
        help="Render estimate poses through sim2real-server and compare RGB frames against captured ground truth",
    )
    sb.add_argument(
        "--endpoint",
        default="tcp://127.0.0.1:5588",
        help="Query endpoint to connect to. Supports tcp://... (ZMQ) and ws://... (WebSocket)",
    )
    sb.add_argument(
        "--run",
        default=None,
        help="Optional localization-run-snapshot JSON exported from the web panel",
    )
    sb.add_argument(
        "--ground-truth",
        default=None,
        help="Path to a route-capture-bundle JSON file. Required unless --run already embeds it",
    )
    sb.add_argument(
        "--estimate",
        default=None,
        help="Path to a localization estimate JSON or TUM/ORB-SLAM text trajectory. Required unless --run embeds it",
    )
    sb.add_argument(
        "--alignment",
        choices=["auto", "index", "timestamp"],
        default="auto",
        help="Pose matching mode before rendering estimate frames",
    )
    sb.add_argument(
        "--metrics",
        nargs="+",
        choices=["psnr", "ssim", "lpips"],
        default=["psnr", "ssim", "lpips"],
        help="Image metrics to compute for each matched frame",
    )
    sb.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="alex",
        help="LPIPS backbone used when --metrics includes lpips",
    )
    sb.add_argument(
        "--device",
        default="cpu",
        help="Torch device for LPIPS. Use auto to prefer CUDA when available",
    )
    sb.add_argument("--timeout-ms", type=int, default=10000, help="Per-frame render timeout in milliseconds")
    sb.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of matched frames")
    sb.add_argument("--output", default=None, help="Optional path for the full benchmark report JSON")

    # localization alignment experiment lab
    el = subparsers.add_parser(
        "experiment-localization-alignment",
        help="Compare multiple localization alignment strategies and optionally refresh experiment docs",
    )
    el.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    el.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    el.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    el.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # render backend selection experiment lab
    er = subparsers.add_parser(
        "experiment-render-backend-selection",
        help="Compare render backend-selection policies and optionally refresh experiment docs",
    )
    er.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    er.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    er.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    er.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # localization estimate import experiment lab
    ei = subparsers.add_parser(
        "experiment-localization-import",
        help="Compare localization estimate import policies and optionally refresh experiment docs",
    )
    ei.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    ei.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    ei.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    ei.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query transport selection experiment lab
    eq = subparsers.add_parser(
        "experiment-query-transport-selection",
        help="Compare query transport policies and optionally refresh experiment docs",
    )
    eq.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eq.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eq.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eq.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query request import experiment lab
    eqr = subparsers.add_parser(
        "experiment-query-request-import",
        help="Compare query request import policies and optionally refresh experiment docs",
    )
    eqr.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqr.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqr.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqr.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # live localization stream import experiment lab
    els = subparsers.add_parser(
        "experiment-live-localization-stream-import",
        help="Compare live localization stream import policies and optionally refresh experiment docs",
    )
    els.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    els.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    els.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    els.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # route capture bundle import experiment lab
    erc = subparsers.add_parser(
        "experiment-route-capture-import",
        help="Compare route capture bundle import policies and optionally refresh experiment docs",
    )
    erc.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    erc.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    erc.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    erc.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # sim2real websocket protocol experiment lab
    esp = subparsers.add_parser(
        "experiment-sim2real-websocket-protocol",
        help="Compare sim2real websocket message protocol policies and optionally refresh experiment docs",
    )
    esp.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    esp.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    esp.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    esp.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # localization review bundle import experiment lab
    erl = subparsers.add_parser(
        "experiment-localization-review-bundle-import",
        help="Compare localization review bundle import policies and optionally refresh experiment docs",
    )
    erl.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    erl.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    erl.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    erl.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query cancellation policy experiment lab
    eqc = subparsers.add_parser(
        "experiment-query-cancellation-policy",
        help="Compare query cancellation policies and optionally refresh experiment docs",
    )
    eqc.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqc.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqc.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqc.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query coalescing policy experiment lab
    eqco = subparsers.add_parser(
        "experiment-query-coalescing-policy",
        help="Compare query dedupe/coalescing policies and optionally refresh experiment docs",
    )
    eqco.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqco.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqco.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqco.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query error mapping experiment lab
    eqem = subparsers.add_parser(
        "experiment-query-error-mapping",
        help="Compare query error mapping policies and optionally refresh experiment docs",
    )
    eqem.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqem.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqem.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqem.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query queue policy experiment lab
    eqq = subparsers.add_parser(
        "experiment-query-queue-policy",
        help="Compare query queue policies and optionally refresh experiment docs",
    )
    eqq.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqq.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqq.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqq.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query source identity experiment lab
    eqsi = subparsers.add_parser(
        "experiment-query-source-identity",
        help="Compare query source identity policies and optionally refresh experiment docs",
    )
    eqsi.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqsi.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqsi.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqsi.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query timeout policy experiment lab
    eqt = subparsers.add_parser(
        "experiment-query-timeout-policy",
        help="Compare query timeout policies and optionally refresh experiment docs",
    )
    eqt.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqt.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqt.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqt.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    # query response build experiment lab
    eqb = subparsers.add_parser(
        "experiment-query-response-build",
        help="Compare query response build policies and optionally refresh experiment docs",
    )
    eqb.add_argument(
        "--repetitions",
        type=int,
        default=200,
        help="Runtime benchmark repetitions per fixture",
    )
    eqb.add_argument(
        "--write-docs",
        action="store_true",
        help="Refresh docs/experiments.md, docs/decisions.md, and docs/interfaces.md",
    )
    eqb.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory where experiment process docs are written when --write-docs is set",
    )
    eqb.add_argument(
        "--output",
        default=None,
        help="Optional path for the full experiment report JSON",
    )

    return parser


def cmd_download(args: argparse.Namespace) -> None:
    """Handle the download subcommand."""
    if args.sample_images:
        from gs_sim2real.common.download import download_sample_images

        output_dir = Path(args.output) if args.output else Path("data/sample")
        download_sample_images(output_dir)
        return

    if args.dataset is None:
        print("Error: --dataset is required (unless using --sample-images).")
        sys.exit(1)

    from gs_sim2real.common.download import download_dataset

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

    if args.method == "waymo":
        from gs_sim2real.datasets.waymo import WaymoLoader

        loader = WaymoLoader(data_dir=str(images_path))
        images_out = loader.extract_frames(
            output_dir=str(output_dir),
            camera=args.camera,
            max_frames=args.max_frames,
            every_n=args.every_n,
        )
        # Convert to COLMAP format if camera_params.json exists
        params_path = output_dir / "camera_params.json"
        if params_path.exists():
            sparse_dir = loader.to_colmap_format(
                camera_params_path=str(params_path),
                output_dir=str(output_dir),
            )
            print(f"Waymo frames extracted to: {images_out}")
            print(f"COLMAP sparse model at: {sparse_dir}")
        else:
            print(f"Waymo frames loaded from: {images_out}")
        return

    if args.method == "frames":
        from gs_sim2real.preprocess.extract_frames import (
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
        from gs_sim2real.preprocess.pose_free import run_pose_free

        # Map CLI method names to PoseFreeProcessor methods
        method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
        run_pose_free(
            image_dir=images_path,
            output_dir=output_dir,
            method=method_map[args.method],
        )
    else:
        from gs_sim2real.preprocess.colmap import run_colmap

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
        from gs_sim2real.common.config import load_config

        config = load_config(args.config)

    if args.method == "gsplat":
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        ply_path = train_gsplat(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
            num_iterations=args.iterations,
        )
        print(f"\nTrained model saved to: {ply_path}")
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        output = train_nerfstudio(
            data_dir=data_dir,
            output_dir=output_dir,
            config=config,
        )
        print(f"\nNerfstudio output at: {output}")


def cmd_view(args: argparse.Namespace) -> None:
    """Handle the view subcommand."""
    from gs_sim2real.viewer.web_viewer import GaussianViewer

    viewer = GaussianViewer(host=args.host, port=args.port)

    if args.colmap:
        viewer.view_colmap(args.model)
    else:
        viewer.view_ply(args.model)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the export subcommand."""
    if args.format == "json":
        from gs_sim2real.viewer.web_export import ply_to_json

        result = ply_to_json(args.model, args.output, max_points=args.max_points)
    elif args.format == "binary":
        from gs_sim2real.viewer.web_export import ply_to_binary

        result = ply_to_binary(args.model, args.output, max_points=args.max_points)
    else:
        from gs_sim2real.viewer.web_export import ply_to_scene_bundle

        result = ply_to_scene_bundle(
            args.model,
            args.output,
            asset_format=args.bundle_asset_format,
            scene_id=args.scene_id,
            label=args.label,
            description=args.description,
            max_points=args.max_points,
        )

    print(f"Exported to: {result}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the benchmark subcommand."""
    from gs_sim2real.benchmark import Benchmark

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
            from gs_sim2real.preprocess.pose_free import run_pose_free

            method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                method=method_map[preprocess_method],
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap

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
        from gs_sim2real.train.gsplat_trainer import train_gsplat

        ply_path = train_gsplat(
            data_dir=colmap_dir,
            output_dir=train_dir,
            num_iterations=args.iterations,
        )
    else:
        from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

        train_nerfstudio(
            data_dir=colmap_dir,
            output_dir=train_dir,
        )

    # Step 3: View
    if not args.no_viewer and ply_path is not None:
        print("\n" + "=" * 60)
        print("Step 3: Viewer")
        print("=" * 60)
        from gs_sim2real.viewer.web_viewer import launch_viewer

        launch_viewer(ply_path, port=args.port)

    print("\nPipeline complete!")


def cmd_demo(args: argparse.Namespace) -> None:
    """Handle the demo subcommand (images -> splat -> DreamWalker teleop)."""
    import subprocess

    ply_path = None

    if args.ply:
        # Use an existing PLY directly
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"Error: PLY file not found: {ply_path}")
            sys.exit(1)
        print(f"Using existing PLY: {ply_path}")
    elif args.images:
        images_dir = Path(args.images)
        output_dir = Path(args.output)
        colmap_dir = output_dir / "colmap"
        train_dir = output_dir / "train"

        # Step 1: Preprocess
        preprocess_method = args.preprocess_method
        print("=" * 60)
        print(f"Step 1/3: Preprocessing ({preprocess_method})")
        print("=" * 60)

        if preprocess_method in ("pose-free", "dust3r", "simple"):
            from gs_sim2real.preprocess.pose_free import run_pose_free

            method_map = {"pose-free": "dust3r", "dust3r": "dust3r", "simple": "simple"}
            run_pose_free(
                image_dir=images_dir,
                output_dir=colmap_dir,
                method=method_map[preprocess_method],
            )
        else:
            from gs_sim2real.preprocess.colmap import run_colmap as _run_colmap

            _run_colmap(image_dir=images_dir, output_dir=colmap_dir)

        # Step 2: Train
        print("\n" + "=" * 60)
        print("Step 2/3: Training")
        print("=" * 60)

        if args.method == "gsplat":
            from gs_sim2real.train.gsplat_trainer import train_gsplat

            ply_path = train_gsplat(
                data_dir=colmap_dir,
                output_dir=train_dir,
                num_iterations=args.iterations,
            )
        else:
            from gs_sim2real.train.nerfstudio_trainer import train_nerfstudio

            train_nerfstudio(data_dir=colmap_dir, output_dir=train_dir)
            ply_path = train_dir / "point_cloud.ply"
    else:
        print("Error: --images or --ply is required.")
        sys.exit(1)

    # Step 3: Stage into DreamWalker
    print("\n" + "=" * 60)
    print("Step 3/3: Staging for DreamWalker")
    print("=" * 60)

    from gs_sim2real.demo.stage_for_dreamwalker import stage_ply

    result = stage_ply(ply_path, fragment=args.fragment)
    print(f"Splat staged: {result['splat_dest']}")
    print(f"Manifest updated: {result['manifest']}")
    print(f"Launch URL: {result['launch_url']}")

    # Launch Vite dev server
    if not args.no_launch:
        dreamwalker_dir = Path(result["manifest"]).parent.parent.parent
        print(f"\nStarting DreamWalker dev server in {dreamwalker_dir} ...")
        print("Open your browser at:", result["launch_url"])
        print("Controls: WASD = move, Mouse = look, R = toggle robot mode")
        subprocess.run(["npm", "run", "dev"], cwd=dreamwalker_dir)
    else:
        print("\nTo launch manually:")
        print(f"  cd {Path(result['manifest']).parent.parent.parent}")
        print("  npm run dev")
        print(f"  Open: {result['launch_url']}")


def cmd_robotics_node(args: argparse.Namespace) -> None:
    """Handle the robotics-node subcommand."""
    from gs_sim2real.robotics.ros2_bridge_node import run_cli

    run_cli(args)


def cmd_sim2real_server(args: argparse.Namespace) -> None:
    """Handle the sim2real-server subcommand."""
    from gs_sim2real.robotics.gsplat_render_server import run_cli

    run_cli(args)


def cmd_sim2real_query(args: argparse.Namespace) -> None:
    """Handle the sim2real-query subcommand."""
    from gs_sim2real.robotics.render_query_client import run_cli

    run_cli(args)


def cmd_sim2real_benchmark_images(args: argparse.Namespace) -> None:
    """Handle the sim2real-benchmark-images subcommand."""
    from gs_sim2real.robotics.localization_image_benchmark import run_cli

    run_cli(args)


def cmd_experiment_localization_alignment(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-alignment subcommand."""
    from gs_sim2real.experiments.localization_alignment_lab import run_cli

    run_cli(args)


def cmd_experiment_render_backend_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-render-backend-selection subcommand."""
    from gs_sim2real.experiments.render_backend_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-import subcommand."""
    from gs_sim2real.experiments.localization_estimate_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_transport_selection(args: argparse.Namespace) -> None:
    """Handle the experiment-query-transport-selection subcommand."""
    from gs_sim2real.experiments.query_transport_selection_lab import run_cli

    run_cli(args)


def cmd_experiment_query_request_import(args: argparse.Namespace) -> None:
    """Handle the experiment-query-request-import subcommand."""
    from gs_sim2real.experiments.query_request_import_lab import run_cli

    run_cli(args)


def cmd_experiment_live_localization_stream_import(args: argparse.Namespace) -> None:
    """Handle the experiment-live-localization-stream-import subcommand."""
    from gs_sim2real.experiments.live_localization_stream_import_lab import run_cli

    run_cli(args)


def cmd_experiment_route_capture_import(args: argparse.Namespace) -> None:
    """Handle the experiment-route-capture-import subcommand."""
    from gs_sim2real.experiments.route_capture_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_sim2real_websocket_protocol(args: argparse.Namespace) -> None:
    """Handle the experiment-sim2real-websocket-protocol subcommand."""
    from gs_sim2real.experiments.sim2real_websocket_protocol_lab import run_cli

    run_cli(args)


def cmd_experiment_localization_review_bundle_import(args: argparse.Namespace) -> None:
    """Handle the experiment-localization-review-bundle-import subcommand."""
    from gs_sim2real.experiments.localization_review_bundle_import_lab import run_cli

    run_cli(args)


def cmd_experiment_query_cancellation_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-cancellation-policy subcommand."""
    from gs_sim2real.experiments.query_cancellation_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_coalescing_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-coalescing-policy subcommand."""
    from gs_sim2real.experiments.query_coalescing_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_error_mapping(args: argparse.Namespace) -> None:
    """Handle the experiment-query-error-mapping subcommand."""
    from gs_sim2real.experiments.query_error_mapping_lab import run_cli

    run_cli(args)


def cmd_experiment_query_queue_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-queue-policy subcommand."""
    from gs_sim2real.experiments.query_queue_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_source_identity(args: argparse.Namespace) -> None:
    """Handle the experiment-query-source-identity subcommand."""
    from gs_sim2real.experiments.query_source_identity_lab import run_cli

    run_cli(args)


def cmd_experiment_query_timeout_policy(args: argparse.Namespace) -> None:
    """Handle the experiment-query-timeout-policy subcommand."""
    from gs_sim2real.experiments.query_timeout_policy_lab import run_cli

    run_cli(args)


def cmd_experiment_query_response_build(args: argparse.Namespace) -> None:
    """Handle the experiment-query-response-build subcommand."""
    from gs_sim2real.experiments.query_response_build_lab import run_cli

    run_cli(args)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the gs-sim2real CLI."""
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
        "demo": cmd_demo,
        "robotics-node": cmd_robotics_node,
        "sim2real-server": cmd_sim2real_server,
        "sim2real-query": cmd_sim2real_query,
        "sim2real-benchmark-images": cmd_sim2real_benchmark_images,
        "experiment-localization-alignment": cmd_experiment_localization_alignment,
        "experiment-render-backend-selection": cmd_experiment_render_backend_selection,
        "experiment-localization-import": cmd_experiment_localization_import,
        "experiment-query-transport-selection": cmd_experiment_query_transport_selection,
        "experiment-query-request-import": cmd_experiment_query_request_import,
        "experiment-live-localization-stream-import": cmd_experiment_live_localization_stream_import,
        "experiment-route-capture-import": cmd_experiment_route_capture_import,
        "experiment-sim2real-websocket-protocol": cmd_experiment_sim2real_websocket_protocol,
        "experiment-localization-review-bundle-import": cmd_experiment_localization_review_bundle_import,
        "experiment-query-cancellation-policy": cmd_experiment_query_cancellation_policy,
        "experiment-query-coalescing-policy": cmd_experiment_query_coalescing_policy,
        "experiment-query-error-mapping": cmd_experiment_query_error_mapping,
        "experiment-query-queue-policy": cmd_experiment_query_queue_policy,
        "experiment-query-source-identity": cmd_experiment_query_source_identity,
        "experiment-query-timeout-policy": cmd_experiment_query_timeout_policy,
        "experiment-query-response-build": cmd_experiment_query_response_build,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
