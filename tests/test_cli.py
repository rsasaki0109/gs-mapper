"""Tests for CLI entry point and subcommand help."""

from __future__ import annotations

import pytest

from gs_sim2real.cli import build_parser, main


class TestCLIHelp:
    """Verify that CLI --help for each subcommand exits cleanly."""

    def test_cli_help(self) -> None:
        """Running main with --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_download_help(self) -> None:
        """Running download --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["download", "--help"])
        assert exc_info.value.code == 0

    def test_cli_preprocess_help(self) -> None:
        """Running preprocess --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["preprocess", "--help"])
        assert exc_info.value.code == 0

    def test_cli_train_help(self) -> None:
        """Running train --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0

    def test_cli_view_help(self) -> None:
        """Running view --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["view", "--help"])
        assert exc_info.value.code == 0

    def test_cli_run_help(self) -> None:
        """Running run --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0

    def test_cli_robotics_node_help(self) -> None:
        """Running robotics-node --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["robotics-node", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_server_help(self) -> None:
        """Running sim2real-server --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-server", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_query_help(self) -> None:
        """Running sim2real-query --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-query", "--help"])
        assert exc_info.value.code == 0

    def test_cli_sim2real_benchmark_images_help(self) -> None:
        """Running sim2real-benchmark-images --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["sim2real-benchmark-images", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_alignment_help(self) -> None:
        """Running experiment-localization-alignment --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-alignment", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_render_backend_selection_help(self) -> None:
        """Running experiment-render-backend-selection --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-render-backend-selection", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_import_help(self) -> None:
        """Running experiment-localization-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_transport_selection_help(self) -> None:
        """Running experiment-query-transport-selection --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-transport-selection", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_request_import_help(self) -> None:
        """Running experiment-query-request-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-request-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_live_localization_stream_import_help(self) -> None:
        """Running experiment-live-localization-stream-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-live-localization-stream-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_route_capture_import_help(self) -> None:
        """Running experiment-route-capture-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-route-capture-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_sim2real_websocket_protocol_help(self) -> None:
        """Running experiment-sim2real-websocket-protocol --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-sim2real-websocket-protocol", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_localization_review_bundle_import_help(self) -> None:
        """Running experiment-localization-review-bundle-import --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-localization-review-bundle-import", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_response_build_help(self) -> None:
        """Running experiment-query-response-build --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-response-build", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_timeout_policy_help(self) -> None:
        """Running experiment-query-timeout-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-timeout-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_queue_policy_help(self) -> None:
        """Running experiment-query-queue-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-queue-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_cancellation_policy_help(self) -> None:
        """Running experiment-query-cancellation-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-cancellation-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_coalescing_policy_help(self) -> None:
        """Running experiment-query-coalescing-policy --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-coalescing-policy", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_error_mapping_help(self) -> None:
        """Running experiment-query-error-mapping --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-error-mapping", "--help"])
        assert exc_info.value.code == 0

    def test_cli_experiment_query_source_identity_help(self) -> None:
        """Running experiment-query-source-identity --help raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["experiment-query-source-identity", "--help"])
        assert exc_info.value.code == 0

    def test_cli_robotics_node_enable_image_relay_flag(self) -> None:
        """robotics-node parser accepts the image relay toggle."""
        args = build_parser().parse_args(["robotics-node", "--enable-image-relay"])
        assert args.enable_image_relay is True

    def test_cli_sim2real_server_static_pose_flags(self) -> None:
        """sim2real-server parser accepts static render settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "static",
                "--static-position",
                "1.0",
                "2.0",
                "3.0",
                "--static-orientation",
                "0.0",
                "0.0",
                "0.0",
                "1.0",
                "--renderer",
                "simple",
                "--run-once",
            ]
        )
        assert args.ply == "scene.ply"
        assert args.pose_source == "static"
        assert args.static_position == [1.0, 2.0, 3.0]
        assert args.static_orientation == [0.0, 0.0, 0.0, 1.0]
        assert args.renderer == "simple"
        assert args.run_once is True

    def test_cli_sim2real_server_query_flags(self) -> None:
        """sim2real-server parser accepts query transport settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "query",
                "--query-transport",
                "zmq",
                "--query-endpoint",
                "tcp://127.0.0.1:6001",
            ]
        )
        assert args.pose_source == "query"
        assert args.query_transport == "zmq"
        assert args.query_endpoint == "tcp://127.0.0.1:6001"

    def test_cli_sim2real_server_auto_query_transport_flag(self) -> None:
        """sim2real-server parser accepts auto query transport selection."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--pose-source",
                "query",
                "--query-transport",
                "auto",
            ]
        )
        assert args.pose_source == "query"
        assert args.query_transport == "auto"

    def test_cli_export_scene_bundle_flags(self) -> None:
        """export parser accepts GitHub Pages scene bundle settings."""
        args = build_parser().parse_args(
            [
                "export",
                "--model",
                "scene.ply",
                "--format",
                "scene-bundle",
                "--output",
                "docs/assets/demo-room",
                "--bundle-asset-format",
                "binary",
                "--scene-id",
                "demo-room",
                "--label",
                "Demo Room",
                "--description",
                "GitHub Pages demo scene",
            ]
        )
        assert args.format == "scene-bundle"
        assert args.bundle_asset_format == "binary"
        assert args.scene_id == "demo-room"
        assert args.label == "Demo Room"
        assert args.description == "GitHub Pages demo scene"

    def test_cli_sim2real_server_websocket_query_flags(self) -> None:
        """sim2real-server parser accepts websocket query settings."""
        args = build_parser().parse_args(
            [
                "sim2real-server",
                "--ply",
                "scene.ply",
                "--query-transport",
                "ws",
                "--query-endpoint",
                "ws://127.0.0.1:8781/sim2real",
            ]
        )
        assert args.query_transport == "ws"
        assert args.query_endpoint == "ws://127.0.0.1:8781/sim2real"

    def test_cli_sim2real_query_flags(self) -> None:
        """sim2real-query parser accepts pose and output settings."""
        args = build_parser().parse_args(
            [
                "sim2real-query",
                "--endpoint",
                "tcp://127.0.0.1:6002",
                "--position",
                "1.0",
                "2.0",
                "3.0",
                "--yaw-degrees",
                "90.0",
                "--width",
                "320",
                "--height",
                "240",
                "--jpeg-out",
                "frame.jpg",
                "--depth-out",
                "depth.npy",
            ]
        )
        assert args.endpoint == "tcp://127.0.0.1:6002"
        assert args.position == [1.0, 2.0, 3.0]
        assert args.yaw_degrees == 90.0
        assert args.width == 320
        assert args.height == 240
        assert args.jpeg_out == "frame.jpg"
        assert args.depth_out == "depth.npy"

    def test_cli_sim2real_benchmark_images_flags(self) -> None:
        """sim2real-benchmark-images parser accepts run and metric settings."""
        args = build_parser().parse_args(
            [
                "sim2real-benchmark-images",
                "--endpoint",
                "ws://127.0.0.1:8781/sim2real",
                "--run",
                "run.json",
                "--alignment",
                "timestamp",
                "--metrics",
                "psnr",
                "ssim",
                "--lpips-net",
                "vgg",
                "--device",
                "auto",
                "--timeout-ms",
                "2500",
                "--max-frames",
                "12",
                "--output",
                "report.json",
            ]
        )
        assert args.endpoint == "ws://127.0.0.1:8781/sim2real"
        assert args.run == "run.json"
        assert args.alignment == "timestamp"
        assert args.metrics == ["psnr", "ssim"]
        assert args.lpips_net == "vgg"
        assert args.device == "auto"
        assert args.timeout_ms == 2500
        assert args.max_frames == 12
        assert args.output == "report.json"

    def test_cli_experiment_localization_alignment_flags(self) -> None:
        """experiment-localization-alignment parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-localization-alignment",
                "--repetitions",
                "32",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "alignment-report.json",
            ]
        )
        assert args.repetitions == 32
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "alignment-report.json"

    def test_cli_experiment_render_backend_selection_flags(self) -> None:
        """experiment-render-backend-selection parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-render-backend-selection",
                "--repetitions",
                "16",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "render-backend-report.json",
            ]
        )
        assert args.repetitions == 16
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "render-backend-report.json"

    def test_cli_experiment_localization_import_flags(self) -> None:
        """experiment-localization-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-localization-import",
                "--repetitions",
                "24",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "localization-import-report.json",
            ]
        )
        assert args.repetitions == 24
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "localization-import-report.json"

    def test_cli_experiment_query_transport_selection_flags(self) -> None:
        """experiment-query-transport-selection parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-transport-selection",
                "--repetitions",
                "20",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-transport-report.json",
            ]
        )
        assert args.repetitions == 20
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-transport-report.json"

    def test_cli_experiment_query_request_import_flags(self) -> None:
        """experiment-query-request-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-request-import",
                "--repetitions",
                "12",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-request-import-report.json",
            ]
        )
        assert args.repetitions == 12
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-request-import-report.json"

    def test_cli_experiment_live_localization_stream_import_flags(self) -> None:
        """experiment-live-localization-stream-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-live-localization-stream-import",
                "--repetitions",
                "10",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "live-stream-import-report.json",
            ]
        )
        assert args.repetitions == 10
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "live-stream-import-report.json"

    def test_cli_experiment_route_capture_import_flags(self) -> None:
        """experiment-route-capture-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-route-capture-import",
                "--repetitions",
                "18",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "route-capture-import-report.json",
            ]
        )
        assert args.repetitions == 18
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "route-capture-import-report.json"

    def test_cli_experiment_sim2real_websocket_protocol_flags(self) -> None:
        """experiment-sim2real-websocket-protocol parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-sim2real-websocket-protocol",
                "--repetitions",
                "14",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "sim2real-websocket-protocol-report.json",
            ]
        )
        assert args.repetitions == 14
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "sim2real-websocket-protocol-report.json"

    def test_cli_experiment_localization_review_bundle_import_flags(self) -> None:
        """experiment-localization-review-bundle-import parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-localization-review-bundle-import",
                "--repetitions",
                "22",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "localization-review-bundle-import-report.json",
            ]
        )
        assert args.repetitions == 22
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "localization-review-bundle-import-report.json"

    def test_cli_experiment_query_response_build_flags(self) -> None:
        """experiment-query-response-build parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-response-build",
                "--repetitions",
                "24",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-response-build-report.json",
            ]
        )
        assert args.repetitions == 24
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-response-build-report.json"

    def test_cli_experiment_query_timeout_policy_flags(self) -> None:
        """experiment-query-timeout-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-timeout-policy",
                "--repetitions",
                "26",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-timeout-policy-report.json",
            ]
        )
        assert args.repetitions == 26
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-timeout-policy-report.json"

    def test_cli_experiment_query_queue_policy_flags(self) -> None:
        """experiment-query-queue-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-queue-policy",
                "--repetitions",
                "28",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-queue-policy-report.json",
            ]
        )
        assert args.repetitions == 28
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-queue-policy-report.json"

    def test_cli_experiment_query_cancellation_policy_flags(self) -> None:
        """experiment-query-cancellation-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-cancellation-policy",
                "--repetitions",
                "30",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-cancellation-policy-report.json",
            ]
        )
        assert args.repetitions == 30
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-cancellation-policy-report.json"

    def test_cli_experiment_query_coalescing_policy_flags(self) -> None:
        """experiment-query-coalescing-policy parser accepts evaluation settings."""
        args = build_parser().parse_args(
            [
                "experiment-query-coalescing-policy",
                "--repetitions",
                "32",
                "--write-docs",
                "--docs-dir",
                "docs-lab",
                "--output",
                "query-coalescing-policy-report.json",
            ]
        )
        assert args.repetitions == 32
        assert args.write_docs is True
        assert args.docs_dir == "docs-lab"
        assert args.output == "query-coalescing-policy-report.json"

    def test_cli_no_command(self) -> None:
        """Running main with no arguments exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
