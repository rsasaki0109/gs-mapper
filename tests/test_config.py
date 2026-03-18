"""Tests for configuration loading and CLI argument parsing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nerf_gs_playground.common.config import (
    get_dataset_config,
    get_project_root,
    load_config,
    load_datasets_config,
    load_training_config,
)


class TestLoadConfig:
    """Tests for the generic load_config function."""

    def test_load_datasets_config(self) -> None:
        """Test that datasets.yaml loads correctly."""
        config = load_datasets_config()
        assert "ggrt" in config
        assert "covla" in config
        assert "mcd" in config

    def test_load_training_config(self) -> None:
        """Test that training.yaml loads correctly with expected keys."""
        config = load_training_config()
        assert config["num_iterations"] == 30000
        assert config["sh_degree"] == 3
        assert "densify_from_iter" in config
        assert "opacity_reset_interval" in config
        assert "learning_rate" in config
        assert "lr_schedule" in config

    def test_load_config_missing_file(self) -> None:
        """Test that loading a missing config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_valid_yaml(self) -> None:
        """Test loading a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key1: value1\nkey2: 42\n")
            f.flush()
            config = load_config(Path(f.name))
        assert config["key1"] == "value1"
        assert config["key2"] == 42
        Path(f.name).unlink()


class TestGetDatasetConfig:
    """Tests for get_dataset_config."""

    def test_get_known_dataset(self) -> None:
        """Test retrieving a known dataset config."""
        config = get_dataset_config("ggrt")
        assert config["name"] == "GGRt"
        assert "description" in config

    def test_get_unknown_dataset_raises(self) -> None:
        """Test that requesting an unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_config("nonexistent_dataset")


class TestGetProjectRoot:
    """Tests for get_project_root."""

    def test_project_root_contains_pyproject(self) -> None:
        """Test that the project root contains pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_project_root_contains_configs(self) -> None:
        """Test that the project root contains configs directory."""
        root = get_project_root()
        assert (root / "configs").is_dir()


class TestTrainingConfigValues:
    """Tests for training.yaml specific values."""

    def test_learning_rates(self) -> None:
        """Test that learning rates are structured correctly."""
        config = load_training_config()
        lr = config["learning_rate"]
        assert "position" in lr
        assert "feature" in lr
        assert "opacity" in lr
        assert "scaling" in lr
        assert "rotation" in lr
        assert all(isinstance(v, float) for v in lr.values())

    def test_save_iterations(self) -> None:
        """Test that save_iterations is a list of integers."""
        config = load_training_config()
        save_iters = config["save_iterations"]
        assert isinstance(save_iters, list)
        assert all(isinstance(v, int) for v in save_iters)

    def test_densification_params(self) -> None:
        """Test densification parameters are present and valid."""
        config = load_training_config()
        assert config["densify_from_iter"] < config["densify_until_iter"]
        assert config["densify_interval"] > 0
        assert config["densify_grad_threshold"] > 0


class TestDatasetsConfigValues:
    """Tests for datasets.yaml specific values."""

    def test_all_datasets_have_required_fields(self) -> None:
        """Test that all datasets have required metadata fields."""
        datasets = load_datasets_config()
        required_fields = {"name", "description", "paper_url", "source_url"}

        for name, config in datasets.items():
            for field in required_fields:
                assert field in config, f"Dataset '{name}' missing field '{field}'"


class TestFrameExtraction:
    """Tests for frame extraction with synthetic data."""

    def test_extract_frames_from_synthetic_video(self) -> None:
        """Test frame extraction using a synthetically generated video."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

        import numpy as np
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            video_path = tmpdir / "test_video.mp4"
            output_dir = tmpdir / "frames"

            # Create a small synthetic video (10 frames, 64x48)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
            for i in range(10):
                frame = np.full((48, 64, 3), fill_value=i * 25, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            assert video_path.exists()

            # Extract all frames
            frames = extract_frames(video_path, output_dir, every_n=1)
            assert len(frames) == 10
            assert all(p.exists() for p in frames)
            assert all(p.suffix == ".png" for p in frames)

    def test_extract_frames_with_max_frames(self) -> None:
        """Test that max_frames limits extraction."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

        import numpy as np
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            video_path = tmpdir / "test_video.mp4"
            output_dir = tmpdir / "frames"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
            for i in range(20):
                frame = np.full((48, 64, 3), fill_value=i * 10, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            frames = extract_frames(video_path, output_dir, every_n=1, max_frames=5)
            assert len(frames) == 5

    def test_extract_frames_missing_video(self) -> None:
        """Test that missing video raises FileNotFoundError."""
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        with pytest.raises(FileNotFoundError):
            extract_frames(Path("/nonexistent/video.mp4"), Path("/tmp/out"))

    def test_extract_frames_both_fps_and_every_n_raises(self) -> None:
        """Test that specifying both fps and every_n raises ValueError."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

        import numpy as np
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            video_path = tmpdir / "test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()

            with pytest.raises(ValueError):
                extract_frames(video_path, tmpdir / "out", fps=2.0, every_n=5)


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_parse_download(self) -> None:
        """Test parsing download subcommand."""
        from nerf_gs_playground.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["download", "--dataset", "ggrt"])
        assert args.command == "download"
        assert args.dataset == "ggrt"

    def test_parse_preprocess(self) -> None:
        """Test parsing preprocess subcommand."""
        from nerf_gs_playground.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["preprocess", "--images", "/data/images", "--method", "colmap"])
        assert args.command == "preprocess"
        assert args.images == "/data/images"
        assert args.method == "colmap"

    def test_parse_train(self) -> None:
        """Test parsing train subcommand."""
        from nerf_gs_playground.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--data", "/data/colmap", "--iterations", "5000"])
        assert args.command == "train"
        assert args.data == "/data/colmap"
        assert args.iterations == 5000

    def test_parse_view(self) -> None:
        """Test parsing view subcommand."""
        from nerf_gs_playground.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["view", "--model", "model.ply", "--port", "9090"])
        assert args.command == "view"
        assert args.model == "model.ply"
        assert args.port == 9090

    def test_parse_run(self) -> None:
        """Test parsing run subcommand."""
        from nerf_gs_playground.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "--images", "/data/imgs", "--no-viewer"])
        assert args.command == "run"
        assert args.images == "/data/imgs"
        assert args.no_viewer is True

    def test_no_command_prints_help(self) -> None:
        """Test that no subcommand exits with code 0."""
        from nerf_gs_playground.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
