"""Tests for frame extraction preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest


def _create_test_video(video_path: Path, num_frames: int = 5, size: tuple[int, int] = (64, 48)) -> Path:
    """Create a tiny synthetic video for testing.

    Args:
        video_path: Path where the video will be written.
        num_frames: Number of frames to write.
        size: (width, height) of the video.

    Returns:
        Path to the created video file.
    """
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 25 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestExtractFrames:
    """Tests for extract_frames."""

    @pytest.fixture(autouse=True)
    def _check_cv2(self) -> None:
        pytest.importorskip("cv2")

    def test_extract_frames_creates_files(self, tmp_path: Path) -> None:
        """Extract frames from a tiny video and verify output files are created."""
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        video_path = _create_test_video(tmp_path / "video.mp4", num_frames=5)
        output_dir = tmp_path / "frames"

        frames = extract_frames(video_path, output_dir, every_n=1)

        assert len(frames) == 5
        assert all(p.exists() for p in frames)
        assert all(p.suffix == ".png" for p in frames)

    def test_extract_frames_max_frames(self, tmp_path: Path) -> None:
        """Setting max_frames=2 extracts exactly 2 frames."""
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        video_path = _create_test_video(tmp_path / "video.mp4", num_frames=10)
        output_dir = tmp_path / "frames"

        frames = extract_frames(video_path, output_dir, every_n=1, max_frames=2)

        assert len(frames) == 2

    def test_extract_frames_nonexistent_video(self) -> None:
        """extract_frames raises FileNotFoundError for a missing video."""
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        with pytest.raises(FileNotFoundError):
            extract_frames(Path("/nonexistent/video.mp4"), Path("/tmp/out"))

    def test_extract_frames_both_fps_and_every_n_raises(self, tmp_path: Path) -> None:
        """Specifying both fps and every_n raises ValueError."""
        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        video_path = _create_test_video(tmp_path / "video.mp4", num_frames=3)

        with pytest.raises(ValueError):
            extract_frames(video_path, tmp_path / "out", fps=2.0, every_n=5)

    def test_extract_frames_output_is_readable(self, tmp_path: Path) -> None:
        """Extracted frames can be read back as valid images."""
        import cv2

        from nerf_gs_playground.preprocess.extract_frames import extract_frames

        video_path = _create_test_video(tmp_path / "video.mp4", num_frames=3)
        output_dir = tmp_path / "frames"

        frames = extract_frames(video_path, output_dir, every_n=1)

        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            assert img is not None
            assert img.shape == (48, 64, 3)
