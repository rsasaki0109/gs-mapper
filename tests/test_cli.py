"""Tests for CLI entry point and subcommand help."""

from __future__ import annotations

import pytest

from nerf_gs_playground.cli import main


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

    def test_cli_no_command(self) -> None:
        """Running main with no arguments exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0
