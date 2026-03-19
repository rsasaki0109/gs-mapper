"""Tests for the benchmark framework."""

from __future__ import annotations

import json
from pathlib import Path

from nerf_gs_playground.benchmark import Benchmark, BenchmarkResult


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict returns a dictionary with all fields."""
        result = BenchmarkResult(
            method="gsplat",
            dataset="test_scene",
            num_iterations=1000,
            training_time_seconds=42.5,
            final_psnr=28.5,
            final_ssim=0.92,
            final_num_gaussians=50000,
            peak_memory_mb=2048.0,
        )
        d = result.to_dict()

        assert d["method"] == "gsplat"
        assert d["dataset"] == "test_scene"
        assert d["num_iterations"] == 1000
        assert d["training_time_seconds"] == 42.5
        assert d["final_psnr"] == 28.5
        assert d["final_ssim"] == 0.92
        assert d["final_num_gaussians"] == 50000
        assert d["peak_memory_mb"] == 2048.0

    def test_to_dict_with_none_values(self) -> None:
        """to_dict handles None values correctly."""
        result = BenchmarkResult(
            method="nerfstudio",
            dataset="default",
            num_iterations=500,
            training_time_seconds=10.0,
            final_psnr=None,
            final_ssim=None,
            final_num_gaussians=None,
            peak_memory_mb=None,
        )
        d = result.to_dict()

        assert d["final_psnr"] is None
        assert d["final_ssim"] is None
        assert d["final_num_gaussians"] is None
        assert d["peak_memory_mb"] is None


class TestBenchmarkCompare:
    """Tests for Benchmark.compare() with mock results."""

    def test_compare_no_results(self, tmp_path: Path) -> None:
        """compare() returns a message when there are no results."""
        bench = Benchmark(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        assert bench.compare() == "No results to compare."

    def test_compare_with_results(self, tmp_path: Path) -> None:
        """compare() generates a formatted table with results."""
        bench = Benchmark(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        bench.results.append(
            BenchmarkResult(
                method="gsplat",
                dataset="scene_a",
                num_iterations=1000,
                training_time_seconds=30.0,
                final_psnr=28.5,
                final_ssim=0.92,
                final_num_gaussians=50000,
                peak_memory_mb=2048.0,
            )
        )
        bench.results.append(
            BenchmarkResult(
                method="nerfstudio",
                dataset="scene_a",
                num_iterations=1000,
                training_time_seconds=45.0,
                final_psnr=None,
                final_ssim=None,
                final_num_gaussians=None,
                peak_memory_mb=None,
            )
        )

        table = bench.compare()
        lines = table.split("\n")

        # Header + separator + 2 data rows
        assert len(lines) == 4
        assert "Method" in lines[0]
        assert "gsplat" in lines[2]
        assert "nerfstudio" in lines[3]
        assert "28.50" in lines[2]
        assert "N/A" in lines[3]

    def test_compare_with_none_metrics(self, tmp_path: Path) -> None:
        """compare() shows N/A for None metric values."""
        bench = Benchmark(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        bench.results.append(
            BenchmarkResult(
                method="gsplat",
                dataset="test",
                num_iterations=100,
                training_time_seconds=5.0,
                final_psnr=None,
                final_ssim=None,
                final_num_gaussians=None,
                peak_memory_mb=None,
            )
        )

        table = bench.compare()
        # Should contain N/A for all None fields
        assert table.count("N/A") == 4


class TestBenchmarkSaveResults:
    """Tests for Benchmark.save_results()."""

    def test_save_results_creates_json(self, tmp_path: Path) -> None:
        """save_results creates a valid JSON file."""
        bench = Benchmark(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        bench.results.append(
            BenchmarkResult(
                method="gsplat",
                dataset="test",
                num_iterations=100,
                training_time_seconds=5.0,
                final_psnr=25.0,
                final_ssim=0.85,
                final_num_gaussians=10000,
                peak_memory_mb=1024.0,
            )
        )

        out_path = str(tmp_path / "results.json")
        bench.save_results(out_path)

        assert Path(out_path).exists()
        with open(out_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["method"] == "gsplat"
        assert data[0]["final_psnr"] == 25.0

    def test_save_results_default_path(self, tmp_path: Path) -> None:
        """save_results uses default path when none is provided."""
        out_dir = tmp_path / "bench_out"
        bench = Benchmark(data_dir=str(tmp_path), output_dir=str(out_dir))
        bench.results.append(
            BenchmarkResult(
                method="gsplat",
                dataset="default",
                num_iterations=50,
                training_time_seconds=2.0,
                final_psnr=None,
                final_ssim=None,
                final_num_gaussians=None,
                peak_memory_mb=None,
            )
        )

        bench.save_results()

        default_path = out_dir / "benchmark_results.json"
        assert default_path.exists()
        with open(default_path) as f:
            data = json.load(f)
        assert len(data) == 1
