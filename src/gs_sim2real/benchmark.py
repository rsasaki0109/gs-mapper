"""Benchmark framework for comparing 3DGS training backends.

Provides tools to measure and compare training performance across
different backends (gsplat vs nerfstudio), tracking metrics such as
training time, PSNR, SSIM, number of Gaussians, and GPU memory usage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    method: str  # "gsplat" or "nerfstudio"
    dataset: str
    num_iterations: int
    training_time_seconds: float
    final_psnr: float | None
    final_ssim: float | None
    final_num_gaussians: int | None
    peak_memory_mb: float | None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return asdict(self)


class Benchmark:
    """Compare training backends (gsplat vs nerfstudio)."""

    def __init__(self, data_dir: str, output_dir: str):
        """Initialize the benchmark.

        Args:
            data_dir: Directory containing input data for training.
            output_dir: Directory where benchmark outputs will be saved.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_gsplat(self, num_iterations: int = 1000, dataset_name: str = "default") -> BenchmarkResult:
        """Run gsplat training and measure performance.

        Args:
            num_iterations: Number of training iterations to run.
            dataset_name: Name label for this dataset run.

        Returns:
            BenchmarkResult with measured metrics.
        """
        from gs_sim2real.train.gsplat_trainer import GsplatTrainer

        output = self.output_dir / f"gsplat_{dataset_name}"
        trainer = GsplatTrainer()

        start = time.time()
        try:
            trainer.train(
                data_dir=str(self.data_dir),
                output_dir=str(output),
                num_iterations=num_iterations,
            )
            elapsed = time.time() - start

            # Try to read final metrics from training log
            psnr, ssim, num_g = self._read_gsplat_metrics(output)
        except Exception as e:
            logger.error("gsplat training failed: %s", e)
            elapsed = time.time() - start
            psnr, ssim, num_g = None, None, None

        memory = self._get_peak_memory()

        result = BenchmarkResult(
            method="gsplat",
            dataset=dataset_name,
            num_iterations=num_iterations,
            training_time_seconds=round(elapsed, 2),
            final_psnr=psnr,
            final_ssim=ssim,
            final_num_gaussians=num_g,
            peak_memory_mb=memory,
        )
        self.results.append(result)
        return result

    def run_nerfstudio(self, num_iterations: int = 1000, dataset_name: str = "default") -> BenchmarkResult:
        """Run nerfstudio splatfacto and measure performance.

        Args:
            num_iterations: Number of training iterations to run.
            dataset_name: Name label for this dataset run.

        Returns:
            BenchmarkResult with measured metrics.
        """
        from gs_sim2real.train.nerfstudio_trainer import NerfstudioTrainer

        output = self.output_dir / f"nerfstudio_{dataset_name}"
        trainer = NerfstudioTrainer()

        start = time.time()
        try:
            trainer.train(
                data_dir=str(self.data_dir),
                output_dir=str(output),
                num_iterations=num_iterations,
            )
            elapsed = time.time() - start
            psnr, ssim, num_g = None, None, None  # parse from nerfstudio output
        except Exception as e:
            logger.error("nerfstudio training failed: %s", e)
            elapsed = time.time() - start
            psnr, ssim, num_g = None, None, None

        result = BenchmarkResult(
            method="nerfstudio",
            dataset=dataset_name,
            num_iterations=num_iterations,
            training_time_seconds=round(elapsed, 2),
            final_psnr=psnr,
            final_ssim=ssim,
            final_num_gaussians=num_g,
            peak_memory_mb=self._get_peak_memory(),
        )
        self.results.append(result)
        return result

    def compare(self) -> str:
        """Generate comparison table.

        Returns:
            Formatted string table comparing all benchmark results.
        """
        if not self.results:
            return "No results to compare."

        header = (
            f"{'Method':<15} {'Dataset':<15} {'Iters':>6} {'Time(s)':>8} "
            f"{'PSNR':>7} {'SSIM':>7} {'Gaussians':>10} {'Mem(MB)':>8}"
        )
        lines = [header, "-" * len(header)]

        for r in self.results:
            psnr = f"{r.final_psnr:.2f}" if r.final_psnr is not None else "N/A"
            ssim = f"{r.final_ssim:.4f}" if r.final_ssim is not None else "N/A"
            num_g = f"{r.final_num_gaussians:,}" if r.final_num_gaussians is not None else "N/A"
            mem = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb is not None else "N/A"
            lines.append(
                f"{r.method:<15} {r.dataset:<15} {r.num_iterations:>6} "
                f"{r.training_time_seconds:>8.1f} {psnr:>7} {ssim:>7} {num_g:>10} {mem:>8}"
            )

        return "\n".join(lines)

    def save_results(self, path: str | None = None) -> None:
        """Save results to JSON.

        Args:
            path: Output path. If None, saves to output_dir/benchmark_results.json.
        """
        path = path or str(self.output_dir / "benchmark_results.json")
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def _read_gsplat_metrics(self, output_dir: Path) -> tuple[float | None, float | None, int | None]:
        """Try to read metrics from gsplat training output."""
        log_path = Path(output_dir) / "training_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            if log:
                last = log[-1] if isinstance(log, list) else log
                return last.get("psnr"), last.get("ssim"), last.get("num_gaussians")
        return None, None, None

    def _get_peak_memory(self) -> float | None:
        """Get peak GPU memory usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        except ImportError:
            pass
        return None
