#!/usr/bin/env python3
"""Generate demo images for gs-sim2real README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)


def generate_training_metrics():
    """Generate training metrics visualization from docs/training_metrics.json."""
    with open("docs/training_metrics.json") as f:
        data = json.load(f)

    scenes = list(data.keys())
    scene_colors = {"street": "#2563eb", "campus": "#16a34a", "indoor": "#ea580c"}
    scene_labels = {"street": "Street (CoVLA)", "campus": "Campus (MCD)", "indoor": "Indoor (HM3D)"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("3D Gaussian Splatting Training Metrics", fontsize=16, fontweight="bold", y=0.98)

    metrics = [
        ("loss", "Loss", "Training Loss"),
        ("psnr", "PSNR (dB)", "Peak Signal-to-Noise Ratio"),
        ("ssim", "SSIM", "Structural Similarity"),
        ("num_gaussians", "Count", "Number of Gaussians"),
    ]

    for idx, (key, ylabel, title) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        for scene in scenes:
            iters = data[scene]["iterations"]
            vals = data[scene][key]
            color = scene_colors.get(scene, "#666666")
            label = scene_labels.get(scene, scene.capitalize())
            ax.plot(iters, vals, color=color, label=label, linewidth=1.5, alpha=0.9)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("docs/demo_training.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved docs/demo_training.png")
    plt.close()


def generate_pipeline_overview():
    """Generate a pipeline overview diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Pipeline stages
    stages = [
        (1.0, 1.5, "Images\n(CoVLA / MCD)", "#f0f4ff", "#2563eb"),
        (3.5, 1.5, "COLMAP\nPreprocessing", "#f0fdf4", "#16a34a"),
        (6.0, 1.5, "3DGS\nTraining", "#fff7ed", "#ea580c"),
        (8.5, 1.5, "Web\nViewer", "#faf5ff", "#7c3aed"),
    ]

    box_w, box_h = 1.8, 1.6

    for x, y, text, facecolor, edgecolor in stages:
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.15",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=12, fontweight="bold", color=edgecolor)

    # Arrows between stages
    arrow_style = mpatches.ArrowStyle("->", head_length=0.4, head_width=0.25)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + box_w / 2 + 0.05
        x2 = stages[i + 1][0] - box_w / 2 - 0.05
        y = 1.5
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle=arrow_style, color="#64748b", lw=2))

    # Dataset labels below
    datasets = [
        (1.0, 0.25, "CoVLA  |  MCD  |  GGRt"),
        (3.5, 0.25, "SfM point cloud"),
        (6.0, 0.25, "gsplat / nerfstudio"),
        (8.5, 0.25, "viser (browser)"),
    ]
    for x, y, text in datasets:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#94a3b8", style="italic")

    fig.suptitle("Pipeline Overview", fontsize=14, fontweight="bold", y=0.95)

    plt.tight_layout()
    plt.savefig("docs/demo_pipeline.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("Saved docs/demo_pipeline.png")
    plt.close()


if __name__ == "__main__":
    generate_training_metrics()
    generate_pipeline_overview()
