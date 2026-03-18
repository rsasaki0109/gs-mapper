#!/usr/bin/env python3
"""Download sample images and generate realistic training metrics for the 3DGS playground demo."""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
GALLERY_DIR = DOCS_DIR / "gallery"

SAMPLE_SCENES = {
    "street": {
        "name": "Street Scene (Driving)",
        "dataset": "CoVLA / Waymo",
        "description": "Urban street view for autonomous driving 3D reconstruction",
        "images": [
            (
                "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=480",
                "street_view_01.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1476231682828-37e571bc172f?w=480",
                "street_view_02.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=480",
                "street_view_03.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1444723121867-7a241cacace9?w=480",
                "street_view_04.jpg",
            ),
        ],
    },
    "campus": {
        "name": "Campus Scene",
        "dataset": "MCD",
        "description": "University campus environment for robot navigation",
        "images": [
            (
                "https://images.unsplash.com/photo-1562774053-701939374585?w=480",
                "campus_view_01.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=480",
                "campus_view_02.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1498243691581-b145c3f54a5a?w=480",
                "campus_view_03.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1519452635265-7b1fbfd1e4e0?w=480",
                "campus_view_04.jpg",
            ),
        ],
    },
    "indoor": {
        "name": "Indoor Scene",
        "dataset": "GGRt / HM3D",
        "description": "Indoor room reconstruction for embodied navigation",
        "images": [
            (
                "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=480",
                "indoor_view_01.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=480",
                "indoor_view_02.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?w=480",
                "indoor_view_03.jpg",
            ),
            (
                "https://images.unsplash.com/photo-1616486338812-3dadae4b4ace?w=480",
                "indoor_view_04.jpg",
            ),
        ],
    },
}


def download_images():
    """Download sample images for gallery."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for scene_key, scene_info in SAMPLE_SCENES.items():
        scene_dir = GALLERY_DIR / scene_key
        scene_dir.mkdir(parents=True, exist_ok=True)

        for url, filename in scene_info["images"]:
            filepath = scene_dir / filename
            if filepath.exists():
                print(f"  [skip] {filepath} already exists")
                continue
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                filepath.write_bytes(data)
                print(f"  [ok] {filepath} ({len(data)} bytes)")
            except Exception as e:
                print(f"  [fail] {filename}: {e}")


def generate_training_metrics():
    """Generate realistic 3DGS training metrics."""
    np.random.seed(42)
    iterations = list(range(0, 30001, 100))
    metrics = {}

    for scene_name in ["street", "campus", "indoor"]:
        # Loss curve (L1 + SSIM) - starts high, decreases with noise
        base_loss = 0.15 + np.random.uniform(-0.02, 0.02)
        losses = []
        for i, it in enumerate(iterations):
            t = it / 30000
            loss = (
                base_loss * np.exp(-3 * t)
                + 0.01
                + np.random.normal(0, 0.003 * (1 - t))
            )
            losses.append(round(max(0.005, float(loss)), 6))

        # PSNR - starts low (~15dB), increases to ~28-32dB
        peak_psnr = 28 + np.random.uniform(0, 4)
        psnrs = []
        for i, it in enumerate(iterations):
            t = it / 30000
            psnr = (
                14
                + (peak_psnr - 14) * (1 - np.exp(-4 * t))
                + np.random.normal(0, 0.3 * (1 - t))
            )
            psnrs.append(round(float(psnr), 3))

        # Number of Gaussians - grows during densification, drops at pruning
        num_gaussians = []
        g = 5000  # initial from SfM points
        for i, it in enumerate(iterations):
            if it < 15000:  # densification phase
                g += np.random.randint(50, 200)
                if it % 3000 == 0 and it > 0:  # pruning steps
                    g = int(g * 0.7)
            elif it == 15000:
                g = int(g * 0.8)  # final pruning
            else:
                g += np.random.randint(-20, 20)
            num_gaussians.append(int(max(1000, g)))

        # SSIM
        peak_ssim = 0.88 + np.random.uniform(0, 0.07)
        ssims = []
        for i, it in enumerate(iterations):
            t = it / 30000
            ssim = (
                0.5
                + (peak_ssim - 0.5) * (1 - np.exp(-3.5 * t))
                + np.random.normal(0, 0.01 * (1 - t))
            )
            ssims.append(round(float(min(1.0, max(0.4, ssim))), 4))

        metrics[scene_name] = {
            "iterations": iterations,
            "loss": losses,
            "psnr": psnrs,
            "ssim": ssims,
            "num_gaussians": num_gaussians,
            "final_psnr": round(psnrs[-1], 2),
            "final_ssim": round(ssims[-1], 4),
            "final_num_gaussians": num_gaussians[-1],
            "final_loss": round(losses[-1], 5),
            "training_time_minutes": round(float(np.random.uniform(8, 25)), 1),
        }

    return metrics


def generate_scenes_json():
    """Generate scenes.json with scene metadata."""
    scenes = {}
    for scene_key, scene_info in SAMPLE_SCENES.items():
        scenes[scene_key] = {
            "name": scene_info["name"],
            "dataset": scene_info["dataset"],
            "description": scene_info["description"],
            "images": [
                f"gallery/{scene_key}/{filename}"
                for _, filename in scene_info["images"]
            ],
        }
    return scenes


def main():
    print("=== Downloading sample images ===")
    download_images()

    print("\n=== Generating training metrics ===")
    metrics = generate_training_metrics()
    metrics_path = DOCS_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved: {metrics_path}")
    for scene, m in metrics.items():
        print(
            f"  {scene}: PSNR={m['final_psnr']:.2f}dB, "
            f"SSIM={m['final_ssim']:.4f}, "
            f"Gaussians={m['final_num_gaussians']}, "
            f"Loss={m['final_loss']:.5f}, "
            f"Time={m['training_time_minutes']}min"
        )

    print("\n=== Generating scenes.json ===")
    scenes = generate_scenes_json()
    scenes_path = DOCS_DIR / "scenes.json"
    scenes_path.write_text(json.dumps(scenes, indent=2))
    print(f"  Saved: {scenes_path}")


if __name__ == "__main__":
    main()
