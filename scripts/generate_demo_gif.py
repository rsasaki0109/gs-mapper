import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import json
from pathlib import Path

OUTPUT = Path("docs/demo.gif")
FRAMES = []


def save_frame(fig, duration_ms=2000):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1)
    img = img.convert("RGB")
    FRAMES.append((img, duration_ms))
    plt.close(fig)


# Load metrics
with open("docs/training_metrics.json") as f:
    metrics = json.load(f)

scene_colors = {"street": "#58a6ff", "campus": "#3fb950", "indoor": "#d29922"}
scene_labels = {"street": "Street (CoVLA)", "campus": "Campus (MCD)", "indoor": "Indoor (GGRt)"}

# --- Frame 1: Title ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.set_facecolor("#0d1117")
ax.text(0.5, 0.6, "gs-sim2real", fontsize=32, ha="center", color="white", fontweight="bold")
ax.text(0.5, 0.42, "Multi-dataset 3D Gaussian Splatting Reconstruction", fontsize=14, ha="center", color="#8b949e")
ax.text(0.5, 0.27, "COLMAP · gsplat · nerfstudio · viser", fontsize=11, ha="center", color="#58a6ff")
save_frame(fig, 2000)

# --- Frame 2: Pipeline ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.set_facecolor("white")
steps = ["Input\nImages", "COLMAP\nSfM", "Point\nCloud", "3DGS\nTraining", "Web\nViewer"]
step_colors = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f85149"]
for i, (step, color) in enumerate(zip(steps, step_colors)):
    x = 0.1 + i * 0.2
    rect = plt.Rectangle(
        (x - 0.07, 0.35), 0.14, 0.3, facecolor=color, alpha=0.15, edgecolor=color, linewidth=2, transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(x, 0.5, step, ha="center", va="center", fontsize=12, fontweight="bold", color=color, transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(x + 0.1, 0.5),
            xytext=(x + 0.06, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#666", lw=2),
        )
ax.set_title("3DGS Reconstruction Pipeline", fontsize=16, fontweight="bold", pad=20)
save_frame(fig, 3000)

# --- Frame 3: Input images ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Sample Input Images", fontsize=16, fontweight="bold")
scene_dirs = {"street": "docs/gallery/street", "campus": "docs/gallery/campus", "indoor": "docs/gallery/indoor"}
img_idx = 0
for scene, dir_path in scene_dirs.items():
    imgs = sorted(Path(dir_path).glob("*.jpg"))
    for img_path in imgs[:1]:
        if img_idx < 4:
            img = mpimg.imread(str(img_path))
            axes[img_idx].imshow(img)
            axes[img_idx].set_title(
                scene_labels.get(scene, scene), fontsize=10, color=scene_colors.get(scene, "#666"), fontweight="bold"
            )
            axes[img_idx].axis("off")
            img_idx += 1
# fill remaining
while img_idx < 4:
    imgs = sorted(Path("docs/gallery/street").glob("*.jpg"))
    if len(imgs) > img_idx:
        img = mpimg.imread(str(imgs[img_idx]))
        axes[img_idx].imshow(img)
        axes[img_idx].set_title("Street", fontsize=10, color="#58a6ff", fontweight="bold")
    axes[img_idx].axis("off")
    img_idx += 1
plt.tight_layout()
save_frame(fig, 3000)

# --- Frame 4: Loss curves ---
fig, ax = plt.subplots(figsize=(10, 6))
for scene in ["street", "campus", "indoor"]:
    ax.plot(
        metrics[scene]["iterations"],
        metrics[scene]["loss"],
        color=scene_colors[scene],
        label=scene_labels[scene],
        linewidth=2,
    )
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training Loss (L1 + SSIM)", fontsize=16, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_frame(fig, 2500)

# --- Frame 5: PSNR ---
fig, ax = plt.subplots(figsize=(10, 6))
for scene in ["street", "campus", "indoor"]:
    ax.plot(
        metrics[scene]["iterations"],
        metrics[scene]["psnr"],
        color=scene_colors[scene],
        label=scene_labels[scene],
        linewidth=2,
    )
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("PSNR (dB)", fontsize=12)
ax.set_title("PSNR Progression", fontsize=16, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_frame(fig, 2500)

# --- Frame 6: 3D point cloud ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
np.random.seed(42)
# Ground
n = 3000
x = np.random.uniform(-5, 5, n)
z = np.random.uniform(-5, 5, n)
y = np.random.normal(0, 0.05, n)
ax.scatter(x, y, z, c="#8B7355", s=0.5, alpha=0.5)
# Building
n = 2000
bx = np.random.uniform(-4, -1, n)
by = np.random.uniform(0, 4, n)
bz = np.random.uniform(-2, 1, n)
ax.scatter(bx, by, bz, c="#CD853F", s=1, alpha=0.6)
# Tree
n = 1500
angles = np.random.uniform(0, 2 * np.pi, n)
radii = np.random.uniform(0, 1.5, n)
tx = 3 + radii * np.cos(angles)
tz = 0 + radii * np.sin(angles)
ty = 2 + np.random.uniform(0, 2, n)
ax.scatter(tx, ty, tz, c="#228B22", s=1, alpha=0.6)
# Trunk
n = 200
ttx = 3 + np.random.normal(0, 0.1, n)
ttz = 0 + np.random.normal(0, 0.1, n)
tty = np.random.uniform(0, 2, n)
ax.scatter(ttx, tty, ttz, c="#8B4513", s=2, alpha=0.8)
ax.set_title("Reconstructed 3D Point Cloud", fontsize=16, fontweight="bold")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=25, azim=135)
plt.tight_layout()
save_frame(fig, 2500)

# --- Frame 7: Final ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.set_facecolor("#0d1117")
ax.text(0.5, 0.6, "github.com/rsasaki0109/gs-sim2real", fontsize=18, ha="center", color="#58a6ff")
ax.text(
    0.5,
    0.4,
    "pip install -e . && gs-sim2real run --images data/",
    fontsize=13,
    ha="center",
    color="#8b949e",
    family="monospace",
)
save_frame(fig, 2000)

# Save GIF
images = [f[0] for f in FRAMES]
durations = [f[1] for f in FRAMES]
images[0].save(str(OUTPUT), save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
print(f"Saved {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB, {len(FRAMES)} frames)")
