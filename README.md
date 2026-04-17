# GS Mapper

Large-scale 3D Gaussian Splatting mapper for robotics, driving, and campus-scale scenes.

One-command pipelines to go from robotics / autonomous-driving dataset images to 3DGS training to an interactive WebGL viewer.

![Demo](docs/demo.gif)

---

ロボティクス・自動運転データセットから 3D Gaussian Splatting の学習、Web ビューアでの可視化までをワンコマンドで実行できるツールです。Python モジュール名は `gs_sim2real` のまま維持し、旧 CLI の `gs-sim2real` も互換エイリアスとして残します。

## Live Demo

GitHub Pages:

- Three.js point viewer: **<https://rsasaki0109.github.io/gs-mapper/>**
- Real WebGL Gaussian Splat viewer: **<https://rsasaki0109.github.io/gs-mapper/splat.html>**

The splat page loads an outdoor scene trained with the MCD pose-import pipeline (GNSS + `/tf_static` + LiDAR-seeded COLMAP, image-projected RGB initialization, 50k-iter gsplat with LiDAR depth supervision) and renders real anisotropic 3D Gaussians, not raw splat centers. Override with `?url=<splat-path>`.

GitHub Pages is deployed by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) on `push` to `main`.

## Concept

```
Images --> Preprocessing --> 3DGS Training --> Web Viewer
  |          (COLMAP /        (gsplat /        (Three.js
  |           GGRt / MCD      nerfstudio)      + antimatter15/splat)
  |           pose-import)
  |
  +-- GGRt (pose-free)
  +-- CoVLA (driving)
  +-- MCD (campus)
  +-- Autoware Leo Drive ISUZU bags (outdoor driving)
```

## Supported Datasets

| Dataset | Type | Description | Pose |
|---------|------|-------------|------|
| [GGRt](https://github.com/abdullahamdi/ggrt) | Pose-free 3DGS | Generalizable Gaussian Splatting over Waymo / RealEstate10K / ACID | Not required |
| [CoVLA](https://github.com/tier4/CoVLA) | Driving scenes | Large-scale driving dataset with front camera images | COLMAP |
| [MCD](https://mcdviral.github.io/) | Campus rosbags | Multi-campus outdoor scenes with stereo, LiDAR, IMU | COLMAP / GNSS + `/tf_static` pose-import |
| Autoware Leo Drive ISUZU | Outdoor driving rosbag | Public multi-camera + LiDAR + GNSS/INS bags from Autoware Foundation | GNSS + `/tf_static` pose-import (see `configs/datasets.yaml`) |

## Installation

```bash
git clone https://github.com/rsasaki0109/gs-mapper.git
cd gs-mapper
pip install -e ".[dev]"
```

For nerfstudio backend:
```bash
pip install -e ".[nerfstudio]"
```

For gsplat backend:
```bash
pip install -e ".[gsplat]"
```

## Demo App

A browser-based Streamlit interface is available for the full 3DGS pipeline
(image upload, COLMAP preprocessing, training, 3D viewer, export).

```bash
pip install -e ".[app]"
streamlit run app.py
```

The app opens at `http://localhost:8501` with a sidebar for pipeline settings
and tabs for each stage of the workflow.

## GitHub Pages 3DGS Viewer

Export a trained PLY as a static scene bundle:

```bash
gs-mapper export \
  --model outputs/train/point_cloud.ply \
  --format scene-bundle \
  --output docs/assets/my-scene \
  --bundle-asset-format binary \
  --scene-id my-scene \
  --label "My Scene"
```

To publish the same PLY into the real WebGL splat viewer (`docs/splat.html`):

```python
from gs_sim2real.viewer.web_export import ply_to_splat

ply_to_splat(
    "outputs/train/point_cloud.ply",
    "docs/assets/my-scene/my-scene.splat",
    max_points=60000,
    normalize_target_extent=30.0,
    min_opacity=0.5,
    max_scale=2.0,
)
```

Open `https://<pages>/splat.html?url=assets/my-scene/my-scene.splat` after deploy.

## Outdoor pipeline quickstart (Autoware Leo Drive)

Download a public bag, preprocess with pose-import + image-projected RGB + LiDAR depth maps, train with depth supervision:

```bash
python3 scripts/download_datasets.py --dataset autoware_leo_drive_bag2 --dest data/

gs-mapper preprocess \
  --images data/autoware_leo_drive_bag2 \
  --output outputs/bag2 \
  --method mcd \
  --image-topic "/lucid_vision/camera_0/raw_image,/lucid_vision/camera_1/raw_image,/lucid_vision/camera_2/raw_image" \
  --lidar-topic /sensing/lidar/concatenated/pointcloud \
  --imu-topic /sensing/imu/imu_data \
  --gnss-topic /gnss/fix \
  --mcd-seed-poses-from-gnss \
  --mcd-tf-use-image-stamps \
  --mcd-export-depth \
  --extract-lidar \
  --max-frames 400 --every-n 1 \
  --matching sequential --no-gpu

gs-mapper train --data outputs/bag2 --output outputs/bag2_train \
  --method gsplat --iterations 50000 \
  --config configs/training_depth_long.yaml
```

See `docs/plan_outdoor_gs.md` for the detailed outdoor pipeline handoff (status, blockers, known-good conditions).

Then either:

- add `docs/assets/my-scene/scene.json` to `docs/assets/scenes.json` so it appears in the default viewer tab list
- or open it directly with `https://<user>.github.io/gs-mapper/?sceneManifest=assets/my-scene/scene.json`

The GitHub Pages viewer also accepts direct exported `.json` / `.bin` point
assets through the `Load Asset` button in `docs/index.html`.

The public demo tabs in `docs/assets/` are regenerated from the repo gallery
with:

```bash
python3 scripts/build_pages_gallery_scenes.py
```

If the live site still shows the old content after merge, check the latest
`Deploy to GitHub Pages` action run and wait for that deployment to complete.

## Prototype Apps

This repository now also hosts browser-first creative prototypes built on top of
Gaussian splat worlds.

- `projects/` contains Unity-native experiments such as DreamWalker.
- `apps/` contains browser-native experiences such as DreamWalker Live.
- `shared/` contains code shared across prototypes.
- `docs/prototypes/dreamwalker-live.md` documents the streamer/photo/browser residency direction.
- `docs/prototypes/dreamwalker-robotics.md` documents the sibling robotics simulation direction.
- `gs-sim2real robotics-node` provides a ROS2-side scaffold that consumes DreamWalker relay topics.
- `configs/robotics/dreamwalker_zones.sample.json` is a starter semantic-zone / costmap config for ROS2-side navigation experiments.

## Experiment-Driven Development

This repository now keeps a small stable core and a discardable experiment lab
for seams such as localization alignment, render backend selection, localization estimate import, localization review bundle import, query cancellation policy, query coalescing policy, query error mapping, query source identity, query transport selection, query request import, query queue policy, query timeout policy, query response build, live localization stream import, route capture bundle import, and sim2real websocket protocol import.

- [docs/experiments.md](docs/experiments.md) tracks the latest side-by-side comparison.
- [docs/decisions.md](docs/decisions.md) records why strategies were kept or deferred.
- [docs/interfaces.md](docs/interfaces.md) defines the minimum stable interface that production code may depend on.

Refresh the comparison and regenerate those docs with either experiment command:

```bash
gs-sim2real experiment-localization-alignment --write-docs --output outputs/localization-alignment-experiment-report.json
gs-sim2real experiment-render-backend-selection --write-docs --output outputs/render-backend-selection-experiment-report.json
gs-sim2real experiment-localization-import --write-docs --output outputs/localization-estimate-import-experiment-report.json
gs-sim2real experiment-query-transport-selection --write-docs --output outputs/query-transport-selection-experiment-report.json
gs-sim2real experiment-query-request-import --write-docs --output outputs/query-request-import-experiment-report.json
gs-sim2real experiment-query-cancellation-policy --write-docs --output outputs/query-cancellation-policy-experiment-report.json
gs-sim2real experiment-query-coalescing-policy --write-docs --output outputs/query-coalescing-policy-experiment-report.json
gs-sim2real experiment-query-error-mapping --write-docs --output outputs/query-error-mapping-experiment-report.json
gs-sim2real experiment-query-queue-policy --write-docs --output outputs/query-queue-policy-experiment-report.json
gs-sim2real experiment-query-source-identity --write-docs --output outputs/query-source-identity-experiment-report.json
gs-sim2real experiment-query-timeout-policy --write-docs --output outputs/query-timeout-policy-experiment-report.json
gs-sim2real experiment-query-response-build --write-docs --output outputs/query-response-build-experiment-report.json
gs-sim2real experiment-live-localization-stream-import --write-docs --output outputs/live-localization-stream-import-experiment-report.json
gs-sim2real experiment-route-capture-import --write-docs --output outputs/route-capture-bundle-import-experiment-report.json
gs-sim2real experiment-sim2real-websocket-protocol --write-docs --output outputs/sim2real-websocket-protocol-experiment-report.json
gs-sim2real experiment-localization-review-bundle-import --write-docs --output outputs/localization-review-bundle-import-experiment-report.json
```

## Docker

Build and run with Docker Compose (requires NVIDIA Container Toolkit):

```bash
docker compose up --build
```

The Streamlit app will be available at `http://localhost:8501`.

To run a specific command inside the container instead of the default app:

```bash
docker compose run playground gs-sim2real run ggrt --backend gsplat
```

Dataset files in `data/` and training outputs in `outputs/` are mounted as volumes, so they persist on the host.

## Quick Start

### Full pipeline (download + preprocess + train + view)

```bash
gs-sim2real run ggrt --backend gsplat
```

### Step-by-step

```bash
# Download a dataset
gs-sim2real download covla --dest data/

# Preprocess with COLMAP
gs-sim2real preprocess --data-dir data/covla --output-dir outputs/colmap

# Train 3DGS
gs-sim2real train --data-dir outputs/colmap --backend gsplat --iterations 30000

# View the result
gs-sim2real view outputs/train/point_cloud.ply --port 8080
```

### Using scripts

```bash
python scripts/download_datasets.py --dataset mcd --dest data/
python scripts/run_demo.py --dataset ggrt --backend gsplat
```

## Project Structure

```
gs-sim2real/
├── apps/                # Browser-native prototypes
├── docs/                # Prototype notes and setup docs
├── projects/            # Unity-native prototypes
├── shared/              # Shared runtime/package code
├── src/gs_sim2real/
│   ├── common/          # Config loading, download utilities
│   ├── preprocess/      # COLMAP, frame extraction
│   ├── train/           # gsplat, nerfstudio training wrappers
│   ├── viewer/          # Viser-based web viewer
│   └── cli.py           # Command-line interface
├── configs/             # Dataset and training YAML configs
├── scripts/             # Standalone helper scripts
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks
├── data/                # Downloaded datasets (gitignored)
└── outputs/             # Training outputs (gitignored)
```

## Citation

If you use this tool in your research, please cite the relevant dataset papers:

```bibtex
@article{li2024ggrt,
  title={GGRt: Towards Pose-free Generalizable 3D Gaussian Splatting in Real-time},
  author={Li, Hao and Jiang, Yuze and Gao, Rui and Luo, Changjian and Zhang, Zhi and Shao, Dingjiang and others},
  journal={arXiv preprint arXiv:2403.10147},
  year={2024}
}

@article{arai2024covla,
  title={CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving},
  author={Arai, Hidehisa and others},
  journal={arXiv preprint arXiv:2408.10680},
  year={2024}
}

@article{lim2024mcd,
  title={MCD: Diverse Large-Scale Multi-Campus Dataset for Robot Perception},
  author={Lim, Thien-Minh and others},
  journal={arXiv preprint arXiv:2403.11755},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
