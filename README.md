# gs-sim2real

Multi-dataset 3D Gaussian Splatting reconstruction playground.

One-command pipelines to go from robotics/autonomous driving dataset images to 3DGS training to interactive web viewer.

![Demo](docs/demo.gif)

---

ロボティクス・自動運転データセットから3D Gaussian Splattingの学習、Webビューアでの可視化までをワンコマンドで実行できるツールです。

## Live Demo

**https://rsasaki0109.github.io/gs-sim2real/**

The demo page features an interactive Three.js 3D point cloud viewer and Plotly.js training metrics charts. You can explore reconstructed scenes directly in the browser without any local setup.

GitHub Pages is deployed by [`.github/workflows/pages.yml`](/media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground/.github/workflows/pages.yml) on `push` to `main`, so changes in a feature branch do not appear until they are merged and the Pages workflow finishes.

The viewer now also accepts exported static scene bundles, so a trained PLY can
be published on GitHub Pages and opened as a browser-only 3DGS space.

## Concept

```
Images --> Preprocessing --> 3DGS Training --> Web Viewer
  |          (COLMAP /        (gsplat /        (viser)
  |           GGRt)          nerfstudio)
  |
  +-- GGRt (pose-free)
  +-- CoVLA (driving)
  +-- MCD (campus)
```

## Supported Datasets

| Dataset | Type | Description | Pose Required |
|---------|------|-------------|---------------|
| [GGRt](https://github.com/abdullahamdi/ggrt) | Pose-free 3DGS | Generalizable 3D Gaussian Splatting using Waymo, RealEstate10K, ACID | No |
| [CoVLA](https://github.com/tier4/CoVLA) | Driving scenes | Large-scale driving dataset with front camera images | Yes (COLMAP) |
| [MCD](https://mcdviral.github.io/) | Campus scenes | Multi-campus outdoor scenes with stereo, LiDAR, IMU | Yes (COLMAP) |

## Installation

```bash
git clone https://github.com/rsasaki0109/gs-sim2real.git
cd gs-sim2real
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
gs-sim2real export \
  --model outputs/train/point_cloud.ply \
  --format scene-bundle \
  --output docs/assets/my-scene \
  --bundle-asset-format binary \
  --scene-id my-scene \
  --label "My Scene"
```

Then either:

- add `docs/assets/my-scene/scene.json` to `docs/assets/scenes.json` so it appears in the default viewer tab list
- or open it directly with `https://<user>.github.io/gs-sim2real/?sceneManifest=assets/my-scene/scene.json`

The GitHub Pages viewer also accepts direct exported `.json` / `.bin` point
assets through the `Load Asset` button in `docs/index.html`.

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
