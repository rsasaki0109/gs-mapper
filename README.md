# nerf-gs-playground

Multi-dataset 3D Gaussian Splatting reconstruction playground.

One-command pipelines to go from robotics/autonomous driving dataset images to 3DGS training to interactive web viewer.

---

ロボティクス・自動運転データセットから3D Gaussian Splattingの学習、Webビューアでの可視化までをワンコマンドで実行できるツールです。

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
git clone https://github.com/rsasaki0109/nerf-gs-playground.git
cd nerf-gs-playground
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

## Quick Start

### Full pipeline (download + preprocess + train + view)

```bash
gs-playground run ggrt --backend gsplat
```

### Step-by-step

```bash
# Download a dataset
gs-playground download covla --dest data/

# Preprocess with COLMAP
gs-playground preprocess --data-dir data/covla --output-dir outputs/colmap

# Train 3DGS
gs-playground train --data-dir outputs/colmap --backend gsplat --iterations 30000

# View the result
gs-playground view outputs/train/point_cloud.ply --port 8080
```

### Using scripts

```bash
python scripts/download_datasets.py --dataset mcd --dest data/
python scripts/run_demo.py --dataset ggrt --backend gsplat
```

## Project Structure

```
nerf-gs-playground/
├── src/nerf_gs_playground/
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
