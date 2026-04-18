# 屋外 3D Gaussian Splatting 開発計画 / 引継ぎメモ

更新日: 2026-04-17（pose-import セッション後）

この文書は、`GS Mapper` リポジトリにおける屋外 3D Gaussian Splatting 対応の現在地を、**Claude / Copilot / その他のコーディングエージェント**がそのまま引き継げる粒度でまとめた handoff 文書です。リポジトリ直下の `CLAUDE.md` は開発コマンド早見、本書は **屋外パイプラインの文脈・判断・失敗の履歴**に重きを置きます。

## エージェント向け: 読み方

1. **初回**: §0 TL;DR → §6 blocker → §7 優先順位 → §8 触るファイル。
2. **データを手元で動かす**: §9 コマンド早見 → §10 GitHub Pages / 検証 → `CLAUDE.md`。
3. **同じ実験を繰り返さない**: §3 empirical findings（特に Autoware bag6 と COLMAP tuning の失敗）。
4. **品質の本丸**: §4 の MCD pose import と §10 の splat viewer。

## 0. TL;DR

- 公開名は `GS Mapper`、repo slug `rsasaki0109/gs-mapper`、Pages `https://rsasaki0109.github.io/gs-mapper/`
- Python import path は **まだ** `gs_sim2real`（rename しない）
- CLI は `gs-mapper`、旧 `gs-sim2real` は legacy alias として維持
- ローカル worktree `nerf-gs-playground` はそのまま
- 屋外向け 7 フェーズ（config / SH 全次数 / depth supervision / DUSt3R / LiDAR SLAM-GNSS-NMEA / dynamic mask / appearance-sky）は全て実装済み
- **2026-04 のセッションで Blocker 1（MCD pose import）と §16 の Pages デモ偽物問題は解決済み**（詳細は §4 / §10）

直近の最大の残課題は以下。

1. Priority B: **Waymo 実データ E2E が未検証**
2. Priority C: NMEA / GNSS / IMU robustness、depth / appearance / sky の比較評価

## 1. リポジトリの現在の識別子

| 種別 | 現在値 | 備考 |
|------|--------|------|
| Public name | `GS Mapper` | README / Pages / docs で使う名前 |
| GitHub repo | `rsasaki0109/gs-mapper` | remote slug 変更済み |
| GitHub Pages | `https://rsasaki0109.github.io/gs-mapper/` | |
| Python package | `gs_sim2real` | **まだ rename しない** |
| Main CLI | `gs-mapper` | |
| Legacy CLI | `gs-sim2real` | 互換 alias |
| Local worktree | `nerf-gs-playground` | そのまま |

## 2. 実装ステータス

### 2.1 元の 7 フェーズ（完了）

屋外用 config、LiDAR depth supervision、DUSt3R、LiDAR SLAM / trajectory import、appearance embedding、dynamic object masking、sky / background model。

### 2.2 2026-04 セッションで追加されたもの

| 項目 | ファイル | 要点 |
|------|---------|------|
| MCD `/tf`, `/tf_static`, `/camera_info`, `/gnss/fix` 抽出 | `src/gs_sim2real/datasets/mcd.py`, `ros_tf.py` | |
| `MCDLoader.merge_lidar_frames_to_world()` | `mcd.py` | per-frame NPY + TUM ENU + `T_base_lidar` で world 点群を合成。identity quaternion の場合は ENU 動きから yaw 推定 |
| `MCDLoader.colorize_lidar_world_from_images()` | `mcd.py` | 各カメラ画像に world LiDAR 点を投影 → bilinearly サンプリングした RGB を累積平均。points3D.txt が grey(128) ではなく実色で seed される（**もや晴らしの本丸**） |
| `MCDLoader.export_lidar_depth_per_image()` | `mcd.py` | 各学習画像に world LiDAR を投影、closest z を per-pixel 保持した float32 (H, W) depth map を `depth/<subdir>/<stem>.npy` に保存 |
| LiDAR-seeded COLMAP sparse (色付き) | `cli.py` `_mcd_gnss_sparse_import` + `_mcd_colorize_seed` | `points3D.txt` を実 LiDAR 点 + 画像由来 RGB で seed。`--mcd-skip-lidar-colorize` で opt-out |
| Nx6 点群の `points3D.txt` 書き出し | `preprocess/lidar_slam.py`, `preprocess/depth_from_lidar.py` | `_write_colmap` / `_write_colmap_multiview` が Nx6（xyz + RGB）を受けて色を書く |
| 新 CLI フラグ | `cli.py` | `--mcd-lidar-frame` / `--mcd-skip-lidar-seed` / `--mcd-skip-lidar-colorize` / `--mcd-export-depth`（preprocess/run/demo すべて） |
| gsplat depth supervision (`render_mode=RGB+D`) | `train/gsplat_trainer.py` | `_render_gsplat(want_depth=True)` で depth を返し、`depth_loss_weight * L1(rendered_depth, gt_depth)` を valid pixel でトレーニングに加算 |
| PLY→.splat 変換 (`ply_to_splat`) | `src/gs_sim2real/viewer/web_export.py` | antimatter15/splat 形式（32 B/gauss）。`normalize_target_extent` で世界スケール→viewer 適合サイズに縮小、`min_opacity` / `max_scale` フィルタで霧除去 |
| WebGL Gaussian Splat viewer | `docs/splat.html`, `docs/splat-viewer/main.js` | antimatter15/splat (MIT) を vendor。gzip 対応 dynamic buffer、`defaultViewMatrix` は outdoor scale、downsample しきい値は 30k |
| `_densify_and_prune` の mask off-by-one 修正 | `src/gs_sim2real/train/gsplat_trainer.py` | clone → split の順で tensor 拡張する際に旧サイズ mask で indexing する IndexError を解消 |
| Three.js viewer のソフトスプライト化 / helper スケール | `docs/index.html` | `PointsMaterial` を radial-alpha disc + sizeAttenuation。grid/axes は scene bounds に合わせて動的に再構築 |
| `configs/training_depth.yaml` | `configs/` | `depth_loss_weight: 0.1` の outdoor + depth 学習プリセット |
| `configs/datasets.yaml` に bag4 / bag6 | `configs/datasets.yaml` | S3 公開バケットの `autoware_leo_drive_bag4` / `bag6` を `scripts/download_datasets.py` から取得可 |

## 3. Public data で「実際に verified された」もの

### 3.1 Autoware Leo Drive - ISUZU sensor data

公開元: <https://autowarefoundation.github.io/autoware-documentation/main/datasets/#leo-drive---isuzu-sensor-data>

- **bag1**: `MCD → COLMAP → gsplat` E2E 成功（`outputs/outdoor_smoke_autoware_seq/`、プラン §6.1 の known-good 条件）。
- **bag6**: プラン初版では **image-only COLMAP で 2 registered images** 止まり（§4.1）。pose-import セッションで **GNSS + /tf_static + LiDAR seed により 180 registered images (3 カメラ × 60 frames) / 100,000 world points** 達成。さらに **250 frames × 3 cam × colorize + 30k iter depth supervision** で 1,024,794 Gaussians まで拡張済み（`outputs/bag6_depth_train/`）。
- **bag4**: 別ルート。**240 frames × 3 cam = 720 registered images + colorize (196k/200k) + per-image depth (720 maps) + 30k iter depth-supervised training** で 932,243 Gaussians を生成し、現在 `/splat.html` のライブ配信に採用中（`outputs/bag4_full_train/`）。

### 3.2 Verified だが品質面は弱いもの

- bag6 / bag4 ともに colorize + depth で Gaussian 数は百万級に到達。ブラウザで trajectory 沿いの 3D 構造は視認できるが、「街並み」として読めるほどは解像していない。学習量/カメラ数/長時間軌跡が足りていないと見ている。

### 3.3 実装済みだが public 実データ E2E 未確認のもの

| 項目 | 実装 | 実データ E2E |
|------|------|-------------|
| Waymo frame / depth / mask extraction | 済み | 未確認 |
| Waymo `run --preprocess-method waymo` | 済み | 未確認 |
| NMEA import + training path | 済み | 未確認 |

## 4. 最近の empirical findings

### 4.1 bag6 の image-only COLMAP チューニングは効かない（プラン初版からの履歴）

`every_n = 1 / 2 / 3 / 5 / 10 / 20`, `matching = sequential / exhaustive` いずれも **2 registered images** 止まり。matcher tuning では解けなかった。

### 4.2 pose-seeded import が効いた

`/tf_static` の `base_link → camera_top/camera_link` 変換 + GNSS 軌跡 + LiDAR 点群 world merge（200k 点）で COLMAP text model を直接書き出し → 3 カメラで 180 frames 全部 registered、100k 点 seed。これが本セッションの突破口。

### 4.3 MCD (mcdviral.github.io) NTU session17 も image-only COLMAP は 2 registered 止まり

`scripts/download_mcd_session.sh` で 633 MB rosbag1 (NTU #17) を取得できるようになった（Drive 確認フォーム自動突破）。ただしこのセッションは handheld D455 ステレオ + IMU のみで **GNSS / /tf_static は含まれない**。150 frame サブサンプルで image-only COLMAP を走らせても Autoware bag6 と同じ「2 registered images / 36 points」症状で詰まる。gsplat 側は 5295 gauss まで学習するが点群が貧弱で demo 品質に達しない。MCD を本格的に使うなら: (a) SLAM 済みポーズを外部で流し込む、(b) もっと画像間重複が多いセッションを選ぶ、(c) DUSt3R pose-free 経路に切り替える、のいずれか。

### 4.4 densification は 100k 級初期点でも Stable に走るよう修正済み

以前は iter 500 以降の densify で `IndexError: mask [N] != tensor [N+num_clone]` が即発生していた。本セッションの修正で bag1 30k iter → 244k Gaussians まで完走。

## 5. 各 subsystem の到達点

- **Training** (`train/gsplat_trainer.py`): `scene_extent` / `scene_auto_scale` / LiDAR point cloud bootstrap / SH 全次数 / depth supervision / appearance / sky / mask-aware L1。densify mask off-by-one 修正済み。
- **Waymo** (`datasets/waymo.py`): frame / depth / mask 抽出 + `to_colmap_format()` + CLI 接続。実 Waymo data 未検証。
- **MCD** (`datasets/mcd.py`): rosbag1/2、image/lidar/imu、camera_info、/tf、/tf_static、NavSatFix、`merge_lidar_frames_to_world`、`extract_lidar(save_timestamps=True)`。bag6 で完全 E2E。
- **LiDAR-SLAM / GNSS / NMEA** (`preprocess/lidar_slam.py`): TUM / KITTI / NMEA / ENU 変換 / yaw 補間。`_write_colmap` / `_write_colmap_multiview` は src==dst の image copy を skip（bag6 single-dir パス対応）。
- **COLMAP glue** (`preprocess/colmap.py`): custom path、CPU-only、sequential / exhaustive、single_camera_per_folder。sparse preflight は `colmap_ready.py`。
- **Smoke script** (`scripts/outdoor_smoke.sh`): `waymo` / `mcd` / `e2e` / `mcd-list` / `nmea` / `verify-colmap`。

## 6. 現在の最大 blocker

### Blocker（残っているもの）

- **NMEA/GNSS/IMU robustness**: GGA / RMC / ENU は入っているが、IMU quaternion / angular velocity 融合や日跨ぎ、logger 時刻ずれが未対応。
- **性能面**: splat viewer のライブ fps が 5–7 程度。100k 超 gaussian + HiDPI で shader 負荷がボトルネック。将来は WebGPU (mkkellogg/Spark 等) or sort/rasterize の optimise を検討。

### 非対象（意図的に scope 外）

- **Waymo real-data E2E**: code path / tests / `scripts/check_waymo_e2e_prereqs.sh` は揃っているが、Python 3.12 環境での SDK build 失敗 + Waymo Open Dataset の利用規約同意 + 手動 tfrecord ダウンロードが必要。プロジェクト方針として公開同意不要データ（Autoware Leo Drive）を優先するため、本セッションで意図的に非対象とした。必要になったら Python 3.10 venv を別途立てて `WAYMO_DATA_DIR` を指定すれば即動く状態。

### 解決済み（2026-04 セッション）

- ~~Blocker 1: MCD calibration / pose import~~ → bag6 で 2→180 registered、bag4 で 720 registered、**5-bag fusion (bag1 + bag2 + bag3 + bag4 + bag6) で 5040 registered**。`merge_lidar_frames_to_world` + `_mcd_gnss_sparse_import` + 3-camera extrinsics + colorize で完結。
- ~~§16.2 デモが campus 共有バイナリ~~ → **5-bag fusion** 由来の 80k gaussian `.splat` を本物の WebGL GS viewer (`/splat.html`) で配信中（15 cameras × 5040 frames, 1.43M Gaussians）。
- ~~「画面が灰色のもや」~~ → 画像由来 RGB による LiDAR 点群初期化 + per-image LiDAR depth supervision + per-image appearance embedding (scale, bias) で color std を 0.06 → 0.15–0.22 に、trajectory 沿いの 3D 構造を可視化。L1 loss も 0.32 → 0.20 と 38% 改善。
- ~~densify bug~~ → `_densify_and_prune` の clone/split 順序を修正、30k–50k iter 学習が安定化。
- ~~Multi-bag fusion~~ → `reference_origin` 共有 + `scripts/merge_mcd_sparse.py` で任意の N bag を 1 つの sparse に結合可能。
- ~~appearance embedding~~ → per-image (scale, bias) RGB affine を trainer に追加、`configs/training_appearance.yaml` で有効化。

## 7. 次の担当者に引き継ぐ順序

### 優先度 A（残り）

1. **WebGPU 版 viewer**: 調査したところ script tag 単体で使える WebGPU GS 実装は現時点で皆無（Spark は ESM import、PlayCanvas Web Components は GS 非対応、cvlab/epfl 版も bundler 必須）。vite 等のビルドステップを導入するか、antimatter15/splat WebGL2 で継続するかの判断が要る。
2. **MCD (mcdviral.github.io)**: 公開・登録不要だが 1 セッション 3.5〜51 GB と巨大。downloader は `aws s3 sync --no-sign-request` ではなく Google Drive 経由なので `configs/datasets.yaml` に入れられない。rosbag 形式が rosbag1 or 2 不明なので `MCDLoader` の AnyReader 互換を実データで確認する必要あり。
3. **NMEA/GNSS/IMU robustness**: IMU quaternion 融合、logger 時刻ずれ、日跨ぎ RMC を堅牢化。

### 優先度 B

1. Waymo real-data E2E（ユーザが Waymo Open Dataset の Terms of Use に同意し、Python 3.10/3.11 環境を立てられる場合のみ）。プロジェクト方針として Autoware 系が主軸のため後回し。
2. Waymo dynamic mask / depth の実使用評価。

### 既に対応済み（refactor の余地はあるが動く）

- ~~画像由来 initial color~~ → `colorize_lidar_world_from_images` で完了
- ~~depth supervision~~ → `_render_gsplat(want_depth=True)` + `configs/training_depth.yaml` で完了
- ~~Multi-bag fusion~~ → `--mcd-reference-origin` / `--mcd-reference-bag` CLI + `scripts/merge_mcd_sparse.py` で実装完了。**5-bag fusion (bag1+2+3+4+6) をライブデモに採用**。1000 images 超で自動的に lazy image loading へ切り替わる
- ~~appearance embedding~~ → per-image (scale, bias) の trainer 実装 + `configs/training_appearance.yaml` で完了
- ~~joint pose refinement (BA)~~ → 各学習画像に 6-DOF (so3 + translation) delta を学習可能化、`joint_pose_start_iter` から有効化。`configs/training_ba.yaml` で depth + appearance + pose の全部 on

## 8. 具体的に触るファイル

### Waymo E2E をやるなら

**前提条件**:

- Python 3.10 or 3.11（`waymo-open-dataset-tf-2-12-0` は Python 3.12 で `pkgutil.ImpImporter` 削除により build 失敗）
- `pip install "waymo-open-dataset-tf-2-12-0"` (TensorFlow 2.12 依存)
- Waymo Open Dataset の利用同意 + `*.tfrecord` ダウンロード（<https://waymo.com/open/download/>）

**セットアップ確認**:

```bash
bash scripts/check_waymo_e2e_prereqs.sh
```

[OK] / [WARN] / [MISS] で Python バージョン、SDK、入力データの状態を表示。

**触るファイル**:

- `src/gs_sim2real/datasets/waymo.py`
- `src/gs_sim2real/cli.py`
- `scripts/outdoor_smoke.sh`
- `tests/test_waymo.py`, `tests/test_cli.py`, `tests/test_waymo_prereqs_script.py`

### bag6 の色改善をやるなら

- `src/gs_sim2real/datasets/mcd.py`（`merge_lidar_frames_to_world` の拡張として image-based colorize）
- `src/gs_sim2real/preprocess/lidar_slam.py`（`_write_colmap` の points3D.txt に RGB を書く経路）
- `src/gs_sim2real/viewer/web_export.py`（`ply_to_splat` は現状 SH DC から RGB、変更不要）

### Splat viewer を深くいじるなら

- `docs/splat-viewer/main.js`（antimatter15/splat の vendored JS）
- 当該ファイルは外部 upstream なので、local patch は最小に。gzip content-length 対応、downsample しきい値、defaultViewMatrix の 3 箇所だけ触っている（git log で差分確認）。

## 9. 参考コマンド集

### 9.1 Repo sanity

```bash
gs-mapper --help
gs-sim2real --help
```

### 9.2 bag6 multi-camera pose-seeded 再現（本セッション）

```bash
gs-mapper preprocess \
  --images data/autoware_leo_drive_bag6 \
  --output outputs/bag6_multicam \
  --method mcd \
  --image-topic "/lucid_vision/camera_0/raw_image,/lucid_vision/camera_1/raw_image,/lucid_vision/camera_2/raw_image" \
  --lidar-topic /sensing/lidar/concatenated/pointcloud \
  --imu-topic /sensing/imu/imu_data \
  --gnss-topic /gnss/fix \
  --mcd-seed-poses-from-gnss \
  --mcd-tf-use-image-stamps \
  --extract-lidar \
  --max-frames 60 \
  --every-n 2 \
  --matching sequential \
  --no-gpu

gs-mapper train --data outputs/bag6_multicam --output outputs/bag6_multicam_train \
  --method gsplat --iterations 15000 --config configs/training.yaml
```

### 9.3 PLY → .splat → Pages デプロイ

```python
from gs_sim2real.viewer.web_export import ply_to_splat
ply_to_splat(
    "outputs/bag6_multicam_train/point_cloud.ply",
    "docs/assets/outdoor-demo/outdoor-demo.splat",
    max_points=60000,
    normalize_target_extent=30.0,
)
```

## 10. GitHub Pages・デモ・E2E の現状

### 10.1 ライブ URL

- サイト: `https://rsasaki0109.github.io/gs-mapper/`
- Three.js 点ビューア（outdoor タブ）: `?scene=outdoor-demo`
- **本物の 3D Gaussian Splat viewer**: `/splat.html`（antimatter15/splat vendored）

### 10.2 Outdoor GS Demo の中身（2026-04 セッション後）

- `docs/assets/outdoor-demo/outdoor-demo.splat` は **5-bag fusion (bag1+2+3+4+6) の実学習結果 80k gaussians**（1.43M Gaussians をフィルタ: `min_opacity ≥ 0.3`, `max_scale ≤ 2 m`, 世界スケール 500 m を 30 単位に正規化）。image-projected RGB 初期化 + per-image LiDAR depth supervision + per-image appearance embedding (scale, bias)。
- `docs/assets/outdoor-demo/outdoor-demo.points.bin`（Three.js 用）は bag1 30k iter の密点群（244k → 60k subsample）。
- 「campus-gallery は 1 枚の写真から合成した擬似点群」という事実は変わっていないので、**ギャラリー系と outdoor-demo の性質差は大きい**。ユーザに説明する際は注意。

### 10.3 Splat viewer の限界

- 100k → 60k に絞ってもレンダ fps は HiDPI 環境で数〜数十 fps。`downsample=1/dpr` から `downsample=1` への切り替えで改善済み。
- 本物の 3D Gaussian rendering（SH 視点依存 / 異方性 2D splat）を行っているが、**scene が駅や道路として識別できるレベルではない**。これは学習量（60 frames × 3 cam / 15k iter）と初期色の質の問題で、viewer 側の問題ではない。

### 10.4 CI と Pages E2E

- Lint: `ruff` + `bash -n`。
- Playwright 系 E2E はデフォで skip（`PLAYWRIGHT_E2E=1` で有効化）。
- `.github/workflows/pages-e2e.yml` は手動実行時に `pages_url` を指定可能。

## 11. validation / test

**推奨の一回まわし（PR 前・引継ぎ時）**

```bash
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
pytest tests/ -v
```

2026-04 時点で main の pytest は 285 passed。

**屋外・データ関連で特に重要なファイル**

- `tests/test_waymo.py`, `tests/test_mcd.py`, `tests/test_lidar_slam.py`, `tests/test_colmap.py`
- `tests/test_pose_free.py`, `tests/test_cli.py`
- `tests/test_pages_assets.py`（outdoor-demo bundle の manifest と binary の存在を検証）
- `tests/e2e/test_github_pages_outdoor.py`（`PLAYWRIGHT_E2E=1` のときのみ）

## 12. 現在の環境メモ

- `COLMAP` は user-space にインストール済み。`--colmap-path` と `--no-gpu` が使える。
- system Python に `pip install --break-system-packages -e .` を利用。
- optional dependency: `dust3r`、`mcd`（rosbag ingest）、Waymo (`waymo-open-dataset-tf-2-12-0`)。

## 13. 今後の non-goal / 触らない方がよいもの

- `gs_sim2real` package path の rename
- local worktree directory 名の rename
- DreamWalker 系の別機能の大規模整理
- `gsplat_trainer.py` の全面分割
- antimatter15/splat の vendored コードへの大改修（upstream が sparse に続いていないので local patch は必要最小限に）

## 14. 関連ドキュメント（このファイルとの役割分担）

| ファイル | 内容 |
|----------|------|
| `CLAUDE.md` | 開発者向けコマンド早見（lint, pytest, E2E, verify-colmap 等）。**毎回の作業入口** |
| `README.md` | 公開向け概要、データセット表、Autoware smoke、ディレクトリ構成 |
| **本書** (`docs/plan_outdoor_gs.md`) | **屋外パイプラインの文脈・判断・失敗の履歴・長引きそうな blocker** |

エージェントは **実装の細部はソースと `CLAUDE.md`、方針と地雷は本書** を参照すると迷いにくい。
