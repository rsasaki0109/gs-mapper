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

### 4.3.1 DUSt3R pose-free 経路は本実装済み（PR #55）、ただし NTU #17 では退化

`src/gs_sim2real/preprocess/pose_free.py` の `_run_dust3r` は以前 stub → simple fallback にフォールスルーしていたが、PR #55 で本実装に差し替えた（`run_dust3r_inference` + `write_colmap_sparse`、`scripts/run_dust3r.py` は CLI ラッパ）。DUSt3R の pairwise pointmap + `PointCloudOptimizer` の MST-seeded global align を 300 iter 回し、per-image PINHOLE `cameras.txt` / c2w→w2c の `images.txt` / confidence-filtered `points3D.txt` を gsplat trainer が読める形で吐く。16 GB 級 GPU では inference 後に model weights を GPU から解放しないと aligner の stacked-pred tensor で OOM するので `del model` + `empty_cache()` を明示挿入。

NTU #17 にそのまま適用した場合、多くのフレームが translation ≈ 0 に退化する（image-only COLMAP と同じ parallax 不足症状）。代替として Autoware bag6 cam0 の 20 frame で回したところ 19/20 が非退化 trajectory を復元し、gsplat 3000 iter で 5.57M gauss まで収束。そこから antimatter15 splat format に 400k × 32 B = 12.8 MB で焼いた成果物が `docs/assets/outdoor-demo/outdoor-demo-dust3r.splat` で、`splat.html?url=...` 経由で GNSS+LiDAR supervised demo とトグル比較できる。GNSS も LiDAR も無しで image だけから出ている点が売り（L1 ≈ 0.15、supervised 版の 0.06〜0.08 には届かないが、ポーズなし構築の比較基準として bundled）。

### 4.3.2 MCD NTU #17 は image-only では詰む（3 通り試して全滅）— 原因は low inter-frame feature match

2026-04-19 に次の 4 経路を全部試した結果、NTU session17 は image-only で成立しない scene だと確定:

| 試行 | Frames | 結果 |
|------|--------|------|
| image-only COLMAP (150 frame 連続) | 150 color | 2 registered / 36 points（初回セッションで既知） |
| DUSt3R mono 30 frame complete | 30 color | 多数が translation=0、scene 非メトリック |
| DUSt3R stereo pair (infra1+infra2 × 10 pair) | 20 IR | 10/20 nonzero、scene extent ~0.2m（D455 baseline しか拾えず） |
| DUSt3R mono 18 frame spread (bag 全域) | 18 color | 3/18 nonzero（最悪。時間差が開くほど退化） |
| ORB-SLAM3 compact (`simple_visual_slam/run_mono --tum`, 600 frame) | 600 color | 初期化不能。`Initializer: Matches found: 2〜13` が全フレームで継続、>100 match に一度も到達せず |

最大要因は ORB の frame-to-frame match が常に <15 個しか取れないこと。原因候補: 歩行による motion blur、NTU キャンパスの繰り返し textures、handheld の揺れによる大きな viewpoint jump。いずれも image-only アプローチを根本的にブロックする。

**結論**: NTU #17 は pose injection 無しでは shippable demo にならない。次に MCD で demo を作るなら `(a)` 別 session 選定（GNSS 付き、重複多い Kirinyaga / Tuas など）、`(b)` MCDVIRAL の GT pose csv をダウンロードして `--trajectory-format tum` で直接食わせる、のどちらか。本セッションではここで撤退し、NTU #17 は "known-failing scene" として scratched。成果物は保持: `outputs/mcd_ntu17_stereo_dust3r/` など（retain しても demo 用途なし）。

### 4.3.3 GNSS+LiDAR 付き MCD session 候補（次セッション向け）

MCDVIRAL のダウンロードページを監査して GNSS (VN100/VN200) + LiDAR (Ouster OS1-64) + camera + /tf_static を全部揃えた session を size 順にリスト化した。NTU #17 撤退後の後継候補は次のとおり:

| Session | Size | Mount | Notes |
|---------|------|-------|-------|
| `tuhh_night_09` | **3.5 GB** | handheld | 最小。GNSS (VN200) + Ouster OS1-64 + D455b + D455t. 同セッションの `ltpb.bag` (312 KB) に calibration + /tf_static が入ると推測 |
| `tuhh_night_07` | 10.2 GB | handheld | |
| `tuhh_day_04`   | 12.5 GB | handheld | day light 条件 |
| `kth_night_05`  | 14.2 GB | handheld | |
| `ntu_day_02`    | **14.8 GB** | vehicle (ATV) | 最小の車載 session。Autoware bag 系と同種の GNSS + LiDAR + camera + /tf_static が期待できる |
| `tuhh_night_08` | 16.8 GB | handheld | |
| `ntu_night_13`  | 17.3 GB | vehicle | |
| ... | ... | ... | full list: https://mcdviral.github.io/Download.html |

ダウンロードは session 単位で Drive **folder** なので、単一 file ID しか扱わなかった `scripts/download_mcd_session.sh` ではなく `scripts/download_mcd_folder.sh` を使う（本 PR で追加）。folder は `https://drive.google.com/drive/folders/<ID>` の `<ID>` を渡せば `gdown --folder` で全 bag を再帰取得する:

```bash
scripts/download_mcd_folder.sh 1nEPiTXkVmLIhmBOVNpwSAEgnAXupnAxx data/mcd/tuhh_night_09/
```

取得後は既存の `--method mcd` + `--mcd-seed-poses-from-gnss` + `--mcd-tf-use-image-stamps` + `--mcd-export-depth` パイプラインがそのまま通る想定（Autoware bag 系で確認済）。ただし MCD handheld session は /tf_static の親 frame 名が Autoware と異なる可能性があるので、preprocess 実行時に `mcd.py` の TF lookup が拾えるか一度確認する（拾えなかった場合、`--mcd-reference-bag` 相当の frame override を足す）。

#### 4.3.3.a `tuhh_night_09` で分かった落とし穴 (2026-04-19)

実際に download してみて分かった MCD 側の注意点:

1. **handheld night session は GPS fix が取れていない**。`tuhh_night_09_vn200.bag` に NavSatFix は 73957 本入っているが、全部 `latitude=longitude=altitude=0.0`、`status.status=0` (NO_FIX)。夜間 + 屋内を渡り歩く取り方なので当然の挙動。`--mcd-seed-poses-from-gnss` は機能しない。**night session 全般を避け、day session (`tuhh_day_04` / `ntu_day_02` 以降) から選ぶ**。
2. **MCD の bag には `/tf` も `/tf_static` も入っていない**。sensor 間 extrinsics は session folder 外の calibration YAML で配布されており、bag 内だけで完結しない。`mcd.py` の TF lookup は空振りになるので、`--mcd-seed-poses-from-gnss` が機能する day session を選んだ後も、camera ↔ imu / camera ↔ lidar の extrinsic を override CLI (`--mcd-reference-bag` 相当の機能、または新規 `--mcd-static-calibration calib.yaml`) で食わせる必要がある。**配布場所は 2026-04-20 時点で MCDVIRAL Download page 本体**（`mcdviral/*` という別リポジトリではない）: `scripts/download_mcd_calibration.sh handheld` / `scripts/download_mcd_calibration.sh atv` で `data/mcd/calibration_<rig>.yaml` に落ちる。
3. session folder は `lua_bagname/<bagname>_*.bag` という二重ディレクトリ構造で展開される（gdown の挙動）。CLI が `--mcd-session data/mcd/<name>/` を要求するなら `<name>/<name>_*.bag` の shape を受け付けるか、単ファイル列挙にするかを決めておく。

これらを踏まえて、次セッションの着手順序:

1. day session 1 本 DL（`tuhh_day_04` 12.5 GB か `ntu_day_02` 14.8 GB）
2. `/vn200/GPS` に valid fix があるか事前に spot-check（`status.status >= 0` が 1% 以上）
3. MCDVIRAL calibration YAML を入手、`--mcd-static-calibration` みたいなフラグを足して `mcd.py` の TF lookup を bypass できるようにする
4. preprocess → gsplat train → splat export → Pages bundle

#### 4.3.3.b `tuhh_day_04` で image-only DUSt3R 経路が通った (2026-04-19)

上の段で "GNSS + calibration YAML を揃えないと詰む" と書いたが、`tuhh_day_04` (12.6 GB 完本 DL) の d455b color を 5558 frame 中から等間隔 20 frame だけ抜いて DUSt3R complete graph で回したら、**20 frame 中 19 frame が非退化 trajectory** を復元した。歩行モーションの norm 列が 0.009 → 0.21 → ... → 1.87 → ... → 0.11 と綺麗なドーム形で、handheld が屋外を往復したと物理的に解釈できる結果。gsplat 3000 iter で 3.23M gauss まで学習収束。

そのまま `ply_to_splat` で 400k / 12.8 MB に焼いて `docs/assets/outdoor-demo/mcd-tuhh-day04.splat` として bundle、`splat.html?url=...mcd-tuhh-day04.splat` でブラウザから toggle できる。

つまり **day session の場合 image-only DUSt3R 一発で MCD demo を作れる**、GNSS seeding も calibration YAML も要らない。NTU #17 (night handheld / GPS denied / motion blur) との差は「昼の屋外 + 歩行速度 + repetitive なし texture」で DUSt3R 前提がそのまま通るかどうか。次 MCD demo を作るときは

1. day session の image bag 1 本だけ DL (5 GB 前後、LiDAR bag は省略可)
2. `photos-to-splat` 一撃 or `run_dust3r.py` → gsplat train → `export --format splat`

で 30 分以内で回る、という playbook が確立した。LiDAR depth supervision / appearance / BA を足して質を詰めるなら、そのとき初めて GNSS + calibration path に戻る。

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

## 15. セッション 2026-04-19〜2026-04-20 の差分要約

Claude + Codex 併走で PR #55〜#75 の 21 本を merge。OSS としての顔と中身を両軸で整えたセッション。

**Pose-free パス (新機能)**: PR #55 で DUSt3R を `stub → real` に差し替え、`PoseFreeProcessor(method="dust3r")` + `scripts/run_dust3r.py` で CLI からも回せるように。PR #66 で MAST3R を第 2 backend として同 shape で wire、`scripts/run_mast3r.py` も PR #70 で対称化。`gs-mapper photos-to-splat` (PR #62) で JPG ディレクトリ → `.splat` が一撃。`gs-mapper export --format splat` も同 PR で追加。

**Bundled demo (5 scene)**:
1. `outdoor-demo.splat` — supervised GNSS + LiDAR 6-bag fused (既存)
2. `outdoor-demo-dust3r.splat` — PR #55、bag6 cam0 DUSt3R pose-free
3. `bag6-mast3r.splat` — PR #66、MAST3R metric + 15k iter (PR #70 で quality push)
4. `mcd-tuhh-day04.splat` — PR #64、MCD 非-Autoware day session DUSt3R
5. `mcd-tuhh-day04-mast3r.splat` — PR #67、MCD MAST3R 15k iter

**Viewer (3 本完全 symmetry)**: `splat.html` / `splat_spark.html` / `splat_webgpu.html` に PR #65 / #71 / #72 で scene picker 移植、PR #73 で `docs/scene-picker.js` + `docs/scenes-list.json` に DRY。Spark 2.0 の blank canvas は PR #71 で解決 (SparkRenderer 追加 + three r179 pin)。PR #74 で iPhone 14 viewport dogfood + `splat_spark.html` viewport meta 追加。

**MCD 経路の知見**:
- NTU #17 は image-only 全滅 (PR #58): COLMAP / DUSt3R ×3 / ORB-SLAM3 全部 frame-to-frame match 不足で詰む。night handheld + GPS-denied + repetitive texture の組み合わせが原因。
- `scripts/download_mcd_folder.sh` (PR #60) で folder 全体を `gdown --folder` で 1 本取得可。
- tuhh_night_09 は vn200 GPS が全 sample lat/lon/alt=0 (PR #61) — **night handheld は GPS 取れてない**。
- tuhh_day_04 は image-only DUSt3R で 19/20 非退化 + gsplat 収束 (PR #64) — day session なら GNSS/calibration 無しで demo 化可能。
- MCDVIRAL calibration YAML は 2026-04-20 に Download page 本体で発見済み (handheld: file `1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym`、ATV: `1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL`)。`scripts/download_mcd_calibration.sh <handheld|atv>` で取得できる。ファイルは `body: { <sensor>: { T:[…4×4…], intrinsics, distortion, rostopic, timeshift_cam_imu } }` 構造で、handheld 版は d455b/d455t + mid70 + os1-64 + vn200、ATV 版は d435i/d455b + mid70 + os1-64 + vn100/vn200 を網羅。NC-SA 4.0 のため repo には同梱せず downloader のみ。

**OSS 顔整備**:
- README hero GIF (PR #68): Playwright で scene picker cycle 録画 → 640×360 / 760 KB GIF。
- README 比較表 + A/B サムネイル (PR #67)、mobile 節 (PR #74)、Credits 節 (PR #73)、Benchmark 表 (PR #75)。
- CLI cleanup (PR #69): `experiment-*` 14 本を `gs-mapper experiment <lab>` nested subparser に隔離、legacy alias は argv rewriter で back-compat、`gs-mapper --help` が痩せた。
- Robotics smoke (PR #59): `scripts/robotics_smoke.py` で PLY → render → bridge payload を ROS なしで貫通、CI 対応。
- GGRt は upstream の CUDA extension 依存で integration 断念、PR #63 で docs を現実に揃えて "reference only" とラベリング。

**次セッション向け着手候補**:
- MCD day session を GNSS + calibration YAML 経由で正規 `--method mcd` に乗せる（§4.3.3.b の playbook から）。
- CoVLA の HF access 承認後、`gs-mapper photos-to-splat --preprocess mast3r` で demo 化。
- WebXR (Enter VR) button を Spark viewer に露出。
- `docs/experiments.md` 系 lab の整理（14 本残存）。
