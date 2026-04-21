# 屋外 3D Gaussian Splatting 開発計画 / 引継ぎメモ

更新日: 2026-04-21（MCD `tuhh_day_04` supervised 検証の訂正、`ntu_day_02` supervised bundle 追加、zero-GNSS guard、COLMAP images parser 修正）

この文書は、`GS Mapper` リポジトリにおける屋外 3D Gaussian Splatting 対応の現在地を、**Claude / Codex / Copilot / その他のコーディングエージェント**がそのまま引き継げる粒度でまとめた handoff 文書です。リポジトリ直下の `CLAUDE.md` は開発コマンド早見、本書は **屋外パイプラインの文脈・判断・失敗の履歴**に重きを置きます。

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
- **2026-04-20 セッションで MCDVIRAL calibration YAML を公式 Download page 本体で発見**（Drive ID は §4.3.3.c / §15.1 参照）。**2026-04-21 時点の本 worktree** では `scripts/download_mcd_calibration.sh` + `ros_tf.load_static_calibration_yaml` / `merge_static_tf_maps` + CLI `--mcd-static-calibration`（`preprocess` / `run` / `demo`）+ 単眼 MCD の colorize/depth 経路 + テストまで **ローカル実装済み**（upstream `main` との差分は `git log` / PR 状態で要確認）。
- **2026-04-21 Codex 追検証**: `tuhh_day_04` supervised 成功扱いは **撤回**。`/vn200/GPS` は 75,173 件すべて `latitude=longitude=altitude=0.0` で、既存 `outputs/tuhh_day04_sup` は静止 GNSS trajectory だった。さらに trainer の `images.txt` parser が空の 2D points 行を捨てて 400 entries 中 200 images しか読んでいなかった。修正内容は **§15.4**。
- **2026-04-21 Codex 追実走**: valid GNSS の `ntu_day_02` を 35 s trim + altitude flatten + ATV calibration + LiDAR seed/depth supervision で preprocess/train/export し、`docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat` を production viewer / README に追加済み（詳細は **§15.5**）。`tuhh_day_04` の zero-GNSS artifact は production picker から除外。

直近の最大の残課題は以下。

1. Priority B: **Waymo 実データ E2E が未検証**
2. Priority C: NMEA / GNSS / IMU robustness、depth / appearance / sky の比較評価
3. Priority D: MCD `ntu_day_02` supervised は production bundle 化済みだが、屋外シーン品質を上げるにはより長い valid-GNSS session / multi-camera / training budget 比較が必要。

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
| **2026-04-21 worktree** `merge_static_tf_maps` / `load_static_calibration_yaml` | `src/gs_sim2real/datasets/ros_tf.py` | MCDVIRAL `body:` YAML → `StaticTfMap`（`T` 4×4）。`merge_static_tf_maps(*maps)` は **後勝ち**で child frame を上書き。`tests/test_ros_tf.py` に単体テスト |
| **2026-04-21** `--mcd-static-calibration` | `src/gs_sim2real/cli.py` | `preprocess` / `run` / `demo` に同フラグ。`_mcd_static_calibration_tf` → `_mcd_gnss_sparse_import` 先頭で bag TF とマージ。マルチカメラ `HybridTfLookup` の `static_topo` にも同じ YAML をマージ |
| **2026-04-21** `_mcd_write_pinhole_from_calibration_yaml` | `cli.py` | bag に **`sensor_msgs/CameraInfo` が無い** MCD セッション（例: `tuhh_day_04` の `*_d455b.bag`）向け。YAML の `intrinsics` + `resolution` + `rostopic` 照合で `calibration/<topic_sanitized>.json` を合成し `extract_camera_info` 失敗分を埋める |
| **2026-04-21** 単眼 `_mcd_gnss_sparse_import` の colorize + depth | `cli.py` | 従来はマルチカメラ分岐にしか `_mcd_colorize_seed` / `_mcd_export_depth_maps` が無く、**単眼 supervised で depth supervision が空振り**していた。単眼でも LiDAR seed 後に同 API を呼ぶ。**`extract_frames` が単 topic のとき画像は `images/frame_*.jpg`（サブディレクトリ無し）**なので、`cameras[]` の `subdir` は **`""`**（`colorize_lidar_world_from_images` / `export_lidar_depth_per_image` のパス規約に合わせる） |
| **2026-04-21** `scripts/download_mcd_calibration.sh` | `scripts/` | `handheld` / `atv` の Drive ID を引くシェル（CC BY-NC-SA — **repo に YAML を commit しない**）。`data/` は `.gitignore` |
| **2026-04-21** `scripts/capture_readme_splat_previews.py` | `scripts/` | `docs/splat.html` を **headed Playwright + CDP `Page.captureScreenshot`** でキャンバスクリップ。README 表用 `docs/images/demo-sweep/0{1-6}_*.png` 再生。headless だと WebGL が真っ黒になりがち → **デフォルト headed**、`DISPLAY=:0` 推奨 |
| **2026-04-21** README / hero 訂正 | `README.md`, `scripts/record_demo_gif.py` | MCD `tuhh_day_04` supervised 成功表記を撤回。valid GNSS の `ntu_day_02` supervised を追加し、Benchmark / scene picker / README thumbnails は production 6 splats 扱い。`mcd-tuhh-day04-supervised.splat` は zero-GNSS diagnostic asset として残すが production picker から除外 |

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
2. **MCD の bag には `/tf` も `/tf_static` も入っていない**。sensor 間 extrinsics は session folder 外の calibration YAML で配布されており、bag 内だけで完結しない。**2026-04-20 更新**: この YAML は MCDVIRAL の Download page 本体に Google Drive 直リンクで置かれている (`mcdviral/*` という別リポジトリではない)。`scripts/download_mcd_calibration.sh handheld` (PR #79) で 6.6 KB 落ちる → `--mcd-static-calibration <calib.yaml>` (PR #80) で `_mcd_gnss_sparse_import` の TF lookup に注入可能。詳細 recipe は §4.3.3.c。
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

#### 4.3.3.c `tuhh_day_04` supervised recipe（2026-04-21 追記: GNSS all-zero のため非推奨）

2026-04-20 セッションで §4.3.3.a の blocker 群が全部解消した:

1. **calibration YAML は公式 Download page 本体に掲載されている** — handheld rig (kth_/tuhh_ 用) が Drive file `1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym`、ATV rig (ntu_ 用) が `1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL`。`scripts/download_mcd_calibration.sh <handheld|atv> [out-path]` で取得可 (PR #79)。
2. **YAML を `mcd.py` の TF lookup に注入する CLI** — `--mcd-static-calibration <calib.yaml>` フラグが `preprocess` / `run` / `demo` 全部に wired (PR #80)。`gs_sim2real.datasets.ros_tf.load_static_calibration_yaml` が `body → <sensor>` edge の `StaticTfMap` を返し、`_mcd_gnss_sparse_import` で bag 由来の空 tree とマージされて `--mcd-camera-frame` / `--mcd-lidar-frame` の lookup を通す。
3. **sensor name 対応表 (handheld rig)**: YAML 側の child key がそのまま `--mcd-camera-frame` / `--mcd-lidar-frame` に渡せる。
   - color cameras: `d455b_color` (rostopic `/d455b/color/image_raw`)、`d455t_color` (`/d455t/color/image_raw`)
   - IR mono: `d455b_infra1/2`、`d455t_infra1/2`
   - IMUs: `d455b_imu`、`d455t_imu`、`vn200_imu` (body 基準はたいてい vn200_imu が identity)
   - LiDAR: `mid70` (Livox `/livox/lidar`)、`os_sensor` (Ouster OS1-64 `/os1_cloud_node/points`)、`os_imu`
4. **2026-04-21 追検証で訂正**: §4.3.3.b で確認した非退化 trajectory は image-only DUSt3R の結果であり、GNSS fix の検証ではなかった。`tuhh_day_04` の `/vn200/GPS` は all-zero なので、この session を GNSS supervised には使わない。

以下は履歴として残すが、**現行コードでは all-zero GNSS を skip するため `tuhh_day_04` では失敗するのが正しい**。topic 名のメモだけは有効 — 完本 DL の `tuhh_day_04_os1_64.bag` では PointCloud2 が **`/os_cloud_node/points`**（`/os1_cloud_node/points` ではない）。VN200 の NavSatFix topic 名は **`/vn200/GPS`**（`/vn200/gps` ではない）が、中身は all-zero。

```bash
# 1) calibration YAML fetch (1 回きり; CC BY-NC-SA — repo に commit しない)
scripts/download_mcd_calibration.sh handheld data/mcd/calibration_handheld.yaml

# 2) supervised preprocess（ローカルは PYTHONPATH=src python3 -m gs_sim2real.cli … でも可）
gs-mapper preprocess \
  --images data/mcd/tuhh_day_04 \
  --output outputs/tuhh_day04_sup \
  --method mcd \
  --image-topic /d455b/color/image_raw \
  --mcd-camera-frame d455b_color \
  --lidar-topic /os_cloud_node/points \
  --mcd-lidar-frame os_sensor \
  --imu-topic /vn200/imu \
  --gnss-topic /vn200/GPS \
  --mcd-static-calibration data/mcd/calibration_handheld.yaml \
  --mcd-seed-poses-from-gnss \
  --mcd-tf-use-image-stamps \
  --mcd-export-depth \
  --extract-lidar \
  --max-frames 400 --every-n 1 \
  --matching sequential --no-gpu

# 3) depth-supervised gsplat training（CLI iterations が yaml の num_iterations より優先されるので明示）
gs-mapper train --data outputs/tuhh_day04_sup --output outputs/tuhh_day04_sup_train \
  --method gsplat --iterations 30000 \
  --config configs/training_depth_long.yaml

# 3b) 長め学習例（2026-04-21 実走: outputs/tuhh_day04_sup_train50k、wall ~1150 s 台、最終 Gaussians ログ上 163,410）
gs-mapper train --data outputs/tuhh_day04_sup --output outputs/tuhh_day04_sup_train50k \
  --method gsplat --iterations 50000 \
  --config configs/training_depth_long.yaml

# 4) 400k / 12.8 MB に export して bundle（どちらの PLY でも可）
gs-mapper export \
  --model outputs/tuhh_day04_sup_train/point_cloud.ply \
  --format splat \
  --output docs/assets/outdoor-demo/mcd-tuhh-day04-supervised.splat \
  --max-points 400000
```

その後の bundle 作業（`CONTRIBUTING.md` §"Bundled demo splats" の triplet）:
- `docs/scenes-list.json` に entry 追加
- `docs/splat.html` / `docs/splat_spark.html` / `docs/splat_webgpu.html` の `<select>` に `<option>` 追加
- `tests/test_pages_assets.py` に `test_mcd_tuhh_day04_supervised_splat_present` 相当を追加

**Pitfalls / 事前チェック**:

1. **CameraInfo の `frame_id` が YAML のセンサー名と一致しないかもしれない**。MCD bag は `/d455b/color/camera_info` の header.frame_id がたとえば `d455b_color_optical_frame` のような `_optical_frame` suffix 付きで publish される可能性があり、そのままだと `tf_map.lookup(base_frame, "d455b_color_optical_frame")` が miss する。その場合は `--mcd-camera-frame d455b_color` を明示的に渡して override すれば YAML 側の key と一致する。`_mcd_gnss_sparse_import` は `CameraInfo.frame_id` を優先し、空なら `--mcd-camera-frame` にフォールバックするので、明示的に frame name を渡すのが安全。
2. **`--mcd-base-frame` のデフォルトは `base_link`** で、`load_static_calibration_yaml(path, base_frame="base_link")` として YAML を読むので、YAML の `body:` key は内部的に `base_link` にリラベルされる。`--mcd-lidar-frame os_sensor` / `--mcd-camera-frame d455b_color` のような child name はそのまま引ける。もし別名を使いたい場合は `--mcd-base-frame body` にして lookup 側も `body` に揃える。
3. **GNSS fix の事前 spot-check**: `scripts/outdoor_smoke.sh mcd-list` 等で **`/vn200/GPS`**（大文字）の `status.status >= 0` サンプル率を確認。day session は取れているはずだが、`tuhh_night_09` のように全 sample fix=0 な bag は `--mcd-seed-poses-from-gnss` で即落ちる。
4. **`tuhh_day_04` の d455b bag には CameraInfo が無い**（2026-04-21 確認）。**`_mcd_write_pinhole_from_calibration_yaml`** 経由で PINHOLE JSON を合成しないと単眼 supervised が弱体化する。実装済み worktree では `extract_camera_info` 失敗後に YAML から補完する。
5. **session folder の二重ディレクトリ** — `gdown --folder` で展開すると `data/mcd/tuhh_day_04/tuhh_day_04_*.bag` という shape になる (§4.3.3.a #3)。`MCDLoader._find_bag_paths` は再帰的に `.bag` を拾うので `--images data/mcd/tuhh_day_04` のまま渡して OK。
6. **LiDAR bag (`*_ouster.bag` など) が無いと depth supervision が無効化される** — image-only bag しか DL していない場合は `--extract-lidar` が空振りし、`mcd-export-depth` は warning を出して skip する。LiDAR 無しで supervised 感は出ないので、session folder 全部 (~12 GB) DL する。

**当時の期待値（2026-04-21 追検証で `tuhh_day_04` には不適合と判明）**:
- registered frames: 240-400 × 1-2 cam = 240-800（day session は single cam でも十分）
- LiDAR world seed: 100k-200k 点 (colorized)
- trained gauss: 500k-1.5M、400k filter 後 12.8 MB
- L1: 0.08-0.12 (bag4 相当を期待、DUSt3R 版の 0.16 より一段良い)
- valid GNSS session であれば、既存 5 本と A/B 比較可能な「同じ scene の supervised vs pose-free」軸が成立する見込みだった。`tuhh_day_04` は all-zero GNSS のためこの用途には使わない。

**2026-04-21 実測（単眼 400 frames、`outputs/tuhh_day04_sup`）** — **§15.4 により診断 artifact 扱い**:
- preprocess: LiDAR colorize **約 32k / 200k** 点が非灰色、per-image depth **400** maps、`sparse/0` 生成
- train **30k**（`outputs/tuhh_day04_sup_train`）: wall **~5600 s** 級、**Final Gaussians ~436k**、ログ終盤 **L1 ~0.19**
- train **50k**（`outputs/tuhh_day04_sup_train50k`）: wall **~1150 s**、**Final Gaussians 163,410**（ログ上 densify 後ほぼ固定）、iter 50000 行 **L1 ~0.25**。**なぜ 30k 実行と Gaussians 最大値が食い違う見えるかは未解決**（同一 preprocess パスを再読みしているつもりでも、コード版・config マージ・ログの見ている run が異なる可能性 — Codex 引継ぎ先で要照合）。

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
- ~~MCDVIRAL calibration YAML は未公開、`--method mcd` の GNSS+LiDAR 経路を回すには別途入手が必要~~ → **2026-04-20 に公式 Download page 本体で発見** (PR #79)、`--mcd-static-calibration` フラグで CLI から食わせられる (PR #80)。詳細は §4.3.3.c。

**OSS 顔整備**:
- README hero GIF (PR #68): Playwright で scene picker cycle 録画 → 640×360 / 760 KB GIF。
- README 比較表 + A/B サムネイル (PR #67)、mobile 節 (PR #74)、Credits 節 (PR #73)、Benchmark 表 (PR #75)。
- CLI cleanup (PR #69): `experiment-*` 14 本を `gs-mapper experiment <lab>` nested subparser に隔離、legacy alias は argv rewriter で back-compat、`gs-mapper --help` が痩せた。
- Robotics smoke (PR #59): `scripts/robotics_smoke.py` で PLY → render → bridge payload を ROS なしで貫通、CI 対応。
- GGRt は upstream の CUDA extension 依存で integration 断念、PR #63 で docs を現実に揃えて "reference only" とラベリング。

**次セッション向け着手候補** (PR #75 当時):
- ~~MCD day session を GNSS + calibration YAML 経由で正規 `--method mcd` に乗せる~~ → **PR #79 / #80 で calibration downloader + `--mcd-static-calibration` までは揃った**。残りは GPU training run だけ、§15.1 / §4.3.3.c 参照。
- CoVLA の HF access 承認後、`gs-mapper photos-to-splat --preprocess mast3r` で demo 化。
- ~~WebXR (Enter VR) button を Spark viewer に露出~~ → PR #76 で完了 (`renderer.xr.enabled = true` + `VRButton.createButton`)、README にも反映済み。
- `docs/experiments.md` 系 lab の整理（14 本残存）。

## 15.1 セッション 2026-04-20（post-#75、OSS 顔 + MCD supervised 経路 開通）

Claude Opus 4.7 で PR #77〜#80 の 4 本。OSS 顔の残り整備 + §4.3.3.a の最大 blocker (calibration YAML 入手不能) を解消した短い session。

**OSS 顔 (housekeeping 2 本)**:
- PR #77 — `.github/dependabot.yml` (GHA + pip 週次、torch/vision/audio は gsplat/DUSt3R/MAST3R upstream と coupling するため pin で ignore) + `.github/PULL_REQUEST_TEMPLATE.md` + `.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.yml` を追加。CONTRIBUTING.md の lint/test incantation を PR 本文テンプレに pre-fill、issue 側は CONTRIBUTING.md §"Reporting issues" の要求項目 (OS / GPU / SDK 版 / minimal repro) を構造化。
- PR #78 — Spark 2.0 の LoD knob を `?lod=auto|high|low` URL param で露出。`LOD_PRESETS = { auto: {}, high: { lodSplatScale: 4.0, lodRenderScale: 0.5 }, low: { lodSplatScale: 0.5, lodRenderScale: 2.0 } }` を `new SparkRenderer({ renderer, ...lodPreset })` に spread。README の viewer 比較表と info pane にも文言追加、`tests/test_pages_assets.py::test_splat_spark_exposes_lod_url_param` で配線を pin。Spark 2.0 の view-dependent LoD / progressive streaming を「謳うだけ」から「触れる機能」にした。

**MCD supervised 経路の blocker 解消 (2 本)**:
- PR #79 — `scripts/download_mcd_calibration.sh <handheld|atv> [out-path]` + `tests/test_download_mcd_calibration_script.py` (5 件 smoke)。Google Drive file ID を MCDVIRAL 公式 Download page の raw HTML scraping で特定 (handheld `1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym` / ATV `1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL`)、`body: { <sensor>: { T:[…4×4…], intrinsics, distortion_*, rostopic, timeshift_cam_imu } }` 構造を確認。`curl -sL` で 6.4-6.6 KB、virus-scan confirm dance 不要。CC BY-NC-SA 4.0 なので repo には committee せず downloader + license reminder only。docs/plan_outdoor_gs.md §4.3.3.a の「別リポジトリ想定」推測を訂正、README Credits の MCDVIRAL 行に script 参照を追記。
- PR #80 — `--mcd-static-calibration <calib.yaml>` フラグを `preprocess` / `run` / `demo` の 3 subcommand に配線。実装は `src/gs_sim2real/datasets/ros_tf.py::load_static_calibration_yaml(path, base_frame=…)` が YAML の `body:` セクションから `StaticTfMap` (parent=`base_frame` → child=`<sensor>` edges) を構築、`merge_static_tf_maps(…)` で bag-derived map とマージ (後者が collision 時に勝つ — MCD では常に empty なので YAML が通る)。`_mcd_gnss_sparse_import` の `tf_map` 構築直後と `HybridTfLookup` 用の `static_topo` 構築直後の両方でマージして、`--mcd-tf-use-image-stamps` と併用しても正しい extrinsics が出るようにした。テスト 6 件: YAML parser (happy path / parent override / lookup round-trip / missing `body:` raises / malformed T skipped) + merge helper (後者が勝つ / 両側エッジ保存 / `None` 許容) + CLI flag presence on 3 subcommands。

**本セッションで変わった blocker 状態**:
- §6 の "非対象" には Waymo E2E が残る。§4.3.3 の MCD supervised 経路は calibration 入手手段と YAML 注入 CLI までは揃った。後続の 2026-04-21 追検証で **`tuhh_day_04` の GNSS fix 確認は誤り**と判明したが（§15.4）、valid GNSS の `ntu_day_02` で同 recipe を実走して production bundle 化済み（§15.5）。
- PR #77 の dependabot が 2026-04-21 月曜に初回発火する。最初の週は pip 側で `urllib3` / `certifi` 等の patch bump が数本出る想定、GHA 側は `actions/setup-python` / `actions/checkout` の最新 minor があれば 1-2 本。マージ前に `ci.yml` / `pages.yml` / `publish.yml` が通ることを必ず確認。

**本セッションで明らかに touch しなかった / あえて残した判断**:
- `docs/experiments.md` (911 行) の prune は「公開 vs 内部 note の線引き」という価値判断が要るので auto mode では踏み込まなかった。次セッションでユーザと目線合わせしてから。
- BYO photos / CoVLA demo は外部依存 (ユーザの写真 / HF access 承認) 待ちで auto 環境で完結しない。
- `--renderer gsplat` の CUDA path は GPU smoke が要るので保留。

## 15.2 次セッションへの短冊 (2026-04-20 時点)

**Priority A (ship できる最短経路)**:

1. ~~**`mcd-tuhh-day04-supervised.splat` を bundle の 6 本目に**~~ — **§15.4 で撤回**。`tuhh_day_04` の `/vn200/GPS` は all-zero。代替として `ntu_day_02` を valid GNSS session として採用し、§15.5 で `mcd-ntu-day02-supervised.splat` を production bundle 化済み。
2. **`docs/experiments.md` の prune** — ユーザ合意後。公開 vs 内部 note の線引き方針を先に決める (前セッションで ROI 低判定、次セッションで議論)。

**Priority B (blocker あり or ROI 中)**:

3. **BYO photos demo** — ユーザ自身の 20-40 枚写真で `photos-to-splat --preprocess mast3r` を回して claim の自己実証。ユーザが写真を提供する必要。
4. **CoVLA mini** — HuggingFace `turing-motors/CoVLA-Dataset-Mini` の access request 承認後 pipeline 実走。driving scene 第 6 demo 候補。
5. **Spark 2.0 LoD knob を scene-picker に promote** — PR #78 は URL param only。CONTRIBUTING.md §"Where to start" で示唆されているとおり、`docs/scene-picker.js` に LoD `<select>` を追加して 3 viewer 共通にしておくと将来 `?cameras=` preset も同じ DRY 経路で足せる。

**Priority C (refactor / 深堀り)**:

6. `--renderer gsplat` CUDA path の smoke テスト (gated、GPU 必要)。
7. README `apps/` / `projects/` プロトタイプ節の最新化 — DreamWalker Live / Robotics の現状と同期していない可能性あり。
8. NMEA / GNSS / IMU robustness (§6 の残 blocker)。日跨ぎ RMC、IMU quaternion 融合、logger 時刻ずれ。

## 15.3 Codex 引き継ぎ — セッション 2026-04-21（Priority A 実走・bundle 6 本目、§15.4 で訂正）

**注意**: この節は作業当時の記録。2026-04-21 の追検証で GNSS all-zero と trainer parser bug が見つかったため、成功扱いは **§15.4 で撤回**。以下の数値は「静止 GNSS trajectory + 旧 parser で 200 images training した診断 artifact」の履歴として読む。

### 確認済み事実（ローカル）

| 項目 | 値 |
|------|-----|
| LiDAR topic | `/os_cloud_node/points`（`/os1_cloud_node/points` ではない） |
| GNSS topic | `/vn200/GPS`（大文字、`status` 要確認） |
| preprocess 出力 | `outputs/tuhh_day04_sup` — colorized LiDAR 約 32k / 200k 非灰色、depth **400** maps、400 frames |
| train 30k | `outputs/tuhh_day04_sup_train` — ログ上最終 Gaussians **~436k**、L1 **~0.19**（スレッド記録） |
| train 50k | `outputs/tuhh_day04_sup_train50k` — 壁時計 **~1152 s**、最終 **163,410** Gaussians、末尾 L1 **~0.2531**、`point_cloud.ply` あり |
| バンドル splat（スレッド時点） | `docs/assets/outdoor-demo/mcd-tuhh-day04-supervised.splat`（**30k** 系 PLY から export、400k cap / ~12.8 MB と記録） |
| 単眼 flat images | `_mcd_gnss_sparse_import` で LiDAR colorize + depth は **`subdir: ""`**（`images/frame_*.jpg`） |
| d455b bag | **CameraInfo なし** → `_mcd_write_pinhole_from_calibration_yaml` 必須（§4.3.3.c Pitfalls 4） |
| README / viewer | §15.5 後は production 6 scenes（`ntu_day_02` supervised 追加）。`tuhh_day_04` zero-GNSS artifact は diagnostic asset としてのみ残し、scene picker / Benchmark / hero script から除外 |

### 実装タッチポイント（worktree）

- `src/gs_sim2real/datasets/ros_tf.py` — `merge_static_tf_maps`, `load_static_calibration_yaml`
- `src/gs_sim2real/cli.py` — `--mcd-static-calibration`、`_mcd_write_pinhole_from_calibration_yaml`、単眼 `subdir: ""`
- `scripts/download_mcd_calibration.sh`, `scripts/capture_readme_splat_previews.py`, `scripts/record_demo_gif.py`（6 splats）
- テスト: `tests/test_ros_tf.py`, `tests/test_cli.py`, `tests/test_pages_assets.py`

### 未確認 / 要調査

1. **同じ preprocess `outputs/tuhh_day04_sup` に対し、30k で ~436k Gaussians、50k で ~163k** — 設定マージ・コード版・ログ解釈のどれか要突合。再現コマンドと `configs/training_depth_long.yaml` の実効値をログと照合すること。
2. **Bundled `.splat` を 50k PLY に差し替えるか** — 品質・ファイルサイズ・README Benchmark 行との整合。
3. **ローカル VLM（Ollama moondream / llava:7b）** — splat スクショの VQA は信頼できず、検証用途には不適。

### 次アクション（Codex / 次エージェント）

1. `git status` / `main` との差分確認 → PR 前に `pytest` + `ruff`。
2. `tuhh_day_04` は all-zero GNSS のため production には戻さない。新しい valid GNSS session を採用する場合のみ、`gs-mapper export` で別名の `.splat` を作り README Benchmark の数値も揃える。
3. Gaussian 数の乖離を潰す（必要なら同一コミット・同一 config で 30k / 50k を再実行してログ比較）。
4. 任意: `record_demo_gif.py` の ffmpeg `palettegen` 警告（`-update 1` 等）の整理。

### 絶対パス（このリポジトリ）

- ルート: `/media/sasaki/aiueo/ai_coding_ws/nerf-gs-playground`
- preprocess: `outputs/tuhh_day04_sup`
- trains: `outputs/tuhh_day04_sup_train`, `outputs/tuhh_day04_sup_train50k`（ログ例: `outputs/tuhh_day04_sup_train50k.log`）
- bundle: `docs/assets/outdoor-demo/mcd-tuhh-day04-supervised.splat`

### 事実と推論（引き継ぎで混同しないこと）

| 区分 | 内容 |
|------|------|
| **直接確認できる** | preprocess 出力ディレクトリの `images/`・`depth/`・`sparse/0` の存在、訓練ログファイルの最終行、`.ply` / `.splat` の mtime とサイズ、`rosbag info` の topic 一覧 |
| **スレッド／メモからの転記** | 「30k で L1 ~0.19」「50k で L1 ~0.2531」「436k vs 163k Gaussians」— **同一マシン・同一コミットのログに再突合するまで確定値ではない** |
| **要再検証** | 30k の wall **~5600 s** と 50k の wall **~1152 s** は、iter 数と単調関係が逆なので **別 GPU・別実行コンテキスト・ログの取り違え**の可能性がある。引き継ぎ先は `time` 付き再実行 or ログ先頭の hostname / git SHA / torch 版を突き合わせること |

### §15.2 との整合（2026-04-21 時点）

- §15.2 の **Priority A #1**（`mcd-tuhh-day04-supervised.splat` を 6 本目に）— **§15.4 で撤回**。現 worktree では diagnostic asset としてのみ残す。
- §15.2 の Priority A #2（`docs/experiments.md` prune）— **未着手のまま**。

### Codex 着任時チェックリスト（推奨順）

1. `git fetch origin && git log --oneline -5` と **`git status`** — ブランチ名・未コミット差分・`main` からの divergence を把握。
2. **MCD 関連の回帰** — `CONTRIBUTING.md` の incantation に従い、少なくとも次を通す（E2E は GPU 次第で skip 可だが、pages テストは軽い）:
   - `ruff format src/ tests/ scripts/` → `ruff check src/ tests/ scripts/` → `pytest tests/ -q --ignore=tests/e2e`
3. **該当テストのピンポイント** — `pytest tests/test_ros_tf.py tests/test_cli.py tests/test_pages_assets.py tests/test_mcd.py tests/test_gsplat_trainer.py -q`（MCD calibration / CLI フラグ / zero-GNSS guard / parser / Pages diagnostic）。
4. **§4.3.3.c の bash ブロック**が現行 CLI と一致するか — フラグリネームが入っていないか `gs-mapper preprocess --help` で確認。
5. **Pages 資産** — `docs/scenes-list.json` が production URL として `assets/outdoor-demo/mcd-ntu-day02-supervised.splat` を含み、`mcd-tuhh-day04-supervised.splat` を含まないこと。ローカルで `python3 -m http.server` 等から `docs/splat.html` を開きシーン切替。
6. **README 表・Benchmark 行** — `MCD ntu_day_02 — supervised` の row が実走値（400 frames / 30k iter / 500 s / 906k→400k / L1 0.195 / 12.8 MB）と一致しているか。食い違うなら README か export 元を直す。

### 検証コマンド早見（データとテスト）

```bash
# topic 実在確認（session folder を path に）
rosbag info data/mcd/tuhh_day_04/*.bag | head -80

# 開発ルートで（venv なら activate 後）
ruff format src/ tests/ scripts/
ruff check src/ tests/ scripts/
pytest tests/ -q --ignore=tests/e2e

# README 用 PNG（WebGL は headed 推奨）
export DISPLAY=:0   # 環境に合わせる
python3 scripts/capture_readme_splat_previews.py

# hero GIF（6 シーン）
python3 scripts/record_demo_gif.py
```

**Headless 注意**: `capture_readme_splat_previews.py` はデフォルト headed。headless のままだとキャンバスが真っ黒・PNG が極小になりうる → CI ではスキップ or 別 job で headed 実行する運用を想定。

### 実装の読みどころ（コードダイブ順）

1. **`_mcd_gnss_sparse_import`**（`cli.py`）— static calib マージ、単眼分岐での `_mcd_colorize_seed` / depth export、**`subdir: ""`**。
2. **`_mcd_write_pinhole_from_calibration_yaml`** — CameraInfo 欠落 bag への PINHOLE JSON 合成。
3. **`load_static_calibration_yaml` / `merge_static_tf_maps`**（`ros_tf.py`）— YAML `body:` → edge、マージ時 **後勝ち**。
4. **`MCDLoader.colorize_lidar_world_from_images` / `export_lidar_depth_per_image`**（`mcd.py`）— path 規約（`images/` flat vs サブディレクトリ）。

### トラブルシュート早見

| 症状 | まず疑うこと |
|------|----------------|
| preprocess で GNSS seed が落ちる | topic が `/vn200/gps` になっていないか（**大文字 `GPS`**）。fix がゼロの night session でないか。 |
| depth maps が 0 / warning のみ | LiDAR bag 未 DL、`--extract-lidar` が空振り。`--lidar-topic` が **`/os_cloud_node/points`** か。 |
| 単眼で colorize が灰色のまま | `cameras[]` の `subdir` が `frame_*.jpg` の実パスと一致しているか（単眼は **`""`**）。 |
| export 後の viewer が真っ白 | `.splat` の URL パス、`scenes-list.json` の相対 URL、ブラザの CORS（`file://` ではなく local server）。 |
| プレビュー PNG が真っ黒 | headless WebGL。`DISPLAY` 付き headed で `capture_readme_splat_previews.py`。 |

### PR 本文に貼れる要約（テンプレ）

- **目的**: MCD `tuhh_day_04` supervised 成功扱いを訂正し、all-zero GNSS から静止 trajectory / false-positive bundle が出ないようにする。
- **変更**: NavSatFix zero placeholder の skip、COLMAP `images.txt` parser の空 2D 行対応、MCD diagnostic relabel、valid GNSS の `ntu_day_02` supervised bundle 追加。
- **検証**: `ruff` + `pytest tests/ -q --ignore=tests/e2e`。実データ spot-check で `/vn200/GPS` は 75,173 / 75,173 samples が all-zero、修正後は `extract_navsat_trajectory` が fail-fast。
- **既知のフォローアップ**: `ntu_day_02` より長い valid-GNSS session / multi-camera で supervised MCD を再実走し、屋外シーン品質を比較する。

## 15.4 Codex 追検証 — `tuhh_day_04` supervised 成功扱いの訂正

**結論**: §15.3 の `tuhh_day_04` supervised bundle は **ship 品質ではない**。原因は 2 つ。

1. **GNSS が all-zero** — `data/mcd/tuhh_day_04/.../tuhh_day_04_vn200.bag` の `/vn200/GPS` は 75,173 samples すべて `latitude=longitude=altitude=0.0`。従来の `extract_navsat_trajectory` は `status.status=0` を valid と見ていたため、ゼロ fix から静止 ENU trajectory を生成していた。`outputs/tuhh_day04_sup/pose/gnss_trajectory*.tum` と `sparse/0/images.txt` は pose が 1 種類しかない。
2. **trainer が 400 entries 中 200 images しか読んでいなかった** — COLMAP text `images.txt` は metadata 行 + 2D points 行のペアだが、pose-seeded import では 2D points 行が空。旧 `_load_images_txt` は空行を捨ててから 2 行ペアで読んでいたため、半数の image metadata を 2D points 行扱いで skip していた。

**実装済み修正**:

- `src/gs_sim2real/datasets/mcd.py::extract_navsat_trajectory` は `lat == 0 && lon == 0` の placeholder NavSatFix を skip する。`tuhh_day_04` は now fail-fast: `Need at least 2 NavSatFix samples ... got 0 from /vn200/GPS`。
- `src/gs_sim2real/train/gsplat_trainer.py::_load_images_txt` は空の 2D points 行を保持したまま metadata 行を parse する。修正後は `outputs/tuhh_day04_sup` を `1 camera / 400 images / 100000 points` と読める。
- `scripts/check_mcd_gnss.py` を追加。NavSatFix の valid / zero-placeholder / invalid-status 件数、ENU translation extent、任意の `image_timestamps.csv` との時刻 overlap を training 前に判定する。
- `README.md` / viewer labels は `mcd-tuhh-day04-supervised.splat` を **zero-GNSS diagnostic** に relabel し、Benchmark から外した。その後 §15.5 で `mcd-ntu-day02-supervised.splat` に置き換え、production picker / README / hero script は production 6 splats を周回する。
- Tests: `tests/test_mcd.py::test_extract_navsat_trajectory_rejects_zero_placeholder_fixes`, `tests/test_gsplat_trainer.py::test_load_images_txt_preserves_entries_with_blank_points_lines`, `tests/test_check_mcd_gnss_script.py`。

**Preflight command**:

```bash
scripts/check_mcd_gnss.py data/mcd/<session> \
  --gnss-topic /vn200/GPS \
  --image-timestamps outputs/<preprocess>/images/image_timestamps.csv
```

`tuhh_day_04` 実測: `total=75173`, `valid=0`, `zero placeholders=75173`, `image overlap=0` → non-zero exit。

**次に supervised MCD を ship する条件**:

- `rosbags` で対象 session の NavSatFix を直接 scan し、`lat/lon != 0` が十分あり、ENU translation extent が数 m 以上あることを training 前に確認する。
- 具体的には `scripts/check_mcd_gnss.py` が `[OK]` になることを preprocess / train の前提にする。
- `GsplatTrainer._load_colmap_data(preprocess_dir)` で images 数が `sparse/0/images.txt` の camera entries 数と一致することを確認する。
- 新しい valid session で preprocess → train → export をやり直すまで、`mcd-tuhh-day04-supervised.splat` は benchmark / README hero の成功例にしない。

### セキュリティ / ライセンス（手戻り防止）

- **MCDVIRAL calibration YAML** — CC BY-NC-SA。**リポジトリに YAML 本体を commit しない**。`scripts/download_mcd_calibration.sh` のみ。
- **大容量 bag / outputs** — `.gitignore` 想定。PR にデータを載せない。

### エージェント向け短い読み順（本ファイルのみ）

1. **§0 TL;DR**（今週の状態）
2. **§15.5**（`ntu_day_02` の valid GNSS probe と trim/flatten 条件）
3. **§15.3**（本セッションのパス・数値・チェックリスト）
4. **§4.3.3.c**（コピペ用コマンドと Pitfalls）
5. **§6**（残ブロッカー）
6. **§15.1–15.2**（PR 履歴と未完了 Priority B/C）

## 15.5 Codex 追検証 — `ntu_day_02` GNSS 候補の cheap probe

**結論**: `ntu_day_02` は `tuhh_day_04` と違い non-zero GNSS がある。ただし raw NavSatFix は開始直後に altitude spike と水平 warm-up jump があるため、そのまま supervised preprocess に使わない。現 worktree では以下を実装して、trim 後の sparse smoke まで通した。

- `MCDLoader.DEFAULT_GNSS_TOPICS` に `/vn200/GPS` / `/vn100/GPS` を追加。
- `scripts/check_mcd_gnss.py` に horizontal / vertical extent、raw altitude span、p95 / max horizontal speed、`--flatten-altitude`、`--start-offset-sec` を追加。
- `MCDLoader.extract_navsat_trajectory(..., flatten_altitude=True, start_offset_sec=...)` を追加。
- CLI `preprocess` / `run` / `demo` に `--mcd-flatten-gnss-altitude` と `--mcd-start-offset-sec` を追加。MCD image / LiDAR extraction と GNSS TUM 出力に同じ start offset を渡す。
- `scripts/download_mcd_session.sh` は小さい Drive file が confirm page ではなく直接返るケースを保存できるようにした。

**取得したローカル subset**（full folder 14.8 GB ではなく supervised 単眼に必要なものだけ）:

| file | source | size |
|------|--------|------|
| `data/mcd/ntu_day_02/ntu_day_02_vn200.bag` | official VN200 file (`1wo1rUuzqDkvFMhXJhx9fnNtn6uyh_F7z`) | 25 MB |
| `data/mcd/ntu_day_02/ntu_day_02_d455b.bag` | official D455 bottom file (`1sfQdn6MGt4BsSx6PQtDdMZSiwfFcsihk`) | 5.0 GB |
| `data/mcd/ntu_day_02/ntu_day_02_os1_128.bag` | official Ouster file (`1jDS84WvHCfM_L73EptXKp-BKPIPKoE0Z`) | 5.0 GB |
| `data/mcd/ntu_day_02/ntu_day_02_ltpb.bag` | official LTPB file (`1a31zWxJK-OgqP6z4IV4WudF2DbcjYRxY`) | 401 KB |
| `data/mcd/calibration_atv.yaml` | `scripts/download_mcd_calibration.sh atv` | 6.3 KB |

**Topic 確認**:

- images: `/d455b/color/image_raw` (6862 frames)
- LiDAR: `/os_cloud_node/points` (2288 frames)
- GNSS: `/vn200/GPS` (91537 samples)
- IMU: `/vn200/imu`
- ATV calibration YAML は `d455b_color`, `os_sensor`, `vn200_imu` を解決できる。`d455b` bag 自体に CameraInfo は無いので YAML → PINHOLE 補完が必要。

**GNSS preflight 実測**:

Raw + altitude flatten only:

```bash
scripts/check_mcd_gnss.py data/mcd/ntu_day_02 --flatten-altitude --json
```

- total / valid: 91537 / 91537
- zero placeholders: 0
- horizontal extent: 1472.72 m
- horizontal path: 2036.44 m
- raw altitude span: 11940.11 m（開始直後が 11760 m、後続は -100 m 台）
- horizontal max speed: 459827 m/s（30.54 sec で 1.36 km warm-up jump）

Start offset 35 sec + altitude flatten:

```bash
scripts/check_mcd_gnss.py data/mcd/ntu_day_02 \
  --flatten-altitude \
  --start-offset-sec 35
```

- total / valid after trim: 77537 / 77537
- horizontal extent: 250.43 m
- horizontal path: 660.76 m
- vertical extent: 0.003 m
- raw altitude span after trim: 112.81 m
- p95 horizontal speed: 6.65 m/s
- Result: `[OK]`

**Sparse smoke 済み**:

```bash
PYTHONPATH=src python3 -m gs_sim2real.cli preprocess \
  --images data/mcd/ntu_day_02 \
  --output outputs/ntu_day02_probe_trimmed2 \
  --method mcd \
  --image-topic /d455b/color/image_raw \
  --mcd-camera-frame d455b_color \
  --gnss-topic /vn200/GPS \
  --mcd-static-calibration data/mcd/calibration_atv.yaml \
  --mcd-seed-poses-from-gnss \
  --mcd-flatten-gnss-altitude \
  --mcd-start-offset-sec 35 \
  --mcd-tf-use-image-stamps \
  --mcd-skip-lidar-seed \
  --max-frames 2 --every-n 1 \
  --matching sequential --no-gpu
```

Output:

- `outputs/ntu_day02_probe_trimmed2/images/image_timestamps.csv` starts at `1644824131.386...`, matching the 35 sec trim.
- `outputs/ntu_day02_probe_trimmed2/pose/origin_wgs84.json` records `"altitude_mode": "flattened_median"` and `"start_offset_sec": 35.0`.
- YAML PINHOLE補完と `base_link <- d455b_color` extrinsics lookup が通り、`sparse/0` 生成成功。

**次に full supervised preprocess を回すなら**:

```bash
PYTHONPATH=src python3 -m gs_sim2real.cli preprocess \
  --images data/mcd/ntu_day_02 \
  --output outputs/ntu_day02_sup \
  --method mcd \
  --image-topic /d455b/color/image_raw \
  --mcd-camera-frame d455b_color \
  --lidar-topic /os_cloud_node/points \
  --mcd-lidar-frame os_sensor \
  --imu-topic /vn200/imu \
  --gnss-topic /vn200/GPS \
  --mcd-static-calibration data/mcd/calibration_atv.yaml \
  --mcd-seed-poses-from-gnss \
  --mcd-flatten-gnss-altitude \
  --mcd-start-offset-sec 35 \
  --mcd-tf-use-image-stamps \
  --extract-lidar --extract-imu --mcd-export-depth \
  --max-frames 400 --every-n 14 \
  --matching sequential --no-gpu
```

`--every-n 14` は trim 後の約193 sec / 6862 frames から 400 frame 近くを拾うための初期値。LiDAR extraction も同じ `every_n` を使うため、LiDAR seed は約 130–140 frames になる見込み。必要なら後続で image と LiDAR の sampling を別指定に分ける。

**2026-04-21 実走結果**:

- preprocess output: `outputs/ntu_day02_sup`
- images: 400 JPG + `image_timestamps.csv`
- LiDAR: 139 `.npy` frames + `timestamps.csv`（`find` の素朴な file count では csv 込みで 140）
- IMU: `imu.csv` 91,539 lines
- GNSS preflight with image timestamps: 400 / 400 overlap, `[OK]`
- LiDAR world seed: 200,000 points → colorized 198,947 / 200,000 with image RGB
- depth maps: 400
- COLMAP sparse: 400 image entries, 1 PINHOLE camera, 100,000 `points3D`

Training:

```bash
PYTHONPATH=src python3 -m gs_sim2real.cli train \
  --data outputs/ntu_day02_sup \
  --output outputs/ntu_day02_sup_train \
  --method gsplat \
  --iterations 30000 \
  --config configs/training_depth_long.yaml
```

- wall time: 500.3 s
- final model: `outputs/ntu_day02_sup_train/point_cloud.ply`
- final Gaussians: 906,291
- final iter log line: `loss=6.1367 l1=0.1951 ssim_loss=0.5354`
- checkpoint: `outputs/ntu_day02_sup_train/point_cloud_iter_20000.ply`

Export:

```bash
PYTHONPATH=src python3 -m gs_sim2real.cli export \
  --model outputs/ntu_day02_sup_train/point_cloud.ply \
  --format splat \
  --output docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat \
  --max-points 400000 \
  --splat-normalize-extent 17.0 \
  --splat-min-opacity 0.02 \
  --splat-max-scale 2.0
```

- output: `docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat`
- size: 12,800,000 bytes = 400,000 gaussians
- viewer direct URL after local/Pages serve: `splat.html?url=assets/outdoor-demo/mcd-ntu-day02-supervised.splat`

Viewer / README wiring:

- `docs/scenes-list.json`, `docs/splat.html`, `docs/splat_spark.html`, `docs/splat_webgpu.html` expose `MCD ntu_day_02 — supervised` as the second supervised MCD production scene.
- `README.md` now documents **six** production bundled scenes and uses `docs/images/demo-sweep/06_mcd-ntu-day02-supervised.png`.
- `scripts/record_demo_gif.py` and `scripts/capture_readme_splat_previews.py` include `mcd-ntu-day02-supervised.splat`; the old `mcd-tuhh-day04-supervised.splat` remains only as a rejected zero-GNSS diagnostic asset and is not listed in the production picker.

Verification after wiring:

```bash
PYTHONPATH=src python3 scripts/capture_readme_splat_previews.py --only 06_mcd-ntu-day02-supervised
ruff format --check src/ tests/ scripts/
ruff check src/ tests/ scripts/
pytest tests/test_pages_assets.py -q
```

- preview: `docs/images/demo-sweep/06_mcd-ntu-day02-supervised.png`, 1280x720, 133,861 bytes, non-black WebGL capture.
- tests: `tests/test_pages_assets.py` = 14 passed.
