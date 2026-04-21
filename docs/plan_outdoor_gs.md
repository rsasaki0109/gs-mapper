# 屋外 3D Gaussian Splatting 開発計画 / 現行 handoff

更新日: 2026-04-21（IMU yaw-rate fallback 反映）

この文書は、次の担当エージェントが屋外 3DGS パイプラインの現在地をすぐ把握するための短い handoff です。古い PR / セッションごとの詳細ログ、失敗履歴、長いコマンド transcript は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に退避しました。

## 読み方

1. まず本書の **TL;DR / 現在の残課題 / 検証コマンド** を読む。
2. MCD calibration、`tuhh_day_04` 誤判定、`ntu_day_02` 実走値、PR #55〜#80 の履歴が必要なときだけ archive を読む。
3. コマンド早見は `CLAUDE.md` が存在する場合はそちらも確認する。リポジトリによっては未配置なので、その場合は本書と `README.md` / `CONTRIBUTING.md` を source of truth にする。

## TL;DR

- 本 repo は **Autoware / MCD の public outdoor data から `.splat` を生成し、GitHub Pages viewer で公開する**ところまで到達済み。
- Production viewer picker は **8 scenes**: 2 supervised、4 pose-free、2 external-SLAM comparison。
- `docs/scenes-list.json` が README table / preview capture / hero GIF / viewer picker の source of truth。`tests/test_pages_assets.py` が drift を検出する。
- `tuhh_day_04` の supervised GNSS 成功扱いは撤回済み。`/vn200/GPS` は all-zero なので production picker には入れない。
- Valid GNSS supervised MCD demo は `ntu_day_02` を採用済み。`docs/assets/outdoor-demo/mcd-ntu-day02-supervised.splat` が production asset。
- External SLAM artifact import は VGGT-SLAM 2.0 / MASt3R-SLAM comparison splat まで実走済み。Pi3 / LoGeR profile も artifact resolver 側に候補追加済み。
- Waymo real-data E2E は未検証。SDK / dataset agreement / manual tfrecord download が blocker。

## 現在の残課題

| Priority | Task | 状態 |
| --- | --- | --- |
| A | BYO photos / CoVLA mini 自己実証 | 外部入力待ち。ユーザ写真または HF access 承認が必要。 |
| A | 公開 docs の継続整理 | `docs/experiments.md` は index 化済み。本書の長い履歴は archive 化済み。今後は古い詳細を archive 側へ追記する。 |
| A | 8-scene viewer smoke 継続運用 | `docs/scenes-list.json` source of truth 化済み。pre-PR で `pytest tests/test_pages_assets.py -q` を通す。 |
| B | Waymo 実データ E2E | code path / prereq script はあるが、実データと Python 3.10 環境が必要。 |
| C | NMEA / GNSS / IMU robustness | IMU orientation CSV normalization、NMEA checksum validation、GNSS timestamp anomaly、IMU angular-velocity yaw fallback の first slices は対応済み。 |
| C | depth / appearance / sky の比較評価 | 実装済み要素はあるが、比較実験としては未整理。 |
| D | MCD `ntu_day_02` quality push | 長い valid-GNSS session、multi-camera、training budget 比較が必要。 |

## 現在の production assets

`docs/scenes-list.json` の production scene list:

1. `assets/outdoor-demo/outdoor-demo.splat` — Autoware 6-bag supervised default
2. `assets/outdoor-demo/outdoor-demo-dust3r.splat` — bag6 DUSt3R pose-free
3. `assets/outdoor-demo/mcd-tuhh-day04.splat` — MCD `tuhh_day_04` DUSt3R pose-free
4. `assets/outdoor-demo/bag6-mast3r.splat` — bag6 MAST3R pose-free metric
5. `assets/outdoor-demo/bag6-vggt-slam-20-15k.splat` — bag6 VGGT-SLAM 2.0 comparison
6. `assets/outdoor-demo/bag6-mast3r-slam-20-15k.splat` — bag6 MASt3R-SLAM comparison
7. `assets/outdoor-demo/mcd-tuhh-day04-mast3r.splat` — MCD `tuhh_day_04` MAST3R pose-free metric
8. `assets/outdoor-demo/mcd-ntu-day02-supervised.splat` — MCD `ntu_day_02` supervised valid-GNSS demo

`assets/outdoor-demo/mcd-tuhh-day04-supervised.splat` may exist as a diagnostic artifact, but it must not be added to the production picker or benchmark table.

## 実装の読みどころ

| Area | Files | Notes |
| --- | --- | --- |
| MCD calibration / static TF | `src/gs_sim2real/datasets/ros_tf.py`, `scripts/download_mcd_calibration.sh` | MCDVIRAL official calibration YAML を downloader 経由で取得。YAML は CC BY-NC-SA なので repo に commit しない。 |
| MCD supervised sparse import | `src/gs_sim2real/cli.py`, `src/gs_sim2real/datasets/mcd.py` | `--mcd-static-calibration`、single-camera colorize/depth、CameraInfo 欠落時の PINHOLE 合成、zero-GNSS guard、IMU orientation CSV normalization、angular-velocity yaw fallback。 |
| External SLAM import | `src/gs_sim2real/preprocess/external_slam.py`, `src/gs_sim2real/preprocess/external_slam_artifacts/` | facade + profile/resolver/materializer/importer 分割済み。VGGT-SLAM / MASt3R-SLAM 実走済み、Pi3 / LoGeR 候補追加済み。 |
| Pages scene contract | `docs/scenes-list.json`, `scripts/pages_scene_manifest.py`, `tests/test_pages_assets.py` | README table、preview capture、hero GIF、3 viewer picker を manifest に揃える。 |
| README preview capture | `scripts/capture_readme_splat_previews.py` | WebGL は headed Chromium 推奨。`--out-dir` で smoke capture を一時出力可能。 |
| Hero GIF | `scripts/record_demo_gif.py` | `docs/scenes-list.json` の production scenes を順に cycle する。 |
| Experiment-process docs | `src/gs_sim2real/experiments/process_docs.py`, `docs/experiments.md`, `docs/experiments.generated.md` | 公開 index と詳細生成表を分離済み。 |

## 検証コマンド

通常の pre-PR:

```bash
ruff format --check src/ tests/ scripts/
ruff check src/ tests/ scripts/
PYTHONPATH=src pytest tests/ -q --ignore=tests/e2e
```

屋外 / Pages まわりを触ったとき:

```bash
PYTHONPATH=src pytest \
  tests/test_pages_assets.py \
  tests/test_viewer.py \
  tests/test_mcd.py \
  tests/test_mcd_gnss_preflight.py \
  tests/test_external_slam.py \
  -q
```

Viewer assets だけなら:

```bash
PYTHONPATH=src pytest tests/test_pages_assets.py tests/test_viewer.py -q
```

README preview PNG を再生成する場合:

```bash
export DISPLAY=:0
python3 scripts/capture_readme_splat_previews.py
```

Hero GIF を再生成する場合:

```bash
python3 scripts/record_demo_gif.py
```

MCD GNSS を新 session で使う前:

```bash
python3 scripts/check_mcd_gnss.py <session-dir> --gnss-topic /vn200/GPS
```

## Scope Boundaries

- Python package path `gs_sim2real` is intentionally kept for compatibility. Do not rename it as part of outdoor-pipeline work.
- Keep the legacy `gs-sim2real` CLI alias unless doing a dedicated deprecation/removal pass.
- Do not commit downloaded MCD calibration YAML, rosbag data, Waymo tfrecords, or generated training outputs.
- Avoid broad DreamWalker / Unity / robotics reorgs from this handoff. Outdoor 3DGS work should stay scoped to preprocessing, training/export, viewer assets, and validation.
- Treat vendored viewer code, especially `docs/splat-viewer/main.js`, as external unless a focused compatibility fix is required.

## 既知の落とし穴

- MCD topic は `/vn200/GPS` の大文字 `GPS`。`/vn200/gps` ではない。
- `tuhh_day_04` の `/vn200/GPS` は all-zero。supervised GNSS demo には使わない。
- MCD calibration YAML は公式 Download page から取得できるが、license 上 repo に YAML を commit しない。
- IMU orientation CSV は zero-length / non-finite quaternion を無視し、全 identity のときだけ姿勢なし扱いにする。一定の non-identity mount orientation は有効な姿勢として残す。orientation が全 identity でも `angular_velocity_z` が非ゼロなら yaw-only fallback として積分する。
- `capture_readme_splat_previews.py` は headless だと WebGL canvas が真っ黒になることがある。CI では静的 contract test、実 capture は headed smoke として扱う。
- `docs/scenes-list.json` に production scene を追加したら、README table、viewer picker 3 種、preview PNG が `tests/test_pages_assets.py` で一致する必要がある。
- Waymo は code path があっても実データ E2E 未検証。Python 3.10 venv と dataset agreement を先に確認する。

## Archive Map

古い詳細は [archive snapshot](archive/plan_outdoor_gs_2026_04_full_handoff.md) に残しています。

| Need | Archive section |
| --- | --- |
| PR #55〜#80 の時系列 | `## 15`, `## 15.1`, `## 15.2` |
| `tuhh_day_04` supervised 誤判定の詳細 | `## 15.3`, `## 15.4` |
| `ntu_day_02` valid-GNSS 実走値 | `## 15.5` |
| MCD calibration YAML discovery / Drive ID | `## 4.3.3.a`, `## 4.3.3.c`, `## 15.1` |
| 8-scene viewer smoke transcript | `## 15.3` |
| Legacy command blocks / one-off output paths | `## 9`, `## 15.*` |

## 関連ドキュメント

| File | Role |
| --- | --- |
| `README.md` | Public-facing overview, live demo, benchmark table |
| `CONTRIBUTING.md` | Development workflow and issue/PR expectations |
| `docs/experiments.md` | Public experiment-process index |
| `docs/experiments.generated.md` | Generated detailed experiment comparison tables |
| `docs/decisions.md` | Accepted/deferred design decisions |
| `docs/interfaces.md` | Stable interfaces that production code may depend on |
| `docs/archive/plan_outdoor_gs_2026_04_full_handoff.md` | Full historical outdoor-GS handoff snapshot |
