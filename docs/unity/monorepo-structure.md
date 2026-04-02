# Unity Gaussian Splat Monorepo 構成案

## ねらい

このリポジトリでは、既存の Python ベース 3DGS 学習・変換ツール群を残しつつ、
Unity 側では複数のゲームプロトタイプを独立プロジェクトとして持つ構成が現実的です。

理由は次の通りです。

- Unity プロジェクトごとに `ProjectSettings/`、Render Pipeline、入力設定、XR 設定が独立して管理できる
- `Library/` や import cache を分離でき、プロトタイプ間で壊れにくい
- 共通コードは UPM ローカルパッケージとして `shared/` に寄せられる
- 巨大な `.spz` / `.ply` / `.splat` は `raw_assets/` に逃がし、Git 管理と切り分けられる

## 推奨構成

```text
gs-sim2real/
├── configs/                              # 既存: 学習・前処理設定
├── docs/
│   ├── prototypes/
│   │   ├── dreamwalker.md                # 各プロトタイプのGDD/メモ
│   │   └── prototype-template.md
│   └── unity/
│       ├── monorepo-structure.md
│       └── dreamwalker-setup.md
├── external/
│   └── UnityGaussianSplatting/           # Aras本体 or Marble推奨forkをclone
├── projects/
│   ├── DreamWalker/                      # Unity 6 URP プロジェクト
│   │   ├── Assets/
│   │   │   ├── Scenes/
│   │   │   ├── Prefabs/
│   │   │   ├── Art/
│   │   │   │   ├── Splats/               # 変換済みGaussianSplatAsset
│   │   │   │   └── ColliderMeshes/       # MarbleのGLB collider mesh
│   │   │   ├── UI/
│   │   │   └── Settings/
│   │   ├── Packages/
│   │   └── ProjectSettings/
│   ├── EchoMaze/
│   └── SkyArchive/
├── raw_assets/
│   ├── marble/
│   │   ├── splats/                       # .spz / .ply / .splat (gitignore)
│   │   ├── colliders/                    # 元のGLB/FBX/obj
│   │   └── reference/                    # 参照画像・メタデータ
│   └── captured/
├── scripts/                              # 既存: Python 補助スクリプト
├── shared/
│   └── unity/
│       └── GaussianAdventureShared/      # 複数Unityプロジェクトで共有するローカルUPM
│           ├── package.json
│           └── Runtime/
├── src/gs_sim2real/               # 既存: 学習・viewer・CLI
├── tests/                                # 既存: Pythonテスト
└── tools/
    ├── marble/                           # 変換メモ・補助CLI
    ├── splat/
    └── validation/
```

## 運用ルール

1. Unity プロトタイプは `projects/<PrototypeName>/` ごとに完全独立で持つ
2. 共通 C# コードは `shared/unity/GaussianAdventureShared` に集約し、各プロジェクトから `file:` 参照する
3. Aras / Marble 由来の外部依存は `external/` にまとめる
4. 生の splat と collider mesh は `raw_assets/` に置き、基本は Git から外す
5. Unity に import した後の `GaussianSplatAsset` は、必要最小限だけ各プロジェクト配下に置く

## DreamWalker で最初に作るべき最小単位

- `projects/DreamWalker`
- `shared/unity/GaussianAdventureShared`
- `external/UnityGaussianSplatting`
- `raw_assets/marble/splats`
- `raw_assets/marble/colliders`

## なぜ `shared/` を UPM パッケージ化するのか

`Assets/SharedScripts` のような直置きでも動きますが、複数プロジェクト運用では次の理由で UPM 化が有利です。

- バージョン差分が追いやすい
- 依存範囲が明確
- 他プロジェクトへの再利用が簡単
- 将来 `Samples~` や Editor 拡張を足しやすい
