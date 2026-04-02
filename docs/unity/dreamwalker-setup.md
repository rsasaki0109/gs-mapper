# DreamWalker セットアップ手順

## 0. 前提

- Unity 6 URP を使う
- Windows では Graphics API を `Direct3D12` または `Vulkan` にする
- Aras の `UnityGaussianSplatting` を使う
- Marble の splat を使う場合は、World Labs が案内している patched fork を優先する

## 1. Unity 6 URP プロジェクトを作る

1. Unity Hub で `New project`
2. テンプレートは `Universal 3D (URP)` を選ぶ
3. Project Name を `DreamWalker` にする
4. 保存先を `projects/DreamWalker` にする
5. 初回起動後、不要な Sample を消して構成を軽くする

## 1.5 入力設定を合わせる

付属の `DreamWalkerFirstPersonController` は旧 Input Manager ベースです。
そのため、まず次を確認してください。

1. `Edit > Project Settings > Player > Active Input Handling`
2. `Both` または `Input Manager (Old)` にする

`Input System Package (New)` のみだと、そのままでは移動入力を拾いません。

## 2. Graphics API を固定する

1. `Edit > Project Settings > Player > Other Settings`
2. `Auto Graphics API for Windows` をオフ
3. `Direct3D11` を外す
4. `Direct3D12` を先頭に置く
5. VR も視野に入れるなら `Vulkan` も追加候補

## 3. URP 側の必須設定

1. 使用中の URP Renderer Asset を開く
2. `Renderer Features` に `GaussianSplatURPFeature` を追加する
3. URP 設定で `Render Graph Compatibility Mode` をオフにする
4. MSAA はオフ推奨
5. VR を触るなら HDR をオンにしておく

## 4. Aras プラグインを入れる

### 安定優先

Marble の `spz` と複数 splat を扱うなら、World Labs ドキュメントで推奨されている fork を `external/UnityGaussianSplatting` に clone し、
`Packages/manifest.json` から `file:` 参照するのが安全です。

```json
{
  "dependencies": {
    "org.nesnausk.gaussian-splatting": "file:../../external/UnityGaussianSplatting/package",
    "com.rsasaki.gaussian-adventure-shared": "file:../../shared/unity/GaussianAdventureShared"
  }
}
```

### 原家元のまま使う場合

UPM の Git URL で入れるなら、`package/` サブフォルダを指す必要があります。

```text
https://github.com/aras-p/UnityGaussianSplatting.git?path=/package
```

## 5. Marble の `.spz` / `.ply` を import する

1. `raw_assets/marble/splats` に `2M spz` または `2M ply` を置く
2. Unity で `Tools > Gaussian Splats > Create GaussianSplatAsset`
3. `Input PLY/SPZ File` に対象ファイルを指定
4. 出力先は `Assets/Art/Splats/` にする
5. 圧縮プリセットを選んで `Create Asset`

### Marble 固有の注意

- `500k spz` は既知の import 問題がある。必要なら一度 PLY に変換してから入れる
- World Labs の世界座標は OpenCV 系で、他ツールと軸の向きが合わないことがある
- splat 見た目は合っても、物理衝突は別に用意しないと歩行できない

## 6. collider mesh を入れる

1. Marble の GLB collider mesh を `raw_assets/marble/colliders` に置く
2. `Assets/Art/ColliderMeshes/` へ import
3. `MeshCollider` を付ける
4. `Walkable` レイヤーに分ける
5. プレイヤーの ground probe はこのレイヤーだけを見る

## 7. シーンの最小構成

```text
DreamWalker_Main
├── Global
│   ├── Directional Light
│   ├── Global Volume
│   └── EventSystem
├── SplatRoot
│   └── GaussianSplatRenderer
├── WalkableProxyRoot
│   └── MarbleColliderMesh (MeshCollider, layer=Walkable)
└── Player
    ├── CharacterController
    ├── SplatRaycastHelper
    ├── DreamWalkerFirstPersonController
    └── Main Camera
```

### 最短セットアップ

共通パッケージ導入後に、Unity メニューから次を実行すると最小シーンの土台を自動生成できます。

```text
Tools > DreamWalker > Create Starter Scene
```

このメニューは次を行います。

- `Walkable` レイヤーの作成
- `Global` / `SplatRoot` / `WalkableProxyRoot` / `Player` の生成
- `Systems` の生成と `DreamStateManager` / `DreamWalkerHUD` の追加
- `CharacterController`、`SplatRaycastHelper`、`DreamWalkerFirstPersonController` の自動追加
- `SplatInteractProbe`、`DreamViewEffects`、`DreamScreenFader` の自動追加
- デバッグ用の仮床 `WalkableDebugFloor` の生成
- 検証用の `SampleShard x3`、`SampleDistortionZone`、`DreamGate`、`SampleEchoNote` の生成
- `DreamGate` 用のローカル転送先 `DreamGateDestination` の生成
- `Assets/Scenes/DreamWalker_Main.unity` の保存

最初はそのまま Play して、`SampleShard` を 3 つ拾うと `DreamGate` が開き、
入るとフェード付きで同一シーン内の destination へローカル転送されることを確認できます。
転送先では `SampleEchoNote` を `E` で読めます。
その後、`WalkableDebugFloor` を削除して Marble の collider mesh に置き換えます。

## 8. 初期調整ポイント

- `GaussianSplatRenderer` の transform は見た目が合うまで回転・スケールを調整する
- プレイヤー開始位置は collider mesh の上に置く
- `CharacterController` は `Height=1.8`, `Radius=0.35`, `Step Offset=0.3` から始める
- `SplatRaycastHelper` は `Walkable` レイヤー限定にする
- 演出用ポストプロセスは後から足す

## 9. DreamWalker の最初のプレイ確認

1. マウスで視点が回る
2. WASD で移動できる
3. Space でジャンプできる
4. `F` で低重力浮遊モードが切り替わる
5. 地面判定が Marble collider mesh 上で安定する
