# DreamWalker Unity Project

この Unity 6 URP プロジェクトは `external/UnityGaussianSplatting/projects/GaussianExample-URP`
をベースに自動展開したものです。

## 依存

- `external/UnityGaussianSplatting/package`
- `shared/unity/GaussianAdventureShared`

## 初回起動時

`Assets/Scenes/DreamWalker_Main.unity` が存在しない場合、
`Assets/Editor/DreamWalkerProjectAutoBootstrap.cs` が自動で starter scene を生成します。

Unity Editor が CLI から使える環境なら、リポジトリ直下で次も使えます。

```bash
./tools/unity/bootstrap_dreamwalker.sh
./tools/unity/open_dreamwalker.sh
```

## 次にやること

1. `Assets/Art/Splats/` に変換済み `GaussianSplatAsset` を置く
2. `Assets/Art/ColliderMeshes/` に Marble collider mesh を入れる
3. `WalkableDebugFloor` を削除して collider mesh に置き換える
4. `SplatRoot` に `GaussianSplatRenderer` を配置する
