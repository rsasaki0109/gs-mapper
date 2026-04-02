# GaussianAdventureShared

複数の Unity プロトタイプから共有利用するローカル UPM パッケージです。

## 追加方法

`projects/<ProjectName>/Packages/manifest.json` に次を追加します。

```json
"com.rsasaki.gaussian-adventure-shared": "file:../../shared/unity/GaussianAdventureShared"
```

## 収録内容

- `IInteractable`
- `SplatInteractProbe`
- `SplatRaycastHelper`
- `DreamWalkerFirstPersonController`
- `DreamViewEffects`
- `DreamDistortionZone`
- `DistortionShard`
- `DreamEchoNote`
- `DreamGate`
- `DreamScreenFader`
- `DreamStateManager`
- `DreamWalkerHUD`
- `Tools/DreamWalker/Create Starter Scene` エディタメニュー

## 方針

生の Gaussian Splat そのものには安定した物理判定を持たせず、
Marble の collider mesh や手置き proxy collider と組み合わせて歩行判定を作ります。
