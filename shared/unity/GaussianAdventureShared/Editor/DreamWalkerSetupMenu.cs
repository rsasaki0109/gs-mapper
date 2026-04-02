using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using NerfGsPlayground.GaussianAdventureShared;
using System.IO;

namespace NerfGsPlayground.GaussianAdventureShared.Editor
{
    public static class DreamWalkerSetupMenu
    {
        private const string WalkableLayerName = "Walkable";
        private const string SceneFolderPath = "Assets/Scenes";
        private const string ScenePath = SceneFolderPath + "/DreamWalker_Main.unity";

        [MenuItem("Tools/DreamWalker/Create Starter Scene")]
        public static void CreateStarterScene()
        {
            CreateStarterSceneInternal(true);
        }

        public static void CreateStarterSceneInternal(bool showCompletionDialog)
        {
            EnsureFolder("Assets", "Scenes");
            EnsureFolder("Assets", "Art");
            EnsureFolder("Assets/Art", "Splats");
            EnsureFolder("Assets/Art", "ColliderMeshes");
            EnsureFolder("Assets", "Prefabs");
            EnsureFolder("Assets", "UI");
            EnsureFolder("Assets", "Editor");

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            int walkableLayer = EnsureLayerExists(WalkableLayerName);
            if (walkableLayer < 0)
            {
                walkableLayer = 0;
                Debug.LogWarning("Walkable レイヤーを自動作成できなかったため Default レイヤーで続行します。");
            }

            GameObject globalRoot = new GameObject("Global");
            CreateDirectionalLight(globalRoot.transform);

            new GameObject("SplatRoot");

            GameObject walkableRoot = new GameObject("WalkableProxyRoot");
            walkableRoot.layer = walkableLayer;
            CreateDebugFloor(walkableRoot.transform, walkableLayer);

            GameObject player = CreatePlayerRig(walkableLayer, out SplatInteractProbe interactProbe, out DreamWalkerFirstPersonController controller);
            CreateSystemsRoot(interactProbe, controller);
            CreateSampleCollectibleLoop(walkableLayer);

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, ScenePath, true);
            EnsureSceneInBuildSettings(ScenePath);
            AssetDatabase.Refresh();
            Selection.activeGameObject = player;

            if (showCompletionDialog)
            {
                EditorUtility.DisplayDialog(
                    "DreamWalker",
                    "Starter Scene を作成しました。\n\n" +
                    "次に行うこと:\n" +
                    "1. GaussianSplatRenderer を SplatRoot 配下に追加\n" +
                    "2. WalkableDebugFloor を Marble collider mesh に置き換え\n" +
                    "3. Active Input Handling を Both または Input Manager (Old) に設定\n" +
                    "4. 3つの SampleShard を拾って DreamGate が開くことを確認\n" +
                    "5. DreamGate の先で SampleEchoNote を読む",
                    "OK");
            }
        }

        private static GameObject CreatePlayerRig(
            int walkableLayer,
            out SplatInteractProbe interactProbe,
            out DreamWalkerFirstPersonController controller)
        {
            GameObject player = new GameObject("Player");
            player.transform.position = new Vector3(0f, 1.8f, -3f);

            CharacterController characterController = player.AddComponent<CharacterController>();
            characterController.height = 1.8f;
            characterController.radius = 0.35f;
            characterController.center = new Vector3(0f, 0.9f, 0f);
            characterController.stepOffset = 0.3f;
            characterController.slopeLimit = 55f;
            characterController.skinWidth = 0.03f;
            characterController.minMoveDistance = 0f;

            SplatRaycastHelper raycastHelper = player.AddComponent<SplatRaycastHelper>();
            SetPrivateLayerMask(raycastHelper, "walkableMask", 1 << walkableLayer);

            controller = player.AddComponent<DreamWalkerFirstPersonController>();
            interactProbe = player.AddComponent<SplatInteractProbe>();
            DreamViewEffects viewEffects = player.AddComponent<DreamViewEffects>();
            player.AddComponent<DreamScreenFader>();

            GameObject cameraObject = new GameObject("Main Camera");
            cameraObject.transform.SetParent(player.transform, false);
            cameraObject.transform.localPosition = new Vector3(0f, 0.72f, 0f);
            cameraObject.tag = "MainCamera";

            Camera cameraComponent = cameraObject.AddComponent<Camera>();
            cameraComponent.nearClipPlane = 0.03f;
            cameraObject.AddComponent<AudioListener>();

            BindControllerReferences(controller, characterController, cameraComponent, raycastHelper);
            BindInteractProbe(interactProbe, cameraComponent);
            BindViewEffects(viewEffects, cameraComponent, controller);
            return player;
        }

        private static void CreateSystemsRoot(SplatInteractProbe interactProbe, DreamWalkerFirstPersonController controller)
        {
            GameObject systemsRoot = new GameObject("Systems");
            DreamStateManager stateManager = systemsRoot.AddComponent<DreamStateManager>();
            DreamWalkerHUD hud = systemsRoot.AddComponent<DreamWalkerHUD>();

            BindHudReferences(hud, stateManager, interactProbe, controller);
        }

        private static void CreateDirectionalLight(Transform parent)
        {
            GameObject lightObject = new GameObject("Directional Light");
            lightObject.transform.SetParent(parent, false);
            lightObject.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            Light lightComponent = lightObject.AddComponent<Light>();
            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 1.1f;
        }

        private static void CreateDebugFloor(Transform parent, int walkableLayer)
        {
            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "WalkableDebugFloor";
            floor.transform.SetParent(parent, false);
            floor.transform.position = new Vector3(0f, -0.5f, 5f);
            floor.transform.localScale = new Vector3(12f, 1f, 24f);
            floor.layer = walkableLayer;
        }

        private static void CreateSampleCollectibleLoop(int walkableLayer)
        {
            GameObject gameplayRoot = new GameObject("SampleGameplay");

            GameObject shardRoot = new GameObject("SampleShards");
            shardRoot.transform.SetParent(gameplayRoot.transform, false);

            CreateSampleShard(shardRoot.transform, "SampleShard_01", new Vector3(-1.75f, 0.85f, 4.5f));
            CreateSampleShard(shardRoot.transform, "SampleShard_02", new Vector3(0f, 0.85f, 6f));
            CreateSampleShard(shardRoot.transform, "SampleShard_03", new Vector3(1.75f, 0.85f, 7.5f));

            CreateSampleDistortionZone(gameplayRoot.transform, new Vector3(0f, 0.75f, 8.75f));
            CreateSampleGate(gameplayRoot.transform, new Vector3(0f, 1.1f, 11.5f), out Transform localDestination);
            CreateSampleDestinationPlatform(gameplayRoot.transform, localDestination, walkableLayer);
            CreateSampleEchoNote(gameplayRoot.transform, new Vector3(0f, 0.95f, 18.2f));
        }

        private static void CreateSampleShard(Transform parent, string objectName, Vector3 position)
        {
            GameObject shard = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            shard.name = objectName;
            shard.transform.SetParent(parent, false);
            shard.transform.position = position;
            shard.transform.localScale = Vector3.one * 0.45f;

            TintObject(shard, new Color(0.6f, 0.95f, 1f, 1f));
            DistortionShard distortionShard = shard.AddComponent<DistortionShard>();
            SetPrivateString(distortionShard, "shardDisplayName", objectName);
        }

        private static void CreateSampleDistortionZone(Transform parent, Vector3 position)
        {
            GameObject zone = GameObject.CreatePrimitive(PrimitiveType.Cube);
            zone.name = "SampleDistortionZone";
            zone.transform.SetParent(parent, false);
            zone.transform.position = position;
            zone.transform.localScale = new Vector3(3.5f, 1.5f, 3.5f);

            BoxCollider zoneCollider = zone.GetComponent<BoxCollider>();
            zoneCollider.isTrigger = true;

            TintObject(zone, new Color(0.45f, 0.75f, 1f, 0.22f));

            DreamDistortionZone distortionZone = zone.AddComponent<DreamDistortionZone>();
            SetPrivateFloat(distortionZone, "intensity", 0.9f);
        }

        private static void CreateSampleGate(Transform parent, Vector3 position, out Transform localDestination)
        {
            GameObject gateRoot = new GameObject("DreamGate");
            gateRoot.transform.SetParent(parent, false);
            gateRoot.transform.position = position;

            BoxCollider entryTrigger = gateRoot.AddComponent<BoxCollider>();
            entryTrigger.isTrigger = true;
            entryTrigger.size = new Vector3(2.5f, 2.5f, 1.6f);

            DreamGate gate = gateRoot.AddComponent<DreamGate>();

            GameObject closedVisual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            closedVisual.name = "ClosedVisual";
            closedVisual.transform.SetParent(gateRoot.transform, false);
            closedVisual.transform.localPosition = new Vector3(0f, 0f, 0f);
            closedVisual.transform.localScale = new Vector3(2.2f, 2.2f, 0.3f);
            RemoveCollider(closedVisual);
            TintObject(closedVisual, new Color(0.95f, 0.55f, 0.75f, 1f));

            GameObject openVisual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            openVisual.name = "OpenVisual";
            openVisual.transform.SetParent(gateRoot.transform, false);
            openVisual.transform.localPosition = new Vector3(0f, 0f, 0f);
            openVisual.transform.localScale = new Vector3(2.3f, 2.3f, 0.08f);
            RemoveCollider(openVisual);
            TintObject(openVisual, new Color(0.55f, 0.95f, 1f, 0.9f));
            openVisual.SetActive(false);

            GameObject blocker = GameObject.CreatePrimitive(PrimitiveType.Cube);
            blocker.name = "GateBlocker";
            blocker.transform.SetParent(gateRoot.transform, false);
            blocker.transform.localPosition = new Vector3(0f, 0f, 0f);
            blocker.transform.localScale = new Vector3(1.85f, 2f, 0.25f);
            TintObject(blocker, new Color(0.08f, 0.1f, 0.16f, 0.35f));

            BoxCollider blockingCollider = blocker.GetComponent<BoxCollider>();

            GameObject destination = new GameObject("DreamGateDestination");
            destination.transform.SetParent(parent, false);
            destination.transform.position = new Vector3(0f, 1.1f, 16.5f);
            destination.transform.rotation = Quaternion.Euler(0f, 180f, 0f);
            localDestination = destination.transform;

            BindGateReferences(gate, blockingCollider, entryTrigger, closedVisual, openVisual, localDestination);
        }

        private static void CreateSampleDestinationPlatform(Transform parent, Transform destination, int walkableLayer)
        {
            if (destination == null)
            {
                return;
            }

            GameObject platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
            platform.name = "DreamGateDestinationPlatform";
            platform.transform.SetParent(parent, false);
            platform.transform.position = new Vector3(destination.position.x, 0f, destination.position.z);
            platform.transform.localScale = new Vector3(5f, 0.8f, 5f);
            platform.layer = walkableLayer;
            TintObject(platform, new Color(0.24f, 0.32f, 0.5f, 1f));

            GameObject beacon = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            beacon.name = "DreamGateDestinationBeacon";
            beacon.transform.SetParent(parent, false);
            beacon.transform.position = destination.position + new Vector3(0f, 0.4f, 0f);
            beacon.transform.localScale = new Vector3(0.35f, 0.8f, 0.35f);
            RemoveCollider(beacon);
            TintObject(beacon, new Color(0.75f, 1f, 0.95f, 1f));
        }

        private static void CreateSampleEchoNote(Transform parent, Vector3 position)
        {
            GameObject note = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            note.name = "SampleEchoNote";
            note.transform.SetParent(parent, false);
            note.transform.position = position;
            note.transform.localScale = new Vector3(0.65f, 0.85f, 0.65f);
            TintObject(note, new Color(0.95f, 0.9f, 0.62f, 1f));

            DreamEchoNote echoNote = note.AddComponent<DreamEchoNote>();
            SetPrivateString(echoNote, "noteTitle", "Echo Note: 目覚め損ねた廊下");
            SetPrivateString(
                echoNote,
                "noteBody",
                "扉の向こうに進んでも、まだ目覚めではない。\n" +
                "美しい splat は記憶の表面で、歩ける床はその裏側にある。\n" +
                "足場を疑い、光を疑い、それでも先へ進め。");
        }

        private static void BindControllerReferences(
            DreamWalkerFirstPersonController controller,
            CharacterController characterController,
            Camera playerCamera,
            SplatRaycastHelper raycastHelper)
        {
            SerializedObject serializedObject = new SerializedObject(controller);
            serializedObject.FindProperty("characterController").objectReferenceValue = characterController;
            serializedObject.FindProperty("playerCamera").objectReferenceValue = playerCamera;
            serializedObject.FindProperty("splatRaycastHelper").objectReferenceValue = raycastHelper;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BindInteractProbe(SplatInteractProbe probe, Camera playerCamera)
        {
            SerializedObject serializedObject = new SerializedObject(probe);
            serializedObject.FindProperty("sourceCamera").objectReferenceValue = playerCamera;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BindHudReferences(
            DreamWalkerHUD hud,
            DreamStateManager stateManager,
            SplatInteractProbe interactProbe,
            DreamWalkerFirstPersonController controller)
        {
            SerializedObject serializedObject = new SerializedObject(hud);
            serializedObject.FindProperty("stateManager").objectReferenceValue = stateManager;
            serializedObject.FindProperty("interactProbe").objectReferenceValue = interactProbe;
            serializedObject.FindProperty("playerController").objectReferenceValue = controller;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BindViewEffects(
            DreamViewEffects viewEffects,
            Camera playerCamera,
            DreamWalkerFirstPersonController controller)
        {
            SerializedObject serializedObject = new SerializedObject(viewEffects);
            serializedObject.FindProperty("targetCamera").objectReferenceValue = playerCamera;
            serializedObject.FindProperty("playerController").objectReferenceValue = controller;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BindGateReferences(
            DreamGate gate,
            Collider blockingCollider,
            Collider entryTrigger,
            GameObject closedVisual,
            GameObject openVisual,
            Transform localDestination)
        {
            SerializedObject serializedObject = new SerializedObject(gate);
            serializedObject.FindProperty("blockingCollider").objectReferenceValue = blockingCollider;
            serializedObject.FindProperty("entryTrigger").objectReferenceValue = entryTrigger;
            serializedObject.FindProperty("closedStateRoot").objectReferenceValue = closedVisual;
            serializedObject.FindProperty("openStateRoot").objectReferenceValue = openVisual;
            serializedObject.FindProperty("localDestination").objectReferenceValue = localDestination;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetPrivateLayerMask(Object target, string propertyName, int maskValue)
        {
            SerializedObject serializedObject = new SerializedObject(target);
            serializedObject.FindProperty(propertyName).intValue = maskValue;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetPrivateString(Object target, string propertyName, string value)
        {
            SerializedObject serializedObject = new SerializedObject(target);
            serializedObject.FindProperty(propertyName).stringValue = value;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetPrivateFloat(Object target, string propertyName, float value)
        {
            SerializedObject serializedObject = new SerializedObject(target);
            serializedObject.FindProperty(propertyName).floatValue = value;
            serializedObject.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void TintObject(GameObject target, Color color)
        {
            Renderer renderer = target.GetComponent<Renderer>();
            if (renderer == null || renderer.sharedMaterial == null)
            {
                return;
            }

            renderer.sharedMaterial = new Material(renderer.sharedMaterial);
            renderer.sharedMaterial.color = color;
        }

        private static void RemoveCollider(GameObject target)
        {
            Collider collider = target.GetComponent<Collider>();
            if (collider != null)
            {
                Object.DestroyImmediate(collider);
            }
        }

        private static void EnsureFolder(string parentFolder, string newFolderName)
        {
            string folderPath = parentFolder + "/" + newFolderName;
            if (!AssetDatabase.IsValidFolder(folderPath))
            {
                AssetDatabase.CreateFolder(parentFolder, newFolderName);
            }
        }

        private static void EnsureSceneInBuildSettings(string scenePath)
        {
            if (!File.Exists(scenePath))
            {
                return;
            }

            EditorBuildSettingsScene[] existingScenes = EditorBuildSettings.scenes;
            string legacySampleScenePath = "Assets/GSTestScene.unity";

            int retainedSceneCount = 1;
            for (int i = 0; i < existingScenes.Length; i++)
            {
                string existingPath = existingScenes[i].path;

                if (existingPath == scenePath || existingPath == legacySampleScenePath)
                {
                    continue;
                }

                retainedSceneCount++;
            }

            EditorBuildSettingsScene[] updatedScenes = new EditorBuildSettingsScene[retainedSceneCount];
            updatedScenes[0] = new EditorBuildSettingsScene(scenePath, true);

            int writeIndex = 1;
            for (int i = 0; i < existingScenes.Length; i++)
            {
                string existingPath = existingScenes[i].path;

                if (existingPath == scenePath || existingPath == legacySampleScenePath)
                {
                    continue;
                }

                updatedScenes[writeIndex] = existingScenes[i];
                writeIndex++;
            }

            EditorBuildSettings.scenes = updatedScenes;
        }

        private static int EnsureLayerExists(string layerName)
        {
            int existingLayer = LayerMask.NameToLayer(layerName);
            if (existingLayer >= 0)
            {
                return existingLayer;
            }

            SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            SerializedProperty layersProperty = tagManager.FindProperty("layers");

            for (int i = 8; i < layersProperty.arraySize; i++)
            {
                SerializedProperty layerProperty = layersProperty.GetArrayElementAtIndex(i);
                if (layerProperty.stringValue == layerName)
                {
                    return i;
                }

                if (!string.IsNullOrEmpty(layerProperty.stringValue))
                {
                    continue;
                }

                layerProperty.stringValue = layerName;
                tagManager.ApplyModifiedProperties();
                return i;
            }

            return -1;
        }
    }
}
