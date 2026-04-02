using System.IO;
using NerfGsPlayground.GaussianAdventureShared.Editor;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DreamWalkerProject.Editor
{
    public static class DreamWalkerProjectCli
    {
        private const string ScenePath = "Assets/Scenes/DreamWalker_Main.unity";

        public static void Bootstrap()
        {
            Debug.Log("DreamWalker CLI: bootstrap start");

            if (!File.Exists(ScenePath))
            {
                DreamWalkerSetupMenu.CreateStarterSceneInternal(false);
                AssetDatabase.Refresh();
            }

            if (File.Exists(ScenePath))
            {
                EditorSceneManager.OpenScene(ScenePath);
                Debug.Log($"DreamWalker CLI: scene ready at {ScenePath}");
                return;
            }

            throw new FileNotFoundException($"DreamWalker CLI: failed to create scene at {ScenePath}");
        }

        public static void OpenMainScene()
        {
            if (!File.Exists(ScenePath))
            {
                throw new FileNotFoundException($"DreamWalker CLI: scene not found at {ScenePath}");
            }

            EditorSceneManager.OpenScene(ScenePath);
            Debug.Log($"DreamWalker CLI: opened {ScenePath}");
        }

        public static void ValidateProject()
        {
            string manifestPath = "Packages/manifest.json";

            if (!File.Exists(manifestPath))
            {
                throw new FileNotFoundException($"DreamWalker CLI: missing manifest {manifestPath}");
            }

            string manifestText = File.ReadAllText(manifestPath);
            bool hasSharedPackage = manifestText.Contains("com.rsasaki.gaussian-adventure-shared");
            bool hasGaussianPackage = manifestText.Contains("org.nesnausk.gaussian-splatting");

            if (!hasSharedPackage || !hasGaussianPackage)
            {
                throw new IOException("DreamWalker CLI: required local packages are missing from manifest.json");
            }

            Debug.Log("DreamWalker CLI: manifest validation passed");
        }
    }
}
