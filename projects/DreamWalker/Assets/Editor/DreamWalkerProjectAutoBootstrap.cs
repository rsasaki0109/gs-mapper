using System.IO;
using NerfGsPlayground.GaussianAdventureShared.Editor;
using UnityEditor;

namespace DreamWalkerProject.Editor
{
    [InitializeOnLoad]
    public static class DreamWalkerProjectAutoBootstrap
    {
        private const string SessionKey = "DreamWalkerProjectAutoBootstrap.Ran";
        private const string ScenePath = "Assets/Scenes/DreamWalker_Main.unity";

        static DreamWalkerProjectAutoBootstrap()
        {
            EditorApplication.delayCall += TryBootstrap;
        }

        private static void TryBootstrap()
        {
            if (SessionState.GetBool(SessionKey, false))
            {
                return;
            }

            SessionState.SetBool(SessionKey, true);

            if (File.Exists(ScenePath))
            {
                return;
            }

            DreamWalkerSetupMenu.CreateStarterSceneInternal(false);
            AssetDatabase.Refresh();
        }
    }
}
