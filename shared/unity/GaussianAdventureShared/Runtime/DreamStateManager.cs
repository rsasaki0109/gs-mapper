using System;
using System.Collections.Generic;
using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// DreamWalker の最小進行状態。
    /// まずは「欠片を集める -> 門が開く」の 1 ループだけを管理する。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream State Manager")]
    public sealed class DreamStateManager : MonoBehaviour
    {
        [SerializeField, Min(1)] private int requiredShardCount = 3;
        [SerializeField] private bool persistAcrossScenes = false;
        [SerializeField] private bool logStateChanges = false;

        private static DreamStateManager instance;

        private readonly HashSet<string> collectedShardIds = new HashSet<string>();
        private int collectedShardCount;

        public static DreamStateManager Instance => instance;

        public int RequiredShardCount => requiredShardCount;
        public int CollectedShardCount => collectedShardCount;
        public int RemainingShardCount => Mathf.Max(0, requiredShardCount - collectedShardCount);
        public bool IsGateUnlocked => collectedShardCount >= requiredShardCount;

        public event Action<int, int> ShardCountChanged;
        public event Action GateUnlocked;
        public event Action StateReset;

        private void Awake()
        {
            if (instance != null && instance != this)
            {
                Destroy(gameObject);
                return;
            }

            instance = this;

            if (persistAcrossScenes)
            {
                DontDestroyOnLoad(gameObject);
            }
        }

        private void OnDestroy()
        {
            if (instance == this)
            {
                instance = null;
            }
        }

        public bool HasCollectedShard(string shardId)
        {
            if (string.IsNullOrEmpty(shardId))
            {
                return false;
            }

            return collectedShardIds.Contains(shardId);
        }

        public bool TryCollectShard(string shardId, int shardValue)
        {
            string safeId = string.IsNullOrWhiteSpace(shardId) ? Guid.NewGuid().ToString("N") : shardId;
            int value = Mathf.Max(1, shardValue);

            if (collectedShardIds.Contains(safeId))
            {
                return false;
            }

            bool wasUnlocked = IsGateUnlocked;

            collectedShardIds.Add(safeId);
            collectedShardCount += value;

            if (logStateChanges)
            {
                Debug.Log($"DreamState: shard collected ({collectedShardCount}/{requiredShardCount}) id={safeId}");
            }

            ShardCountChanged?.Invoke(collectedShardCount, requiredShardCount);

            if (!wasUnlocked && IsGateUnlocked)
            {
                if (logStateChanges)
                {
                    Debug.Log("DreamState: gate unlocked");
                }

                GateUnlocked?.Invoke();
            }

            return true;
        }

        [ContextMenu("Reset Dream State")]
        public void ResetState()
        {
            collectedShardIds.Clear();
            collectedShardCount = 0;
            ShardCountChanged?.Invoke(collectedShardCount, requiredShardCount);
            StateReset?.Invoke();
        }
    }
}
