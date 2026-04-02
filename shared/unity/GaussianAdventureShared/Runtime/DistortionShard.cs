using UnityEngine;
using UnityEngine.Events;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// 収集対象の欠片。
    /// まずは軽い浮遊アニメーションと収集イベントだけを持つ。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Distortion Shard")]
    public sealed class DistortionShard : MonoBehaviour, IInteractable
    {
        [Header("識別情報")]
        [SerializeField] private string shardId = string.Empty;
        [SerializeField] private string shardDisplayName = "歪みの欠片";
        [SerializeField, Min(1)] private int shardValue = 1;

        [Header("操作")]
        [SerializeField] private string interactionPrompt = "[E] 歪みの欠片に触れる";
        [SerializeField] private string collectedToastFormat = "{0} を回収した";

        [Header("演出")]
        [SerializeField] private bool animateIdle = true;
        [SerializeField] private Vector3 idleRotationSpeed = new Vector3(0f, 55f, 0f);
        [SerializeField, Min(0f)] private float bobAmplitude = 0.12f;
        [SerializeField, Min(0f)] private float bobSpeed = 1.8f;
        [SerializeField] private bool disableObjectOnCollect = true;
        [SerializeField] private UnityEvent onCollected;

        private Vector3 initialLocalPosition;
        private DreamStateManager stateManager;
        private bool isCollected;

        private void Awake()
        {
            initialLocalPosition = transform.localPosition;
            stateManager = ResolveStateManager();
        }

        private void Start()
        {
            RefreshCollectedState();
        }

        private void Update()
        {
            if (!animateIdle || isCollected)
            {
                return;
            }

            transform.Rotate(idleRotationSpeed * Time.deltaTime, Space.Self);

            float offsetY = Mathf.Sin(Time.time * bobSpeed) * bobAmplitude;
            Vector3 animatedPosition = initialLocalPosition;
            animatedPosition.y += offsetY;
            transform.localPosition = animatedPosition;
        }

        public bool CanInteract(SplatInteractProbe probe)
        {
            return !isCollected;
        }

        public string GetInteractionPrompt(SplatInteractProbe probe)
        {
            return isCollected ? string.Empty : interactionPrompt;
        }

        public void Interact(SplatInteractProbe probe)
        {
            if (isCollected)
            {
                return;
            }

            if (stateManager == null)
            {
                stateManager = ResolveStateManager();
            }

            if (stateManager == null)
            {
                Debug.LogWarning($"DistortionShard '{name}' could not find DreamStateManager.");
                return;
            }

            string resolvedShardId = ResolveShardId();
            if (!stateManager.TryCollectShard(resolvedShardId, shardValue))
            {
                ApplyCollectedState();
                return;
            }

            DreamWalkerHUD hud = DreamWalkerHUD.Instance != null
                ? DreamWalkerHUD.Instance
                : FindFirstObjectByType<DreamWalkerHUD>();

            if (hud != null && !string.IsNullOrWhiteSpace(collectedToastFormat))
            {
                hud.ShowToast(string.Format(collectedToastFormat, shardDisplayName), 2.2f);
            }

            DreamViewEffects viewEffects = probe != null ? probe.GetComponent<DreamViewEffects>() : null;
            if (viewEffects != null)
            {
                viewEffects.TriggerPulse(1f);
            }

            onCollected?.Invoke();
            ApplyCollectedState();
        }

        private void RefreshCollectedState()
        {
            if (stateManager == null)
            {
                stateManager = ResolveStateManager();
            }

            if (stateManager == null)
            {
                return;
            }

            if (!stateManager.HasCollectedShard(ResolveShardId()))
            {
                return;
            }

            ApplyCollectedState();
        }

        private void ApplyCollectedState()
        {
            if (isCollected)
            {
                return;
            }

            isCollected = true;

            Collider[] colliders = GetComponentsInChildren<Collider>(true);
            for (int i = 0; i < colliders.Length; i++)
            {
                colliders[i].enabled = false;
            }

            Renderer[] renderers = GetComponentsInChildren<Renderer>(true);
            for (int i = 0; i < renderers.Length; i++)
            {
                renderers[i].enabled = false;
            }

            if (disableObjectOnCollect)
            {
                gameObject.SetActive(false);
            }
        }

        private DreamStateManager ResolveStateManager()
        {
            if (DreamStateManager.Instance != null)
            {
                return DreamStateManager.Instance;
            }

            return FindFirstObjectByType<DreamStateManager>();
        }

        private string ResolveShardId()
        {
            if (!string.IsNullOrWhiteSpace(shardId))
            {
                return shardId;
            }

            shardId = $"{gameObject.scene.name}:{BuildHierarchyPath(transform)}:{shardDisplayName}";
            return shardId;
        }

        private static string BuildHierarchyPath(Transform current)
        {
            string path = current.name;

            while (current.parent != null)
            {
                current = current.parent;
                path = current.name + "/" + path;
            }

            return path;
        }
    }
}
