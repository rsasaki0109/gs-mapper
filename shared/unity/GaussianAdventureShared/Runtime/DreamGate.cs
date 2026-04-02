using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.SceneManagement;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// 欠片を一定数集めると開く出口。
    /// まずは簡単な visual 切り替えとシーン遷移のみを担当する。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream Gate")]
    public sealed class DreamGate : MonoBehaviour, IInteractable
    {
        [Header("参照")]
        [SerializeField] private DreamStateManager stateManager;
        [SerializeField] private Collider blockingCollider;
        [SerializeField] private Collider entryTrigger;
        [SerializeField] private GameObject closedStateRoot;
        [SerializeField] private GameObject openStateRoot;

        [Header("操作")]
        [SerializeField] private string lockedPrompt = "門は閉ざされている";
        [SerializeField] private string openPrompt = "[E] 夢の門へ入る";
        [SerializeField] private bool allowTriggerEntryWhenOpen = true;

        [Header("遷移")]
        [SerializeField] private string targetSceneName = string.Empty;
        [SerializeField] private Transform localDestination;
        [SerializeField] private bool alignPlayerRotationToDestination = true;
        [SerializeField] private bool projectDestinationToWalkable = true;
        [SerializeField] private bool resetStateAfterLocalTransfer = false;
        [SerializeField, Min(0.1f)] private float localDestinationProbeLift = 2f;
        [SerializeField, Min(0.5f)] private float localDestinationProbeDistance = 8f;

        [Header("演出")]
        [SerializeField] private DreamScreenFader screenFader;
        [SerializeField] private bool useTransitionFade = true;
        [SerializeField, Min(0f)] private float fadeOutDuration = 0.45f;
        [SerializeField, Min(0f)] private float fadeHoldDuration = 0.08f;
        [SerializeField, Min(0f)] private float fadeInDuration = 0.55f;
        [SerializeField] private string gateOpenedToastMessage = "夢の門が開いた";

        [SerializeField] private UnityEvent onGateOpened;
        [SerializeField] private UnityEvent onGateEntered;

        private bool isOpen;
        private bool isEntering;

        public bool IsOpen => isOpen;

        private void Reset()
        {
            entryTrigger = GetComponent<Collider>();
        }

        private void Awake()
        {
            if (stateManager == null)
            {
                stateManager = ResolveStateManager();
            }

            if (screenFader == null)
            {
                screenFader = ResolveScreenFader(null);
            }

            RefreshState();
        }

        private void OnEnable()
        {
            if (stateManager == null)
            {
                stateManager = ResolveStateManager();
            }

            if (stateManager != null)
            {
                stateManager.GateUnlocked += HandleGateUnlocked;
                stateManager.StateReset += HandleStateReset;
            }

            if (screenFader == null)
            {
                screenFader = ResolveScreenFader(null);
            }

            RefreshState();
        }

        private void OnDisable()
        {
            if (stateManager != null)
            {
                stateManager.GateUnlocked -= HandleGateUnlocked;
                stateManager.StateReset -= HandleStateReset;
            }
        }

        public bool CanInteract(SplatInteractProbe probe)
        {
            return isOpen;
        }

        public string GetInteractionPrompt(SplatInteractProbe probe)
        {
            if (isOpen)
            {
                return openPrompt;
            }

            if (stateManager == null)
            {
                return lockedPrompt;
            }

            int remaining = stateManager.RemainingShardCount;
            return remaining > 0 ? $"{lockedPrompt}  あと {remaining} 個" : lockedPrompt;
        }

        public void Interact(SplatInteractProbe probe)
        {
            if (!isOpen)
            {
                return;
            }

            DreamWalkerFirstPersonController playerController = probe != null
                ? probe.GetComponent<DreamWalkerFirstPersonController>()
                : null;

            EnterGate(playerController);
        }

        private void OnTriggerEnter(Collider other)
        {
            if (!allowTriggerEntryWhenOpen || !isOpen)
            {
                return;
            }

            if (!IsPlayerCollider(other))
            {
                return;
            }

            EnterGate(other.GetComponentInParent<DreamWalkerFirstPersonController>());
        }

        private void HandleGateUnlocked()
        {
            OpenGate();
        }

        private void HandleStateReset()
        {
            RefreshState();
        }

        private void RefreshState()
        {
            bool shouldBeOpen = stateManager != null && stateManager.IsGateUnlocked;

            if (shouldBeOpen)
            {
                OpenGate();
                return;
            }

            isOpen = false;

            if (blockingCollider != null)
            {
                blockingCollider.enabled = true;
            }

            if (entryTrigger != null)
            {
                entryTrigger.enabled = true;
                entryTrigger.isTrigger = true;
            }

            if (closedStateRoot != null)
            {
                closedStateRoot.SetActive(true);
            }

            if (openStateRoot != null)
            {
                openStateRoot.SetActive(false);
            }
        }

        private void OpenGate()
        {
            if (isOpen)
            {
                return;
            }

            isOpen = true;

            if (blockingCollider != null)
            {
                blockingCollider.enabled = false;
            }

            if (entryTrigger != null)
            {
                entryTrigger.enabled = true;
                entryTrigger.isTrigger = true;
            }

            if (closedStateRoot != null)
            {
                closedStateRoot.SetActive(false);
            }

            if (openStateRoot != null)
            {
                openStateRoot.SetActive(true);
            }

            DreamWalkerHUD hud = DreamWalkerHUD.Instance != null
                ? DreamWalkerHUD.Instance
                : FindFirstObjectByType<DreamWalkerHUD>();

            if (hud != null && !string.IsNullOrWhiteSpace(gateOpenedToastMessage))
            {
                hud.ShowToast(gateOpenedToastMessage, 3f);
            }

            onGateOpened?.Invoke();
        }

        private void EnterGate(DreamWalkerFirstPersonController playerController)
        {
            if (isEntering)
            {
                return;
            }

            isEntering = true;
            onGateEntered?.Invoke();

            if (screenFader == null)
            {
                screenFader = ResolveScreenFader(playerController);
            }

            if (useTransitionFade && screenFader != null)
            {
                StartCoroutine(EnterGateRoutine(playerController));
                return;
            }

            CompleteEntryWithoutFade(playerController);
        }

        private IEnumerator EnterGateRoutine(DreamWalkerFirstPersonController playerController)
        {
            yield return screenFader.FadeToAlpha(1f, fadeOutDuration);

            if (fadeHoldDuration > 0f)
            {
                yield return WaitForSecondsUnscaled(fadeHoldDuration);
            }

            GateEntryResult result = ExecuteGateEntry(playerController);

            if (result == GateEntryResult.SceneLoad)
            {
                yield break;
            }

            yield return screenFader.FadeToAlpha(0f, fadeInDuration);
            isEntering = false;
        }

        private void CompleteEntryWithoutFade(DreamWalkerFirstPersonController playerController)
        {
            GateEntryResult result = ExecuteGateEntry(playerController);

            if (result != GateEntryResult.SceneLoad)
            {
                isEntering = false;
            }
        }

        private GateEntryResult ExecuteGateEntry(DreamWalkerFirstPersonController playerController)
        {
            if (!string.IsNullOrWhiteSpace(targetSceneName))
            {
                try
                {
                    SceneManager.LoadScene(targetSceneName);
                    return GateEntryResult.SceneLoad;
                }
                catch (System.Exception exception)
                {
                    Debug.LogException(exception, this);
                }
            }

            if (TryTransferLocally(playerController))
            {
                if (resetStateAfterLocalTransfer && stateManager != null)
                {
                    stateManager.ResetState();
                }

                return GateEntryResult.LocalTransfer;
            }

            Debug.Log("DreamGate entered. targetSceneName が未設定なのでシーン遷移は行いません。");
            return GateEntryResult.None;
        }

        private bool TryTransferLocally(DreamWalkerFirstPersonController playerController)
        {
            if (localDestination == null || playerController == null)
            {
                return false;
            }

            Vector3 destinationPosition = localDestination.position;

            if (projectDestinationToWalkable && playerController.GroundProbe != null)
            {
                bool projected = playerController.GroundProbe.TryProjectPointToWalkable(
                    localDestination.position,
                    playerController.transform.up,
                    out Vector3 projectedPoint,
                    out _,
                    localDestinationProbeLift,
                    localDestinationProbeDistance);

                if (projected)
                {
                    destinationPosition = projectedPoint;
                }
            }

            Quaternion destinationRotation = alignPlayerRotationToDestination
                ? localDestination.rotation
                : playerController.transform.rotation;

            playerController.TeleportToPose(destinationPosition, destinationRotation);
            return true;
        }

        private static IEnumerator WaitForSecondsUnscaled(float duration)
        {
            float endTime = Time.unscaledTime + Mathf.Max(0f, duration);
            while (Time.unscaledTime < endTime)
            {
                yield return null;
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

        private DreamScreenFader ResolveScreenFader(DreamWalkerFirstPersonController playerController)
        {
            if (playerController != null)
            {
                DreamScreenFader playerFader = playerController.GetComponent<DreamScreenFader>();
                if (playerFader != null)
                {
                    return playerFader;
                }
            }

            return FindFirstObjectByType<DreamScreenFader>();
        }

        private static bool IsPlayerCollider(Collider other)
        {
            if (other == null)
            {
                return false;
            }

            if (other.GetComponentInParent<DreamWalkerFirstPersonController>() != null)
            {
                return true;
            }

            return other.GetComponentInParent<CharacterController>() != null;
        }

        private enum GateEntryResult
        {
            None = 0,
            LocalTransfer = 1,
            SceneLoad = 2
        }
    }
}
