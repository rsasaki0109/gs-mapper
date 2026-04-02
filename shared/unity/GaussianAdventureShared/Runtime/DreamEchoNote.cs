using UnityEngine;
using UnityEngine.Events;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// DreamWalker 用の簡易テキスト断片。
    /// まずは HUD に直接表示して、世界観の断片を置けるようにする。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream Echo Note")]
    public sealed class DreamEchoNote : MonoBehaviour, IInteractable
    {
        [Header("内容")]
        [SerializeField] private string noteTitle = "Echo Note";
        [SerializeField, TextArea(4, 10)] private string noteBody = "夢の断片をここに書く。";
        [SerializeField] private Color accentColor = new Color(0.95f, 0.92f, 0.62f, 1f);

        [Header("操作")]
        [SerializeField] private string unreadPrompt = "[E] 残響に耳を澄ます";
        [SerializeField] private string rereadPrompt = "[E] 残響を読み返す";
        [SerializeField] private string closePrompt = "[E / Esc] 閉じる";

        [Header("状態")]
        [SerializeField] private bool markAsReadOnOpen = true;
        [SerializeField] private string firstReadToastMessage = "記憶の残響が聞こえた";
        [SerializeField] private UnityEvent onFirstRead;

        private bool hasBeenRead;

        public bool CanInteract(SplatInteractProbe probe)
        {
            return true;
        }

        public string GetInteractionPrompt(SplatInteractProbe probe)
        {
            return hasBeenRead ? rereadPrompt : unreadPrompt;
        }

        public void Interact(SplatInteractProbe probe)
        {
            DreamWalkerHUD hud = DreamWalkerHUD.Instance != null
                ? DreamWalkerHUD.Instance
                : FindFirstObjectByType<DreamWalkerHUD>();

            if (hud == null)
            {
                Debug.LogWarning($"DreamEchoNote '{name}' could not find DreamWalkerHUD.");
                return;
            }

            hud.ShowEchoNote(noteTitle, noteBody, accentColor, closePrompt);

            DreamViewEffects viewEffects = probe != null ? probe.GetComponent<DreamViewEffects>() : null;
            if (viewEffects != null)
            {
                viewEffects.TriggerPulse(0.45f);
            }

            if (hasBeenRead)
            {
                return;
            }

            if (markAsReadOnOpen)
            {
                hasBeenRead = true;
            }

            if (!string.IsNullOrWhiteSpace(firstReadToastMessage))
            {
                hud.ShowToast(firstReadToastMessage, 2.6f);
            }

            onFirstRead?.Invoke();
        }
    }
}
