using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// TMPro 依存を持たない簡易 HUD。
    /// まずはプロトタイプ検証向けに OnGUI で最小情報を表示する。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/DreamWalker HUD")]
    public sealed class DreamWalkerHUD : MonoBehaviour
    {
        private static readonly Color DefaultNoteAccent = new Color(0.62f, 0.9f, 1f, 1f);

        [Header("参照")]
        [SerializeField] private DreamStateManager stateManager;
        [SerializeField] private SplatInteractProbe interactProbe;
        [SerializeField] private DreamWalkerFirstPersonController playerController;

        [Header("表示")]
        [SerializeField] private string title = "DreamWalker";
        [SerializeField] private bool showControls = true;
        [SerializeField] private bool showCrosshair = true;

        private static DreamWalkerHUD instance;

        private GUIStyle titleStyle;
        private GUIStyle bodyStyle;
        private GUIStyle promptStyle;
        private GUIStyle crosshairStyle;
        private GUIStyle toastStyle;
        private GUIStyle noteTitleStyle;
        private GUIStyle noteBodyStyle;

        private string toastMessage = string.Empty;
        private float toastHideTime;
        private bool isNoteOpen;
        private string activeNoteTitle = string.Empty;
        private string activeNoteBody = string.Empty;
        private string activeNoteClosePrompt = "[E / Esc] 閉じる";
        private Color activeNoteAccentColor = DefaultNoteAccent;

        public static DreamWalkerHUD Instance => instance;
        public bool HasModalViewOpen => isNoteOpen;
        public string CurrentModalPrompt => isNoteOpen ? activeNoteClosePrompt : string.Empty;

        private void Awake()
        {
            instance = this;
            ResolveReferences();
        }

        private void OnEnable()
        {
            instance = this;
        }

        private void OnDisable()
        {
            if (instance == this)
            {
                instance = null;
            }
        }

        private void OnGUI()
        {
            ResolveReferences();
            RefreshToastState();
            EnsureStyles();

            DrawStatusPanel();

            if (showControls)
            {
                DrawControlsPanel();
            }

            if (showCrosshair && !HasModalViewOpen)
            {
                DrawCrosshair();
            }

            DrawInteractionPrompt();
            DrawActiveNote();
            DrawToast();
        }

        private void ResolveReferences()
        {
            if (stateManager == null)
            {
                stateManager = DreamStateManager.Instance != null
                    ? DreamStateManager.Instance
                    : FindFirstObjectByType<DreamStateManager>();
            }

            if (playerController == null)
            {
                playerController = FindFirstObjectByType<DreamWalkerFirstPersonController>();
            }

            if (interactProbe == null)
            {
                interactProbe = FindFirstObjectByType<SplatInteractProbe>();
            }
        }

        private void DrawStatusPanel()
        {
            Rect panelRect = new Rect(16f, 16f, 340f, 118f);
            GUI.Box(panelRect, string.Empty);

            GUILayout.BeginArea(new Rect(panelRect.x + 14f, panelRect.y + 10f, panelRect.width - 28f, panelRect.height - 20f));
            GUILayout.Label(title, titleStyle);

            if (stateManager != null)
            {
                GUILayout.Label(
                    $"歪みの欠片: {stateManager.CollectedShardCount} / {stateManager.RequiredShardCount}",
                    bodyStyle);

                GUILayout.Label(
                    stateManager.IsGateUnlocked ? "出口: 開放済み" : $"出口: あと {stateManager.RemainingShardCount} 個必要",
                    bodyStyle);
            }
            else
            {
                GUILayout.Label("DreamStateManager 未接続", bodyStyle);
            }

            if (playerController != null)
            {
                GUILayout.Label(
                    playerController.IsLowGravityEnabled ? "重力モード: 低重力浮遊" : "重力モード: 通常",
                    bodyStyle);
            }

            GUILayout.EndArea();
        }

        private void DrawControlsPanel()
        {
            Rect panelRect = new Rect(16f, Screen.height - 112f, 360f, 96f);
            GUI.Box(panelRect, string.Empty);

            GUILayout.BeginArea(new Rect(panelRect.x + 14f, panelRect.y + 10f, panelRect.width - 28f, panelRect.height - 20f));
            GUILayout.Label("WASD: 移動   Shift: 走る   Space: ジャンプ", bodyStyle);
            GUILayout.Label("F: 低重力浮遊   Ctrl/C: 下降   E: 調べる", bodyStyle);
            GUILayout.Label("Esc: カーソル解除 / ノートを閉じる", bodyStyle);
            GUILayout.EndArea();
        }

        private void DrawCrosshair()
        {
            Rect crosshairRect = new Rect((Screen.width * 0.5f) - 10f, (Screen.height * 0.5f) - 14f, 20f, 28f);
            GUI.Label(crosshairRect, "+", crosshairStyle);
        }

        private void DrawInteractionPrompt()
        {
            string prompt = HasModalViewOpen
                ? CurrentModalPrompt
                : interactProbe != null
                    ? interactProbe.CurrentInteractionPrompt
                    : string.Empty;

            if (string.IsNullOrWhiteSpace(prompt))
            {
                return;
            }

            Rect promptRect = new Rect((Screen.width * 0.5f) - 220f, Screen.height - 84f, 440f, 40f);
            GUI.Box(promptRect, string.Empty);
            GUI.Label(promptRect, prompt, promptStyle);
        }

        private void DrawToast()
        {
            if (string.IsNullOrWhiteSpace(toastMessage))
            {
                return;
            }

            Rect toastRect = new Rect((Screen.width * 0.5f) - 210f, 20f, 420f, 38f);
            GUI.Box(toastRect, string.Empty);
            GUI.Label(toastRect, toastMessage, toastStyle);
        }

        private void DrawActiveNote()
        {
            if (!HasModalViewOpen)
            {
                return;
            }

            Rect panelRect = new Rect(
                Mathf.Max(64f, (Screen.width * 0.5f) - 260f),
                Mathf.Max(72f, (Screen.height * 0.5f) - 170f),
                Mathf.Min(520f, Screen.width - 128f),
                Mathf.Min(340f, Screen.height - 144f));

            Color previousColor = GUI.color;

            GUI.color = new Color(0f, 0f, 0f, 0.82f);
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture);

            GUI.color = new Color(activeNoteAccentColor.r, activeNoteAccentColor.g, activeNoteAccentColor.b, 0.95f);
            GUI.DrawTexture(new Rect(panelRect.x, panelRect.y, panelRect.width, 6f), Texture2D.whiteTexture);

            GUI.color = previousColor;
            GUI.Box(panelRect, string.Empty);

            GUILayout.BeginArea(new Rect(panelRect.x + 20f, panelRect.y + 18f, panelRect.width - 40f, panelRect.height - 36f));
            GUILayout.Label(activeNoteTitle, noteTitleStyle);
            GUILayout.Space(10f);
            GUILayout.Label(activeNoteBody, noteBodyStyle);
            GUILayout.FlexibleSpace();
            GUILayout.Label(activeNoteClosePrompt, bodyStyle);
            GUILayout.EndArea();
        }

        private void EnsureStyles()
        {
            if (titleStyle != null)
            {
                return;
            }

            titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 22,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.white }
            };

            bodyStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 15,
                normal = { textColor = new Color(0.92f, 0.95f, 1f, 1f) }
            };

            promptStyle = new GUIStyle(GUI.skin.label)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 18,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.white }
            };

            crosshairStyle = new GUIStyle(GUI.skin.label)
            {
                alignment = TextAnchor.MiddleCenter,
                fontSize = 26,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.white }
            };

            toastStyle = new GUIStyle(promptStyle)
            {
                fontSize = 16,
                wordWrap = true
            };

            noteTitleStyle = new GUIStyle(titleStyle)
            {
                fontSize = 24,
                wordWrap = true
            };

            noteBodyStyle = new GUIStyle(bodyStyle)
            {
                fontSize = 16,
                wordWrap = true,
                richText = false
            };
        }

        private void RefreshToastState()
        {
            if (string.IsNullOrWhiteSpace(toastMessage))
            {
                return;
            }

            if (Time.unscaledTime <= toastHideTime)
            {
                return;
            }

            toastMessage = string.Empty;
        }

        public void ShowToast(string message, float durationSeconds = 2.5f)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return;
            }

            toastMessage = message;
            toastHideTime = Time.unscaledTime + Mathf.Max(0.1f, durationSeconds);
        }

        public void ShowEchoNote(string noteTitle, string noteBody, Color accentColor, string closePrompt = "[E / Esc] 閉じる")
        {
            activeNoteTitle = string.IsNullOrWhiteSpace(noteTitle) ? "Echo Note" : noteTitle;
            activeNoteBody = string.IsNullOrWhiteSpace(noteBody) ? "......" : noteBody;
            activeNoteAccentColor = accentColor.a > 0f ? accentColor : DefaultNoteAccent;
            activeNoteClosePrompt = string.IsNullOrWhiteSpace(closePrompt) ? "[E / Esc] 閉じる" : closePrompt;
            isNoteOpen = true;
        }

        public void HideModal()
        {
            isNoteOpen = false;
        }

        public bool TryConsumeModalCloseInput(KeyCode interactKey)
        {
            if (!HasModalViewOpen)
            {
                return false;
            }

            if (Input.GetKeyDown(interactKey) || Input.GetKeyDown(KeyCode.Escape))
            {
                HideModal();
            }

            return true;
        }
    }
}
