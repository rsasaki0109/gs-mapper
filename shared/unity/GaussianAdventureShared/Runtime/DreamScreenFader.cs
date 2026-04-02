using System.Collections;
using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// 依存を増やさずに使える簡易フルスクリーンフェーダー。
    /// Scene UI をまだ組んでいない段階でも、門の遷移や目覚め演出を入れやすくする。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream Screen Fader")]
    public sealed class DreamScreenFader : MonoBehaviour
    {
        [Header("起動時フェード")]
        [SerializeField] private bool fadeInOnStart = true;
        [SerializeField, Min(0f)] private float startupFadeInDuration = 0.75f;

        [Header("表示")]
        [SerializeField] private Color fadeColor = Color.black;

        private float currentAlpha;
        private int fadeTicket;

        public bool IsFading { get; private set; }
        public float CurrentAlpha => currentAlpha;

        private void Awake()
        {
            currentAlpha = fadeInOnStart ? 1f : 0f;
        }

        private IEnumerator Start()
        {
            if (!fadeInOnStart)
            {
                yield break;
            }

            yield return FadeToAlpha(0f, startupFadeInDuration);
        }

        private void OnGUI()
        {
            if (currentAlpha <= 0.001f)
            {
                return;
            }

            Color previousColor = GUI.color;
            int previousDepth = GUI.depth;

            GUI.depth = -1000;
            GUI.color = new Color(fadeColor.r, fadeColor.g, fadeColor.b, currentAlpha);
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height), Texture2D.whiteTexture);

            GUI.color = previousColor;
            GUI.depth = previousDepth;
        }

        public void SetAlphaImmediate(float alpha)
        {
            fadeTicket++;
            currentAlpha = Mathf.Clamp01(alpha);
            IsFading = false;
        }

        public IEnumerator FadeToAlpha(float targetAlpha, float duration)
        {
            int ticket = ++fadeTicket;
            targetAlpha = Mathf.Clamp01(targetAlpha);

            if (duration <= 0f)
            {
                currentAlpha = targetAlpha;
                IsFading = false;
                yield break;
            }

            float startAlpha = currentAlpha;
            float elapsed = 0f;
            IsFading = true;

            while (elapsed < duration)
            {
                if (ticket != fadeTicket)
                {
                    yield break;
                }

                elapsed += Time.unscaledDeltaTime;
                float normalized = Mathf.Clamp01(elapsed / duration);
                currentAlpha = Mathf.Lerp(startAlpha, targetAlpha, normalized);
                yield return null;
            }

            if (ticket == fadeTicket)
            {
                currentAlpha = targetAlpha;
                IsFading = false;
            }
        }
    }
}
