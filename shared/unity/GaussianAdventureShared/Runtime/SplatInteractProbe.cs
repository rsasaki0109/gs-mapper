using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// 画面中央から前方へ raycast し、interaction 可能な対象を拾う。
    /// splat そのものを触るのではなく、コライダーを持つギミックや proxy を相手にする。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Splat Interact Probe")]
    public sealed class SplatInteractProbe : MonoBehaviour
    {
        [Header("参照")]
        [SerializeField] private Camera sourceCamera;

        [Header("判定")]
        [SerializeField] private LayerMask interactMask = ~0;
        [SerializeField, Min(0.1f)] private float maxDistance = 4.0f;
        [SerializeField] private QueryTriggerInteraction triggerInteraction = QueryTriggerInteraction.Collide;

        [Header("入力")]
        [SerializeField] private KeyCode interactKey = KeyCode.E;

        [Header("デバッグ")]
        [SerializeField] private bool drawDebugRay = true;

        private IInteractable currentInteractable;
        private RaycastHit currentHit;
        private string currentPrompt = string.Empty;

        public Camera SourceCamera => sourceCamera;
        public RaycastHit CurrentHit => currentHit;
        public bool HasTarget => currentInteractable != null;
        public bool CanInteractCurrent => currentInteractable != null && currentInteractable.CanInteract(this);
        public string CurrentInteractionPrompt => currentPrompt;

        private void Reset()
        {
            sourceCamera = GetComponentInChildren<Camera>();
        }

        private void Awake()
        {
            if (sourceCamera == null)
            {
                sourceCamera = GetComponentInChildren<Camera>();
            }
        }

        private void Update()
        {
            if (TryHandleHudModal())
            {
                return;
            }

            ResolveTarget();

            if (currentInteractable == null)
            {
                return;
            }

            if (Input.GetKeyDown(interactKey) && currentInteractable.CanInteract(this))
            {
                currentInteractable.Interact(this);

                if (!TryHandleHudModal())
                {
                    ResolveTarget();
                }
            }
        }

        public bool TryGetCurrentInteractable(out IInteractable interactable, out RaycastHit hit)
        {
            interactable = currentInteractable;
            hit = currentHit;
            return interactable != null;
        }

        private void ResolveTarget()
        {
            currentInteractable = null;
            currentHit = default;
            currentPrompt = string.Empty;

            if (sourceCamera == null)
            {
                return;
            }

            Vector3 origin = sourceCamera.transform.position;
            Vector3 direction = sourceCamera.transform.forward;

            bool hasHit = Physics.Raycast(
                origin,
                direction,
                out RaycastHit hit,
                maxDistance,
                interactMask,
                triggerInteraction);

            if (drawDebugRay)
            {
                Color rayColor = hasHit ? Color.yellow : Color.gray;
                Debug.DrawLine(origin, origin + direction * maxDistance, rayColor);
            }

            if (!hasHit)
            {
                return;
            }

            IInteractable interactable = FindInteractable(hit.collider);
            if (interactable == null)
            {
                return;
            }

            currentInteractable = interactable;
            currentHit = hit;
            currentPrompt = interactable.GetInteractionPrompt(this) ?? string.Empty;

            if (drawDebugRay)
            {
                Debug.DrawRay(hit.point, hit.normal * 0.35f, CanInteractCurrent ? Color.green : Color.red);
            }
        }

        private static IInteractable FindInteractable(Collider targetCollider)
        {
            if (targetCollider == null)
            {
                return null;
            }

            MonoBehaviour[] behaviours = targetCollider.GetComponentsInParent<MonoBehaviour>(true);
            for (int i = 0; i < behaviours.Length; i++)
            {
                if (behaviours[i] is IInteractable interactable)
                {
                    return interactable;
                }
            }

            return null;
        }

        private bool TryHandleHudModal()
        {
            DreamWalkerHUD hud = DreamWalkerHUD.Instance;
            if (hud == null)
            {
                return false;
            }

            if (!hud.TryConsumeModalCloseInput(interactKey))
            {
                return false;
            }

            currentInteractable = null;
            currentHit = default;
            currentPrompt = hud.CurrentModalPrompt;
            return true;
        }
    }
}
