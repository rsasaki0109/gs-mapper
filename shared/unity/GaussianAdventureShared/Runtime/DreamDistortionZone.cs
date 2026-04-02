using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// プレイヤーが入ると dream view を歪ませる trigger zone。
    /// collider mesh の上に置く軽い演出として使う。
    /// </summary>
    [RequireComponent(typeof(Collider))]
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream Distortion Zone")]
    public sealed class DreamDistortionZone : MonoBehaviour
    {
        [SerializeField, Range(0f, 1f)] private float intensity = 0.85f;
        [SerializeField] private bool forceLowGravityWhileInside = true;
        [SerializeField] private bool pulseOnEnter = true;
        [SerializeField, Range(0f, 1f)] private float enterPulseIntensity = 0.65f;

        public float Intensity => intensity;
        public bool ForceLowGravityWhileInside => forceLowGravityWhileInside;

        private void Reset()
        {
            Collider zoneCollider = GetComponent<Collider>();
            zoneCollider.isTrigger = true;
        }

        private void OnTriggerEnter(Collider other)
        {
            DreamViewEffects effects = other.GetComponentInParent<DreamViewEffects>();
            if (effects == null)
            {
                return;
            }

            effects.RegisterZone(this);

            if (pulseOnEnter)
            {
                effects.TriggerPulse(enterPulseIntensity);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            DreamViewEffects effects = other.GetComponentInParent<DreamViewEffects>();
            if (effects == null)
            {
                return;
            }

            effects.UnregisterZone(this);
        }
    }
}
