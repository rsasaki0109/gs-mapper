using System.Collections.Generic;
using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// 重いポストプロセスに頼らず、FOV 変化だけで dream 感を少し足す軽量演出。
    /// Gaussian Splat との相性を崩しにくい範囲で抑えている。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Dream View Effects")]
    public sealed class DreamViewEffects : MonoBehaviour
    {
        [Header("参照")]
        [SerializeField] private Camera targetCamera;
        [SerializeField] private DreamWalkerFirstPersonController playerController;

        [Header("FOV")]
        [SerializeField] private bool readBaseFovFromCameraOnStart = true;
        [SerializeField, Min(1f)] private float baseFieldOfView = 60f;
        [SerializeField, Min(0f)] private float lowGravityFovBoost = 3.5f;
        [SerializeField, Min(0f)] private float zoneFovBoost = 8f;
        [SerializeField, Min(0f)] private float pulseFovBoost = 7f;
        [SerializeField, Min(0.1f)] private float fovResponseSpeed = 7f;
        [SerializeField, Min(0.1f)] private float pulseDecaySpeed = 1.8f;

        private readonly List<DreamDistortionZone> activeZones = new List<DreamDistortionZone>();
        private float pulseAmount;

        public Camera TargetCamera => targetCamera;

        private void Reset()
        {
            targetCamera = GetComponentInChildren<Camera>();
            playerController = GetComponent<DreamWalkerFirstPersonController>();
        }

        private void Awake()
        {
            if (targetCamera == null)
            {
                targetCamera = GetComponentInChildren<Camera>();
            }

            if (playerController == null)
            {
                playerController = GetComponent<DreamWalkerFirstPersonController>();
            }
        }

        private void Start()
        {
            if (targetCamera != null && readBaseFovFromCameraOnStart)
            {
                baseFieldOfView = targetCamera.fieldOfView;
            }
        }

        private void Update()
        {
            if (targetCamera == null)
            {
                return;
            }

            pulseAmount = Mathf.MoveTowards(pulseAmount, 0f, pulseDecaySpeed * Time.deltaTime);

            float strongestZoneIntensity = GetStrongestZoneIntensity();
            bool shouldForceLowGravity = HasForcedLowGravityZone();

            if (playerController != null)
            {
                playerController.SetForcedLowGravity(shouldForceLowGravity);
            }

            float targetFov = baseFieldOfView;

            if (playerController != null && playerController.IsLowGravityEnabled)
            {
                targetFov += lowGravityFovBoost;
            }

            targetFov += strongestZoneIntensity * zoneFovBoost;
            targetFov += pulseAmount * pulseFovBoost;

            float lerpFactor = 1f - Mathf.Exp(-fovResponseSpeed * Time.deltaTime);
            targetCamera.fieldOfView = Mathf.Lerp(targetCamera.fieldOfView, targetFov, lerpFactor);
        }

        public void TriggerPulse(float normalizedIntensity = 1f)
        {
            pulseAmount = Mathf.Max(pulseAmount, Mathf.Clamp01(normalizedIntensity));
        }

        public void RegisterZone(DreamDistortionZone zone)
        {
            if (zone == null || activeZones.Contains(zone))
            {
                return;
            }

            activeZones.Add(zone);
        }

        public void UnregisterZone(DreamDistortionZone zone)
        {
            if (zone == null)
            {
                return;
            }

            activeZones.Remove(zone);
        }

        private float GetStrongestZoneIntensity()
        {
            float strongest = 0f;

            for (int i = activeZones.Count - 1; i >= 0; i--)
            {
                DreamDistortionZone zone = activeZones[i];

                if (zone == null || !zone.isActiveAndEnabled)
                {
                    activeZones.RemoveAt(i);
                    continue;
                }

                strongest = Mathf.Max(strongest, zone.Intensity);
            }

            return strongest;
        }

        private bool HasForcedLowGravityZone()
        {
            for (int i = activeZones.Count - 1; i >= 0; i--)
            {
                DreamDistortionZone zone = activeZones[i];

                if (zone == null || !zone.isActiveAndEnabled)
                {
                    activeZones.RemoveAt(i);
                    continue;
                }

                if (zone.ForceLowGravityWhileInside)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
