using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// Gaussian Splat 世界用のハイブリッド地面判定ヘルパー。
    /// 実際に raycast する相手は「生の splat 点」ではなく、
    /// Marble の collider mesh や手置き proxy collider を想定している。
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/Splat Raycast Helper")]
    public sealed class SplatRaycastHelper : MonoBehaviour
    {
        [Header("歩行判定に使うレイヤー")]
        [SerializeField] private LayerMask walkableMask = ~0;

        [Header("SphereCast 設定")]
        [SerializeField, Min(0.05f)] private float probeRadius = 0.35f;
        [SerializeField, Min(0.05f)] private float probeStartOffset = 0.25f;
        [SerializeField, Min(0.1f)] private float defaultProbeDistance = 2.0f;

        [Header("歩行可能面の条件")]
        [SerializeField, Range(1f, 89f)] private float maxWalkableSlope = 55f;
        [SerializeField] private QueryTriggerInteraction triggerInteraction = QueryTriggerInteraction.Ignore;

        [Header("補助設定")]
        [SerializeField, Min(0.01f)] private float groundSnapDistance = 0.5f;
        [SerializeField] private bool drawDebugLines = true;

        private readonly RaycastHit[] hitBuffer = new RaycastHit[16];

        public LayerMask WalkableMask => walkableMask;
        public float ProbeRadius => probeRadius;
        public float ProbeStartOffset => probeStartOffset;
        public float DefaultProbeDistance => defaultProbeDistance;
        public float GroundSnapDistance => groundSnapDistance;

        /// <summary>
        /// 足元の歩行可能面を取るメイン API。
        /// まず SphereCast で広めに拾い、薄い面や段差で抜けた時だけ Raycast にフォールバックする。
        /// </summary>
        public bool TryGetWalkableHit(Vector3 origin, Vector3 up, out RaycastHit bestHit, float maxDistance = -1f)
        {
            float castDistance = maxDistance > 0f ? maxDistance : defaultProbeDistance;
            Vector3 upAxis = GetSafeUp(up);
            Vector3 castOrigin = origin + upAxis * probeStartOffset;

            int sphereHitCount = Physics.SphereCastNonAlloc(
                castOrigin,
                probeRadius,
                -upAxis,
                hitBuffer,
                castDistance + probeStartOffset,
                walkableMask,
                triggerInteraction);

            bool foundHit = TrySelectClosestWalkableHit(sphereHitCount, upAxis, out bestHit);

            if (!foundHit)
            {
                foundHit = TryRaycastWalkable(castOrigin, upAxis, out bestHit, castDistance + probeStartOffset);
            }

            if (drawDebugLines)
            {
                Debug.DrawLine(castOrigin, castOrigin - upAxis * (castDistance + probeStartOffset), foundHit ? Color.green : Color.red);

                if (foundHit)
                {
                    Debug.DrawRay(bestHit.point, bestHit.normal * 0.5f, Color.cyan);
                }
            }

            return foundHit;
        }

        /// <summary>
        /// 正確な接地距離が欲しい時の Raycast 版。
        /// Ground snap や interaction 用はこちらを使う。
        /// </summary>
        public bool TryRaycastWalkable(Vector3 origin, Vector3 up, out RaycastHit bestHit, float maxDistance)
        {
            Vector3 upAxis = GetSafeUp(up);
            int rayHitCount = Physics.RaycastNonAlloc(
                origin,
                -upAxis,
                hitBuffer,
                maxDistance,
                walkableMask,
                triggerInteraction);

            return TrySelectClosestWalkableHit(rayHitCount, upAxis, out bestHit);
        }

        /// <summary>
        /// 任意の点を「歩ける面」に投影したい時に使う。
        /// ワープ先の補正、スポーン位置補正、テレポート先の安全確認に便利。
        /// </summary>
        public bool TryProjectPointToWalkable(
            Vector3 point,
            Vector3 up,
            out Vector3 projectedPoint,
            out Vector3 projectedNormal,
            float lift = 2.0f,
            float maxDistance = 6.0f)
        {
            Vector3 upAxis = GetSafeUp(up);
            Vector3 castOrigin = point + upAxis * lift;

            if (TryRaycastWalkable(castOrigin, upAxis, out RaycastHit hit, maxDistance))
            {
                projectedPoint = hit.point;
                projectedNormal = hit.normal;
                return true;
            }

            projectedPoint = point;
            projectedNormal = upAxis;
            return false;
        }

        /// <summary>
        /// 画面中央の前方表面を拾う。
        /// 今後の interaction / examine 系の基本ヘルパーとして使える。
        /// </summary>
        public bool TryGetForwardSurface(Camera sourceCamera, float maxDistance, out RaycastHit hit)
        {
            if (sourceCamera == null)
            {
                hit = default;
                return false;
            }

            return Physics.Raycast(
                sourceCamera.transform.position,
                sourceCamera.transform.forward,
                out hit,
                maxDistance,
                walkableMask,
                triggerInteraction);
        }

        public bool IsWalkable(Vector3 up, Vector3 surfaceNormal)
        {
            return Vector3.Angle(GetSafeUp(up), surfaceNormal) <= maxWalkableSlope;
        }

        private bool TrySelectClosestWalkableHit(int hitCount, Vector3 up, out RaycastHit bestHit)
        {
            bestHit = default;
            bool found = false;
            float closestDistance = float.PositiveInfinity;

            for (int i = 0; i < hitCount; i++)
            {
                RaycastHit candidate = hitBuffer[i];

                if (candidate.collider == null)
                {
                    continue;
                }

                if (!IsWalkable(up, candidate.normal))
                {
                    continue;
                }

                if (candidate.distance >= closestDistance)
                {
                    continue;
                }

                closestDistance = candidate.distance;
                bestHit = candidate;
                found = true;
            }

            return found;
        }

        private static Vector3 GetSafeUp(Vector3 up)
        {
            if (up.sqrMagnitude < 0.0001f)
            {
                return Vector3.up;
            }

            return up.normalized;
        }
    }
}
