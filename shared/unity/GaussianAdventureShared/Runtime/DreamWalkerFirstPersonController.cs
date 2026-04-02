using UnityEngine;

namespace NerfGsPlayground.GaussianAdventureShared
{
    /// <summary>
    /// DreamWalker 向けの軽量 FPS コントローラ。
    /// Gaussian Splat の見た目と collider mesh の物理を分離する前提で作っている。
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    [DisallowMultipleComponent]
    [AddComponentMenu("DreamWalker/DreamWalker First Person Controller")]
    public sealed class DreamWalkerFirstPersonController : MonoBehaviour
    {
        [Header("参照")]
        [SerializeField] private CharacterController characterController;
        [SerializeField] private Camera playerCamera;
        [SerializeField] private SplatRaycastHelper splatRaycastHelper;

        [Header("視点")]
        [SerializeField, Min(0.01f)] private float mouseSensitivity = 2.2f;
        [SerializeField, Range(30f, 89f)] private float lookPitchLimit = 85f;
        [SerializeField] private bool invertY = false;
        [SerializeField] private bool lockCursorOnStart = true;

        [Header("移動")]
        [SerializeField, Min(0.1f)] private float walkSpeed = 4.5f;
        [SerializeField, Min(0.1f)] private float sprintSpeed = 6.75f;
        [SerializeField, Min(0.1f)] private float groundAcceleration = 18f;
        [SerializeField, Min(0.1f)] private float airAcceleration = 6f;
        [SerializeField, Min(0.1f)] private float maxAirSpeed = 5.5f;
        [SerializeField, Min(0.1f)] private float jumpHeight = 1.35f;
        [SerializeField, Min(0.1f)] private float gravity = 22f;
        [SerializeField, Min(0.1f)] private float groundedStickForce = 3f;
        [SerializeField, Min(0f)] private float coyoteTime = 0.12f;

        [Header("Splat 接地補助")]
        [SerializeField, Min(0.01f)] private float feetProbeLift = 0.2f;
        [SerializeField, Min(0.01f)] private float manualGroundSnapDistance = 0.5f;

        [Header("低重力浮遊")]
        [SerializeField] private bool allowLowGravityFloat = true;
        [SerializeField] private KeyCode toggleFloatKey = KeyCode.F;
        [SerializeField, Min(0.1f)] private float floatGravity = 4f;
        [SerializeField, Min(0.1f)] private float floatMoveMultiplier = 1.15f;
        [SerializeField, Min(0.1f)] private float floatAscendSpeed = 2.6f;
        [SerializeField, Min(0.1f)] private float floatDescendSpeed = 2.2f;
        [SerializeField, Min(0.1f)] private float floatVerticalResponse = 8f;

        private Vector3 planarVelocity;
        private float verticalVelocity;
        private float lookPitch;
        private float lastGroundedTime = float.NegativeInfinity;
        private bool lowGravityEnabled;
        private bool forcedLowGravityEnabled;
        private RaycastHit currentGroundHit;

        public bool IsGrounded { get; private set; }
        public bool IsLowGravityEnabled => lowGravityEnabled || forcedLowGravityEnabled;
        public Camera PlayerCamera => playerCamera;
        public SplatRaycastHelper GroundProbe => splatRaycastHelper;

        private void Reset()
        {
            characterController = GetComponent<CharacterController>();
            playerCamera = GetComponentInChildren<Camera>();
            splatRaycastHelper = GetComponent<SplatRaycastHelper>();
        }

        private void Awake()
        {
            if (characterController == null)
            {
                characterController = GetComponent<CharacterController>();
            }

            if (playerCamera == null)
            {
                playerCamera = GetComponentInChildren<Camera>();
            }

            if (splatRaycastHelper == null)
            {
                splatRaycastHelper = GetComponent<SplatRaycastHelper>();
            }
        }

        private void Start()
        {
            if (lockCursorOnStart)
            {
                LockCursor(true);
            }
        }

        private void Update()
        {
            UpdateCursorState();
            HandleLook();
            HandleModeToggle();
            RefreshGroundState();
            HandleMovement(Time.deltaTime);
        }

        private void HandleLook()
        {
            float mouseX = Input.GetAxisRaw("Mouse X") * mouseSensitivity;
            float mouseY = Input.GetAxisRaw("Mouse Y") * mouseSensitivity;
            float yDirection = invertY ? 1f : -1f;

            transform.Rotate(Vector3.up, mouseX, Space.Self);

            lookPitch = Mathf.Clamp(lookPitch + mouseY * yDirection, -lookPitchLimit, lookPitchLimit);

            if (playerCamera != null)
            {
                playerCamera.transform.localRotation = Quaternion.Euler(lookPitch, 0f, 0f);
            }
        }

        private void HandleModeToggle()
        {
            if (!allowLowGravityFloat)
            {
                lowGravityEnabled = false;
                return;
            }

            if (Input.GetKeyDown(toggleFloatKey))
            {
                lowGravityEnabled = !lowGravityEnabled;
            }
        }

        private void HandleMovement(float deltaTime)
        {
            Vector2 moveInput = GetMoveInput();
            bool wantsSprint = Input.GetKey(KeyCode.LeftShift);
            bool jumpPressed = Input.GetButtonDown("Jump");
            bool jumpedThisFrame = false;

            Vector3 desiredMove = BuildMoveDirection(moveInput);
            float moveSpeed = wantsSprint ? sprintSpeed : walkSpeed;

            if (IsLowGravityEnabled && !IsGrounded)
            {
                moveSpeed *= floatMoveMultiplier;
            }

            Vector3 targetPlanarVelocity = desiredMove * moveSpeed;
            float acceleration = IsGrounded ? groundAcceleration : airAcceleration;

            if (!IsGrounded && targetPlanarVelocity.magnitude > maxAirSpeed)
            {
                targetPlanarVelocity = targetPlanarVelocity.normalized * maxAirSpeed;
            }

            planarVelocity = Vector3.MoveTowards(planarVelocity, targetPlanarVelocity, acceleration * deltaTime);

            if (CanJump(jumpPressed))
            {
                // 低重力モードでもジャンプの初速はしっかり確保し、
                // 空中でだけ重力を弱めると操作感が安定する。
                verticalVelocity = Mathf.Sqrt(2f * gravity * jumpHeight);
                IsGrounded = false;
                jumpedThisFrame = true;
            }

            ApplyVerticalForces(deltaTime, jumpedThisFrame);

            Vector3 frameVelocity = planarVelocity + transform.up * verticalVelocity;
            CollisionFlags collisionFlags = characterController.Move(frameVelocity * deltaTime);

            if ((collisionFlags & CollisionFlags.Above) != 0 && verticalVelocity > 0f)
            {
                verticalVelocity = 0f;
            }

            RefreshGroundState();

            if (!jumpedThisFrame && verticalVelocity <= 0f)
            {
                TrySnapToGround();
            }
        }

        private void ApplyVerticalForces(float deltaTime, bool jumpedThisFrame)
        {
            if (IsGrounded && !jumpedThisFrame)
            {
                verticalVelocity = -groundedStickForce;
                return;
            }

            if (IsLowGravityEnabled)
            {
                float ascendInput = 0f;

                if (Input.GetKey(KeyCode.Space))
                {
                    ascendInput += 1f;
                }

                if (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.C))
                {
                    ascendInput -= 1f;
                }

                float targetVerticalSpeed = 0f;

                if (ascendInput > 0f)
                {
                    targetVerticalSpeed = floatAscendSpeed;
                }
                else if (ascendInput < 0f)
                {
                    targetVerticalSpeed = -floatDescendSpeed;
                }

                verticalVelocity = Mathf.MoveTowards(verticalVelocity, targetVerticalSpeed, floatVerticalResponse * deltaTime);
                verticalVelocity -= floatGravity * deltaTime;
                return;
            }

            verticalVelocity -= gravity * deltaTime;
        }

        private void RefreshGroundState()
        {
            bool groundedByController = characterController.isGrounded;
            bool groundedByProbe = false;

            if (splatRaycastHelper != null)
            {
                Vector3 rayOrigin = GetFeetPosition() + transform.up * feetProbeLift;

                if (splatRaycastHelper.TryRaycastWalkable(
                    rayOrigin,
                    transform.up,
                    out RaycastHit probeHit,
                    feetProbeLift + manualGroundSnapDistance))
                {
                    currentGroundHit = probeHit;
                    groundedByProbe = probeHit.distance <= feetProbeLift + manualGroundSnapDistance;
                }
            }

            IsGrounded = groundedByController || groundedByProbe;

            if (IsGrounded)
            {
                lastGroundedTime = Time.time;
            }
        }

        private void TrySnapToGround()
        {
            if (splatRaycastHelper == null)
            {
                return;
            }

            Vector3 rayOrigin = GetFeetPosition() + transform.up * feetProbeLift;

            if (!splatRaycastHelper.TryRaycastWalkable(
                    rayOrigin,
                    transform.up,
                    out RaycastHit hit,
                    feetProbeLift + manualGroundSnapDistance))
            {
                return;
            }

            float snapDistance = hit.distance - feetProbeLift;

            if (snapDistance <= 0.001f)
            {
                IsGrounded = true;
                currentGroundHit = hit;
                verticalVelocity = -groundedStickForce;
                return;
            }

            characterController.Move(-transform.up * snapDistance);
            currentGroundHit = hit;
            IsGrounded = true;
            lastGroundedTime = Time.time;
            verticalVelocity = -groundedStickForce;
        }

        private bool CanJump(bool jumpPressed)
        {
            if (!jumpPressed)
            {
                return false;
            }

            if (IsGrounded)
            {
                return true;
            }

            return Time.time - lastGroundedTime <= coyoteTime;
        }

        private Vector2 GetMoveInput()
        {
            float x = Input.GetAxisRaw("Horizontal");
            float y = Input.GetAxisRaw("Vertical");
            return Vector2.ClampMagnitude(new Vector2(x, y), 1f);
        }

        private Vector3 BuildMoveDirection(Vector2 moveInput)
        {
            Vector3 move = transform.right * moveInput.x + transform.forward * moveInput.y;

            if (move.sqrMagnitude <= 0.0001f)
            {
                return Vector3.zero;
            }

            move.Normalize();

            if (IsGrounded && currentGroundHit.collider != null)
            {
                // collider mesh の斜面に沿って進むため、接地中は法線へ投影しておく。
                move = Vector3.ProjectOnPlane(move, currentGroundHit.normal).normalized;
            }

            return move;
        }

        private Vector3 GetFeetPosition()
        {
            Vector3 worldCenter = transform.TransformPoint(characterController.center);
            float halfHeight = Mathf.Max(characterController.height * 0.5f, characterController.radius);
            float bottomToCenter = halfHeight - characterController.radius;
            Vector3 bottomSphereCenter = worldCenter - transform.up * bottomToCenter;
            return bottomSphereCenter - transform.up * (characterController.radius - characterController.skinWidth);
        }

        private void UpdateCursorState()
        {
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                LockCursor(false);
                return;
            }

            if (!lockCursorOnStart)
            {
                return;
            }

            if (Cursor.lockState != CursorLockMode.Locked && Input.GetMouseButtonDown(0))
            {
                LockCursor(true);
            }
        }

        private static void LockCursor(bool shouldLock)
        {
            Cursor.lockState = shouldLock ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !shouldLock;
        }

        /// <summary>
        /// zone など外部要因から低重力を強制したい時の入口。
        /// 手動トグルとは独立して扱う。
        /// </summary>
        public void SetForcedLowGravity(bool enabled)
        {
            forcedLowGravityEnabled = enabled;
        }

        /// <summary>
        /// CharacterController を壊さずに安全に位置合わせするための簡易テレポート。
        /// 夢の門やイベント遷移用に使う。
        /// </summary>
        public void TeleportToPose(Vector3 worldPosition, Quaternion worldRotation)
        {
            if (characterController == null)
            {
                characterController = GetComponent<CharacterController>();
            }

            Quaternion yawOnlyRotation = Quaternion.Euler(0f, worldRotation.eulerAngles.y, 0f);

            planarVelocity = Vector3.zero;
            verticalVelocity = 0f;
            currentGroundHit = default;
            lastGroundedTime = Time.time;

            bool wasEnabled = characterController != null && characterController.enabled;

            if (characterController != null)
            {
                characterController.enabled = false;
            }

            transform.SetPositionAndRotation(worldPosition, yawOnlyRotation);

            if (characterController != null)
            {
                characterController.enabled = wasEnabled;
            }

            RefreshGroundState();

            if (wasEnabled)
            {
                TrySnapToGround();
            }
        }
    }
}
