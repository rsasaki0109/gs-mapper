"""Stable backend-selection interfaces for headless sim2real rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RenderBackendCapabilities:
    """Runtime capabilities that constrain renderer selection."""

    has_gaussian_splat: bool
    gsplat_available: bool
    cuda_available: bool


@dataclass(frozen=True)
class RenderBackendPreferences:
    """Optional workload preferences for backend-selection policies."""

    prefer_low_startup_latency: bool = False
    prefer_visual_fidelity: bool = True


@dataclass(frozen=True)
class RenderBackendRequest:
    """Stable input contract for backend-selection decisions."""

    requested_backend: str
    capabilities: RenderBackendCapabilities
    preferences: RenderBackendPreferences = RenderBackendPreferences()


@dataclass(frozen=True)
class RenderBackendSelection:
    """Resolved renderer backend and the reason it was chosen."""

    name: str
    reason: str


class RenderBackendPolicy(Protocol):
    """Minimal interface for interchangeable backend-selection policies."""

    name: str
    label: str
    style: str
    tier: str
    capabilities: dict[str, bool]

    def select(self, request: RenderBackendRequest) -> RenderBackendSelection:
        """Choose a backend or raise when an explicit request cannot be satisfied."""


class BalancedCapabilityRenderBackendPolicy:
    """Stable render backend policy used by production code."""

    name = "balanced"
    label = "Balanced Capability Gate"
    style = "capability-gated"
    tier = "core"
    capabilities = {
        "respectsExplicitRequests": True,
        "usesRuntimeCapabilities": True,
        "usesWorkloadPreferences": True,
        "supportsFastPreviewBias": True,
    }

    def select(self, request: RenderBackendRequest) -> RenderBackendSelection:
        requested_backend = str(request.requested_backend or "auto").strip() or "auto"
        runtime = request.capabilities
        preferences = request.preferences

        if requested_backend == "simple":
            return RenderBackendSelection("simple", "forced by --renderer simple")

        if requested_backend == "gsplat":
            if not runtime.has_gaussian_splat:
                raise RuntimeError("renderer=gsplat requires Gaussian parameters in the input PLY")
            if not runtime.gsplat_available:
                raise RuntimeError("renderer=gsplat requires the optional `gsplat` package to be installed")
            if not runtime.cuda_available:
                raise RuntimeError("renderer=gsplat requires CUDA because gsplat uses a CUDA rasterization backend")
            return RenderBackendSelection("gsplat", "forced by --renderer gsplat")

        if requested_backend != "auto":
            raise RuntimeError(f"unsupported renderer selection mode: {requested_backend}")

        if not runtime.has_gaussian_splat:
            return RenderBackendSelection(
                "simple",
                "fallback because the PLY does not contain Gaussian scale/rotation parameters",
            )
        if not runtime.gsplat_available:
            return RenderBackendSelection(
                "simple",
                "fallback because the optional `gsplat` package is not installed",
            )
        if not runtime.cuda_available:
            return RenderBackendSelection("simple", "fallback because CUDA is not available for gsplat")
        if preferences.prefer_low_startup_latency and not preferences.prefer_visual_fidelity:
            return RenderBackendSelection(
                "simple",
                "auto-selected simple because the workload prefers low startup latency over maximum fidelity",
            )
        return RenderBackendSelection(
            "gsplat",
            "auto-selected because gsplat, CUDA, and Gaussian PLY parameters are available",
        )


CORE_RENDER_BACKEND_POLICIES: dict[str, RenderBackendPolicy] = {
    "balanced": BalancedCapabilityRenderBackendPolicy(),
}


def select_render_backend(
    request: RenderBackendRequest,
    *,
    policy: str = "balanced",
) -> RenderBackendSelection:
    """Select the render backend under the current runtime and workload constraints."""
    if policy not in CORE_RENDER_BACKEND_POLICIES:
        raise RuntimeError(
            f"unsupported render backend policy: {policy}. "
            f"Expected one of {', '.join(sorted(CORE_RENDER_BACKEND_POLICIES))}"
        )
    return CORE_RENDER_BACKEND_POLICIES[policy].select(request)
