"""Resolve external visual SLAM output files without importing those projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gs_sim2real.preprocess.external_slam_artifacts.profiles import PROFILES, normalize_system


_SKIP_TEXT_NAMES = ("readme", "license", "config", "metrics", "results", "log")


@dataclass(frozen=True, slots=True)
class ExternalSLAMArtifacts:
    """Resolved files ready for conversion through the existing trajectory importer."""

    system: str
    trajectory_path: Path
    trajectory_format: str
    pointcloud_path: Path | None = None
    pinhole_calib_path: Path | None = None


def resolve_external_slam_artifacts(
    *,
    system: str = "generic",
    artifact_dir: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    trajectory_format: str | None = None,
    pointcloud_path: str | Path | None = None,
    pinhole_calib_path: str | Path | None = None,
) -> ExternalSLAMArtifacts:
    """Resolve external SLAM output paths without importing the external project."""

    system_key = normalize_system(system)
    profile = PROFILES[system_key]
    base_dir = _optional_existing_dir(artifact_dir, role="external SLAM output")

    resolved_trajectory = _resolve_required_file(
        explicit=trajectory_path,
        base_dir=base_dir,
        candidates=profile.trajectory_candidates,
        role=f"{profile.display_name} trajectory",
    )
    resolved_pointcloud = _resolve_optional_file(
        explicit=pointcloud_path,
        base_dir=base_dir,
        candidates=profile.pointcloud_candidates,
        role=f"{profile.display_name} point cloud",
    )
    resolved_calib = _resolve_optional_file(
        explicit=pinhole_calib_path,
        base_dir=base_dir,
        candidates=(),
        role="PINHOLE calibration",
    )

    return ExternalSLAMArtifacts(
        system=system_key,
        trajectory_path=resolved_trajectory,
        trajectory_format=trajectory_format or profile.default_trajectory_format,
        pointcloud_path=resolved_pointcloud,
        pinhole_calib_path=resolved_calib,
    )


def _optional_existing_dir(path: str | Path | None, *, role: str) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{role} directory not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{role} path is not a directory: {candidate}")
    return candidate


def _resolve_required_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path:
    resolved = _resolve_optional_file(explicit=explicit, base_dir=base_dir, candidates=candidates, role=role)
    if resolved is not None:
        return resolved
    if base_dir is None:
        raise ValueError(f"{role} is required. Pass --trajectory or --external-slam-output.")
    raise FileNotFoundError(f"Could not find {role} under {base_dir}")


def _resolve_optional_file(
    *,
    explicit: str | Path | None,
    base_dir: Path | None,
    candidates: tuple[str, ...],
    role: str,
) -> Path | None:
    if explicit not in (None, ""):
        return _resolve_explicit_file(Path(explicit), base_dir=base_dir, role=role)
    if base_dir is None:
        return None
    for pattern in candidates:
        matches = _candidate_matches(base_dir, pattern)
        if matches:
            return matches[0]
    return None


def _resolve_explicit_file(path: Path, *, base_dir: Path | None, role: str) -> Path:
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_file():
                raise FileNotFoundError(f"{role} path is not a file: {candidate}")
            return candidate
    raise FileNotFoundError(f"{role} file not found: {path}")


def _candidate_matches(base_dir: Path, pattern: str) -> list[Path]:
    matches = sorted(p for p in base_dir.rglob(pattern) if p.is_file())
    if pattern.endswith(".txt") or pattern == "*.txt":
        matches = [p for p in matches if not _looks_like_non_trajectory_text(p)]
    return matches


def _looks_like_non_trajectory_text(path: Path) -> bool:
    name = path.name.lower()
    return any(skip in name for skip in _SKIP_TEXT_NAMES)
