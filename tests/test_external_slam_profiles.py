"""Schema invariants for registered external SLAM artifact profiles."""

from __future__ import annotations

import pytest

from gs_sim2real.preprocess.external_slam_artifacts.profiles import (
    ALIASES,
    PROFILES,
    SYSTEM_CHOICES,
    normalize_system,
)


_SUPPORTED_TRAJECTORY_FORMATS = {"tum", "kitti", "nmea"}


def test_system_choices_match_registered_profiles() -> None:
    assert SYSTEM_CHOICES == tuple(PROFILES)
    assert "generic" in PROFILES


def test_every_profile_has_usable_default_format() -> None:
    for key, profile in PROFILES.items():
        assert profile.key == key, f"profile {key!r} key mismatch"
        assert profile.display_name, f"profile {key!r} has empty display_name"
        assert profile.default_trajectory_format in _SUPPORTED_TRAJECTORY_FORMATS, (
            f"profile {key!r} declares unsupported trajectory format {profile.default_trajectory_format!r}"
        )


def test_every_profile_has_non_empty_trajectory_candidates() -> None:
    for key, profile in PROFILES.items():
        assert profile.trajectory_candidates, (
            f"profile {key!r} has no trajectory candidates; resolver would always fail"
        )
        assert all(isinstance(pattern, str) and pattern for pattern in profile.trajectory_candidates), (
            f"profile {key!r} has empty/non-string trajectory candidate"
        )


def test_every_profile_has_specific_trajectory_candidate_before_wildcards() -> None:
    """Resolver picks the first matching pattern; wildcard-only profiles would misroute."""

    for key, profile in PROFILES.items():
        specific = [p for p in profile.trajectory_candidates if not p.startswith("*")]
        if key == "generic":
            # Generic is intentionally permissive.
            assert specific, "generic profile should still list named trajectory files first"
            continue
        assert specific, (
            f"profile {key!r} only lists wildcard trajectory candidates; "
            f"this would cause arbitrary text files to be selected before named ones"
        )


def test_profile_candidate_patterns_have_no_path_separators() -> None:
    for key, profile in PROFILES.items():
        for pattern in profile.trajectory_candidates + profile.pointcloud_candidates:
            assert "/" not in pattern, f"profile {key!r} pattern {pattern!r} contains a path separator"
            assert "\\" not in pattern, f"profile {key!r} pattern {pattern!r} contains a backslash"


def test_aliases_resolve_to_registered_profiles() -> None:
    for alias, target in ALIASES.items():
        assert target in PROFILES, f"alias {alias!r} points to unknown profile {target!r}"
        assert alias not in PROFILES, (
            f"alias {alias!r} shadows a registered profile key; remove the alias or rename the profile"
        )
        assert normalize_system(alias) == target


def test_normalize_system_roundtrips_registered_keys() -> None:
    for key in PROFILES:
        assert normalize_system(key) == key
        assert normalize_system(key.upper()) == key


def test_normalize_system_rejects_unknown_systems() -> None:
    with pytest.raises(ValueError, match="Unsupported external SLAM system"):
        normalize_system("not-a-real-slam-system")
