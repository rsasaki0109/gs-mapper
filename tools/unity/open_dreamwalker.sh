#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_path="${repo_root}/projects/DreamWalker"

unity_bin="$("${repo_root}/tools/unity/find_unity.sh" 2>/dev/null || true)"

if [[ -z "${unity_bin}" ]]; then
  echo "Unity Editor not found."
  echo "Set UNITY_EDITOR=/path/to/Unity or install Unity ${UNITY_VERSION:-6000.0.23f1}."
  exit 1
fi

exec "${unity_bin}" -projectPath "${project_path}"
