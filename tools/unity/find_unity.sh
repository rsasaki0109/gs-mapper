#!/usr/bin/env bash
set -euo pipefail

preferred_version="${UNITY_VERSION:-6000.0.23f1}"

declare -a candidates=()

if [[ -n "${UNITY_EDITOR:-}" ]]; then
  candidates+=("${UNITY_EDITOR}")
fi

for pattern in \
  "$HOME/Unity/Hub/Editor/${preferred_version}/Editor/Unity" \
  "$HOME/Unity/Hub/Editor"/*/Editor/Unity \
  "$HOME/.local/share/Unity/Hub/Editor/${preferred_version}/Editor/Unity" \
  "$HOME/.local/share/Unity/Hub/Editor"/*/Editor/Unity \
  "/opt/Unity/Hub/Editor/${preferred_version}/Editor/Unity" \
  "/opt/Unity/Hub/Editor"/*/Editor/Unity \
  "/Applications/Unity/Hub/Editor/${preferred_version}/Unity.app/Contents/MacOS/Unity" \
  "/Applications/Unity/Hub/Editor"/*"/Unity.app/Contents/MacOS/Unity"
do
  for candidate in $pattern; do
    candidates+=("${candidate}")
  done
done

for candidate in "${candidates[@]}"; do
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

exit 1
