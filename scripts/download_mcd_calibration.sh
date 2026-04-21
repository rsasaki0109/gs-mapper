#!/usr/bin/env bash
# Fetch MCDVIRAL rig calibration YAML from the official Download page (Google Drive).
# Dataset: CC BY-NC-SA 4.0 — do not commit the YAML; use this script locally.
set -euo pipefail

usage() {
  echo "usage: $0 <handheld|atv> [output.yaml]" >&2
  exit 1
}

kind="${1:-}"
out="${2:-}"

[[ -n "$kind" ]] || usage

case "$kind" in
  handheld)
    id="1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym"
    default_out="data/mcd/calibration_handheld.yaml"
    ;;
  atv)
    id="1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL"
    default_out="data/mcd/calibration_atv.yaml"
    ;;
  *)
    usage
    ;;
esac

[[ -n "$out" ]] || out="$default_out"
mkdir -p "$(dirname "$out")"
curl -sL "https://drive.google.com/uc?export=download&id=${id}" -o "$out"
echo "Wrote ${out}"
