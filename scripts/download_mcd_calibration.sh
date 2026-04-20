#!/usr/bin/env bash
# Download the MCDVIRAL sensor calibration YAML for a given rig from the
# upstream Google Drive folder linked on https://mcdviral.github.io/Download.html .
#
# Usage:
#   scripts/download_mcd_calibration.sh <rig> [<out-path>]
#     rig: "handheld"  -> for kth_/tuhh_ sessions (d455b + d455t + mid70 + os1-64 + vn200)
#          "atv"       -> for ntu_ sessions       (d435i + d455b + mid70 + os1-64 + vn100/vn200)
#   out-path defaults to data/mcd/calibration_<rig>.yaml
#
# Example:
#   scripts/download_mcd_calibration.sh handheld
#   scripts/download_mcd_calibration.sh atv data/mcd/calib_atv.yaml
#
# The YAMLs are small (<10 KB) so this bypasses the "virus scan warning" form
# that download_mcd_session.sh handles for larger rosbags.
#
# IMPORTANT: MCDVIRAL (the YAMLs included) is distributed under CC BY-NC-SA
# 4.0 — do NOT redistribute the downloaded file in commercial contexts. See
# https://mcdviral.github.io/ for the dataset license.

set -eu

rig=${1:-}
out=${2:-}
case "${rig}" in
    handheld)
        id="1htr26EE-Y1sHS5J4zaSbauC1XFgIh3Ym"
        ;;
    atv)
        id="1zVTBqh4cA1DciWBj5n7BGiexbfan1BBL"
        ;;
    *)
        printf "usage: %s <handheld|atv> [<out-path>]\n" "$0" >&2
        exit 2
        ;;
esac

if [ -z "${out}" ]; then
    out="data/mcd/calibration_${rig}.yaml"
fi
mkdir -p "$(dirname "${out}")"

curl -sL -o "${out}" "https://drive.google.com/uc?export=download&id=${id}"

# Sanity-check: the YAML should start with "body:" and declare at least one
# sensor; if the Drive page served an HTML error instead we want to fail loud.
if ! head -1 "${out}" | grep -qE '^body:' ; then
    printf "downloaded file does not look like an MCD calibration YAML: %s\n" "${out}" >&2
    printf "(check that the file ID is still valid on the upstream Download page)\n" >&2
    exit 1
fi

printf "Saved %s (%s bytes)\n" "${out}" "$(wc -c < "${out}" | tr -d ' ')"
printf "License reminder: MCDVIRAL is CC BY-NC-SA 4.0 — non-commercial use only.\n"
