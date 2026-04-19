#!/usr/bin/env bash
# Download an entire MCD session folder (multiple .bag files) from Google Drive.
#
# The per-file download_mcd_session.sh only handles one ID at a time. MCDVIRAL
# ships each vehicle-mounted / handheld session as a *folder* containing
# camera.bag + LiDAR.bag + GNSS/IMU.bag + calibration siblings. This wrapper
# uses gdown to pull the whole folder recursively, skipping files that already
# exist so it can resume partial downloads.
#
# Usage:
#   scripts/download_mcd_folder.sh <google-drive-folder-id> <output-dir>
#
# Example (tuhh_night_09, smallest MCDVIRAL session with GNSS + LiDAR, 3.5 GB):
#   scripts/download_mcd_folder.sh 1nEPiTXkVmLIhmBOVNpwSAEgnAXupnAxx data/mcd/tuhh_night_09/
#
# Folder IDs come from https://mcdviral.github.io/Download.html (each session
# row links to a Drive folder; copy the /folders/<ID> segment).

set -eu
id=${1:-}
out=${2:-}
if [ -z "${id}" ] || [ -z "${out}" ]; then
    printf "usage: %s <folder-id> <out-dir>\n" "$0" >&2
    exit 2
fi
mkdir -p "${out}"

if ! command -v gdown >/dev/null 2>&1; then
    printf "gdown not found; install with: pip install gdown\n" >&2
    exit 1
fi

gdown --folder --continue --remaining-ok -O "${out}" "https://drive.google.com/drive/folders/${id}"

printf "\nSaved MCD session folder to %s\n" "${out}"
ls -lh "${out}" | tail -n +2
