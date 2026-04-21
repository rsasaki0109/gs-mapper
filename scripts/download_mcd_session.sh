#!/usr/bin/env bash
# Download a single MCD rosbag session from Google Drive without a login.
#
# Usage:
#   scripts/download_mcd_session.sh <google-drive-file-id> <output-path>
#
# Example (NTU session 17, one of the smaller bags):
#   scripts/download_mcd_session.sh 1VmHgEj6GI0mPhLOA-gJo1UG8G9FrM26c data/mcd/ntu_session_17.bag
#
# Session IDs are listed on https://mcdviral.github.io/Download.html . The
# script follows the "virus scan warning" form POST that Drive serves for
# files larger than ~100 MB, so gdown's default retrieval path is bypassed.

set -eu
id=${1:-}
out=${2:-}
if [ -z "${id}" ] || [ -z "${out}" ]; then
    printf "usage: %s <file-id> <out-path>\n" "$0" >&2
    exit 2
fi
mkdir -p "$(dirname "${out}")"

tmp_html=$(mktemp)
trap "rm -f ${tmp_html}" EXIT

curl -sL -o "${tmp_html}" "https://drive.google.com/uc?export=download&id=${id}"
uuid=$(grep -oE 'name="uuid" value="[^"]+"' "${tmp_html}" | head -1 | sed 's/.*value="//;s/"//')
if [ -z "${uuid}" ]; then
    if ! grep -qi '<html' "${tmp_html}"; then
        mv "${tmp_html}" "${out}"
        trap - EXIT
        printf "\nSaved %s\n" "${out}"
        exit 0
    fi
    printf "could not extract uuid from Drive confirm page (is the file public?)\n" >&2
    exit 1
fi

curl -L --progress-bar --max-time 7200 \
    -o "${out}" \
    "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t&uuid=${uuid}"

printf "\nSaved %s\n" "${out}"
