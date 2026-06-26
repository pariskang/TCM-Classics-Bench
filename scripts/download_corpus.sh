#!/usr/bin/env bash
# Download and extract the 中醫笈成 / Jicheng-TCM corpus snapshot that
# TCM-Classics-Bench is built on (version book-20180111).
#
# Result: corpus_src/book/<書名>/...  (one directory per classical text)
#
# The extracted text is NOT committed to this repo. Run this script, then
# `python scripts/build_release.py --root corpus_src/book` to regenerate the
# data/ artifacts, or `python -m tcm_bench ingest` for the full corpus.
set -euo pipefail

ARCHIVE_URL="https://jicheng.tw/files/jcw/book-20180111.7z"
DEST="${1:-corpus_src}"
ARCHIVE="${DEST}/book-20180111.7z"

mkdir -p "${DEST}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading ${ARCHIVE_URL} ..."
  curl -fSL --retry 4 --retry-delay 2 -o "${ARCHIVE}" "${ARCHIVE_URL}"
else
  echo "Archive already present: ${ARCHIVE}"
fi

if ! command -v 7z >/dev/null 2>&1; then
  echo "ERROR: 7z not found. Install p7zip-full (Debian/Ubuntu) or 'pip install py7zr'." >&2
  exit 1
fi

echo "Extracting ..."
7z x -y "${ARCHIVE}" -o"${DEST}" >/dev/null
echo "Done. Books are under ${DEST}/book/"
