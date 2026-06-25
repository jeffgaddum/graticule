#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data}"
mkdir -p "${DEST}"
echo "place IHDP / NSW / ACIC / WDI / PWT extracts under ${DEST}"
echo "see data-sources.txt for the reachable source links"
