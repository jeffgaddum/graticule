#!/usr/bin/env bash
set -euo pipefail
SHEET="${1:-sheets/main.cfg}"
CKPT="${2:-engravings/atlas.pt}"
python3 -m graticule.burin read --sheet "${SHEET}" --ckpt "${CKPT}"
python3 -m graticule.burin proof --sheet "${SHEET}" --ckpt "${CKPT}"
