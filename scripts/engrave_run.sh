#!/usr/bin/env bash
set -euo pipefail
SHEET="${1:-sheets/main.cfg}"
STEPS="${2:-490000}"
torchrun --standalone --nproc_per_node=16 -m graticule.burin engrave \
  --sheet "${SHEET}" --steps "${STEPS}" --out engravings/atlas.pt
