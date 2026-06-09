#!/usr/bin/env bash
# =============================================================================
# build_multiarch.sh
# =============================================================================
# Builds all ISA-tuned variants of the C++ extension IN ONE GO.
# Run this once on the cluster LOGIN NODE (or a build node via an interactive
# job) BEFORE submitting any batch jobs.
#
# The login node does not need to support all ISAs itself — the compiler
# handles cross-arch compilation with explicit -march flags.
#
# Usage:
#   bash scripts/build_multiarch.sh [--clean]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() { echo "[build_multiarch] $*"; }

# Optional --clean flag
CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Optionally wipe previous builds
# ---------------------------------------------------------------------------
if [[ "$CLEAN" == "true" ]]; then
    log "Cleaning previous builds..."
    rm -rf build/
    find . -name "stuart_landau_simulator_*.so" -delete
    find . -name "stuart_landau_simulator_*.pyd" -delete
    log "Clean done."
fi

# ---------------------------------------------------------------------------
# Verify build tools
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    log "ERROR: python3 not found. Activate your conda environment first."
    exit 1
fi

if ! python3 -c "import pybind11" &>/dev/null; then
    log "pybind11 not found. Installing..."
    pip install pybind11
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
log "Building all ISA tiers via setup_multiarch.py ..."
python3 setup_multiarch.py build_ext --inplace 2>&1 | tee build_multiarch.log

# ---------------------------------------------------------------------------
# Report what was produced
# ---------------------------------------------------------------------------
log "----------------------------------------------------------------------"
log "Built .so files:"
find . -name "stuart_landau_simulator_*.so" -o \
       -name "stuart_landau_simulator_*.pyd" 2>/dev/null \
  | sort | while read -r f; do
    echo "  $f"
done

log "----------------------------------------------------------------------"
log "Testing which tier THIS node can run:"
python3 source/core/select_backend.py

log "Done. Ship the .so files alongside your source tree to the cluster."
