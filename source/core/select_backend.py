"""
select_backend.py
=================
Runtime CPU-feature detection and ISA-tier selector.

Usage (in simulation_engine.py)
--------------------------------
    from source.core.select_backend import load_best_simulator
    simulator = load_best_simulator()   # returns the fastest importable variant

The module reads /proc/cpuinfo (Linux) or uses ``sysctl`` (macOS) to obtain
the CPU flags, then walks the ISA_TIERS list from fastest to slowest and
returns the first variant whose .so file exists AND whose required CPU flags
are all present.

Falls back to the pure-Python implementation if nothing loads.
"""

from __future__ import annotations

import importlib
import logging
import platform
import subprocess
import sys
from pathlib import Path
#from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the project root (where the .so files live) is on sys.path.
# This file sits at <root>/source/core/select_backend.py → root is ../../
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent      # <root>/source/core/
_PROJECT_ROOT = _HERE.parent.parent          # <root>/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    logger.debug("select_backend: inserted %s into sys.path", _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Must match the order / names in setup_multiarch.py  (fastest → slowest)
# ---------------------------------------------------------------------------
ISA_TIERS: list[tuple[str, list[str]]] = [
    ("avx512",   ["avx512f"]),
    ("avx2",     ["avx2", "fma"]),
    ("avx",      ["avx"]),
    ("sse42",    ["sse4_2"]),
    ("baseline", []),
]


# ---------------------------------------------------------------------------
# CPU-flag detection
# ---------------------------------------------------------------------------
def _cpu_flags_linux() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("flags"):
                return set(line.split(":")[1].split())
    except OSError:
        pass
    return set()


def _cpu_flags_macos() -> set[str]:
    flags: set[str] = set()
    feature_map = {
        "hw.optional.avx512f":  "avx512f",
        "hw.optional.avx2_0":   "avx2",
        "hw.optional.fma":       "fma",
        "hw.optional.avx1_0":   "avx",
        "hw.optional.sse4_2":   "sse4_2",
    }
    for sysctl_key, flag_name in feature_map.items():
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", sysctl_key],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out == "1":
                flags.add(flag_name)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return flags


def get_cpu_flags() -> set[str]:
    """Return the set of CPU feature flags for the current CPU."""
    system = platform.system()
    if system == "Linux":
        return _cpu_flags_linux()
    if system == "Darwin":
        return _cpu_flags_macos()
    # Windows / other: assume only baseline
    return set()


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
def load_best_simulator():
    """
    Import and return the fastest available ISA variant of the simulator.

    Returns the module object, or None if every import fails.
    The caller is responsible for falling back to pure Python.
    """
    cpu_flags = get_cpu_flags()
    logger.debug("Detected CPU flags: %s", sorted(cpu_flags))
    logger.debug("sys.path includes: %s", _PROJECT_ROOT)

    # Log which .so files are actually found on disk (aids debugging)
    found_so = sorted(_PROJECT_ROOT.glob("stuart_landau_simulator_*.so"))
    if found_so:
        logger.debug("Found .so files: %s", [f.name for f in found_so])
    else:
        logger.warning(
            "No stuart_landau_simulator_*.so files found under %s", _PROJECT_ROOT
        )

    for suffix, required in ISA_TIERS:
        module_name = f"stuart_landau_simulator_{suffix}"

        # Check required CPU features
        missing = [f for f in required if f not in cpu_flags]
        if missing:
            logger.debug(
                "Skipping %s: missing CPU flags %s", module_name, missing
            )
            continue

        # Try to import
        try:
            mod = importlib.import_module(module_name)
            logger.info("Backend: C++ (accelerated) [%s]", suffix.upper())
            return mod
        except ImportError as exc:
            logger.debug("Could not import %s: %s", module_name, exc)
            continue

    logger.warning(
        "No compiled C++ backend found. Falling back to pure Python."
    )
    return None


# ---------------------------------------------------------------------------
# CLI helper: print the best tier this node can run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    flags = get_cpu_flags()
    print(f"CPU flags detected: {len(flags)} flags")
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"sys.path[0]: {sys.path[0]}")
    print()

    so_files = sorted(_PROJECT_ROOT.glob("stuart_landau_simulator_*.so"))
    print(f"Found .so files ({len(so_files)}):")
    for f in so_files:
        print(f"  {f.name}")
    print()

    chosen = None
    for suffix, required in ISA_TIERS:
        missing = [f for f in required if f not in flags]
        status = "✓ COMPATIBLE" if not missing else f"✗ missing: {missing}"
        print(f"  {suffix:10s}  {status}")
        if not missing and chosen is None:
            chosen = suffix

    print()
    mod = load_best_simulator()
    if mod:
        print(f"Loaded module: {mod.__name__}")
        sys.exit(0)
    else:
        print("No tier loaded. Check paths and .so files above.")
        sys.exit(1)
        