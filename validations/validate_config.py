#!/usr/bin/env python3
"""
validate_config.py
==================
Stand-alone validation script for ``config/config.json``.

Run before any PSO job to catch misconfigurations early.

Usage
-----
    python validations/validate_config.py
    python validations/validate_config.py --config path/to/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from source.utils.config_validator import load_and_validate_config, ConfigValidationError
from source.utils.logging_utils import setup_logging, get_logger

setup_logging(log_dir=_PROJECT_ROOT / "logs", log_filename="run.log")
logger = get_logger("validate_config")


def _check_paths(cfg: dict) -> list[str]:
    """Return list of warning messages for non-existent paths."""
    warnings = []
    for key in ("data_dir", "signals_file"):
        p = Path(cfg.get(key, ""))
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            warnings.append(f"Path '{key}' does not exist: {p}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate config/config.json")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=_PROJECT_ROOT / "config" / "config.json",
        help="Path to config.json",
    )
    args = parser.parse_args()

    logger.info("Validating config: %s", args.config)

    try:
        cfg = load_and_validate_config(args.config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ConfigValidationError as exc:
        logger.error("Config is INVALID:\n%s", exc)
        sys.exit(1)

    logger.info("Schema validation: PASSED")

    # Path existence warnings (non-fatal)
    warnings = _check_paths(cfg)
    for w in warnings:
        logger.warning("%s", w)

    # Summarise key parameters
    logger.info("Key parameters:")
    logger.info("  n_particles  = %s", cfg.get("n_particles"))
    logger.info("  max_iter     = %s", cfg.get("max_iter"))
    logger.info("  n_rois       = %s", cfg.get("n_rois"))
    logger.info("  use_cpp      = %s", cfg.get("use_cpp"))
    logger.info("  tmax         = %s s", cfg.get("tmax"))
    logger.info("  invalid_rois = %s", cfg.get("invalid_rois"))

    if warnings:
        logger.warning("Config is valid but some paths are missing (%d warning(s)).", len(warnings))
        sys.exit(0)

    logger.info("Config is fully valid. Ready to run.")
    sys.exit(0)


if __name__ == "__main__":
    main()
