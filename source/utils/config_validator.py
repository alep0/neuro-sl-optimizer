"""
config_validator.py
===================
Validates ``config/config.json`` against the expected schema before any
optimisation run is launched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from source.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Required keys and their expected Python types
# ---------------------------------------------------------------------------
_REQUIRED: Dict[str, type] = {
    "data_dir":          str,
    "signals_file":      str,
    "output_dir":        str,
    "tmax":              (int, float),
    "tman_samples":      int,
    "n_rois":            int,
    "n_particles":       int,
    "max_iter":          int,
    "pso_w":             (int, float),
    "pso_c1":            (int, float),
    "pso_c2":            (int, float),
    "filter_high_freq":  (int, float),
    "filter_low_freq":   (int, float),
    "filter_fs":         (int, float),
    "filter_order":      int,
    "downsample_step":   int,
    "cross_corr_frac":   (int, float),
    "invalid_rois":      list,
    "use_cpp":           bool,
}

_POSITIVE_FLOATS = {
    "tmax", "filter_high_freq", "filter_low_freq", "filter_fs",
    "cross_corr_frac", "pso_w", "pso_c1", "pso_c2",
}
_POSITIVE_INTS = {
    "n_rois", "n_particles", "max_iter", "filter_order",
    "downsample_step", "tman_samples",
}


class ConfigValidationError(ValueError):
    """Raised when the configuration file fails validation."""


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate *cfg* dict in-place.

    Parameters
    ----------
    cfg : dict
        Parsed JSON configuration.

    Raises
    ------
    ConfigValidationError
        On any missing key, wrong type, or out-of-range value.
    """
    errors: List[str] = []

    # Check required keys and types
    for key, expected_type in _REQUIRED.items():
        if key not in cfg:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(cfg[key], expected_type):
            errors.append(
                f"'{key}' must be {expected_type}, got {type(cfg[key]).__name__}"
            )

    # Value-range checks (only if types are correct)
    for key in _POSITIVE_FLOATS:
        if key in cfg and isinstance(cfg[key], (int, float)) and cfg[key] <= 0:
            errors.append(f"'{key}' must be > 0, got {cfg[key]}")
    for key in _POSITIVE_INTS:
        if key in cfg and isinstance(cfg[key], int) and cfg[key] <= 0:
            errors.append(f"'{key}' must be > 0, got {cfg[key]}")

    # Filter-specific logic
    if ("filter_low_freq" in cfg and "filter_high_freq" in cfg
            and isinstance(cfg["filter_low_freq"], (int, float))
            and isinstance(cfg["filter_high_freq"], (int, float))):
        if cfg["filter_low_freq"] >= cfg["filter_high_freq"]:
            errors.append(
                f"filter_low_freq ({cfg['filter_low_freq']}) must be "
                f"< filter_high_freq ({cfg['filter_high_freq']})"
            )

    if errors:
        msg = "Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        logger.error(msg)
        raise ConfigValidationError(msg)

    logger.info("Config validation passed (%d keys checked).", len(_REQUIRED))


def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load *config_path* from disk and validate it.

    Parameters
    ----------
    config_path : Path
        Path to ``config.json``.

    Returns
    -------
    dict
        Validated configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    validate_config(cfg)
    return cfg
