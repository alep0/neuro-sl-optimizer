"""
logging_utils.py
================
Centralised logging configuration for the neuro-SL-optimizer project.

Usage
-----
    from source.utils.logging_utils import get_logger, setup_logging

    setup_logging(log_dir=Path("logs"), level=logging.INFO)
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Parameters
    ----------
    name : str
        Typically ``__name__``.
    """
    return logging.getLogger(name)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_filename: str = "run.log",
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Configure the root logger with file and/or console handlers.

    Parameters
    ----------
    log_dir : Path, optional
        Directory where the log file will be written.  When *None*, only the
        console handler is added (if ``console=True``).
    log_filename : str
        Name of the log file inside *log_dir*.
    level : int
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    console : bool
        Whether to add a ``StreamHandler`` to *stderr*.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called multiple times
    root.handlers.clear()

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(_FORMATTER)
        root.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        fh.setFormatter(_FORMATTER)
        root.addHandler(fh)
        root.info("Logging to %s/%s", log_dir, log_filename)
