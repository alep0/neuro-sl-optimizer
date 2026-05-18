#!/usr/bin/env python3
"""
run_pso.py
==========
Command-line entry point for the Stuart-Landau PSO optimisation pipeline.

This script replaces the legacy ``PSO_step_v1.sh`` bash launcher.
It validates inputs, configures logging, and delegates to
:func:`source.core.pso_optimizer.run_pso_optimisation`.

Usage
-----
    python scripts/run_pso.py --realization 1 --op-corr 1 --op-net 3 --op-model 1

    # Multiple realizations:
    python scripts/run_pso.py --realization 1 2 3 --op-corr 1 --op-net 3 --op-model 1

    # Custom config:
    python scripts/run_pso.py --realization 1 --config config/config.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from source.core.pso_optimizer import run_pso_optimisation
from source.utils.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pso",
        description=(
            "Stuart-Landau PSO functional-connectivity optimiser.\n\n"
            "Optimises coupling and frequency parameters of a network of "
            "Stuart-Landau oscillators to match empirical fMRI correlation matrices."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--realization", "-r",
        nargs="+",
        default=["1"],
        metavar="ID",
        help="One or more realization identifiers (default: 1).",
    )
    parser.add_argument(
        "--op-corr", "-c",
        type=int,
        choices=[1, 2],
        default=1,
        help="Correlation mode: 1=Pearson (default), 2=cross-correlation.",
    )
    parser.add_argument(
        "--op-net", "-n",
        type=int,
        choices=[2, 3, 4],
        default=3,
        help="Network connectivity mode: 2=velocity, 3=tau (default), 4=bimodal.",
    )
    parser.add_argument(
        "--op-model", "-m",
        type=int,
        choices=[1, 2],
        default=1,
        help="Model variant: 1=fixed frequencies (default), 2=connectivity-derived.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config.json (default: <project_root>/config/config.json).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_PROJECT_ROOT / "logs",
        metavar="DIR",
        help="Directory for log files (default: <project_root>/logs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace) -> None:
    """Raise SystemExit with a helpful message on invalid arguments."""
    if args.config is not None and not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    for r in args.realization:
        if not r.strip():
            print("ERROR: Realization ID must not be empty.", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(
        log_dir=args.log_dir,
        log_filename="run.log",
        level=numeric_level,
    )

    logger = logging.getLogger("run_pso")
    logger.info("Starting PSO pipeline")
    logger.info(
        "realizations=%s | op_corr=%d | op_net=%d | op_model=%d",
        args.realization, args.op_corr, args.op_net, args.op_model,
    )

    any_failure = False
    for realization in args.realization:
        logger.info("─" * 60)
        logger.info("Processing realization: %s", realization)
        exit_code = run_pso_optimisation(
            realization_index=realization,
            op_corr=args.op_corr,
            op_net=args.op_net,
            op_model=args.op_model,
            config_path=args.config,
        )
        if exit_code != 0:
            logger.error(
                "Realization %s finished with error code %d", realization, exit_code
            )
            any_failure = True
        else:
            logger.info("Realization %s completed successfully.", realization)

    if any_failure:
        logger.error("One or more realizations failed. Check logs for details.")
        sys.exit(1)

    logger.info("All realizations completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
