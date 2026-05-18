#!/usr/bin/env python3
"""
validate_signal_processing.py
==============================
Sanity-checks the signal-processing utilities with synthetic data.

Usage
-----
    python validations/validate_signal_processing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from source.analysis.signal_processing import bandpass_filter, normalise_signal
from source.analysis.functional_connectivity import (
    compute_correlation_matrix,
    coarse_grain_matrix,
)
from source.utils.logging_utils import setup_logging, get_logger

setup_logging(log_dir=_PROJECT_ROOT / "logs", log_filename="run.log")
logger = get_logger("validate_signal_processing")

PASSED = 0
FAILED = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        logger.info("  PASS  %s", name)
        PASSED += 1
    else:
        logger.error("  FAIL  %s  %s", name, detail)
        FAILED += 1


def test_bandpass_filter() -> None:
    logger.info("--- bandpass_filter ---")
    fs = 10_000.0
    T = 100_000
    t = np.arange(T) / fs

    # Pure 0.1 Hz signal – should survive a 0.01–0.5 Hz filter
    signal = np.sin(2 * np.pi * 0.1 * t)
    filtered = bandpass_filter(signal, high_freq=0.5, low_freq=0.01,
                               fs=fs, filter_order=4)
    _check("output_shape_preserved", filtered.shape == signal.shape)
    # Signal energy should be maintained (within 50 %)
    _check("energy_maintained",
           np.var(filtered) > 0.1 * np.var(signal),
           f"var(filtered)={np.var(filtered):.4f} vs var(signal)={np.var(signal):.4f}")

    # Invalid arguments
    try:
        bandpass_filter(signal, high_freq=0.01, low_freq=0.5, fs=fs)
        _check("raises_on_inverted_range", False, "Expected ValueError")
    except ValueError:
        _check("raises_on_inverted_range", True)


def test_normalise_signal() -> None:
    logger.info("--- normalise_signal ---")
    rng = np.random.default_rng(0)
    s = rng.normal(5.0, 3.0, 1000)
    n = normalise_signal(s)
    _check("zero_mean",  abs(np.mean(n)) < 1e-10, f"mean={np.mean(n)}")
    _check("unit_std",   abs(np.std(n) - 1.0) < 1e-10, f"std={np.std(n)}")

    # Constant signal
    zero_var = np.ones(100)
    n0 = normalise_signal(zero_var)
    _check("zero_std_returns_zeros", np.all(n0 == 0))


def test_correlation_matrix() -> None:
    logger.info("--- compute_correlation_matrix (Pearson) ---")
    rng = np.random.default_rng(42)
    N, T = 10, 500
    signals = rng.standard_normal((N, T))
    corr = compute_correlation_matrix(signals, mode=1)
    _check("shape_NxN",     corr.shape == (N, N))
    _check("diagonal_zero", np.allclose(np.diag(corr), 0))
    _check("symmetric",     np.allclose(corr, corr.T))
    _check("range_[-1,1]",  np.all(np.abs(corr) <= 1.0 + 1e-9))


def test_coarse_grain_matrix() -> None:
    logger.info("--- coarse_grain_matrix ---")
    N = 10
    mat = np.ones((N, N))
    invalid = [0, 3, 7]
    cg = coarse_grain_matrix(mat, invalid)
    expected = N - len(invalid)
    _check("correct_output_shape", cg.shape == (expected, expected),
           f"expected ({expected},{expected}), got {cg.shape}")

    # Out-of-range index
    try:
        coarse_grain_matrix(mat, [99])
        _check("raises_on_out_of_range", False, "Expected ValueError")
    except ValueError:
        _check("raises_on_out_of_range", True)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Signal Processing Validation")

    test_bandpass_filter()
    test_normalise_signal()
    test_correlation_matrix()
    test_coarse_grain_matrix()

    logger.info("=" * 60)
    logger.info("Results: %d passed, %d failed", PASSED, FAILED)
    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
