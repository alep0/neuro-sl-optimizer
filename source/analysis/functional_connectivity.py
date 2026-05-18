"""
functional_connectivity.py
==========================
Functional connectivity estimation for the Stuart-Landau pipeline.

Provides Pearson correlation and maximum cross-correlation matrices,
plus a coarse-graining utility to remove invalid ROIs.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from source.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cross-correlation helpers
# ---------------------------------------------------------------------------

def _cross_corr_at_lags(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Compute normalised cross-correlation between *x* and *y* for
    lags in [-max_lag, +max_lag].

    Parameters
    ----------
    x, y : np.ndarray, shape (T,)
        Equal-length signals.
    max_lag : int
        Maximum number of samples to shift.

    Returns
    -------
    np.ndarray, shape (2*max_lag + 1,)
        Normalised cross-correlation values.
    """
    if len(x) != len(y):
        raise ValueError("Signals x and y must have the same length.")

    N = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    norm = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if norm == 0:
        return np.zeros(len(lags))

    corr = np.empty(len(lags))
    for k, lag in enumerate(lags):
        if lag < 0:
            corr[k] = np.dot(x[:N + lag], y[-lag:N])
        elif lag > 0:
            corr[k] = np.dot(x[lag:N], y[:N - lag])
        else:
            corr[k] = np.dot(x, y)
    return corr / norm


def cross_correlation_matrix(
    signals: np.ndarray,
    frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the maximum-lag cross-correlation matrix.

    For each pair (i, j) with i > j, find the lag in [-max_lag, +max_lag]
    that maximises |CC(i, j)| and record that value (symmetrised).

    Parameters
    ----------
    signals : np.ndarray, shape (N, T)
        Multi-node time series.
    frac : float
        Fraction of *T* used as the maximum lag.

    Returns
    -------
    lag_matrix : np.ndarray, shape (N, N)
        Optimal lag (in samples, centred at zero) for each pair.
    cross_corr : np.ndarray, shape (N, N)
        Cross-correlation value at the optimal lag (symmetric).
    """
    N, T = signals.shape
    max_lag = int(T * frac)
    cross_corr = np.zeros((N, N))
    lag_matrix  = np.zeros((N, N))

    for i in range(N):
        for j in range(i):
            cc = _cross_corr_at_lags(signals[i], signals[j], max_lag)
            best_k = int(np.argmax(np.abs(cc)))
            cross_corr[i, j] = cc[best_k]
            cross_corr[j, i] = cross_corr[i, j]
            lag_matrix[i, j]  = best_k - max_lag
            lag_matrix[j, i]  = lag_matrix[i, j]

    logger.debug("Cross-correlation matrix computed: shape=%s", cross_corr.shape)
    return lag_matrix, cross_corr


# ---------------------------------------------------------------------------
# Unified correlation dispatcher
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    signals: np.ndarray,
    mode: int = 1,
    frac: float = 0.2,
) -> np.ndarray:
    """Compute a functional connectivity matrix.

    Parameters
    ----------
    signals : np.ndarray, shape (N, T)
        Multi-node time series.
    mode : int
        1 = Pearson (diagonal zeroed), 2 = max cross-correlation.
    frac : float
        Max-lag fraction (only used when mode=2).

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric connectivity matrix with zeroed diagonal.

    Raises
    ------
    ValueError
        If *mode* is not 1 or 2.
    """
    if mode == 1:
        corr = np.corrcoef(signals) - np.eye(len(signals))
        logger.debug("Pearson correlation matrix computed: shape=%s", corr.shape)
        return corr
    elif mode == 2:
        _, corr = cross_correlation_matrix(signals, frac=frac)
        return corr
    else:
        raise ValueError(f"Unknown correlation mode: {mode}. Use 1 (Pearson) or 2 (cross-corr).")


# ---------------------------------------------------------------------------
# Coarse-graining
# ---------------------------------------------------------------------------

def coarse_grain_matrix(
    matrix: np.ndarray,
    invalid_indices: List[int],
) -> np.ndarray:
    """Remove rows and columns corresponding to invalid ROIs.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Full connectivity matrix.
    invalid_indices : list of int
        Row/column indices to exclude.

    Returns
    -------
    np.ndarray, shape (N - len(invalid_indices), N - len(invalid_indices))
        Reduced connectivity matrix.

    Raises
    ------
    ValueError
        If any invalid index is out of range.
    """
    N = matrix.shape[0]
    bad = set(invalid_indices)
    out_of_range = [i for i in bad if i < 0 or i >= N]
    if out_of_range:
        raise ValueError(
            f"Invalid ROI indices out of range [0, {N-1}]: {out_of_range}"
        )

    valid = [i for i in range(N) if i not in bad]
    coarse = matrix[np.ix_(valid, valid)]
    logger.debug(
        "Coarse-graining: %d -> %d nodes (removed %d ROIs)",
        N, len(valid), len(bad),
    )
    return coarse
