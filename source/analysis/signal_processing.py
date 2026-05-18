"""
signal_processing.py
====================
Band-pass filtering and related signal-processing utilities for the
Stuart-Landau neuro-simulator pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt

from source.utils.logging_utils import get_logger

logger = get_logger(__name__)


def bandpass_filter(
    signal: np.ndarray,
    high_freq: float,
    low_freq: float,
    fs: float,
    filter_order: int = 50,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to *signal*.

    The filter is implemented as a cascade of a high-pass (removing
    frequencies below *low_freq*) followed by a low-pass (removing
    frequencies above *high_freq*).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        1-D time series to filter.
    high_freq : float
        Upper cut-off frequency in Hz.
    low_freq : float
        Lower cut-off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    filter_order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered signal.

    Raises
    ------
    ValueError
        If *low_freq* >= *high_freq* or either exceeds the Nyquist limit.
    """
    nyquist = fs / 2.0
    if low_freq <= 0 or high_freq <= 0:
        raise ValueError("Filter frequencies must be positive.")
    if low_freq >= high_freq:
        raise ValueError(
            f"low_freq ({low_freq}) must be < high_freq ({high_freq})."
        )
    if high_freq >= nyquist:
        raise ValueError(
            f"high_freq ({high_freq}) must be < Nyquist ({nyquist})."
        )

    sos_hp = butter(filter_order, low_freq,  btype="hp", fs=fs, output="sos")
    filtered = sosfilt(sos_hp, signal)
    sos_lp = butter(filter_order, high_freq, btype="lp", fs=fs, output="sos")
    filtered = sosfilt(sos_lp, filtered)

    logger.debug(
        "Band-pass filter applied: [%.4f, %.4f] Hz, order=%d, fs=%.1f Hz",
        low_freq, high_freq, filter_order, fs,
    )
    return filtered


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    np.ndarray
        Normalised signal with zero mean and unit variance.
        Returns zeros if standard deviation is zero.
    """
    mean = np.mean(signal)
    std  = np.std(signal)
    if std == 0:
        logger.warning("Signal has zero standard deviation; returning zeros.")
        return np.zeros_like(signal, dtype=float)
    return (signal - mean) / std
