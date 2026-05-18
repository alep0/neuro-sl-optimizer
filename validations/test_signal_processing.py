"""
test_signal_processing.py
=========================
Unit tests for signal processing and functional connectivity modules.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from source.analysis.signal_processing import bandpass_filter, normalise_signal
from source.analysis.functional_connectivity import (
    compute_correlation_matrix,
    coarse_grain_matrix,
    cross_correlation_matrix,
)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    FS = 10_000.0
    T  = 50_000

    def _signal(self, freq: float) -> np.ndarray:
        t = np.arange(self.T) / self.FS
        return np.sin(2 * np.pi * freq * t)

    def test_output_shape_unchanged(self):
        s = self._signal(0.1)
        out = bandpass_filter(s, high_freq=0.5, low_freq=0.01, fs=self.FS)
        assert out.shape == s.shape

    def test_in_band_signal_passes(self):
        s = self._signal(0.1)
        out = bandpass_filter(s, high_freq=0.5, low_freq=0.01, fs=self.FS, filter_order=4)
        # At least 10% of original energy must survive
        assert np.var(out) > 0.1 * np.var(s)

    def test_raises_inverted_range(self):
        s = self._signal(0.1)
        with pytest.raises(ValueError):
            bandpass_filter(s, high_freq=0.01, low_freq=0.5, fs=self.FS)

    def test_raises_above_nyquist(self):
        s = self._signal(0.1)
        with pytest.raises(ValueError):
            bandpass_filter(s, high_freq=self.FS, low_freq=0.01, fs=self.FS)

    def test_raises_non_positive_freq(self):
        s = self._signal(0.1)
        with pytest.raises(ValueError):
            bandpass_filter(s, high_freq=0.5, low_freq=-0.01, fs=self.FS)


# ---------------------------------------------------------------------------
# normalise_signal
# ---------------------------------------------------------------------------

class TestNormaliseSignal:
    def test_zero_mean_unit_std(self):
        rng = np.random.default_rng(7)
        s = rng.normal(10.0, 4.0, 2000)
        n = normalise_signal(s)
        assert abs(np.mean(n)) < 1e-10
        assert abs(np.std(n) - 1.0) < 1e-10

    def test_constant_returns_zeros(self):
        s = np.ones(100)
        n = normalise_signal(s)
        assert np.all(n == 0.0)


# ---------------------------------------------------------------------------
# compute_correlation_matrix
# ---------------------------------------------------------------------------

class TestComputeCorrelationMatrix:
    def test_pearson_shape_and_diagonal(self):
        rng = np.random.default_rng(1)
        sig = rng.standard_normal((8, 400))
        corr = compute_correlation_matrix(sig, mode=1)
        assert corr.shape == (8, 8)
        assert np.allclose(np.diag(corr), 0.0)

    def test_pearson_symmetric(self):
        rng = np.random.default_rng(2)
        sig = rng.standard_normal((5, 300))
        corr = compute_correlation_matrix(sig, mode=1)
        assert np.allclose(corr, corr.T)

    def test_pearson_range(self):
        rng = np.random.default_rng(3)
        sig = rng.standard_normal((6, 500))
        corr = compute_correlation_matrix(sig, mode=1)
        assert np.all(np.abs(corr) <= 1.0 + 1e-9)

    def test_unknown_mode_raises(self):
        sig = np.ones((4, 100))
        with pytest.raises(ValueError):
            compute_correlation_matrix(sig, mode=99)


# ---------------------------------------------------------------------------
# coarse_grain_matrix
# ---------------------------------------------------------------------------

class TestCoarseGrainMatrix:
    def test_correct_output_size(self):
        mat = np.ones((10, 10))
        cg = coarse_grain_matrix(mat, [0, 5, 9])
        assert cg.shape == (7, 7)

    def test_identity_with_no_removal(self):
        mat = np.arange(25).reshape(5, 5).astype(float)
        cg = coarse_grain_matrix(mat, [])
        assert np.array_equal(cg, mat)

    def test_out_of_range_raises(self):
        mat = np.ones((5, 5))
        with pytest.raises(ValueError):
            coarse_grain_matrix(mat, [10])

    def test_symmetry_preserved(self):
        rng = np.random.default_rng(4)
        raw = rng.standard_normal((8, 8))
        mat = (raw + raw.T) / 2
        cg = coarse_grain_matrix(mat, [1, 4])
        assert np.allclose(cg, cg.T)
