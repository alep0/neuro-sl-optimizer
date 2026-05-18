"""
test_pso_optimizer.py
=====================
Unit tests for the PSO optimiser core logic.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from source.core.pso_optimizer import PSOConfig, EvaluationContext, CorrelationPSO


# ---------------------------------------------------------------------------
# PSOConfig tests
# ---------------------------------------------------------------------------

class TestPSOConfig:
    def test_default_validate_passes(self):
        cfg = PSOConfig()
        cfg.validate()  # should not raise

    def test_invalid_n_particles(self):
        cfg = PSOConfig(n_particles=0)
        with pytest.raises(ValueError, match="n_particles"):
            cfg.validate()

    def test_invalid_max_iter(self):
        cfg = PSOConfig(max_iter=-1)
        with pytest.raises(ValueError, match="max_iter"):
            cfg.validate()

    def test_bad_bounds_shape(self):
        cfg = PSOConfig(bounds=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="bounds"):
            cfg.validate()

    def test_lower_ge_upper_raises(self):
        bounds = np.array([[5.0, 1.0]])  # lower > upper
        cfg = PSOConfig(bounds=bounds)
        with pytest.raises(ValueError, match="lower bounds"):
            cfg.validate()


# ---------------------------------------------------------------------------
# CorrelationPSO tests (mocked evaluate)
# ---------------------------------------------------------------------------

class _MockEvalCtx(EvaluationContext):
    """EvaluationContext whose evaluate always returns a simple quadratic."""
    pass


def _make_simple_pso(n_particles: int = 4, max_iter: int = 5) -> CorrelationPSO:
    """Build a PSO with a mock evaluate that returns sum-of-squares loss."""
    bounds = np.array([[  -5.0, 5.0],
                        [-5.0, 5.0]])
    pso_cfg = PSOConfig(n_particles=n_particles, max_iter=max_iter, bounds=bounds)
    target = np.eye(5)  # dummy

    ctx = EvaluationContext(
        target_corr=target,
        sim_config_kwargs={},
        filter_kwargs={},
    )
    pso = CorrelationPSO.__new__(CorrelationPSO)
    pso._cfg = pso_cfg
    pso._ctx = ctx
    pso._rng = np.random.default_rng(0)

    lo, hi = bounds[:, 0], bounds[:, 1]
    pso.positions = pso._rng.uniform(lo, hi, (n_particles, 2))
    pso.velocities = np.zeros_like(pso.positions)
    pso.pbest_positions = pso.positions.copy()

    # Monkeypatch evaluate to a simple quadratic
    pso._evaluate = lambda p: float(np.sum(p ** 2))

    pso.pbest_values = np.array([pso._evaluate(p) for p in pso.positions])
    best_idx = int(np.argmin(pso.pbest_values))
    pso.gbest_position = pso.pbest_positions[best_idx].copy()
    pso.gbest_value = float(pso.pbest_values[best_idx])
    return pso


class TestCorrelationPSO:
    def test_optimise_returns_correct_types(self):
        pso = _make_simple_pso()
        gbest_pos, gbest_val, err_hist, pos_hist = pso.optimise()
        assert isinstance(gbest_pos, np.ndarray)
        assert isinstance(gbest_val, float)
        assert len(err_hist) == 5
        assert len(pos_hist) == 5

    def test_error_history_is_non_increasing(self):
        pso = _make_simple_pso(n_particles=6, max_iter=10)
        _, _, err_hist, _ = pso.optimise()
        for i in range(1, len(err_hist)):
            assert err_hist[i] <= err_hist[i - 1] + 1e-12, (
                f"Error increased at iteration {i}: {err_hist[i-1]} -> {err_hist[i]}"
            )

    def test_gbest_value_is_non_negative(self):
        pso = _make_simple_pso()
        _, gbest_val, _, _ = pso.optimise()
        assert gbest_val >= 0.0
