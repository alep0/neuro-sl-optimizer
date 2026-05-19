"""
pso_optimizer.py
================
Particle Swarm Optimisation (PSO) for Stuart-Landau neural network
functional-connectivity fitting.

Optimises a parameter vector [K, a, omega_0, omega_1, ..., omega_N-1] so
that the simulated correlation matrix matches an empirical fMRI target.

Public API
----------
    CorrelationPSO        – PSO class (evaluate / optimise)
    run_pso_optimisation  – top-level convenience function

Changelog
---------
v1.0.0  Refactored from PSO_corr_mat_loss_v0_9_n2.py
        - Structured logging, validated config, typed API
        - Moved signal processing helpers to source.analysis.signal_processing
        - Moved coarse-graining helpers to source.analysis.functional_connectivity
        - All paths relative to project root (no hard-coded /mnt/c/… paths)
        - Thread-safe evaluate(); inf returned on any simulation failure
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from source.analysis.functional_connectivity import (
    compute_correlation_matrix,
    coarse_grain_matrix,
)
from source.analysis.signal_processing import bandpass_filter
from source.core.simulation_engine import SimulationConfig, run_simulation
from source.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PSO hyper-parameter configuration
# ---------------------------------------------------------------------------

@dataclass
class PSOConfig:
    """Hyper-parameters that govern the PSO run.

    Parameters
    ----------
    n_particles : int
        Swarm size.
    max_iter : int
        Number of optimisation iterations.
    w : float
        Inertia weight.
    c1 : float
        Cognitive acceleration coefficient.
    c2 : float
        Social acceleration coefficient.
    bounds : np.ndarray, shape (D, 2)
        Lower/upper bounds for each dimension.
    """
    n_particles: int = 4
    max_iter: int = 10
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    bounds: np.ndarray = field(default_factory=lambda: np.array([
        [1e4, 1e5],   # K  – global coupling
        [-10.0, 10.0],  # a  – bifurcation parameter
        [20.0, 60.0],   # ω  – shared frequency (single-freq model)
    ]))

    def validate(self) -> None:
        if self.n_particles < 1:
            raise ValueError("n_particles must be >= 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (D, 2).")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("All lower bounds must be strictly < upper bounds.")
        logger.debug("PSOConfig validated: %d particles, %d iters, dim=%d",
                     self.n_particles, self.max_iter, self.bounds.shape[0])


# ---------------------------------------------------------------------------
# Evaluation context (shared state injected into PSO)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationContext:
    """Everything the PSO evaluator needs besides the parameter vector.

    Parameters
    ----------
    target_corr : np.ndarray
        Normalised empirical correlation matrix (no diagonal).
    sim_config_kwargs : dict
        Keyword arguments forwarded to :class:`SimulationConfig`.
    filter_kwargs : dict
        Keyword arguments forwarded to :func:`bandpass_filter`
        (``high_freq``, ``low_freq``, ``fs``, ``filter_order``).
    downsample_step : int
        Keep every *n*-th sample after filtering.
    op_corr : int
        1 = Pearson, 2 = cross-correlation.
    cross_corr_frac : float
        Fraction of signal length to use as max lag (cross-corr only).
    invalid_rois : List[int]
        Row/column indices to remove in coarse-graining step.
    op_net : int
        Network mode (2, 3, or 4); forwarded to SimulationConfig.
    op_model : int
        Model variant (1 or 2); forwarded to SimulationConfig.
    """
    target_corr: np.ndarray
    sim_config_kwargs: Dict[str, Any]
    filter_kwargs: Dict[str, Any]
    downsample_step: int = 20_000
    op_corr: int = 1
    cross_corr_frac: float = 0.2
    invalid_rois: List[int] = field(default_factory=lambda: [
        0, 1, 2, 41, 78, 79, 80, 81, 120, 157
    ])
    op_net: int = 3
    op_model: int = 1


# ---------------------------------------------------------------------------
# PSO class
# ---------------------------------------------------------------------------

class CorrelationPSO:
    """Particle Swarm Optimiser that minimises the MSE between the simulated
    and empirical functional-connectivity matrices.

    Parameters
    ----------
    pso_cfg : PSOConfig
        Swarm hyper-parameters and search bounds.
    eval_ctx : EvaluationContext
        Shared evaluation context (target matrix, simulation settings, …).
    seed : Optional[int]
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        pso_cfg: PSOConfig,
        eval_ctx: EvaluationContext,
        seed: Optional[int] = None,
    ) -> None:
        pso_cfg.validate()
        self._cfg = pso_cfg
        self._ctx = eval_ctx
        self._rng = np.random.default_rng(seed)

        D = pso_cfg.bounds.shape[0]
        lo, hi = pso_cfg.bounds[:, 0], pso_cfg.bounds[:, 1]

        self.positions: np.ndarray = self._rng.uniform(lo, hi, (pso_cfg.n_particles, D))
        self.velocities: np.ndarray = np.zeros((pso_cfg.n_particles, D))
        self.pbest_positions: np.ndarray = self.positions.copy()
        self.pbest_values: np.ndarray = np.array(
            [self._evaluate(p) for p in self.positions]
        )
        best_idx = int(np.argmin(self.pbest_values))
        self.gbest_position: np.ndarray = self.pbest_positions[best_idx].copy()
        self.gbest_value: float = float(self.pbest_values[best_idx])

        logger.info(
            "PSO initialised: %d particles, dim=%d, initial gbest=%.6f",
            pso_cfg.n_particles, D, self.gbest_value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, params: np.ndarray) -> float:
        """Return MSE loss for *params*; returns ``inf`` on any error."""
        ctx = self._ctx
        try:
            cfg = SimulationConfig(
                K=float(params[0]),
                a=float(params[1]),
                f=params[2:],
                op_net=ctx.op_net,
                op_model=ctx.op_model,
                **ctx.sim_config_kwargs,
            )
            if not cfg.use_cpp:
                logger.warning("C++ backend unavailable; skipping evaluation.")
                return float("inf")

            trajectory = run_simulation(cfg)

            # Band-pass filter
            """
            trajectory_filt = np.stack([
                bandpass_filter(trajectory[i], **ctx.filter_kwargs)
                for i in range(len(trajectory))
            ])
            """

            # Downsample
            #trajectory_filt = trajectory_filt[:, :: ctx.downsample_step]
            trajectory_filt = trajectory[:, :: ctx.downsample_step]
            #trajectory_filt = trajectory

            #print(trajectory_filt[0,:])
            print(trajectory_filt)
            print( np.shape( trajectory_filt ) )
            # Correlation
            corr = compute_correlation_matrix(
                trajectory_filt, mode=ctx.op_corr, frac=ctx.cross_corr_frac
            )

            print(corr)
            # Coarse-grain
            corr_cg = coarse_grain_matrix(corr, ctx.invalid_rois)

            print(corr_cg)
            # Normalise and compute MSE
            corr_norm = corr_cg / np.max(np.abs(corr_cg))
            
            print(corr_norm)
            mse = float(np.mean((corr_norm - ctx.target_corr) ** 2))
            return mse

        except Exception as exc:  # noqa: BLE001
            logger.warning("Evaluation failed (params=%s): %s", params, exc)
            return float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimise(self) -> Tuple[np.ndarray, float, List[float], List[np.ndarray]]:
        """Run the PSO loop.

        Returns
        -------
        gbest_position : np.ndarray
            Best parameter vector found.
        gbest_value : float
            Best (lowest) MSE found.
        error_history : list of float
            Global-best error at each iteration.
        position_history : list of np.ndarray
            Global-best position at each iteration.
        """
        cfg = self._cfg
        lo, hi = cfg.bounds[:, 0], cfg.bounds[:, 1]
        error_history: List[float] = []
        position_history: List[np.ndarray] = []

        for iteration in range(cfg.max_iter):
            iter_start = time.time()
            for i in range(cfg.n_particles):
                r1 = self._rng.random(self.positions.shape[1])
                r2 = self._rng.random(self.positions.shape[1])

                cognitive = cfg.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social    = cfg.c2 * r2 * (self.gbest_position     - self.positions[i])
                self.velocities[i] = (
                    cfg.w * self.velocities[i] + cognitive + social
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lo, hi)

                loss = self._evaluate(self.positions[i])

                if loss < self.pbest_values[i]:
                    self.pbest_values[i] = loss
                    self.pbest_positions[i] = self.positions[i].copy()

                    if loss < self.gbest_value:
                        self.gbest_value = loss
                        self.gbest_position = self.positions[i].copy()

            error_history.append(self.gbest_value)
            position_history.append(self.gbest_position.copy())

            elapsed = time.time() - iter_start
            logger.info(
                "Iteration %3d/%d | gbest=%.6f | wall=%.1fs",
                iteration + 1, cfg.max_iter, self.gbest_value, elapsed,
            )

        logger.info(
            "PSO complete. Best error=%.6f | params=%s",
            self.gbest_value, self.gbest_position,
        )
        return self.gbest_position, self.gbest_value, error_history, position_history


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def run_pso_optimisation(
    realization_index: str,
    op_corr: int,
    op_net: int,
    op_model: int,
    config_path: Optional[Path] = None,
) -> int:
    """Execute a full PSO optimisation run.

    Parameters
    ----------
    realization_index : str
        Identifier for this run (used in output subdirectory naming).
    op_corr : int
        Correlation mode: 1 = Pearson, 2 = cross-correlation.
    op_net : int
        Network connectivity mode: 2, 3, or 4.
    op_model : int
        Model variant: 1 (fixed frequencies) or 2 (connectivity-derived).
    config_path : Path, optional
        Path to ``config.json``.  Defaults to ``<project_root>/config/config.json``.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    run_start = time.time()
    logger.info("=" * 70)
    logger.info("PSO Optimisation START")
    logger.info(
        "realization=%s | op_corr=%d | op_net=%d | op_model=%d",
        realization_index, op_corr, op_net, op_model,
    )

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "config" / "config.json"

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        cfg_raw: Dict[str, Any] = json.load(fh)

    logger.info("Config loaded from %s", config_path)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    data_dir = Path(cfg_raw.get("data_dir", str(project_root / "data" / "processed")))
    signals_file = Path(cfg_raw.get("signals_file", str(data_dir / "signals.json")))
    output_base = Path(cfg_raw.get("output_dir", str(project_root / "results")))
    output_dir = output_base / f"M{op_net}_r{realization_index}_c{op_corr}_f{op_model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Load empirical signals
    # ------------------------------------------------------------------
    if not signals_file.exists():
        logger.error("Signals file not found: %s", signals_file)
        return 1

    with signals_file.open("r", encoding="utf-8") as fh:
        zsave = json.load(fh)

    #real_signals = np.array(zsave["signal_data"])
    real_signals = np.array( list( zsave['signal_data'].values() ) )
    tman: int = cfg_raw.get("tman_samples", 30)
    real_signals = real_signals[:, :tman]
    logger.info("Empirical signals loaded: shape=%s", real_signals.shape)

    # ------------------------------------------------------------------
    # Build target correlation matrix
    # ------------------------------------------------------------------
    invalid_rois: List[int] = cfg_raw.get(
        "invalid_rois", [0, 1, 2, 41, 78, 79, 80, 81, 120, 157]
    )
    cross_corr_frac: float = cfg_raw.get("cross_corr_frac", 0.2)

    print( np.shape( real_signals ) )
    target_corr_full = compute_correlation_matrix(
        real_signals, mode=op_corr, frac=cross_corr_frac
    )
    #target_corr_cg = coarse_grain_matrix(target_corr_full, invalid_rois)
    target_corr_cg = target_corr_full
    target_corr = target_corr_cg / np.max(np.abs(target_corr_cg))
    logger.info("Target correlation matrix built: shape=%s", target_corr.shape)

    # ------------------------------------------------------------------
    # Assemble PSO & evaluation configs from config.json
    # ------------------------------------------------------------------
    n_rois: int = cfg_raw.get("n_rois", 158)
    bounds_base = np.array(cfg_raw.get("bounds_base", [
        [1e4, 1e5], [-10.0, 10.0], [20.0, 60.0]
    ]))
    freq_bounds = np.tile(
        cfg_raw.get("freq_bounds", [20.0, 60.0]), (n_rois - 1, 1)
    )
    bounds = np.vstack([bounds_base, freq_bounds])

    pso_cfg = PSOConfig(
        n_particles=cfg_raw.get("n_particles", 4),
        max_iter=cfg_raw.get("max_iter", 10),
        w=cfg_raw.get("pso_w", 0.7),
        c1=cfg_raw.get("pso_c1", 1.5),
        c2=cfg_raw.get("pso_c2", 1.5),
        bounds=bounds,
    )

    filter_kwargs: Dict[str, Any] = {
        "high_freq": cfg_raw.get("filter_high_freq", 0.5),
        "low_freq":  cfg_raw.get("filter_low_freq",  0.01),
        "fs":        cfg_raw.get("filter_fs",        10_000.0),
        "filter_order": cfg_raw.get("filter_order",  50),
    }

    sim_config_kwargs: Dict[str, Any] = {
        "tmax":       cfg_raw.get("tmax", 60.0),
        "data_dir":   data_dir,
        "output_dir": output_dir,
        "use_cpp":    cfg_raw.get("use_cpp", True),
    }

    eval_ctx = EvaluationContext(
        target_corr=target_corr,
        sim_config_kwargs=sim_config_kwargs,
        filter_kwargs=filter_kwargs,
        downsample_step=cfg_raw.get("downsample_step", 20_000),
        op_corr=op_corr,
        cross_corr_frac=cross_corr_frac,
        invalid_rois=invalid_rois,
        op_net=op_net,
        op_model=op_model,
    )

    # ------------------------------------------------------------------
    # Run PSO
    # ------------------------------------------------------------------
    seed: Optional[int] = cfg_raw.get("seed", None)
    pso = CorrelationPSO(pso_cfg, eval_ctx, seed=seed)
    best_params, best_error, error_history, position_history = pso.optimise()

    # ------------------------------------------------------------------
    # Final simulation with best parameters
    # ------------------------------------------------------------------
    logger.info("Running final simulation with best parameters …")
    final_cfg = SimulationConfig(
        K=float(best_params[0]),
        a=float(best_params[1]),
        f=best_params[2:],
        op_net=op_net,
        op_model=op_model,
        **sim_config_kwargs,
    )

    if not final_cfg.use_cpp:
        logger.error("C++ backend required but unavailable. Aborting.")
        return 2

    trajectory = run_simulation(final_cfg)

    """
    trajectory_filt = np.stack([
        bandpass_filter(trajectory[i], **filter_kwargs)
        for i in range(len(trajectory))
    ])
    trajectory_filt = trajectory_filt[:, :: cfg_raw.get("downsample_step", 20_000)]
    """
    trajectory_filt = trajectory[:, :: cfg_raw.get("downsample_step", 20_000)]
    #trajectory_filt = trajectory

    print( np.shape( trajectory_filt ) )
    corr_final = compute_correlation_matrix(
        trajectory_filt, mode=op_corr, frac=cross_corr_frac
    )
    corr_cg = coarse_grain_matrix(corr_final, invalid_rois)
    corr_opt = corr_cg / np.max(np.abs(corr_cg))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    _save_results(
        output_dir=output_dir,
        best_params=best_params,
        best_error=best_error,
        error_history=error_history,
        position_history=position_history,
        target_corr=target_corr,
        corr_opt=corr_opt,
        realization_index=realization_index,
        op_corr=op_corr,
        op_net=op_net,
        op_model=op_model,
    )

    elapsed = time.time() - run_start
    logger.info("PSO Optimisation COMPLETE | wall=%.2f min", elapsed / 60)
    logger.info("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def _save_results(
    output_dir: Path,
    best_params: np.ndarray,
    best_error: float,
    error_history: List[float],
    position_history: List[np.ndarray],
    target_corr: np.ndarray,
    corr_opt: np.ndarray,
    realization_index: str,
    op_corr: int,
    op_net: int,
    op_model: int,
) -> None:
    """Persist optimisation results (parameters, arrays, plots)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Parameters text file
    param_file = output_dir / "optimal_parameters.txt"
    with param_file.open("w", encoding="utf-8") as fh:
        fh.write("# PSO Optimisation Results\n")
        fh.write(f"# realization={realization_index}, op_corr={op_corr}, "
                 f"op_net={op_net}, op_model={op_model}\n\n")
        fh.write("## Position history (iteration, params)\n")
        for idx, pos in enumerate(position_history):
            fh.write(f"{idx} - {pos.tolist()}\n")
        fh.write("\n## Best parameters\n")
        fh.write(f"K    = {best_params[0]}\n")
        fh.write(f"a    = {best_params[1]}\n")
        fh.write(f"freq = {best_params[2:].tolist()}\n")
        fh.write(f"\n## Best MSE = {best_error}\n")
    logger.info("Parameters saved to %s", param_file)

    # Numpy arrays
    np.save(str(output_dir / "target_correlation.npy"), target_corr)
    np.save(str(output_dir / "optimal_correlation.npy"), corr_opt)
    np.save(str(output_dir / "error_history.npy"), np.array(error_history))
    logger.info("Numpy arrays saved to %s", output_dir)

    # JSON summary
    summary = {
        "realization": realization_index,
        "op_corr": op_corr,
        "op_net": op_net,
        "op_model": op_model,
        "best_error": best_error,
        "best_K": float(best_params[0]),
        "best_a": float(best_params[1]),
        "best_freq": best_params[2:].tolist(),
        "n_iterations": len(error_history),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary JSON saved.")

    # Plots
    def _heatmap(matrix: np.ndarray, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, cmap="coolwarm", vmin=-1, vmax=1,
                    square=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        fig.tight_layout()
        fig.savefig(str(output_dir / filename), dpi=150)
        plt.close(fig)

    _heatmap(target_corr, "Target Correlation Matrix", "target_correlation.png")
    _heatmap(corr_opt,    "Optimal Correlation Matrix", "optimal_correlation.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(error_history, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_title("PSO Convergence")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "convergence.png"), dpi=150)
    plt.close(fig)
    logger.info("Figures saved to %s", output_dir)
