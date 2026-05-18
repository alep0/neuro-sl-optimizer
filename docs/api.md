# API Reference

## Overview

The public Python API is organised into three layers:

| Layer | Module | Responsibility |
|---|---|---|
| **Entry point** | `scripts/run_pso.py` | CLI argument parsing, logging setup |
| **Optimiser** | `source/core/pso_optimizer.py` | PSO loop, loss evaluation |
| **Simulator** | `source/core/simulation_engine.py` | Stuart-Landau ODE integration |
| **Analysis** | `source/analysis/` | Signal filtering, FC computation |
| **Utilities** | `source/utils/` | Logging, config validation |

---

## `source.core.pso_optimizer`

### `PSOConfig`

```python
@dataclass
class PSOConfig:
    n_particles: int = 4
    max_iter:    int = 10
    w:           float = 0.7    # inertia weight
    c1:          float = 1.5    # cognitive coefficient
    c2:          float = 1.5    # social coefficient
    bounds:      np.ndarray     # shape (D, 2)  lower/upper per dimension
```

**Methods**

- `validate() -> None` — raises `ValueError` on invalid configuration.

---

### `EvaluationContext`

```python
@dataclass
class EvaluationContext:
    target_corr:        np.ndarray          # empirical FC matrix
    sim_config_kwargs:  Dict[str, Any]      # forwarded to SimulationConfig
    filter_kwargs:      Dict[str, Any]      # forwarded to bandpass_filter
    downsample_step:    int   = 20_000
    op_corr:            int   = 1           # 1=Pearson, 2=cross-corr
    cross_corr_frac:    float = 0.2
    invalid_rois:       List[int]           # ROI indices to exclude
    op_net:             int   = 3
    op_model:           int   = 1
```

---

### `CorrelationPSO`

```python
class CorrelationPSO:
    def __init__(
        self,
        pso_cfg:  PSOConfig,
        eval_ctx: EvaluationContext,
        seed:     Optional[int] = None,
    ): ...

    def optimise(self) -> Tuple[
        np.ndarray,   # gbest_position
        float,        # gbest_value (MSE)
        List[float],  # error_history  [n_iter]
        List[np.ndarray],  # position_history  [n_iter]
    ]: ...
```

---

### `run_pso_optimisation`

```python
def run_pso_optimisation(
    realization_index: str,
    op_corr:           int,
    op_net:            int,
    op_model:          int,
    config_path:       Optional[Path] = None,
) -> int:
    """
    Top-level convenience function.

    Returns 0 on success, 1 on configuration error, 2 if C++ is required
    but unavailable.
    """
```

**Example:**

```python
from source.core.pso_optimizer import run_pso_optimisation

exit_code = run_pso_optimisation(
    realization_index="1",
    op_corr=1,
    op_net=3,
    op_model=1,
)
```

---

## `source.core.simulation_engine`

### `SimulationConfig`

```python
@dataclass
class SimulationConfig:
    K:          float = 1e5        # global coupling strength
    f:          np.ndarray         # oscillation frequencies (Hz), shape (N,)
    a:          float = -5.0       # bifurcation parameter
    sig_noise:  float = 1e-3       # noise amplitude
    tmax:       float = 60.0       # simulation duration (s)
    t_prev:     float = 0.0        # warm-up period (s)
    dt:         float = 1e-4       # integration step (s)
    dt_save:    float = 1e-4       # output sampling interval (s)
    mean_vel:   float = 5.8        # axonal velocity (m/s)
    op_net:     int = 3            # 2=velocity, 3=tau, 4=bimodal
    op_model:   int = 1            # 1=fixed freq, 2=derived
    data_dir:   Optional[Path]     # connectivity files directory
    output_dir: Optional[Path]     # results directory
    save_data:  bool = False       # save trajectory.npy
    use_cpp:    bool = True        # attempt C++ backend
```

### `run_simulation`

```python
def run_simulation(config: Optional[SimulationConfig] = None) -> np.ndarray:
    """
    Run a Stuart-Landau network simulation.

    Returns
    -------
    trajectory : np.ndarray, shape (N_nodes, n_time_points)
        Real part of oscillator trajectories sampled at dt_save.
    """
```

---

## `source.analysis.signal_processing`

### `bandpass_filter`

```python
def bandpass_filter(
    signal:       np.ndarray,
    high_freq:    float,
    low_freq:     float,
    fs:           float,
    filter_order: int = 50,
) -> np.ndarray:
    """Butterworth band-pass filter. Returns filtered signal of same shape."""
```

### `normalise_signal`

```python
def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise. Returns zeros if std == 0."""
```

---

## `source.analysis.functional_connectivity`

### `compute_correlation_matrix`

```python
def compute_correlation_matrix(
    signals: np.ndarray,   # shape (N, T)
    mode:    int = 1,      # 1=Pearson, 2=cross-corr
    frac:    float = 0.2,  # max-lag fraction (mode=2 only)
) -> np.ndarray:           # shape (N, N), diagonal zeroed
```

### `coarse_grain_matrix`

```python
def coarse_grain_matrix(
    matrix:          np.ndarray,   # shape (N, N)
    invalid_indices: List[int],    # row/col indices to remove
) -> np.ndarray:                   # shape (N-k, N-k)
```

---

## `source.utils.logging_utils`

### `setup_logging`

```python
def setup_logging(
    log_dir:      Optional[Path] = None,
    log_filename: str = "run.log",
    level:        int = logging.INFO,
    console:      bool = True,
) -> None:
```

### `get_logger`

```python
def get_logger(name: str) -> logging.Logger:
```

---

## `source.utils.config_validator`

### `load_and_validate_config`

```python
def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate config.json. Raises ConfigValidationError on failure."""
```

### `validate_config`

```python
def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate an already-loaded config dict. Raises ConfigValidationError."""
```
