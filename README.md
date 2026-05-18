# neuro-sl-optimizer

> **Stuart-Landau Neural Network Simulator with PSO Functional-Connectivity Fitting**

A research pipeline that optimises a coupled network of noisy Stuart-Landau oscillators so that its simulated functional connectivity (FC) matches an empirical resting-state fMRI target. The optimisation backbone is **Particle Swarm Optimisation (PSO)**; the forward simulation is accelerated by a **C++ / pybind11** extension.

---

## Key Features

| Feature | Details |
|---|---|
| **C++ accelerated simulator** | pybind11 extension (`stuart_landau_simulator`) for fast ODE integration with axonal delays |
| **Python fallback** | Pure-NumPy simulator when the C++ build is unavailable |
| **PSO optimiser** | Optimises `[K, a, ω₀ … ωₙ]` to minimise MSE between simulated and empirical FC |
| **Pearson & cross-correlation FC** | Selectable via `--op-corr` flag |
| **Three network modes** | Velocity delays (2), tau-matrix delays (3), bimodal connectivity (4) |
| **Structured logging** | Every script logs to `logs/run.log` |
| **Docker-ready** | One-command containerised runs |
| **CI/CD** | GitHub Actions workflow included |

---

## Quick Links

| Document | Description |
|---|---|
| [QUICKSTART.md](docs/QUICKSTART.md) | Get running in minutes |
| [installation.md](docs/installation.md) | Full installation guide (Conda, pip, Docker) |
| [api.md](docs/api.md) | Python API reference |
| [Docker.md](docs/Docker.md) | Docker & docker-compose guide |
| [GitHub.md](docs/GitHub.md) | Contributing & CI/CD |

---

## Project Layout

```
neuro-sl-optimizer/
├── config/
│   └── config.json              # All tunable parameters
├── data/
│   ├── raw/                     # Connectivity matrices (*.txt)
│   └── processed/               # Empirical signals (signals.json)
├── docs/                        # Documentation
├── logs/                        # Runtime logs (run.log)
├── results/                     # PSO output (per-run subdirectories)
├── scripts/
│   ├── run_pso.py               # Python CLI entry point
│   └── run_pso.sh               # Bash wrapper with validation
├── source/
│   ├── analysis/
│   │   ├── functional_connectivity.py
│   │   └── signal_processing.py
│   ├── core/
│   │   ├── pso_optimizer.py
│   │   └── simulation_engine.py
│   └── utils/
│       ├── config_validator.py
│       └── logging_utils.py
├── validations/
│   ├── validate_config.py
│   ├── validate_signal_processing.py
│   ├── test_pso_optimizer.py
│   └── test_signal_processing.py
├── Dockerfile
├── docker-compose.yml
├── environment.yml
├── setup.py
└── .github/workflows/ci.yml
```

---

## Minimal Usage

```bash
# 1. Activate environment
conda activate pso_opt_env

# 2. Build C++ extension
python setup.py build_ext --inplace

# 3. Edit config
nano config/config.json

# 4. Run
python scripts/run_pso.py --realization 1 --op-corr 1 --op-net 3 --op-model 1
```

See [QUICKSTART.md](docs/QUICKSTART.md) for detailed instructions.

---

## Citation

If you use this software in your research, please cite the original work by Alejandro Aguado and collaborators.

## License

MIT — see `LICENSE`.
