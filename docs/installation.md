# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Linux, macOS, Windows 10 | Ubuntu 22.04 LTS |
| Python | 3.9 | 3.11 |
| RAM | 4 GB | 16 GB+ |
| CPU | 4 cores | 8+ cores (OpenMP) |
| C++ compiler | GCC 9 / Clang 11 / MSVC 2019 | GCC 12 |

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/neuro-sl-optimizer.git
cd neuro-sl-optimizer
```

---

## 2. Python Environment

### Via Conda (recommended)

```bash
conda env create -f config/environment.yml
conda activate pso_opt_env
```

To update an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

### Via pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt    # generate from environment.yml if needed
```

---

## 3. Build the C++ Extension

The C++ accelerated simulator must be compiled before first use.

```bash
python setup.py build_ext --inplace
```

### Linux (GCC + OpenMP)

```bash
# Install build tools if needed
sudo apt-get install build-essential libomp-dev

python setup.py build_ext --inplace
```

### macOS (Clang + Homebrew libomp)

```bash
brew install libomp
python setup.py build_ext --inplace
```

### Windows (MSVC)

Open a *Developer Command Prompt for Visual Studio*, then:

```cmd
python setup.py build_ext --inplace
```

### Verify the build

```bash
python -c "import stuart_landau_simulator; print('OK')"
```

If this fails the pipeline will automatically use the pure-Python fallback
(identical results, ~10–100× slower depending on problem size).

---

## 4. Install as an editable package (optional)

```bash
pip install -e .
```

This allows importing `source.*` modules from anywhere in the project.

---

## 5. Verify the full installation

```bash
# Run unit tests
pytest validations/ -v

# Validate configuration
python validations/validate_config.py

# Validate signal processing utilities
python validations/validate_signal_processing.py
```

All tests should pass before running any optimisation.

---

## 6. Docker (alternative)

See [Docker.md](Docker.md).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'stuart_landau_simulator'`

The C++ extension was not built or the build artefact is not on `sys.path`.
Run `python setup.py build_ext --inplace` from the project root.

### OpenMP not found on macOS

```bash
brew install libomp
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
python setup.py build_ext --inplace
```

### Permission denied on `run_pso.sh`

```bash
chmod +x scripts/run_pso.sh
```
