# Docker Guide

The project ships with a `Dockerfile` and `docker-compose.yml` so you can run
the full pipeline in an isolated, reproducible container without installing
any dependencies locally.

---

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) ≥ 24
- [docker compose](https://docs.docker.com/compose/install/) ≥ 2 (usually
  bundled with Docker Desktop)

---

## Quick Start

### 1. Build the image

```bash
docker build -t neuro-sl-optimizer:latest .
```

This single command:

1. Installs Miniconda and creates the `pso_opt_env` Conda environment.
2. Compiles the C++ pybind11 extension.
3. Runs the validation suite to confirm the build is healthy.

### 2. Run a single realization

```bash
docker run --rm \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/logs:/app/logs" \
    -v "$(pwd)/config:/app/config:ro" \
    neuro-sl-optimizer \
    python scripts/run_pso.py \
        --realization 1 \
        --op-corr 1 \
        --op-net 3 \
        --op-model 1
```

### 3. Run with docker-compose

Edit `docker-compose.yml` if you need to change environment variables or
mount different data paths, then:

```bash
docker-compose up
```

To run in the background:

```bash
docker-compose up -d
docker-compose logs -f
```

---

## Volume Mounts

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `./data` | `/app/data` | read-only | Connectivity matrices and signals |
| `./results` | `/app/results` | read-write | PSO output (auto-created) |
| `./logs` | `/app/logs` | read-write | Log files |
| `./config` | `/app/config` | read-only | `config.json` |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PSO_REALIZATION` | `1` | Realization ID passed to `run_pso.py` |
| `PSO_OP_CORR` | `1` | Correlation mode (1 or 2) |
| `PSO_OP_NET` | `3` | Network mode (2, 3, or 4) |
| `PSO_OP_MODEL` | `1` | Model variant (1 or 2) |
| `PSO_LOG_LEVEL` | `INFO` | Logging verbosity |

Override from the command line:

```bash
docker-compose run \
    -e PSO_REALIZATION=5 \
    -e PSO_OP_NET=2 \
    pso_optimizer
```

---

## Multi-GPU / Multi-Core

The C++ extension uses OpenMP automatically.  Set the number of threads:

```bash
docker run --rm \
    -e OMP_NUM_THREADS=8 \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/results:/app/results" \
    neuro-sl-optimizer \
    python scripts/run_pso.py --realization 1
```

---

## Rebuilding After Code Changes

```bash
docker-compose build --no-cache
```

Or for faster iteration during development:

```bash
docker build --target builder -t neuro-sl-optimizer:dev .
```

---

## Troubleshooting

### `Exec format error` on Apple Silicon

Build with the `--platform` flag to target AMD64:

```bash
docker build --platform linux/amd64 -t neuro-sl-optimizer .
```

### C++ extension build failure inside Docker

Check that the base image has `gcc` and `libomp-dev`.  The provided
`Dockerfile` installs these automatically.

### Results directory not created

Docker creates the volume mount directory on the host automatically; ensure
the host user has write permission:

```bash
mkdir -p results logs
chmod 777 results logs
```
