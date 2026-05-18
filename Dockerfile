# =============================================================================
# Dockerfile
# neuro-sl-optimizer
# =============================================================================
# Multi-stage build:
#   builder  – installs system deps + Conda environment + compiles C++ ext
#   runtime  – lean image for production runs
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM continuumio/miniconda3:23.10.0-1 AS builder

LABEL maintainer="Alejandro Aguado"
LABEL description="Stuart-Landau PSO functional-connectivity optimiser"

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libomp-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency definitions first (better layer caching)
COPY environment.yml setup.py ./
COPY source/ ./source/

# Create Conda environment
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make conda env the default Python
SHELL ["conda", "run", "-n", "pso_opt_env", "/bin/bash", "-c"]

# Build C++ extension
COPY stuart_landau_simulator.cpp ./source/core/
RUN python setup.py build_ext --inplace

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM continuumio/miniconda3:23.10.0-1 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libomp5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built Conda environment from builder
COPY --from=builder /opt/conda/envs/pso_opt_env /opt/conda/envs/pso_opt_env

# Copy the application source and compiled extension
COPY --from=builder /app /app

# Make sure the Conda environment is active by default
ENV PATH="/opt/conda/envs/pso_opt_env/bin:$PATH"
ENV CONDA_DEFAULT_ENV="pso_opt_env"

# Create runtime directories (will be overridden by volume mounts)
RUN mkdir -p /app/data/raw /app/data/processed \
             /app/results /app/logs /app/config

# Default environment variables (override at runtime)
ENV PSO_REALIZATION="1"
ENV PSO_OP_CORR="1"
ENV PSO_OP_NET="3"
ENV PSO_OP_MODEL="1"
ENV PSO_LOG_LEVEL="INFO"

# Run validation on container start to confirm the build is healthy
RUN python -c "import numpy, scipy, matplotlib; print('Scientific stack OK')" \
    && python -c "import stuart_landau_simulator; print('C++ backend OK')" || \
       echo "WARNING: C++ backend not available; pure-Python fallback will be used."

# Default command — can be overridden in docker run / docker-compose
CMD ["bash", "-c", \
     "python scripts/run_pso.py \
         --realization ${PSO_REALIZATION} \
         --op-corr ${PSO_OP_CORR} \
         --op-net ${PSO_OP_NET} \
         --op-model ${PSO_OP_MODEL} \
         --log-level ${PSO_LOG_LEVEL}"]
