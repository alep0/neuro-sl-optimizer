# QUICKSTART

Get the PSO functional-connectivity pipeline running in five minutes. (folded connectome, multiarch, Eigen)

---

## Prerequisites

- Git
- One of: **Conda** (recommended), plain Python ≥ 3.9, or **Docker**
- A C++ compiler with OpenMP support (GCC ≥ 9, Clang ≥ 11, or MSVC ≥ 2019)

---

## Option A — Conda (recommended)

### 1. Clone

```bash
git clone https://github.com/your-org/neuro-sl-optimizer.git
cd neuro-sl-optimizer
```

### 2. Create and activate environment

```bash
conda env create -f config/environment.yml
conda activate pso_opt_env
```

The `environment.yml` pins all scientific dependencies (NumPy, SciPy,
Matplotlib, Seaborn, pybind11).

### 3. Build the C++ extension

```bash

bash scripts/build_multiarch.sh

python3 setup_multiarch.py build_ext --inplace
python source/core/select_backend.py
```

Verify the build:

```bash
python -c "import stuart_landau_simulator; print('C++ backend OK')"
```

If the build fails, the pipeline will fall back to the pure-Python simulator
(slower but functionally identical).

### 4. Prepare your data

Place your connectivity matrices in `data/raw/` and your empirical signal
file at the path specified by `signals_file` in `config/config.json`.

Default expected layout:

```
data/
├── raw/
│   ├── th-0.0_R01_w.txt
│   ├── th-0.0_R01_tau.txt
│   └── th-0.0_R01_v.txt
└── processed/
    └── signals.json          ← {"signal_data": [[...], ...]}
```

DATA
scp -r /mnt/c/Users/aleph/Desktop/Katana/Katana_MSI_Windows_/Job/Code/Multiarch/neuro_sl_optimizer/data aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer

CHECKPOINT
rat=R02
mkdir -p /data/workspaces/aaaguado/experimental_multiarch/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r1_c1_f1
cp /data/workspaces/aaaguado/experimental_combined/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r1_c1_f1/checkpoint_iter_0499.* /data/workspaces/aaaguado/experimental_multiarch/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r1_c1_f1

rat=R16
mkdir -p /data/workspaces/aaaguado/experimental_multiarch/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r1_c1_f1
cp /data/workspaces/aaaguado/experimental_combined/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r3_c1_f1/checkpoint_iter_0499.* /data/workspaces/aaaguado/experimental_multiarch/neuro-sl-optimizer/results/checkpoints_C/results_t1_500_20_${rat}/M3_r1_c1_f1

FINE-TUNING
mkdir -p /data/workspaces/aaaguado/experimental_fine-tuning/neuro-sl-optimizer/data/external/t1/R15
cp -r /data/workspaces/aaaguado/experimental_fine-tuning/neuro-sl-optimizer/data/raw/t1/R15 /data/workspaces/aaaguado/experimental_fine-tuning/neuro-sl-optimizer/data/external/t1

DOWNLOAD
ssh aaaguado@ifisc.uib-csic.es "cd /data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results \
&& tar --exclude='*.npz' --exclude='*.pkl' -cvf - ." | tar -xf - -C /mnt/c/Users/aleph/Desktop/Nuredduna

ssh aaaguado@ifisc.uib-csic.es "cd /data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/checkpoints_C && tar --exclude='*.npz' --exclude='*.pkl' -cvf - ." | tar -xf - -C /mnt/c/Users/aleph/Desktop/Nuredduna/checkpoints_C

ssh aaaguado@ifisc.uib-csic.es "cd /data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_C/results_t1_2000_20_R14/M3_r3_c1_f1 && tar --exclude='*.npz' --exclude='*.pkl' -cvf - ." | tar -xf - -C /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_R14

rat=R19
r=1
file=optimal_correlation.png
mkdir -p /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1
file=target_correlation.png
mkdir -p /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1

file=th-0.0_${rat}_w_gen.png
mkdir -p /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1

file=FC_iteration_14*
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_C/results_t1_2000_20_${rat}/M3_r${r}_c1_f1

ls /data/workspaces/aaaguado/experimental_combined/neuro-sl-optimizer/results/optimization_C/results_t1_500_20_R16/M3_r3_c1_f1


file=checkpoints_t2_C
mkdir -p /mnt/c/Users/aleph/Desktop/Nuredduna/${file}
ssh aaaguado@ifisc.uib-csic.es "cd /data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/${file} && tar --exclude='*.npz' --exclude='*.pkl' -cvf - ." | tar -xf - -C /mnt/c/Users/aleph/Desktop/Nuredduna/${file}

rat=R19
r=5
iterations=1000

mkdir -p /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1

file=target_correlation.png
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1

file=optimal_correlation.png
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1

file=th-0.0_${rat}_w_gen.png
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1


file=FC_iteration_9*
scp aaaguado@ifisc.uib-csic.es:/data/workspaces/aaaguado/experimental_folding/neuro-sl-optimizer/results/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1/${file} /mnt/c/Users/aleph/Desktop/Nuredduna/optimization_t2_C/results_t2_${iterations}_20_${rat}/M3_r${r}_c1_f1

### 5. Edit configuration

```bash
nano config/config.json
```

Key fields to review:

| Key | Default | Description |
|---|---|---|
| `signals_file` | `data/processed/signals.json` | Path to empirical signals |
| `data_dir` | `data/processed` | Directory with connectivity files |
| `n_particles` | `4` | PSO swarm size |
| `max_iter` | `10` | PSO iterations |
| `tmax` | `60.0` | Simulation duration (s) |
| `use_cpp` | `true` | Use C++ backend |

### 6. Validate configuration

```bash
python validations/validate_config.py
```

### 7. Run the optimisation

**Python CLI (recommended):**

```bash
python scripts/run_pso.py \
    --realization 1 \
    --op-corr 1 \
    --op-net 3 \
    --op-model 1
```

**Bash wrapper:**

```bash
chmod +x Nuredduna_run_pso.sh
chmod +x scripts/run_pso.sh

nano config/config.json

bash scripts/run_pso.sh --rats "R01 R02" --realizations "1 2" --op-corr 1 --op-net 3 --op-model 1
bash scripts/run_pso.sh --rats "R03 R04 R05 R06 R07 R08 R09 R10 R12 R13 R14 R15 R16 R17 R18 R19" \
                        --realizations "1" --op-corr 1 --op-net 3 --op-model 1

./Nuredduna_run_pso.sh "R01 R02" "1 2" "1" "3" "1"
./Nuredduna_run_pso.sh "R03 R04 R05 R06 R07 R08 R09 R10 R12 R13 R14 R15 R16 R17 R18 R19" \
                        "1 2 3 4 5 6 7 8 9 10" \
                        "1" "3" "1"

```

**Multiple realizations in one command:**

```bash
python scripts/run_pso.py --realization 1 2 3 4 5 --op-corr 1 --op-net 3
```

### 8. Check results

Results are written to `results/M<op_net>_r<realization>_c<op_corr>_f<op_model>/`:

```
results/M3_r1_c1_f1/
├── optimal_parameters.txt
├── summary.json
├── target_correlation.npy
├── optimal_correlation.npy
├── error_history.npy
├── target_correlation.png
├── optimal_correlation.png
└── convergence.png
```

Logs are written to `logs/run.log`.

---

## Option B — Docker

```bash
# Build image
docker build -t neuro-sl-optimizer .

# Run a single realization
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/logs:/app/logs" \
    neuro-sl-optimizer \
    python scripts/run_pso.py --realization 1 --op-corr 1 --op-net 3 --op-model 1
```

Or with docker-compose:

```bash
docker-compose up
```

See [Docker.md](Docker.md) for full details.

---

## Option C — pip (no Conda)

```bash
pip install numpy scipy matplotlib seaborn pybind11
python setup.py build_ext --inplace
```

---

## Running Tests

```bash
pytest validations/ -v
```

Or run individual validation scripts:

```bash
python validations/validate_signal_processing.py
python validations/validate_config.py
```
