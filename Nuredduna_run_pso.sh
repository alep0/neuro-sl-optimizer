#!/usr/bin/env sh
# =============================================================================
# Nuredduna_run_pso.sh  (multi-arch edition)
# =============================================================================
# Submits PSO jobs to SLURM, automatically selecting the right SLURM
# constraint for the compiled ISA tier — or using the multi-arch build
# that picks the best .so at runtime.
#
# Strategy (choose one via ARCH_STRATEGY below):
#
#   "runtime"   — always submit with no --constraint; each node imports the
#                 best .so it can run.  Needs the multi-arch build in place.
#                 Recommended: simplest, works everywhere.
#
#   "probe"     — submit a 1-minute probe job first to find what SLURM
#                 constraints are available on idle nodes, then pin future
#                 jobs to compatible partitions/features.
#
#   "constraint"— hard-code a SLURM --constraint (e.g. avx2).  Fastest
#                 turnaround when you already know which nodes to target.
#
# Usage:
#   ./scripts/Nuredduna_run_pso.sh <rats> <realizations> <op_corr> <op_net> <op_model>
#
# Example:
#   ./scripts/Nuredduna_run_pso.sh "R01" "1 2" "1" "3" "1"
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Choose: "runtime" | "constraint" | "probe"
ARCH_STRATEGY="runtime"

# Used only when ARCH_STRATEGY="constraint"
# Valid SLURM feature values on your cluster (check with: sinfo -o "%f" | sort -u)
SLURM_CONSTRAINT=""        # e.g. "avx2" or "avx512" — leave empty for none

# Retry / polling
SLEEP_BETWEEN_POLLS=5
MAX_ATTEMPTS=0             # 0 = retry indefinitely

# SLURM resource request
JOB_TIME="123:30"
JOB_CPUS=1
JOB_MEM=16                 # GB

# ---------------------------------------------------------------------------
# Parse positional args
# ---------------------------------------------------------------------------
echo "$1 $2 $3 $4 $5"

rats=$1
rea_list=$2
oc=$3
on=$4
om=$5

# ---------------------------------------------------------------------------
# Build the --constraint flag string (empty if not needed)
# ---------------------------------------------------------------------------
constraint_flag=""
if [ "$ARCH_STRATEGY" = "constraint" ] && [ -n "$SLURM_CONSTRAINT" ]; then
    constraint_flag="--constraint=$SLURM_CONSTRAINT"
    echo "[arch] Using SLURM constraint: $SLURM_CONSTRAINT"
elif [ "$ARCH_STRATEGY" = "runtime" ]; then
    echo "[arch] Using runtime ISA selection (multi-arch build)."
fi

# ---------------------------------------------------------------------------
# Main submission loop
# ---------------------------------------------------------------------------
for r in $rea_list; do
  for rat in $rats; do

    attempt=0

    while :; do
      attempt=$((attempt + 1))
      echo "[submit] rat=$rat rea=$r attempt=$attempt"

      MY_JOB="./scripts/run_pso.sh --rats \"$rat\" --realizations \"$r\" --op-corr \"$oc\" --op-net \"$on\" --op-model \"$om\""

      # Build run command (cluster-specific scheduler)
      # shellcheck disable=SC2086
      SUB_OUT=$(run \
          -t "$JOB_TIME" \
          -c "$JOB_CPUS" \
          -m "$JOB_MEM" \
          -j "run_pso_c${JOB_CPUS}_m${JOB_MEM}_${rat}" \
          $constraint_flag \
          "$MY_JOB" 2>&1)

      RETCODE=$?
      echo "$SUB_OUT"

      if [ $RETCODE -ne 0 ]; then
        echo "[error] run returned code $RETCODE"
        if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
          echo "[abort] Max attempts reached for rat=$rat rea=$r"
          break
        fi
        sleep $SLEEP_BETWEEN_POLLS
        continue
      fi

      JOBID=$(printf "%s\n" "$SUB_OUT" | awk '/Submitted batch job/ {print $NF; exit}')
      if [ -z "$JOBID" ]; then
        echo "[error] Could not parse job ID from: $SUB_OUT"
        if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
          echo "[abort] Max attempts reached for rat=$rat rea=$r"
          break
        fi
        sleep $SLEEP_BETWEEN_POLLS
        continue
      fi

      LOGFILE_e="$(pwd)/scripts/run_pso.sh.e${JOBID}"
      LOGFILE_o="$(pwd)/scripts/run_pso.sh.o${JOBID}"
      echo "[wait] Job $JOBID submitted. Polling for $LOGFILE_o ..."

      # Wait for output log to appear
      while [ ! -f "$LOGFILE_o" ]; do
        sleep $SLEEP_BETWEEN_POLLS
      done

      # Poll log for success / failure patterns
      job_done=false
      job_retry=false

      while :; do
        # ---- Hard failure: illegal instruction (wrong ISA) ----------------
        # With the multi-arch build this should no longer happen, but we keep
        # the guard for safety.
        if grep -q -E "Illegal instruction" "$LOGFILE_e" 2>/dev/null; then
          echo "[warn] Illegal instruction on job $JOBID. Node incompatible with any .so?"
          echo "       Check that build_multiarch.sh has been run and baseline .so exists."
          job_retry=true
          break
        fi

        # ---- Success: C++ backend loaded (any ISA tier) -------------------
        if grep -q -E "| INFO     | source.core.simulation_engine | Backend: C++ (accelerated) " "$LOGFILE_e" 2>/dev/null; then
          tier=$(grep -oE "Backend: C\+\+ \(accelerated\) \[.*\]" "$LOGFILE_e" | head -1)
          echo "[ok] Job $JOBID succeeded. $tier  rat=$rat rea=$r"
          job_done=true
          break
        fi

        # ---- Soft failure: fell back to pure Python -----------------------
        # This means no .so matched — rebuild or check paths.
        if grep -q -E "| INFO     | source.core.simulation_engine | Backend: Python (pure NumPy)" "$LOGFILE_e" 2>/dev/null; then
          echo "[warn] Job $JOBID running on pure-Python backend."
          echo "       Performance severely degraded. Check multi-arch build."
          # Treat as success (result is still valid) — remove 'job_retry=true'
          # if you prefer to accept slow results rather than retry forever.
          job_done=true
          break
        fi

        # ---- Job left the queue without a clear signal --------------------
        if ! squeue -j "$JOBID" -h >/dev/null 2>&1; then
          echo "[warn] Job $JOBID left queue with no recognised pattern."
          echo "       Last 20 lines of $LOGFILE_e:"
          tail -n 20 "$LOGFILE_e" 2>/dev/null || true
          job_retry=true
          break
        fi

        sleep $SLEEP_BETWEEN_POLLS
      done

      if [ "$job_done" = "true" ]; then
        break   # move to next rat/rea
      fi

      # Retry limit check
      if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "[abort] Max attempts ($MAX_ATTEMPTS) reached for rat=$rat rea=$r"
        break
      fi

      sleep $SLEEP_BETWEEN_POLLS
    done  # while retry

  done  # for rat
done  # for rea
