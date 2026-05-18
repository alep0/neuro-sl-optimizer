#!/usr/bin/env bash
# =============================================================================
# run_pso.sh
# =============================================================================
# Bash wrapper for the PSO functional-connectivity optimisation pipeline.
# Replaces the legacy PSO_step_v1.sh with proper error handling, logging,
# input validation, and configurable parameters.
#
# Usage:
#   ./scripts/run_pso.sh [OPTIONS]
#
# Options:
#   -r, --realizations  "1 2 3"   Space-separated list of realization IDs
#                                  (default: "1")
#   -c, --op-corr       1|2       Correlation mode (default: 1)
#   -n, --op-net        2|3|4     Network mode (default: 3)
#   -m, --op-model      1|2       Model variant (default: 1)
#   -C, --config        PATH      Path to config.json
#   -l, --log-level     DEBUG|INFO|WARNING|ERROR  (default: INFO)
#   -h, --help                    Print this help message
#
# Example:
#   ./scripts/run_pso.sh --realizations "1 2 3" --op-corr 1 --op-net 3 --op-model 1
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REALIZATIONS="1"
OP_CORR=1
OP_NET=3
OP_MODEL=1
CONFIG_ARG=""
LOG_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/run.log"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
_timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log_info()    { echo "$(_timestamp) | INFO     | run_pso.sh | $*" | tee -a "$LOG_FILE"; }
log_warning() { echo "$(_timestamp) | WARNING  | run_pso.sh | $*" | tee -a "$LOG_FILE" >&2; }
log_error()   { echo "$(_timestamp) | ERROR    | run_pso.sh | $*" | tee -a "$LOG_FILE" >&2; }

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \?//' | sed '1d'
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -r|--realizations)
                REALIZATIONS="$2"; shift 2 ;;
            -c|--op-corr)
                OP_CORR="$2"; shift 2 ;;
            -n|--op-net)
                OP_NET="$2"; shift 2 ;;
            -m|--op-model)
                OP_MODEL="$2"; shift 2 ;;
            -C|--config)
                CONFIG_ARG="--config $2"; shift 2 ;;
            -l|--log-level)
                LOG_LEVEL="$2"; shift 2 ;;
            -h|--help)
                usage ;;
            *)
                log_error "Unknown argument: $1"
                usage ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
validate_inputs() {
    local valid=true

    if ! [[ "$OP_CORR" =~ ^[12]$ ]]; then
        log_error "op-corr must be 1 or 2, got: $OP_CORR"
        valid=false
    fi

    if ! [[ "$OP_NET" =~ ^[234]$ ]]; then
        log_error "op-net must be 2, 3, or 4, got: $OP_NET"
        valid=false
    fi

    if ! [[ "$OP_MODEL" =~ ^[12]$ ]]; then
        log_error "op-model must be 1 or 2, got: $OP_MODEL"
        valid=false
    fi

    if ! [[ "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
        log_error "log-level must be DEBUG|INFO|WARNING|ERROR, got: $LOG_LEVEL"
        valid=false
    fi

    if [[ -n "${CONFIG_ARG}" ]]; then
        cfg_path="${CONFIG_ARG#--config }"
        if [[ ! -f "$cfg_path" ]]; then
            log_error "Config file not found: $cfg_path"
            valid=false
        fi
    fi

    if [[ "$valid" == false ]]; then
        log_error "Validation failed. Exiting."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Check Python environment
# ---------------------------------------------------------------------------
check_environment() {
    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found. Please activate your conda environment."
        exit 1
    fi

    local py_version
    py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $py_version"

    if ! python3 -c "import numpy, scipy, matplotlib" &>/dev/null; then
        log_error "Required Python packages missing. Run: conda env create -f environment.yml"
        exit 1
    fi
    log_info "Python environment OK."
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    log_info "============================================================"
    log_info "PSO Pipeline START"
    log_info "realizations: $REALIZATIONS"
    log_info "op_corr=$OP_CORR | op_net=$OP_NET | op_model=$OP_MODEL"
    log_info "log_level=$LOG_LEVEL"
    log_info "project_root=$PROJECT_ROOT"

    validate_inputs
    check_environment

    # Build realization list
    read -ra REA_LIST <<< "$REALIZATIONS"
    log_info "Realization list: ${REA_LIST[*]}"

    local failures=0
    local total=${#REA_LIST[@]}

    for rea in "${REA_LIST[@]}"; do
        log_info "------------------------------------------------------------"
        log_info "Running realization: $rea"

        if python3 "${SCRIPT_DIR}/run_pso.py" \
            --realization "$rea" \
            --op-corr "$OP_CORR" \
            --op-net "$OP_NET" \
            --op-model "$OP_MODEL" \
            --log-dir "$LOG_DIR" \
            --log-level "$LOG_LEVEL" \
            ${CONFIG_ARG}; then
            log_info "Realization $rea: SUCCESS"
        else
            log_error "Realization $rea: FAILED"
            (( failures++ )) || true
        fi
    done

    log_info "============================================================"
    log_info "Pipeline complete: $((total - failures))/$total realizations succeeded."

    if [[ $failures -gt 0 ]]; then
        log_error "$failures realization(s) failed. See $LOG_FILE for details."
        exit 1
    fi

    log_info "All realizations completed successfully."
    exit 0
}

main "$@"
