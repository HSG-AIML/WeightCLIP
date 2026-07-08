#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") GPU_ID MAX_USED_MIB POLL_SECONDS LOG_DIR -- command1 [args ...] -- command2 [args ...] ..."
    echo
    echo "Example:"
    echo "  $(basename "$0") 0 2048 30 logs -- python3 script1.py --arg x -- python3 script2.py --arg y"
}

is_executable_command() {
    local cmd="$1"
    if [[ "$cmd" == */* ]]; then
        [[ -x "$cmd" ]]
        return
    fi
    command -v "$cmd" >/dev/null 2>&1
}

print_command() {
    printf '%q' "$1"
    shift
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

query_gpu_memory_all() {
    "$NVIDIA_SMI_BIN" --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
}

wait_for_gpus() {
    local gpu_array
    IFS=',' read -ra gpu_array <<< "$GPU_IDS"

    while true; do
        local query_output
        query_output="$(query_gpu_memory_all)"

        local all_ready=1
        local timestamp
        timestamp="$(date '+%F %T')"

        echo "[$timestamp] Checking GPUs: $GPU_IDS"

        for gpu in "${gpu_array[@]}"; do
            local line used_mib total_mib

            line="$(awk -F', *' -v gpu="$gpu" '$1 == gpu {print $2 " " $3; found=1; exit} END {if (!found) exit 1}' <<< "$query_output")" || {
                echo "Failed to query memory for GPU $gpu via $NVIDIA_SMI_BIN" >&2
                exit 1
            }

            read -r used_mib total_mib <<< "$line"

            echo "[$timestamp] GPU $gpu memory.used=${used_mib} MiB memory.total=${total_mib} MiB"

            if (( used_mib > MAX_USED_MIB )); then
                all_ready=0
            fi
        done

        if (( all_ready == 1 )); then
            echo "All requested GPUs are below threshold."
            return 0
        fi

        sleep "$POLL_SECONDS"
    done
}

run_command() {
    local idx="$1"
    shift

    local log_file="${LOG_DIR}/command_${idx}.log"

    echo
    echo "============================================================"
    echo "Running command $idx"
    echo "Log file: $log_file"
    echo -n "Command: "
    print_command "$@"
    echo "============================================================"

    {
        echo "============================================================"
        echo "Started: $(date '+%F %T')"
        echo -n "Command: "
        print_command "$@"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "============================================================"
    } >> "$log_file"

    "$@" >> "$log_file" 2>&1

    {
        echo
        echo "============================================================"
        echo "Finished: $(date '+%F %T')"
        echo "Exit code: 0"
        echo "============================================================"
    } >> "$log_file"
}

if [[ $# -lt 6 ]]; then
    usage >&2
    exit 2
fi

GPU_IDS="$1"
MAX_USED_MIB="$2"
POLL_SECONDS="$3"
LOG_DIR="$4"
shift 4

if [[ "$1" != "--" ]]; then
    usage >&2
    exit 2
fi
shift

if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be a comma-separated list of non-negative integers, got: $GPU_IDS" >&2
    echo "Example: 0,1,2" >&2
    exit 2
fi

if ! [[ "$MAX_USED_MIB" =~ ^[0-9]+$ ]]; then
    echo "MAX_USED_MIB must be a non-negative integer in MiB, got: $MAX_USED_MIB" >&2
    exit 2
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "POLL_SECONDS must be a non-negative number, got: $POLL_SECONDS" >&2
    exit 2
fi

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PIN_CUDA_VISIBLE_DEVICES="${PIN_CUDA_VISIBLE_DEVICES:-1}"

if ! is_executable_command "$NVIDIA_SMI_BIN"; then
    echo "nvidia-smi command not found or not executable: $NVIDIA_SMI_BIN" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

COMMANDS=()
current_cmd=()

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        if [[ ${#current_cmd[@]} -eq 0 ]]; then
            echo "Empty command found between -- separators" >&2
            exit 2
        fi

        COMMANDS+=("$(printf '%q ' "${current_cmd[@]}")")
        current_cmd=()
        shift
        continue
    fi

    current_cmd+=("$1")
    shift
done

if [[ ${#current_cmd[@]} -gt 0 ]]; then
    COMMANDS+=("$(printf '%q ' "${current_cmd[@]}")")
fi

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    usage >&2
    exit 2
fi

echo "Waiting for GPUs $GPU_IDS memory.used to be <= ${MAX_USED_MIB} MiB"
echo "Number of commands: ${#COMMANDS[@]}"
echo "Log directory: $LOG_DIR"

wait_for_gpus

if [[ "$PIN_CUDA_VISIBLE_DEVICES" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

idx=1
for cmd_string in "${COMMANDS[@]}"; do
    eval "cmd_array=($cmd_string)"

    if ! is_executable_command "${cmd_array[0]}"; then
        echo "Command not found or not executable: ${cmd_array[0]}" >&2
        exit 1
    fi

    run_command "$idx" "${cmd_array[@]}"
    idx=$((idx + 1))
done

echo
echo "All commands finished successfully."