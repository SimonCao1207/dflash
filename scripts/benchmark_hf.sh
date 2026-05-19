#!/usr/bin/env bash
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-/sensei-fs-3/users/thanhl/d-spec/outputs/continual-qwen3-4b-dflash-perfectblend-chat/epoch_2_step_23110}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NPROC="${NPROC:-1}"
LOG_DIR="${LOG_DIR:-logs/$(date +%Y%m%d_%H%M%S)}"
LOG_PREFIX="${LOG_PREFIX:-continual_qwen3_4b_chat}"

cd "$(dirname "$0")/.."
mkdir -p "${LOG_DIR}"

TASKS=(
    "gsm8k:128"
    "math500:128"
    "humaneval:164"
    "mbpp:128"
    "mt-bench:80"
    "alpaca:128"
)

# Allow overriding the task list: `./benchmark_hf.sh gsm8k:128 mbpp:128`
if [ "$#" -gt 0 ]; then
    TASKS=("$@")
fi

echo "Target model:  ${TARGET_MODEL}"
echo "Draft model:   ${DRAFT_MODEL}"
echo "Logs:          ${LOG_DIR}"
echo "Tasks:         ${TASKS[*]}"
echo

for entry in "${TASKS[@]}"; do
    dataset="${entry%%:*}"
    samples="${entry##*:}"
    log_file="${LOG_DIR}/${LOG_PREFIX}_${dataset}.log"

    echo "==== ${dataset} (max_samples=${samples}) ===="
    torchrun --nproc_per_node="${NPROC}" -m dflash.benchmark \
        --backend transformers \
        --model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --dataset "${dataset}" \
        --max-samples "${samples}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" 2>&1 | tee "${log_file}"
    echo
done

echo "All benchmarks done. Logs at ${LOG_DIR}/"
