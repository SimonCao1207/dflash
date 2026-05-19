#!/usr/bin/env bash
set -euo pipefail

# Bench every checkpoint dir under a training run root.
# Each checkpoint runs the full task list defined in benchmark_hf.sh
# and writes logs to ${LOG_DIR}/<ckpt-name>_<dataset>.log.

CKPT_ROOT="${CKPT_ROOT:-/sensei-fs-3/users/thanhl/d-spec/outputs/continual-qwen3-4b-dflash-perfectblend-chat}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
RUN_NAME="${RUN_NAME:-continual_qwen3_4b_chat}"
LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}/$(date +%Y%m%d_%H%M%S)}"
# Number of checkpoints to bench, starting from the latest step and walking
# backwards. Set to 0 to run every checkpoint in CKPT_ROOT.
MAX_CKPTS="${MAX_CKPTS:-8}"

cd "$(dirname "$0")/.."
mkdir -p "${LOG_DIR}"

if [ ! -d "${CKPT_ROOT}" ]; then
    echo "ERROR: checkpoint root not found: ${CKPT_ROOT}" >&2
    exit 1
fi

# Sort by step DESCENDING so latest checkpoint runs first.
mapfile -t CKPTS < <(
    find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*_step_*' \
        | sort -t_ -k4n -r
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no epoch_*_step_* directories found in ${CKPT_ROOT}" >&2
    exit 1
fi

total=${#CKPTS[@]}
if [ "${MAX_CKPTS}" -gt 0 ] && [ "${total}" -gt "${MAX_CKPTS}" ]; then
    CKPTS=("${CKPTS[@]:0:${MAX_CKPTS}}")
    echo "Selected latest ${#CKPTS[@]} of ${total} checkpoints (newest first):"
else
    echo "Found ${#CKPTS[@]} checkpoint(s) (newest first):"
fi
for c in "${CKPTS[@]}"; do echo "  - $(basename "${c}")"; done
echo "Target model:  ${TARGET_MODEL}"
echo "Logs:          ${LOG_DIR}"
echo

for ckpt in "${CKPTS[@]}"; do
    name="$(basename "${ckpt}")"
    echo "############################################"
    echo "# ${name}"
    echo "############################################"
    TARGET_MODEL="${TARGET_MODEL}" \
    DRAFT_MODEL="${ckpt}" \
    LOG_DIR="${LOG_DIR}" \
    LOG_PREFIX="${name}" \
        ./scripts/benchmark_hf.sh "$@"
    echo
done

echo "All checkpoints done. Logs at ${LOG_DIR}/"
