#!/usr/bin/env bash
set -euo pipefail

# Bench every checkpoint dir under a training run root.
# Each checkpoint runs the full task list defined in benchmark_hf.sh
# and writes logs to ${LOG_DIR}/<ckpt-name>_<dataset>.log.

CKPT_ROOT="${CKPT_ROOT:-/sensei-fs-3/users/thanhl/d-spec/outputs/continual-qwen3-4b-dflash-perfectblend-chat}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
RUN_NAME="${RUN_NAME:-continual_qwen3_4b_chat}"
# Pretrained DFlash checkpoint to bench as the "before continual training"
# baseline. Set to empty to skip. Accepts a HF hub id or a local path.
BASELINE_DRAFT="${BASELINE_DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}/$(date +%Y%m%d_%H%M%S)}"
# Number of checkpoints to bench (evenly spaced across training, always
# includes the first and last). Latest checkpoint runs first so partial runs
# still surface the most-recent number. Set to 0 to run every checkpoint.
MAX_CKPTS="${MAX_CKPTS:-8}"

cd "$(dirname "$0")/.."
mkdir -p "${LOG_DIR}"

if [ ! -d "${CKPT_ROOT}" ]; then
    echo "ERROR: checkpoint root not found: ${CKPT_ROOT}" >&2
    exit 1
fi

# Sort by step ascending so spaced sampling spans the whole training run.
mapfile -t CKPTS < <(
    find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*_step_*' \
        | sort -t_ -k4n
)

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "ERROR: no epoch_*_step_* directories found in ${CKPT_ROOT}" >&2
    exit 1
fi

total=${#CKPTS[@]}
if [ "${MAX_CKPTS}" -gt 0 ] && [ "${total}" -gt "${MAX_CKPTS}" ]; then
    selected=()
    if [ "${MAX_CKPTS}" -eq 1 ]; then
        selected+=("${CKPTS[$((total - 1))]}")
    else
        for ((i=0; i<MAX_CKPTS; i++)); do
            idx=$(( i * (total - 1) / (MAX_CKPTS - 1) ))
            selected+=("${CKPTS[$idx]}")
        done
    fi
    CKPTS=("${selected[@]}")
    msg="Sampled ${#CKPTS[@]} of ${total} checkpoints (evenly spaced)"
else
    msg="Found ${#CKPTS[@]} checkpoint(s)"
fi

# Reverse so the latest checkpoint runs first.
reversed=()
for ((i=${#CKPTS[@]}-1; i>=0; i--)); do reversed+=("${CKPTS[$i]}"); done
CKPTS=("${reversed[@]}")

# Insert the pretrained baseline right after the latest continual checkpoint
# so both critical comparison points run early.
if [ -n "${BASELINE_DRAFT}" ]; then
    if [ "${#CKPTS[@]}" -ge 1 ]; then
        CKPTS=("${CKPTS[0]}" "${BASELINE_DRAFT}" "${CKPTS[@]:1}")
    else
        CKPTS=("${BASELINE_DRAFT}")
    fi
    msg="${msg} + baseline ${BASELINE_DRAFT}"
fi

echo "${msg}, running newest first:"
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
