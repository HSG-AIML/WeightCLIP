#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/train_resnet18_datasetpt_zoo.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DEFAULT_DATASETS=(
    "land_cover_resnet"
    "cactus_aerial_resnet"
    "ct_images_resnet"
    "lego_vs_generic_resnet"
    "cassava_leaf_resnet"
    "asl_resnet"
    "artworks_resnet"
    "blood_cells_resnet"
    "casting_resnet"
    "breakhis_resnet"
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
else
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

SOURCE_ZOO_ROOT="${SOURCE_ZOO_ROOT:-/local/zoos/metatrain_resnet18_updated}"
INIT_TYPE="${INIT_TYPE:-normal}"
if [[ -n "${ZOO_ROOT:-}" ]]; then
    OUTPUT_ZOO_ROOT="$ZOO_ROOT"
elif [[ "$INIT_TYPE" == "uniform" ]]; then
    OUTPUT_ZOO_ROOT="/local/zoos/metatrain_resnet18"
elif [[ "$INIT_TYPE" == "normal" ]]; then
    OUTPUT_ZOO_ROOT="/local/zoos/metatrain_resnet18"
else
    echo "Unsupported INIT_TYPE: $INIT_TYPE" >&2
    exit 1
fi

NUM_SEEDS="${NUM_SEEDS:-50}"
LR_SWEEP_SEEDS="${LR_SWEEP_SEEDS:-2}"
LR_CANDIDATES=(${LR_CANDIDATES:-5e-2 7e-2 1e-1 2e-1 3e-1 5e-1 7e-1})
FORCE_LR_SWEEP="${FORCE_LR_SWEEP:-0}"
NO_LR_SWEEP="${NO_LR_SWEEP:-0}"
EPOCHS="${EPOCHS:-45}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-50}"
CHECKPOINT_START="${CHECKPOINT_START:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.05}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
WIDTH_MULT="${WIDTH_MULT:-0.5}"
NLIN="${NLIN:-relu}"
DROPOUT="${DROPOUT:-0.15}"
MODELS_PER_GPU="${MODELS_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"

EXTRA_ARGS=("$@")
SWEEP_ARGS=()
if [[ "$NO_LR_SWEEP" == "1" ]]; then
    SWEEP_ARGS+=(--no-lr-sweep)
else
    SWEEP_ARGS+=(--lr-sweep-seeds "$LR_SWEEP_SEEDS" --lr-candidates "${LR_CANDIDATES[@]}")
    if [[ "$FORCE_LR_SWEEP" == "1" ]]; then
        SWEEP_ARGS+=(--force-lr-sweep)
    fi
fi

echo "========================================"
echo "Regenerating MetaTrain ResNet18 Zoos"
echo "========================================"
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "Source zoo root: $SOURCE_ZOO_ROOT"
echo "Output zoo root: $OUTPUT_ZOO_ROOT"
echo "Seeds per dataset: $NUM_SEEDS"
if [[ "$NO_LR_SWEEP" == "1" ]]; then
    echo "LR sweep: disabled"
else
    echo "LR sweep seeds: $LR_SWEEP_SEEDS"
    echo "LR candidates: ${LR_CANDIDATES[*]}"
    echo "Force LR sweep: $FORCE_LR_SWEEP"
fi
echo "Epochs: $EPOCHS"
echo "Scheduler epochs: $SCHEDULER_EPOCHS"
echo "Checkpoint start epoch: $CHECKPOINT_START"
echo "Eval every: $EVAL_EVERY"
echo "Batch size: $BATCH_SIZE"
echo "LR: $LR"
echo "Momentum: $MOMENTUM"
echo "Weight decay: $WEIGHT_DECAY"
echo "Width multiplier: $WIDTH_MULT"
echo "Nonlinearity: $NLIN"
echo "Dropout: $DROPOUT"
echo "Init type: $INIT_TYPE"
echo "Models per GPU: $MODELS_PER_GPU"
echo "Num workers: $NUM_WORKERS"
echo "Gradient clip: $GRAD_CLIP"
echo "Python: $PYTHON_BIN"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra args: ${EXTRA_ARGS[*]}"
fi
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Training: $dataset"
    echo "========================================"

    "$PYTHON_BIN" "$SCRIPT" \
        --dataset "$dataset" \
        --source-zoo-root "$SOURCE_ZOO_ROOT" \
        --zoo-root "$OUTPUT_ZOO_ROOT" \
        --num-seeds "$NUM_SEEDS" \
        --epochs "$EPOCHS" \
        --scheduler-epochs "$SCHEDULER_EPOCHS" \
        --checkpoint-start-epoch "$CHECKPOINT_START" \
        --eval-every "$EVAL_EVERY" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --momentum "$MOMENTUM" \
        --weight-decay "$WEIGHT_DECAY" \
        --width-mult "$WIDTH_MULT" \
        --nlin "$NLIN" \
        --dropout "$DROPOUT" \
        --init-type "$INIT_TYPE" \
        --models-per-gpu "$MODELS_PER_GPU" \
        --num-workers "$NUM_WORKERS" \
        --grad-clip "$GRAD_CLIP" \
        "${SWEEP_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "Completed: $dataset"
    echo ""
done

echo "========================================"
echo "All requested ResNet18 zoos completed"
echo "========================================"