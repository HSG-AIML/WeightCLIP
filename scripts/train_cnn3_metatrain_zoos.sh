#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/train_cnn3_datasetpt_zoo.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=(
    "proptit_aif_homework_cnn3"
    "cactus_aerial_cnn3"
    "dl2020_cnn3"
    "four_shapes_cnn3"
    "e4040_assignment_cnn3"
    "car_classification_cnn3"
    "simpsons_characters_cnn3"
    "simpsons_challenge_cnn3"
    "breakhis_cnn3"
    "mushrooms_cnn3"
    "artworks_cnn3"
    "numta_bengali_cnn3"
    "lego_bricks_cnn3"
    "ads5035_cnn3"
    "day3_kaggle_cnn3"
    "ct_images_cnn3"
    "land_cover_cnn3"
    "casting_cnn3"
    "asl_cnn3"
    "cassava_leaf_cnn3"
)

SOURCE_ZOO_ROOT="/local/zoos/metatrain_cnn3_updated"
ZOO_ROOT="/local/zoos/metatrain_cnn3"
NUM_SEEDS=100
LR_SWEEP_SEEDS=5
LR_CANDIDATES=(3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1)
EPOCHS=35
SCHEDULER_EPOCHS=50
CHECKPOINT_START=30
EVAL_EVERY=1
BATCH_SIZE=128
WEIGHT_DECAY=3e-3
NLIN="gelu"
DROPOUT=0.0
INIT_TYPE="normal"
MODELS_PER_GPU=2
WARMUP=3
GRAD_CLIP=1.0

EXTRA_ARGS=("$@")

echo "========================================"
echo "Regenerating MetaTrain CNN3 Zoos"
echo "========================================"
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo "Source zoo root: $SOURCE_ZOO_ROOT"
echo "Output zoo root: $ZOO_ROOT"
echo "Seeds per dataset: $NUM_SEEDS"
echo "LR sweep seeds: $LR_SWEEP_SEEDS"
echo "LR candidates: ${LR_CANDIDATES[*]}"
echo "Epochs: $EPOCHS"
echo "Scheduler epochs: $SCHEDULER_EPOCHS"
echo "Checkpoint start epoch: $CHECKPOINT_START"
echo "Eval every: $EVAL_EVERY"
echo "Batch size: $BATCH_SIZE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Nonlinearity: $NLIN"
echo "Dropout: $DROPOUT"
echo "Init type: $INIT_TYPE"
echo "Models per GPU: $MODELS_PER_GPU"
echo "Warmup epochs: $WARMUP"
echo "Gradient clip: $GRAD_CLIP"
echo "Python: $PYTHON_BIN"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Training: $dataset"
    echo "========================================"

    "$PYTHON_BIN" "$SCRIPT" \
        --dataset "$dataset" \
        --source-zoo-root "$SOURCE_ZOO_ROOT" \
        --zoo-root "$ZOO_ROOT" \
        --num-seeds "$NUM_SEEDS" \
        --lr-sweep-seeds "$LR_SWEEP_SEEDS" \
        --lr-candidates "${LR_CANDIDATES[@]}" \
        --epochs "$EPOCHS" \
        --scheduler-epochs "$SCHEDULER_EPOCHS" \
        --checkpoint-start-epoch "$CHECKPOINT_START" \
        --eval-every "$EVAL_EVERY" \
        --batch-size "$BATCH_SIZE" \
        --weight-decay "$WEIGHT_DECAY" \
        --nlin "$NLIN" \
        --dropout "$DROPOUT" \
        --init-type "$INIT_TYPE" \
        --models-per-gpu "$MODELS_PER_GPU" \
        --warmup-epochs "$WARMUP" \
        --grad-clip "$GRAD_CLIP" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "Completed: $dataset"
    echo ""
done

echo "========================================"
echo "All requested CNN3 zoos completed"
echo "========================================"
