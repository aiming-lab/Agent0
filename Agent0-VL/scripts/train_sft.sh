#!/bin/bash
# ============================================================================
# Agent0-VL SFT Training (verl FSDP route, text trajectories)
#
# Trains on reasoning trajectories where tool calls and tool observations are
# rendered as text (<think> tags + JSON tool calls). Optimizes language loss
# only. For full vision-language SFT with images — the pipeline used in the
# paper — use scripts/sft_stage1.sh and scripts/sft_stage2.sh (ms-swift).
#
# Paper SFT hyperparameters: lr 1e-5, global batch 128, 3 epochs, warmup 0.05.
# ============================================================================

set -e

# Default configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA_PATH="${DATA_PATH:-data/sft}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/sft}"
NUM_GPUS="${NUM_GPUS:-8}"

# Training hyperparameters from paper (global batch = 4 x 4 x 8 = 128)
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --num_epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=================================="
echo "Agent0-VL SFT Training (verl FSDP)"
echo "=================================="
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "LR: ${LEARNING_RATE}"
echo "Global batch: ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "Epochs: ${NUM_EPOCHS}"
echo "=================================="

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="agent0-vl-sft"

VAL_ARG=""
if [[ -f "${DATA_PATH}/val.parquet" ]]; then
    VAL_ARG="--val_data ${DATA_PATH}/val.parquet"
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    verl/trainer/agent0_sft_trainer.py \
    --model_path "${MODEL_PATH}" \
    --train_data "${DATA_PATH}/train.parquet" \
    ${VAL_ARG} \
    --output_dir "${OUTPUT_DIR}" \
    --learning_rate "${LEARNING_RATE}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --gradient_checkpointing true \
    --dataloader_num_workers 4

echo "=================================="
echo "SFT Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=================================="
