#!/bin/bash
# ============================================================================
# Agent0-VL SFT Stage 1: Tool-usage and image-manipulation trajectories
#
# Paper (Appendix B): SFT is trained first on tool-usage / image manipulation
# data, then annealed on mathematical code reasoning data (see sft_stage2.sh).
# Stage 1 uses lr 1e-5, 3 epochs, warmup 0.05, global batch 128
# (per_device 1 x grad_accum 16 x 8 GPUs).
#
# Requires ms-swift (https://github.com/modelscope/ms-swift):
#   pip install ms-swift
#
# Data: a jsonl/dataset in swift "messages" format with images.
# Set SFT_DATA to your dataset path.
# ============================================================================

nproc_per_node=${NPROC_PER_NODE:-8}

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
SFT_DATA=${SFT_DATA:-data/sft/stage1_tool_usage.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/sft_stage1}

bsz=${BSZ:-1}

FPS_MAX_FRAMES=10 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
MAX_PIXELS=3211264 \
swift sft \
    --model "$MODEL" \
    --dataset "$SFT_DATA" \
    --train_type full \
    --torch_dtype bfloat16 \
    --system scripts/prompt.txt \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $bsz) \
    --save_strategy epoch \
    --max_length 10240 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --report_to wandb
