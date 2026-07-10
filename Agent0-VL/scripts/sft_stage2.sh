#!/bin/bash
# ============================================================================
# Agent0-VL SFT Stage 2: Annealing on mathematical code-reasoning data
#
# Paper (Appendix B): after stage-1 tool-usage SFT, the model is gradually
# annealed on mathematical code reasoning data with a lower learning rate
# (1e-6). Initialize MODEL from the stage-1 output checkpoint.
#
# Requires ms-swift: pip install ms-swift
# ============================================================================

nproc_per_node=${NPROC_PER_NODE:-8}

export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8

MODEL=${MODEL:-checkpoints/sft_stage1}          # stage-1 checkpoint
SFT_DATA=${SFT_DATA:-data/sft/stage2_math_code.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/sft_stage2}

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
    --learning_rate 1e-6 \
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
