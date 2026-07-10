#!/bin/bash
set -x

# ============================================================================
# Agent0-VL Self-Evolving RL Training (SERC)
#
# Launches GRPO training with the Agent0-VL rollout worker:
#   Solver -> Tool execution -> Verifier -> confidence-gated Self-Repair
#
# Paper hyperparameters (Appendix B): lr 5e-7, batch 256, GRPO N=8,
# beta_KL=0.001, beta_ent=0.01, repetition penalty 1.05, tau_c=0.7, eta=0.05,
# 1 epoch per self-evolution iteration on 8x H200 (bfloat16).
#
# Prerequisites:
#   - data/agent0/train.parquet and data/agent0/val.parquet
#     (verl-compatible parquet: prompt / images / reward_model / data_source)
#   - Optional: SANDBOX_ENDPOINT for an HTTP sandbox service; without it,
#     tool calls run in local subprocesses.
#
# Self-evolution: run this script 3 times (ITERATION=1,2,3), each time
# initializing MODEL_PATH from the previous iteration's checkpoint.
# ============================================================================

# Data paths (parquet files)
train_data=${TRAIN_DATA:-data/agent0/train.parquet}
val_data=${VAL_DATA:-data/agent0/val.parquet}

# Project and experiment configuration
PROJECT_NAME='Agent0-VL'
CKPT_PATH=${CKPT_PATH:-checkpoints}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
ITERATION=${ITERATION:-1}
EXPERIMENT_NAME="agent0_vl_serc_iter${ITERATION}"

# GRPO group size (paper: N=8)
n=8

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 -m verl.trainer.main_ppo \
    --config-name=agent0_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.rollout_type=agent0 \
    actor_rollout_ref.rollout.max_reasoning_steps=8 \
    actor_rollout_ref.rollout.repair_threshold=0.7 \
    actor_rollout_ref.rollout.enable_tool_execution=True \
    actor_rollout_ref.rollout.enable_verification=True \
    actor_rollout_ref.rollout.enable_self_repair=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    reward_model.reward_manager=agent0 \
    reward_model.lambda_tool=0.3 \
    reward_model.alpha_out=1.0 \
    reward_model.gamma=0.99 \
    reward_model.tau_c=0.7 \
    reward_model.eta=0.05 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=${N_GPUS:-8} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    "$@"
