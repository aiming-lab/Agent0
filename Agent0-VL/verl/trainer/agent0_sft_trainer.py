# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agent0-VL SFT Trainer (text-trajectory variant)

Supervised fine-tuning entry point built on verl's FSDPSFTTrainer. It trains
on Agent0-VL reasoning trajectories (<think> tags + tool-call JSON + tool
observations rendered as text).

NOTE: this trainer optimizes the *language* loss only. For full
vision-language SFT (images + text), use the ms-swift pipeline in
``scripts/sft_stage1.sh`` / ``scripts/sft_stage2.sh``, which is the pipeline
described in the paper (stage 1: tool usage, stage 2: math-code annealing).
"""

import os
import argparse
import logging

from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer

logger = logging.getLogger(__name__)


def _pick_auto_model_cls(hf_config):
    """Select the correct Auto class: VLM models (e.g. Qwen2.5-VL) are not in
    the CausalLM mapping and must be loaded via AutoModelForImageTextToText."""
    model_type = getattr(hf_config, 'model_type', None)
    if model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


class Agent0SFTTrainer(FSDPSFTTrainer):
    """FSDP SFT trainer using the Agent0SFTDataset (chat-format prompts +
    trajectory responses, tensor-only outputs)."""

    def _build_dataloader(self):
        """Build train dataloader with the Agent0 SFT dataset.

        Replaces the parent implementation entirely (the parent would
        overwrite ``self.train_dataset`` with the plain SFTDataset).
        """
        config = self.config
        from verl.utils.dataset import Agent0SFTDataset

        self.train_dataset = Agent0SFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            processor=None,  # text loss only; see module docstring
            prompt_key=config.data.prompt_key,
            response_key=config.data.response_key,
            max_length=config.data.max_length,
            truncation=config.data.truncation,
            tensor_only=True,
        )
        if config.data.val_files:
            self.val_dataset = Agent0SFTDataset(
                parquet_files=config.data.val_files,
                tokenizer=self.tokenizer,
                processor=None,
                prompt_key=config.data.prompt_key,
                response_key=config.data.response_key,
                max_length=config.data.max_length,
                truncation=config.data.truncation,
                tensor_only=True,
            )

        # Data distribution mirrors FSDPSFTTrainer._build_dataloader
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=self.config.data.get('dataloader_num_workers', 8),
                                           pin_memory=True,
                                           drop_last=True)

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """VL-aware loss computation.

        Qwen2.5-VL and other VL backbones apply 3D mRoPE and derive position_ids
        internally; the 2D position_ids the SFT dataset builds for text batches
        would corrupt mRoPE. For VL models on the standard (non sequence-parallel)
        path we drop position_ids and let the model compute them from the
        attention mask (correct for text-only sequences). All other cases defer
        to the parent implementation unchanged.
        """
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if not (getattr(self, '_is_vl_model', False) and not use_sp):
            return super()._compute_loss_and_backward(batch, do_backward=do_backward)

        import torch.nn as nn

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            labels = input_ids[:, 1:].contiguous()
            # position_ids intentionally omitted -> model computes mRoPE positions
            output = self.fsdp_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels.contiguous().view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)

            valid_token_this_rank = torch.sum(loss_mask)
            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = torch.distributed.get_world_size()
            else:
                dp_size = 1
            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def _build_model_optimizer(self):
        """Mirror of FSDPSFTTrainer._build_model_optimizer that also supports
        vision-language checkpoints (Qwen2.5-VL etc.), which are registered
        under AutoModelForImageTextToText rather than AutoModelForCausalLM."""
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        auto_model_cls = _pick_auto_model_cls(hf_config)
        # VL backbones (Qwen2.5-VL) use 3D mRoPE and compute position_ids
        # internally; _compute_loss_and_backward must not forward the 2D
        # text position_ids to them.
        self._is_vl_model = auto_model_cls.__name__ != 'AutoModelForCausalLM'

        init_context = get_init_weight_context_manager(use_meta_tensor=not hf_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = auto_model_cls.from_pretrained(
                local_model_path,
                config=hf_config,
                torch_dtype=torch.float32,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code)

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = torch.optim.AdamW(self.fsdp_model.parameters(),
                                           lr=self.config.optim.lr,
                                           betas=self.config.optim.betas,
                                           weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f'Number of steps/epoch {self.steps_per_epoch}, number of epochs '
                  f'{self.config.trainer.total_epochs}, total number of steps {self.total_steps}')

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps)


def convert_args_to_config(args: argparse.Namespace, world_size: int) -> DictConfig:
    """Build an FSDPSFTTrainer-compatible config from CLI arguments.

    Every key FSDPSFTTrainer reads (see verl/trainer/config/sft_trainer.yaml)
    must be present here.
    """
    global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size

    config_dict = {
        'model': {
            'partial_pretrain': args.model_path,
            'trust_remote_code': True,
            'enable_gradient_checkpointing': args.gradient_checkpointing,
            'external_lib': None,
            'fsdp_config': {
                'wrap_policy': {
                    'min_num_params': 0,
                },
                'cpu_offload': False,
                'offload_params': False,
            },
            'lora_rank': 0,
            'lora_alpha': 16,
            'target_modules': 'all-linear',
            'use_liger': False,
        },
        'data': {
            'train_files': [args.train_data],
            'val_files': [args.val_data] if args.val_data else [],
            'train_batch_size': global_batch_size,
            'micro_batch_size': None,
            'micro_batch_size_per_gpu': args.per_device_train_batch_size,
            'max_length': args.max_length,
            'truncation': 'right',
            'prompt_key': 'prompt',
            'response_key': 'response',
            'multiturn': {
                'enable': False,
                'messages_key': 'messages',
            },
            'balance_dp_token': False,
            'chat_template': None,
            'dataloader_num_workers': args.dataloader_num_workers,
            'custom_cls': {
                'path': None,
                'name': None,
            },
        },
        'optim': {
            'lr': args.learning_rate,
            'betas': [0.9, 0.95],
            'weight_decay': args.weight_decay,
            'warmup_steps_ratio': args.warmup_ratio,
            'clip_grad': args.max_grad_norm,
        },
        'ulysses_sequence_parallel_size': args.ulysses_sequence_parallel_size,
        'use_remove_padding': args.use_remove_padding,
        'trainer': {
            'total_epochs': args.num_train_epochs,
            'total_training_steps': None,
            'project_name': 'agent0-vl-sft',
            'experiment_name': args.experiment_name,
            'logger': ['console', 'wandb'] if args.report_to_wandb else ['console'],
            'default_local_dir': args.output_dir,
            'default_hdfs_dir': None,
            'resume_path': None,
            'seed': args.seed,
        },
    }

    return OmegaConf.create(config_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Agent0-VL Supervised Fine-Tuning Trainer (text trajectories)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def str2bool(x):
        return str(x).lower() in ('true', '1', 'yes')

    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--train_data', type=str, required=True, help='Training parquet file')
    parser.add_argument('--val_data', type=str, default=None, help='Validation parquet file (optional)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sft')
    parser.add_argument('--experiment_name', type=str, default='sft-training')

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=4096, help='Max sequence length (prompt + response)')

    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=True)

    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--ulysses_sequence_parallel_size', type=int, default=1)
    parser.add_argument('--use_remove_padding', type=str2bool, default=False)
    parser.add_argument('--report_to_wandb', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    local_rank, rank, world_size = initialize_global_process_group()

    if rank == 0:
        logger.info("Starting Agent0-VL SFT Training")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Train data: {args.train_data}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Global batch size: {args.per_device_train_batch_size} x "
                    f"{args.gradient_accumulation_steps} x {world_size} = "
                    f"{args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = convert_args_to_config(args, world_size)

    device_mesh = init_device_mesh(
        device_type='cuda',
        mesh_shape=(world_size,),
        mesh_dim_names=('fsdp',),
    )
    dp_size = world_size // args.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type='cuda',
        mesh_shape=(dp_size, args.ulysses_sequence_parallel_size),
        mesh_dim_names=('dp', 'sp'),
    )

    trainer = Agent0SFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
    )
    trainer.fit()

    if rank == 0:
        logger.info("Training complete! Model saved to %s", args.output_dir)


if __name__ == '__main__':
    main()
