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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMMutliTurnRollout(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        trust_remote_code = kwargs.get('trust_remote_code', False)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.num_turns = config.get('num_turns', 2)
        self.feedback_options = {
            "default": "There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution.",
            "genrm": "Please carefully check whether the solution process of the math problem is correct. If it is incorrect, please point out the wrong step, explain the reason for the error. Then conclude with 'The answer is wrong'. If it is correct, simply state 'The answer is correct'."
        } # max token 100
        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length * self.num_turns + (self.num_turns - 1) * 100
        max_model_len = int(max_model_len)
        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=42,
        )

        # Offload vllm model to reduce peak memory usage
        # self.inference_engine.sleep(level=1)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        if "num_turns" in prompts.meta_info:
            num_turns = prompts.meta_info['num_turns']
        else:
            num_turns = self.num_turns


        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # only for multiturn, vllm n is not support for next turn
        if do_sample and self.config.n > 1 and not is_validate:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            current_inputs = []
            for i in range(batch_size):
                current_inputs.append(_pre_process_inputs(self.pad_token_id, idx[i]))
            kwargs['n'] = 1
        else:
            current_inputs = []
            for i in range(batch_size):
                current_inputs.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        all_responses = []
        all_responses_mask = []
        
        feedback_type = prompts.meta_info.get('feedback_type', 'default')
        feedback = self.feedback_options.get(feedback_type, self.feedback_options['default'])
        
        messages = [{"role": "user", "content": feedback}]
        chat_template = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        # for Qwen, remove system prompt
        if chat_template.startswith("<|im_start|>system"):
            system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
            chat_template = chat_template[system_end:].lstrip()
        # For llama, remove bos
        if chat_template.startswith("<｜begin▁of▁sentence｜>"):
            chat_template = chat_template[len(""):]
        chat_template = "\n\n" + chat_template
        feedback_tokens = self.tokenizer.encode(chat_template, add_special_tokens=False)

        for turn in range(num_turns):
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=current_inputs,
                    use_tqdm=False)

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
                # breakpoint()
                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        response.append(output.outputs[sample_id].token_ids)

                response = pad_2d_list_to_length(response, self.pad_token_id,
                                                max_length=self.config.response_length).to(idx.device)
                response_mask = get_response_mask(response, eos_token_id)
                
                all_responses.append(response)
                all_responses_mask.append(response_mask)
                
                # If not the last turn, prepare input for the next turn
                if turn < num_turns - 1:
                    next_inputs = []
                    for i in range(batch_size):
                        current_response = response[i][:sum(response_mask[i])].tolist()
                        next_inputs.append(current_inputs[i] + current_response + feedback_tokens)
                    current_inputs = next_inputs

        total_length = num_turns * self.config.response_length + (num_turns - 1) * len(feedback_tokens)
        
        # Combine all turns into one tensor
        combined_response = torch.full((batch_size, total_length), self.pad_token_id, device=idx.device)
        multiturn_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        response_attention_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        current_pos = torch.zeros(batch_size, dtype=torch.long, device=idx.device)
        for turn in range(num_turns):
            response_lengths = all_responses_mask[turn].sum(dim=1)  # (bs,)

            for i in range(batch_size):
                actual_length = response_lengths[i]
                combined_response[i, current_pos[i]:current_pos[i] + actual_length] = all_responses[turn][i, :actual_length]
                multiturn_mask[i, current_pos[i]:current_pos[i] + actual_length] = True
                response_attention_mask[i, current_pos[i]:current_pos[i] + actual_length] = True
            
                current_pos[i] += actual_length
            
            # If not the last turn, add feedback tokens
            if turn < num_turns - 1:
                feedback_tensor = torch.tensor(feedback_tokens, device=idx.device)
                for i in range(batch_size):
                    combined_response[i, current_pos[i]:current_pos[i] + len(feedback_tokens)] = feedback_tensor
                    response_attention_mask[i, current_pos[i]:current_pos[i] + len(feedback_tokens)] = True
                    current_pos[i] += len(feedback_tokens)

        # Update attention_mask and position_ids
        seq = torch.cat([idx, combined_response], dim=-1)
        delta_position_id = torch.arange(1, combined_response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': combined_response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'multiturn_mask': multiturn_mask
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
