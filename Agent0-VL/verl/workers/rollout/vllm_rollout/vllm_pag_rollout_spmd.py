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
vLLM Genrm Rollout - Multi-turn generation logic:
1. First turn: prompt -> answer
2. Verify previous answer -> correct/wrong judgment
3. If "wrong", next turn: request regeneration -> new answer
4. Repeat until specified turns or GenRM considers answer correct
"""
import numpy as np
import re
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

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from input token sequence"""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:].tolist()


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMPAGRollout(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """vLLM GenRM inference engine
        
        Args:
            model_path: Model path
            config: Config dict
            tokenizer: Tokenizer
            model_hf_config: HuggingFace model config
            **kwargs: Additional parameters
        """
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        # Initialize Megatron parallel state
        if kwargs.get('train_tp', None) is not None:
            import os
            os.environ.update({
                'CUDA_TIMER_STREAM_KAFKA_ENABLE': '0',
                'MEGATRON_IMPORT_TIMERS': '0'
            })
            train_tp = kwargs.get('train_tp')
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp
            )

        # Configure sampling parameters
        sampling_kwargs = {
            'n': 1,
            'logprobs': 0,
            'max_tokens': config.response_length,
        }

        if vllm_version != '0.3.1':
            sampling_kwargs['detokenize'] = False

        # Add sampling params from config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                sampling_kwargs[k] = config.get(k)

        print(f"Sampling kwargs: {sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Initialize basic attributes
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.num_turns = config.get('num_turns', 2)
        self.is_only_genrm = config.get('is_only_genrm', False)
        self.end_with_verifer = config.get('end_with_verifer', False)
        
        # Prompt templates
        self.prompt_templates = {
            "verify": "Check the math solution step-by-step. If you find a mistake: state the wrong step, explain why it's wrong, and end your response with 'The answer is wrong'. If all steps are correct, end your response with 'The answer is correct'.",
            "regenerate": "You indicated that your previous answer was wrong. Please provide the correct solution to the math problem."
        }

        # Calculate max model length
        if self.end_with_verifer:
            max_model_len = self.config.response_length * self.num_turns * 2 + config.prompt_length
            max_model_len = min(max_model_len, model_hf_config.max_position_embeddings)
        else:
            max_model_len = (config.prompt_length + 
                           self.num_turns * self.config.response_length + 
                           (self.num_turns - 1) * (200 + self.config.response_length))
        
        if config.get('max_model_len', None) is not None:
            max_model_len = config.get('max_model_len')
        assert model_hf_config.max_position_embeddings >= max_model_len, \
            "model context length should be greater than total sequence length"

        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len')

        # Initialize inference engine
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
            trust_remote_code=kwargs.get('trust_remote_code', False),
            seed=42,
        )

        # Offload model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        # Pre-compute token IDs for keywords
        self.correct_token_ids = self._find_token_ids_for_word("correct")
        self.wrong_token_ids = self._find_token_ids_for_word("wrong")
        print(f"Token IDs for 'correct': {self.correct_token_ids}")
        print(f"Token IDs for 'wrong': {self.wrong_token_ids}")

    def _find_token_ids_for_word(self, word):
        """Find all possible token IDs representing the word"""
        variants = [word, f" {word}"]
        token_ids = []
        
        for variant in variants:
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:  # Only add single tokens
                token_ids.append(ids[0])
        
        return token_ids
    
    def _get_template_tokens(self, template_key):
        """Convert prompt template to token ids"""
        template = self.prompt_templates.get(template_key, "")
        messages = [{"role": "user", "content": template}]
        
        chat_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        
        # for Qwen, remove system prompt
        if chat_template.startswith("<|im_start|>system"):
            system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
            chat_template = chat_template[system_end:].lstrip()
        # For llama, remove bos
        if chat_template.startswith("<｜begin▁of▁sentence｜>"):
            chat_template = chat_template[len(""):]
        
        chat_template = "\n\n" + chat_template
        return self.tokenizer.encode(chat_template, add_special_tokens=False)

    def _extract_judgment_and_probability(self, output, text):
        """Extract 'correct' or 'wrong' judgment and token probability"""
        pattern = r"The answer is (correct|wrong)\.$"
        matches = []
        
        try:
            matches = [(match.group(1).lower(), match.group(0), match.start()) 
                      for match in re.finditer(pattern, text)]
        except:
            pass
        
        if not matches:
            return None, None
        
        # Get the last judgment
        judgment, _, _ = matches[-1]
        target_token_ids = self.correct_token_ids if judgment == "correct" else self.wrong_token_ids
        
        # Search backwards for corresponding token probability
        token_ids = output.outputs[0].token_ids
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] in target_token_ids:
                token_logprobs = output.outputs[0].logprobs[i]
                prob = np.exp(token_logprobs[token_ids[i]].logprob)
                return judgment, prob
        
        print(f"Warning: prob is None for judgment: {judgment}, text: {text}")
        return judgment, None

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate multi-turn dialogue sequences"""
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)
        num_turns = prompts.meta_info.get('num_turns', self.num_turns)
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        # Configure sampling parameters
        if not do_sample:
            kwargs = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }

        # Handle multiple sampling
        if do_sample and self.config.n > 1 and not is_validate:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            kwargs['n'] = 1

        # Preprocess inputs
        current_inputs = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        # Initialize variables
        final_generation_turn = [0] * batch_size
        verify_probs = [0] * batch_size
        full_verify_probs = [[] for _ in range(batch_size)]
        verify_tokens = self._get_template_tokens("verify")
        regenerate_tokens = self._get_template_tokens("regenerate")

        # Calculate total length
        if self.end_with_verifer:
            max_total_length = (2 * num_turns * self.config.response_length + 
                              num_turns * (len(verify_tokens) + len(regenerate_tokens)))
        else:
            max_total_length = (num_turns * self.config.response_length + 
                              (num_turns - 1) * (len(verify_tokens) + self.config.response_length + len(regenerate_tokens)))

        # Initialize response tensors
        combined_response = torch.full((batch_size, max_total_length), self.pad_token_id, device=idx.device)
        multiturn_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        response_attention_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        
        current_positions = [0] * batch_size
        turns_positions = [[0] for _ in range(batch_size)]

        # === First turn: user question -> model answer ===
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=current_inputs,
                use_tqdm=False
            )

        response = [output.outputs[sample_id].token_ids 
                   for output in outputs for sample_id in range(len(output.outputs))]
        
        current_response = pad_2d_list_to_length(response, self.pad_token_id,
                                               max_length=self.config.response_length).to(idx.device)
        current_response_mask = get_response_mask(current_response, eos_token_id)
        
        # Fill combined response tensor
        for i in range(batch_size):
            pos = current_positions[i]
            response_length = current_response_mask[i].sum().item()
            combined_response[i, pos:pos + response_length] = current_response[i, :response_length]
            multiturn_mask[i, pos:pos + response_length] = True
            response_attention_mask[i, pos:pos + response_length] = True
            current_positions[i] = pos + response_length
            turns_positions[i].append(pos + response_length)
            
        active_samples = list(range(batch_size))
        kwargs_for_verification = kwargs.copy()
        kwargs_for_verification["logprobs"] = 1
        
        # === Multi-turn generation loop ===
        max_turns = num_turns + 1 if self.end_with_verifer else num_turns
        for answer_turn in range(1, max_turns):
            # GenRM inference
            next_inputs = []
            for i, original_idx in enumerate(active_samples):
                # Add verification tokens
                pos = current_positions[original_idx]
                verify_tensor = torch.tensor(verify_tokens, device=idx.device)
                combined_response[original_idx, pos:pos + len(verify_tokens)] = verify_tensor
                response_attention_mask[original_idx, pos:pos + len(verify_tokens)] = True
                current_positions[original_idx] = pos + len(verify_tokens)
                turns_positions[original_idx].append(pos + len(verify_tokens))

                # Build GenRM input
                response_tokens = combined_response[original_idx, :current_positions[original_idx]].tolist()
                history = current_inputs[original_idx] + response_tokens
                next_inputs.append(history)

            # GenRM inference
            with self.update_sampling_params(**kwargs_for_verification):
                outputs = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=next_inputs,
                    use_tqdm=False
                )
            
            verification_response = [output.outputs[sample_id].token_ids 
                                   for output in outputs for sample_id in range(len(output.outputs))]
            active_verification = pad_2d_list_to_length(verification_response, self.pad_token_id,
                                                      max_length=self.config.response_length).to(idx.device)
            active_verification_mask = get_response_mask(active_verification, eos_token_id)
            
            # Fill verification response
            for i, original_idx in enumerate(active_samples):
                pos = current_positions[original_idx]
                verification_length = active_verification_mask[i].sum().item()
                combined_response[original_idx, pos:pos + verification_length] = active_verification[i, :verification_length]
                multiturn_mask[original_idx, pos:pos + verification_length] = True
                response_attention_mask[original_idx, pos:pos + verification_length] = True
                current_positions[original_idx] = pos + verification_length
                turns_positions[original_idx].append(pos + verification_length)
            
            if self.is_only_genrm and not is_validate:
                break
            
            # Process policy input
            new_active_samples = []
            next_inputs = []
            for i, original_idx in enumerate(active_samples):
                verification_length = active_verification_mask[i].sum().item()
                verification_tokens = active_verification[i][:verification_length].tolist()
                verification_text = self.tokenizer.decode(verification_tokens, skip_special_tokens=True)
                
                # Judge verification result
                judgment, prob = self._extract_judgment_and_probability(outputs[i], verification_text)
                if answer_turn == 1:
                    verify_probs[original_idx] = prob

                if not hasattr(full_verify_probs[original_idx], 'append'):
                    full_verify_probs[original_idx] = []
                full_verify_probs[original_idx].append(prob)
                
                if judgment == "wrong" or (is_validate and judgment != "correct"):
                    new_active_samples.append(original_idx)
                    # Add regeneration tokens
                    pos = current_positions[original_idx]
                    regenerate_tensor = torch.tensor(regenerate_tokens, device=idx.device)
                    combined_response[original_idx, pos:pos + len(regenerate_tokens)] = regenerate_tensor
                    response_attention_mask[original_idx, pos:pos + len(regenerate_tokens)] = True
                    current_positions[original_idx] = pos + len(regenerate_tokens)
                    turns_positions[original_idx].append(pos + len(regenerate_tokens))

                    response_tokens = combined_response[original_idx, :current_positions[original_idx]].tolist()
                    history = current_inputs[original_idx] + response_tokens
                    next_inputs.append(history)
                else:
                    # Correct, record and stop generation for this sample
                    final_generation_turn[original_idx] = answer_turn - 1
            
            active_samples = new_active_samples
            
            if not active_samples or answer_turn == num_turns:
                break
            
            # Policy inference
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=next_inputs,
                    use_tqdm=False
                )
            
            regenerated_response = [output.outputs[sample_id].token_ids 
                                  for output in outputs for sample_id in range(len(output.outputs))]
            active_response = pad_2d_list_to_length(regenerated_response, self.pad_token_id,
                                                  max_length=self.config.response_length).to(idx.device)
            active_response_mask = get_response_mask(active_response, eos_token_id)
            
            # Fill regenerated response
            for i, original_idx in enumerate(active_samples):
                pos = current_positions[original_idx]
                response_length = active_response_mask[i].sum().item()
                combined_response[original_idx, pos:pos + response_length] = active_response[i, :response_length]
                multiturn_mask[original_idx, pos:pos + response_length] = True
                response_attention_mask[original_idx, pos:pos + response_length] = True
                current_positions[original_idx] = pos + response_length
                turns_positions[original_idx].append(pos + response_length)
                final_generation_turn[original_idx] = answer_turn
        
        # Update attention_mask and position_ids
        seq = torch.cat([idx, combined_response], dim=-1)
        delta_position_id = torch.arange(1, combined_response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict({
            'prompts': idx,
            'responses': combined_response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'multiturn_mask': multiturn_mask
        }, batch_size=batch_size)

        # Free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
            
        return DataProto(batch=batch, non_tensor_batch={
            "final_generation_turn": np.array(final_generation_turn, dtype=np.int32), 
            "verify_probs": np.array(verify_probs, dtype=np.float32),
            "full_verify_probs": np.array(full_verify_probs, dtype=object)
        }) 