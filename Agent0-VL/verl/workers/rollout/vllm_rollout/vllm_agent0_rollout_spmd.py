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
Agent0-VL Rollout Worker implementing SERC (Self-Evolving Reasoning Cycle)

SERC inner loop (paper Algorithm 1):
1. Solver samples a reasoning step a_t (optionally containing a tool call)
2. The tool executes in a sandbox -> observation o_t
3. Verifier evaluates the step -> V_t = (score, confidence, critique, tool_check)
4. If confidence < tau_c: a repair instruction Delta_t is generated and the
   Solver re-samples a corrected segment a'_t

Token bookkeeping:
- ``multiturn_mask`` is True only on model-generated tokens (Solver steps,
  Verifier outputs, repair instructions, and repaired segments). Injected
  prompts and tool observations are False and therefore excluded from the
  policy loss.
- Per-step verification results and reward-relevant positions are exported in
  ``non_tensor_batch['step_data']`` for the Agent0 reward manager.
"""

import asyncio
import numpy as np
import re
import json
import os
from typing import List, Dict, Any, Optional, Union

from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl.third_party.vllm import vllm_version

# Sandbox for tool execution. `sandbox.local_sandbox` talks to an HTTP
# sandbox service (SANDBOX_ENDPOINT); `sandbox.internal_sandbox` runs code in
# local subprocesses and requires no infrastructure.
try:
    if os.getenv("SANDBOX_ENDPOINT", None) is not None:
        from sandbox.local_sandbox import parallel_sandbox
    else:
        from sandbox.internal_sandbox import parallel_sandbox
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    parallel_sandbox = None


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from input token sequence"""
    non_pad = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if non_pad.numel() == 0:
        return []
    return prompt_token_ids[non_pad[0][0]:].tolist()


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def truncate_content(content: str, max_length: int = 2000) -> str:
    """Truncate content to max length keeping head and tail"""
    if len(content) <= max_length:
        return content
    return (
        content[: max_length // 2]
        + f"\n..._Truncated to {max_length} chars_...\n"
        + content[-max_length // 2:]
    )


class vLLMAgent0Rollout(vLLMRollout):
    """
    Agent0-VL rollout worker implementing the Self-Evolving Reasoning Cycle.

    Extends vLLMRollout with:
    - Multi-step reasoning with sandboxed tool execution
    - Step-by-step tool-grounded verification
    - Confidence-gated self-repair with segment re-sampling
    - Vision-language (multi-image) input handling
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # Agent0-VL specific configuration (read before super().__init__ so we
        # can extend max_model_len for the multi-turn budget).
        self.max_reasoning_steps = int(config.get('max_reasoning_steps', config.get('num_turns', 8)))
        self.repair_threshold = float(config.get('repair_threshold', 0.7))
        self.enable_tool_execution = bool(config.get('enable_tool_execution', True))
        self.enable_step_verification = bool(config.get('enable_verification',
                                                        config.get('enable_step_verification', True)))
        self.enable_self_repair = bool(config.get('enable_self_repair', True))
        self.max_obs_length = int(config.get('max_obs_length', 512))
        self.sandbox_timeout = int(config.get('sandbox_timeout', 10))
        self.max_repairs_per_trajectory = int(config.get('max_repairs_per_trajectory', 2))

        # The whole trajectory (all steps' Solver / tool-observation / Verifier /
        # repair tokens) is written into a single response buffer bounded by
        # ``max_total_response_length`` (or max_reasoning_steps * response_length).
        # The model context window therefore only needs to fit prompt + that
        # buffer, plus a small slack for the last in-flight generation. Sizing it
        # from an inflated per-step worst case would push max_model_len far above
        # what is ever used, blow up KV-cache memory, and trip vLLM's
        # ``max_num_batched_tokens < max_model_len`` chunked-prefill check.
        buffer_budget = int(config.get('max_total_response_length',
                                       self.max_reasoning_steps * config.response_length))
        buffer_budget = max(buffer_budget, config.response_length)
        extended_response_length = buffer_budget + self.max_obs_length + config.response_length

        original_max_model_len = config.get('max_model_len', None)
        if original_max_model_len is None:
            desired = config.prompt_length + extended_response_length
        else:
            desired = max(int(original_max_model_len), config.prompt_length + config.response_length)
        config.max_model_len = min(desired, model_hf_config.max_position_embeddings)

        # vLLM (and verl's base rollout) require max_num_batched_tokens >=
        # max_model_len when chunked prefill is enabled. The multi-turn budget
        # can push max_model_len above the configured token budget for large
        # ``max_total_response_length`` settings, so raise the budget to match
        # (a no-op for the default config where it is already large enough).
        if config.get('enable_chunked_prefill', True):
            mnbt = int(config.get('max_num_batched_tokens', 8192))
            if mnbt < int(config.max_model_len):
                config.max_num_batched_tokens = int(config.max_model_len)

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.tokenizer = tokenizer
        self._load_prompt_templates()

        # Response buffer capacity. Bounded separately from max_model_len so a
        # large position-embedding budget doesn't blow up the response tensor.
        default_total = min(
            int(self.config.max_model_len) - int(self.config.prompt_length),
            self.max_reasoning_steps * int(self.config.response_length),
        )
        self.max_total_length = max(
            int(self.config.response_length),
            int(config.get('max_total_response_length', default_total)),
        )

        print(f"Agent0-VL Rollout initialized: max_steps={self.max_reasoning_steps}, "
              f"repair_threshold={self.repair_threshold}, tools={self.enable_tool_execution}, "
              f"verification={self.enable_step_verification}, self_repair={self.enable_self_repair}, "
              f"sandbox_available={SANDBOX_AVAILABLE}, max_model_len={self.config.max_model_len}")

    # ------------------------------------------------------------------
    # Prompt templates (aligned with the paper's Appendix C system prompts)
    # ------------------------------------------------------------------
    def _load_prompt_templates(self):
        """Load prompt templates for the Verifier and Self-Repair roles.

        JSON braces are escaped as ``{{ }}`` for use with ``str.format``.
        """
        self.verifier_prompt_template = """Now switch to the Verifier role. Verify the reasoning step above using the available evidence.

Step to evaluate:
{step_content}

Tool outputs (if any):
{tool_outputs}

Principles:
- Ground verification on objective tool evidence.
- Penalize unsupported or inconsistent reasoning.
- High confidence requires agreement between tool and text.

Output exactly one JSON line:
{{"step_index": {step_index}, "score": <-1 to 1>, "confidence": <0 to 1>, "critique": "<at most 2 sentences>", "tool_check": <true|false>}}"""

        self.repair_prompt_template = """Now switch to the Self-Repair role. The Verifier flagged the reasoning step with low confidence.

Verification result:
- Score: {score}
- Confidence: {confidence}
- Critique: {critique}

Original step:
{original_step}

Propose a minimal local patch that fixes the specific error WITHOUT rewriting validated context.
Output exactly one JSON line, either:
{{"action": "PATCH", "target_step": {step_index}, "patch_type": "<text|code|tool_call|parameter>", "new_content": "<minimal replacement>", "justification": "<at most 2 sentences>"}}
or:
{{"action": "NO_CHANGE", "target_step": {step_index}, "reason": "<why repair is not warranted>"}}"""

        self.regenerate_prompt_template = """A repair instruction has been issued for the previous step:
{repair_instruction}

Switch back to the Solver role. Re-derive the corrected reasoning step applying this patch, then continue toward the final answer."""

    def _get_template_tokens(self, template_key: str, **kwargs) -> List[int]:
        """Convert prompt template to token ids with variable substitution"""
        template = getattr(self, f"{template_key}_prompt_template", "")
        try:
            formatted = template.format(**kwargs)
        except (KeyError, IndexError) as e:
            print(f"Warning: prompt template formatting failed ({e}); using raw template")
            formatted = template

        messages = [{"role": "user", "content": formatted}]
        chat_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Remove the duplicated system prompt that apply_chat_template inserts
        # for Qwen models (the conversation already has one).
        if chat_template.startswith("<|im_start|>system"):
            system_end = chat_template.find("<|im_end|>") + len("<|im_end|>")
            chat_template = chat_template[system_end:].lstrip()
        if chat_template.startswith("<|begin_of_sentence|>"):
            chat_template = chat_template[len("<|begin_of_sentence|>"):]

        chat_template = "\n\n" + chat_template
        return self.tokenizer.encode(chat_template, add_special_tokens=False)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _execute_code_in_sandbox(self, code_blocks: List[str]) -> List[Dict[str, Any]]:
        """Execute code blocks in the sandbox and return results"""
        if not code_blocks:
            return []
        if not SANDBOX_AVAILABLE:
            return [{"success": False, "stdout": "", "stderr": "Sandbox not available"}] * len(code_blocks)

        try:
            success_list, stdout_list, stderr_list = asyncio.run(
                parallel_sandbox(code_blocks, num_processes=min(256, len(code_blocks)),
                                 run_timeout=self.sandbox_timeout)
            )
            results = []
            for success, stdout, stderr in zip(success_list, stdout_list, stderr_list):
                results.append({
                    "success": bool(success),
                    "stdout": truncate_content(str(stdout), 512),
                    "stderr": truncate_content(str(stderr), 512) if stderr else ""
                })
            return results
        except Exception as e:
            return [{"success": False, "stdout": "", "stderr": str(e)}] * len(code_blocks)

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text.

        Tolerant of an optional ``python``/``py`` language tag and of code
        fences without a trailing newline before the closing ``` (so single-line
        snippets are captured too)."""
        pattern = r"```(?:python|py)?[ \t]*\r?\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from solver output"""
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, text)
        if matches:
            return matches[-1].strip()

        final_pattern = r"FINAL_ANSWER:\s*(.+?)(?:\n|$)"
        matches = re.findall(final_pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    # ------------------------------------------------------------------
    # Verification / repair parsing
    # ------------------------------------------------------------------
    def _parse_verification_output(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse verification JSON from verifier output"""
        json_pattern = r'\{[^{}]*"step_index"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            try:
                verification = json.loads(matches[-1])
                required = ['step_index', 'score', 'confidence']
                if all(k in verification for k in required):
                    verification['score'] = max(-1.0, min(1.0, float(verification.get('score', 0))))
                    verification['confidence'] = max(0.0, min(1.0, float(verification.get('confidence', 0.5))))
                    return verification
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return None

    def _parse_repair_instruction(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse repair instruction JSON from repair output"""
        json_pattern = r'\{[^{}]*"action"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            try:
                repair = json.loads(matches[-1])
                if repair.get('action') in ['PATCH', 'NO_CHANGE']:
                    return repair
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate multi-step SERC trajectories.

        Returns a DataProto with responses, attention_mask, position_ids,
        multiturn_mask (model-generated tokens only) and per-step metadata in
        ``non_tensor_batch['step_data']``.
        """
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info.get('eos_token_id', self.tokenizer.eos_token_id)

        batch_size = idx.size(0)
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        non_tensor_batch = dict(prompts.non_tensor_batch)

        # Sampling parameter overrides for this call
        if not do_sample:
            sample_kwargs = {'best_of': 1, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'temperature': 0, 'n': 1}
        elif is_validate:
            sample_kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,
            }
        else:
            sample_kwargs = {'n': 1}

        # Handle GRPO group sampling (n > 1): repeat everything upfront and
        # generate one sample per repeated row.
        n_repeat = self.config.n if (do_sample and self.config.n > 1 and not is_validate) else 1
        if n_repeat > 1:
            idx = _repeat_interleave(idx, n_repeat)
            attention_mask = _repeat_interleave(attention_mask, n_repeat)
            position_ids = _repeat_interleave(position_ids, n_repeat)
            batch_size = batch_size * n_repeat
            for key in list(non_tensor_batch.keys()):
                non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], n_repeat)

        # Build vLLM prompt token ids. For multimodal models the raw
        # (unexpanded) prompt ids must be used; the expanded ``input_ids`` are
        # only for FSDP training.
        if 'raw_prompt_ids' in non_tensor_batch:
            current_inputs = []
            for raw in non_tensor_batch['raw_prompt_ids']:
                current_inputs.append(list(raw) if not isinstance(raw, list) else list(raw))
        else:
            current_inputs = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        multi_modal_data = non_tensor_batch.get('multi_modal_data', None)

        # Response buffer
        max_total_length = self.max_total_length
        combined_response = torch.full((batch_size, max_total_length), self.pad_token_id,
                                       dtype=idx.dtype, device=idx.device)
        multiturn_mask = torch.zeros_like(combined_response, dtype=torch.bool)
        response_attention_mask = torch.zeros_like(combined_response, dtype=torch.bool)

        current_positions = [0] * batch_size

        # Per-sample bookkeeping
        step_data: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        verify_probs: List[List[float]] = [[] for _ in range(batch_size)]
        final_answers: List[Optional[str]] = [None] * batch_size
        num_steps = [0] * batch_size
        num_repairs = [0] * batch_size
        active_samples = list(range(batch_size))

        verify_kwargs = dict(sample_kwargs)

        def _append_tokens(global_idx: int, tokens: List[int], trainable: bool) -> int:
            """Append tokens to a sample's response buffer with bounds checks.

            Returns the number of tokens actually written."""
            if not tokens:
                return 0
            pos = current_positions[global_idx]
            room = max_total_length - pos
            if room <= 0:
                return 0
            tokens = tokens[:room]
            tensor = torch.tensor(tokens, dtype=combined_response.dtype, device=combined_response.device)
            combined_response[global_idx, pos:pos + len(tokens)] = tensor
            if trainable:
                multiturn_mask[global_idx, pos:pos + len(tokens)] = True
            response_attention_mask[global_idx, pos:pos + len(tokens)] = True
            current_positions[global_idx] = pos + len(tokens)
            return len(tokens)

        def _has_room(global_idx: int, needed: int) -> bool:
            """Whether the sample still has response-buffer and model-window room."""
            if current_positions[global_idx] + needed > max_total_length:
                return False
            ctx_len = len(current_inputs[global_idx]) + needed
            return ctx_len < int(self.config.max_model_len) - 8

        def _batched_generate(vllm_inputs: List[dict], gen_kwargs: dict) -> List:
            """Run vLLM generation with a per-call max_tokens that fits the window."""
            max_input_len = max(len(v['prompt_token_ids']) for v in vllm_inputs)
            room = int(self.config.max_model_len) - max_input_len - 8
            call_kwargs = dict(gen_kwargs)
            call_kwargs['max_tokens'] = max(16, min(self.config.response_length, room))
            with self.update_sampling_params(**call_kwargs):
                return self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)

        def _build_vllm_inputs(sample_indices: List[int]) -> List[dict]:
            inputs = []
            for g in sample_indices:
                entry = {'prompt_token_ids': list(current_inputs[g])}
                if multi_modal_data is not None:
                    entry['multi_modal_data'] = multi_modal_data[g]
                inputs.append(entry)
            return inputs

        # === SERC main loop ===
        for step_idx in range(self.max_reasoning_steps):
            # Drop samples that ran out of window/buffer room
            active_samples = [g for g in active_samples
                              if _has_room(g, min(256, self.config.response_length))]
            if not active_samples:
                break

            # --- Step 1: Solver generates a reasoning step ---
            solver_outputs = _batched_generate(_build_vllm_inputs(active_samples), sample_kwargs)

            solver_text_by_g: Dict[int, str] = {}
            tool_output_by_g: Dict[int, str] = {}
            tool_success_by_g: Dict[int, bool] = {}
            newly_completed = []

            solver_responses = [list(output.outputs[0].token_ids) for output in solver_outputs]

            for local_idx, global_idx in enumerate(active_samples):
                tokens = solver_responses[local_idx]
                # strip trailing eos for context growth, but keep it in buffer
                written = _append_tokens(global_idx, tokens, trainable=True)
                current_inputs[global_idx].extend(tokens[:written])
                num_steps[global_idx] += 1

                solver_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                solver_text_by_g[global_idx] = solver_text

                final_answer = self._extract_final_answer(solver_text)
                if final_answer is not None:
                    final_answers[global_idx] = final_answer
                    newly_completed.append(global_idx)

            # --- Step 2: Execute tools for still-active samples ---
            exec_candidates = [g for g in active_samples if g not in newly_completed]
            if self.enable_tool_execution and exec_candidates:
                code_blocks_batch = []
                samples_with_code = []
                for g in exec_candidates:
                    code_blocks = self._extract_code_blocks(solver_text_by_g.get(g, ""))
                    if code_blocks:
                        code_blocks_batch.extend(code_blocks)
                        samples_with_code.append((g, len(code_blocks)))

                if code_blocks_batch:
                    exec_results = self._execute_code_in_sandbox(code_blocks_batch)
                    result_idx = 0
                    for g, num_blocks in samples_with_code:
                        sample_results = exec_results[result_idx:result_idx + num_blocks]
                        result_idx += num_blocks

                        tool_output_text = "\n[Code Execution Result]\n"
                        any_success = False
                        for result in sample_results:
                            if result['stderr']:
                                tool_output_text += f"Error: {result['stderr']}\n"
                            elif result['stdout']:
                                tool_output_text += f"Output: {result['stdout']}\n"
                                any_success = any_success or result['success']
                            else:
                                tool_output_text += "No output\n"
                                any_success = any_success or result['success']

                        tool_success_by_g[g] = any_success
                        tool_output_by_g[g] = tool_output_text

                        tool_tokens = self.tokenizer.encode(tool_output_text, add_special_tokens=False)
                        tool_tokens = tool_tokens[:self.max_obs_length]
                        written = _append_tokens(g, tool_tokens, trainable=False)
                        current_inputs[g].extend(tool_tokens[:written])

                        # A tool result may print the boxed final answer
                        boxed = self._extract_final_answer(tool_output_text)
                        if boxed is not None and final_answers[g] is None:
                            final_answers[g] = boxed

            # --- Step 3: Verifier evaluates the step ---
            verification_by_g: Dict[int, Optional[Dict[str, Any]]] = {}
            verify_candidates = [g for g in active_samples
                                 if self.enable_step_verification and _has_room(g, 256)]
            if verify_candidates:
                for g in verify_candidates:
                    verify_prompt_tokens = self._get_template_tokens(
                        "verifier",
                        step_content=truncate_content(solver_text_by_g.get(g, ""), 2000),
                        tool_outputs=truncate_content(tool_output_by_g.get(g, "None"), 1000),
                        step_index=step_idx + 1,
                    )
                    written = _append_tokens(g, verify_prompt_tokens, trainable=False)
                    current_inputs[g].extend(verify_prompt_tokens[:written])

                verify_outputs = _batched_generate(_build_vllm_inputs(verify_candidates), verify_kwargs)

                for local_idx, g in enumerate(verify_candidates):
                    tokens = list(verify_outputs[local_idx].outputs[0].token_ids)
                    written = _append_tokens(g, tokens, trainable=True)
                    current_inputs[g].extend(tokens[:written])

                    verify_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                    verification = self._parse_verification_output(verify_text)
                    verification_by_g[g] = verification

                    verify_prob = verification.get('confidence', 0.5) if verification else 0.5
                    verify_probs[g].append(float(verify_prob))

            # --- Step 4: Confidence-gated self-repair ---
            repair_candidates = []
            if self.enable_self_repair:
                for g in verify_candidates:
                    if g in newly_completed:
                        continue
                    verification = verification_by_g.get(g)
                    confidence = verification.get('confidence', 0.5) if verification else 0.5
                    if (confidence < self.repair_threshold
                            and num_repairs[g] < self.max_repairs_per_trajectory
                            and _has_room(g, 512)):
                        repair_candidates.append(g)

            repaired_by_g: Dict[int, bool] = {}
            if repair_candidates:
                # 4a: generate repair instructions
                for g in repair_candidates:
                    verification = verification_by_g.get(g)
                    repair_prompt_tokens = self._get_template_tokens(
                        "repair",
                        score=verification.get('score', 0) if verification else 0,
                        confidence=verification.get('confidence', 0.5) if verification else 0.5,
                        critique=(verification.get('critique', 'Low confidence')
                                  if verification else 'Verification failed'),
                        original_step=truncate_content(solver_text_by_g.get(g, ""), 1000),
                        step_index=step_idx + 1,
                    )
                    written = _append_tokens(g, repair_prompt_tokens, trainable=False)
                    current_inputs[g].extend(repair_prompt_tokens[:written])

                repair_outputs = _batched_generate(_build_vllm_inputs(repair_candidates), sample_kwargs)

                regen_candidates = []
                for local_idx, g in enumerate(repair_candidates):
                    tokens = list(repair_outputs[local_idx].outputs[0].token_ids)
                    written = _append_tokens(g, tokens, trainable=True)
                    current_inputs[g].extend(tokens[:written])

                    repair_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                    repair = self._parse_repair_instruction(repair_text)
                    if repair is not None and repair.get('action') == 'PATCH' and _has_room(g, 512):
                        regen_candidates.append((g, repair))

                # 4b: Solver re-samples the corrected segment a'_t
                if regen_candidates:
                    for g, repair in regen_candidates:
                        regen_prompt_tokens = self._get_template_tokens(
                            "regenerate",
                            repair_instruction=truncate_content(json.dumps(repair, ensure_ascii=False), 800),
                        )
                        written = _append_tokens(g, regen_prompt_tokens, trainable=False)
                        current_inputs[g].extend(regen_prompt_tokens[:written])

                    regen_indices = [g for g, _ in regen_candidates]
                    regen_outputs = _batched_generate(_build_vllm_inputs(regen_indices), sample_kwargs)

                    for local_idx, g in enumerate(regen_indices):
                        tokens = list(regen_outputs[local_idx].outputs[0].token_ids)
                        written = _append_tokens(g, tokens, trainable=True)
                        current_inputs[g].extend(tokens[:written])

                        regen_text = self.tokenizer.decode(tokens[:written], skip_special_tokens=True)
                        solver_text_by_g[g] = regen_text  # corrected step replaces the old one
                        num_repairs[g] += 1
                        repaired_by_g[g] = True

                        final_answer = self._extract_final_answer(regen_text)
                        if final_answer is not None:
                            final_answers[g] = final_answer
                            if g not in newly_completed:
                                newly_completed.append(g)

            # --- Record per-step data for the reward manager ---
            for g in active_samples:
                verification = verification_by_g.get(g)
                step_data[g].append({
                    'step_index': step_idx + 1,
                    'score': float(verification.get('score', 0.0)) if verification else 0.0,
                    'confidence': float(verification.get('confidence', 0.5)) if verification else 0.5,
                    'verified': verification is not None,
                    'tool_used': g in tool_output_by_g,
                    'tool_success': bool(tool_success_by_g.get(g, False)),
                    'was_repaired': bool(repaired_by_g.get(g, False)),
                    'step_end_pos': current_positions[g],
                })

            # Remove completed samples
            active_samples = [g for g in active_samples if final_answers[g] is None]

        # === Finalize outputs ===
        max_response_len = max(max(current_positions), 1) if current_positions else 1
        # pad to multiple of 8 for kernel friendliness
        max_response_len = min(max_total_length, (max_response_len + 7) // 8 * 8)
        combined_response = combined_response[:, :max_response_len]
        multiturn_mask = multiturn_mask[:, :max_response_len]
        response_attention_mask = response_attention_mask[:, :max_response_len]

        seq = torch.cat([idx, combined_response], dim=-1)

        delta_position_id = torch.arange(1, combined_response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask.to(attention_mask.dtype)), dim=-1)

        batch = TensorDict({
            'prompts': idx,
            'responses': combined_response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'multiturn_mask': multiturn_mask,
        }, batch_size=batch_size)

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # Non-tensor outputs. Everything is already repeated to batch_size.
        max_verify_len = max((len(vp) for vp in verify_probs), default=0)
        max_verify_len = max(max_verify_len, 1)
        verify_probs_array = np.zeros((batch_size, max_verify_len), dtype=np.float32)
        for i, vp in enumerate(verify_probs):
            verify_probs_array[i, :len(vp)] = vp

        step_data_array = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            step_data_array[i] = step_data[i]
        final_answers_array = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            final_answers_array[i] = final_answers[i]

        output_non_tensor = {
            'step_data': step_data_array,
            'verify_probs': verify_probs_array,
            'num_steps': np.array(num_steps, dtype=np.int32),
            'num_repairs': np.array(num_repairs, dtype=np.int32),
            'final_answers': final_answers_array,
            'final_generation_step': np.array([max(n - 1, 0) for n in num_steps], dtype=np.int32),
        }
        # Preserve inputs the trainer needs back (e.g. multi_modal_inputs for
        # log-prob recomputation), already repeated to batch_size.
        for key, value in non_tensor_batch.items():
            if key in ('raw_prompt_ids', 'multi_modal_data'):
                continue
            if key not in output_non_tensor:
                output_non_tensor[key] = value

        return DataProto(batch=batch, non_tensor_batch=output_non_tensor)
